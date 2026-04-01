import os
import time
import copy
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

SEED = 42
VER = 501
DATA_DIR = Path(__file__).resolve().parent
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
NN_OUT_DIR = DATA_DIR / "nn"
TARGET_COL = "Churn"

FEATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_FEATS = [c for c in FEATS if c not in NUMS]
RARE_MIN_COUNT = 25
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# One place to tune all models
MODEL_CONFIGS = {
    "emb_mlp_snap": {
        "epochs": 20, "batch_size": 256, "lr": 2.5e-4, "weight_decay": 3e-4,
        "emb_dropout": 0.10, "dropout": 0.30, "hidden": [512, 256], "patience": 4
    },
    "resnet_tabular": {
        "epochs": 22, "batch_size": 256, "lr": 2.0e-4, "weight_decay": 2e-4,
        "width": 384, "depth": 4, "dropout": 0.20, "patience": 4
    },
    "ft_transformer": {
        "epochs": 18, "batch_size": 256, "lr": 1.5e-4, "weight_decay": 1e-4,
        "d_token": 64, "n_blocks": 3, "n_heads": 8, "ffn_mult": 4, "dropout": 0.15, "patience": 4
    },
    "dcnv2_tabular": {
        "epochs": 20, "batch_size": 256, "lr": 2.0e-4, "weight_decay": 2e-4,
        "cross_layers": 3, "mlp_hidden": [384, 192], "dropout": 0.20, "patience": 4
    },
    "tab_moe": {
        "epochs": 20, "batch_size": 256, "lr": 1.8e-4, "weight_decay": 2e-4,
        "n_experts": 4, "expert_hidden": [384, 192], "gate_hidden": 128, "dropout": 0.20, "patience": 4
    },
}

# ============================================================
# REPRO
# ============================================================
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


# ============================================================
# HELPERS
# ============================================================
def emb_dim_from_card(card: int) -> int:
    d = int(round(1.8 * (card ** 0.25)))
    return int(np.clip(d, 4, 64))


def make_vocab_maps(train_df: pd.DataFrame, cols: List[str]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    maps: Dict[str, Dict[str, int]] = {}
    sizes: Dict[str, int] = {}
    for c in cols:
        uniq = pd.Series(train_df[c].values).astype(str).unique().tolist()
        v2i = {v: i + 1 for i, v in enumerate(uniq)}  # 0 reserved for UNK
        maps[c] = v2i
        sizes[c] = len(v2i) + 1
    return maps, sizes


def encode_with_maps(df: pd.DataFrame, cols: List[str], maps: Dict[str, Dict[str, int]]) -> np.ndarray:
    x = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        s = pd.Series(df[c].values).astype(str).map(maps[c]).fillna(0).astype(np.int64).values
        x[:, j] = s
    return x


def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)

    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)

    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        x = pd.to_numeric(series_any, errors="coerce").astype(np.float32).values
        is_nan = np.isnan(x)
        is_rare = np.ones_like(x, dtype=np.int32)

        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if float(v) in frequent_set else 1

        x_snapped = x.copy()
        idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
        if idx_snap.size > 0 and frequent_vals.size > 0:
            v = x[idx_snap]
            pos = np.searchsorted(frequent_vals, v)
            pos = np.clip(pos, 0, len(frequent_vals) - 1)
            left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
            right = pos
            left_vals = frequent_vals[left]
            right_vals = frequent_vals[right]
            choose_right = np.abs(v - right_vals) <= np.abs(v - left_vals)
            nearest = np.where(choose_right, right_vals, left_vals)
            x_snapped[idx_snap] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform


# ============================================================
# DATASET
# ============================================================
class TabDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: Optional[np.ndarray] = None):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x_cat.shape[0]

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_cat[idx], self.x_num[idx]
        return self.x_cat[idx], self.x_num[idx], self.y[idx]


# ============================================================
# SHARED BUILDING BLOCKS
# ============================================================
class SmoothBCE(nn.Module):
    def __init__(self, eps: float = 0.02):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.eps) + 0.5 * self.eps
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)


class CatNumEncoder(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, emb_dropout: float = 0.1):
        super().__init__()
        self.emb_layers = nn.ModuleList()
        self.emb_out_dim = 0
        for card in cat_cardinals:
            d = emb_dim_from_card(card)
            emb = nn.Embedding(card, d)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.emb_layers.append(emb)
            self.emb_out_dim += d
        self.n_num = n_num
        self.emb_drop = nn.Dropout(emb_dropout)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.emb_layers)]
        if len(embs) > 0:
            z_cat = torch.cat(embs, dim=1)
            z_cat = self.emb_drop(z_cat)
            return torch.cat([z_cat, x_num], dim=1)
        return x_num


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ============================================================
# MODEL 1: BASELINE EMBEDDING MLP + NUMERIC SNAP PROXIES
# ============================================================
class EmbMLPSnap(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, hidden: List[int], emb_dropout: float, dropout: float):
        super().__init__()
        self.encoder = CatNumEncoder(cat_cardinals, n_num, emb_dropout=emb_dropout)
        in_dim = self.encoder.emb_out_dim + n_num
        layers: List[nn.Module] = []
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x_cat, x_num)
        return self.mlp(z).squeeze(1)


# ============================================================
# MODEL 2: RESNET-STYLE TABULAR NETWORK
# Inspired by strong ResNet-style baselines in tabular DL
# ============================================================
class ResNetTabular(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, width: int, depth: int, dropout: float):
        super().__init__()
        self.encoder = CatNumEncoder(cat_cardinals, n_num, emb_dropout=dropout * 0.5)
        in_dim = self.encoder.emb_out_dim + n_num
        self.stem = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.ReLU(inplace=True), nn.Linear(width, 1))

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x_cat, x_num)
        z = self.stem(z)
        z = self.blocks(z)
        return self.head(z).squeeze(1)


# ============================================================
# MODEL 3: FT-TRANSFORMER STYLE TOKENIZED TABULAR MODEL
# ============================================================
class NumericalTokenizer(nn.Module):
    def __init__(self, n_num: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_num, d_token))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        # [B, N] -> [B, N, D]
        return x_num.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, ffn_mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ffn_mult, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.ffn(self.ln2(x))
        return x


class FTTransformerTabular(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, d_token: int, n_blocks: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_token) for card in cat_cardinals])
        for emb in self.cat_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        self.num_tokenizer = NumericalTokenizer(n_num, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.Sequential(*[
            FTTransformerBlock(d_token=d_token, n_heads=n_heads, dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(inplace=True),
            nn.Linear(d_token, 1),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        tokens = []
        for j, emb in enumerate(self.cat_embeddings):
            tokens.append(emb(x_cat[:, j]).unsqueeze(1))
        num_tokens = self.num_tokenizer(x_num)
        if len(tokens) > 0:
            x = torch.cat(tokens + [num_tokens], dim=1)
        else:
            x = num_tokens
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.blocks(x)
        cls_out = x[:, 0]
        return self.head(cls_out).squeeze(1)


# ============================================================
# MODEL 4: DCN V2 STYLE CROSS + DEEP NETWORK
# ============================================================
class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.w(x) + self.b + x


class DCNv2Tabular(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, cross_layers: int, mlp_hidden: List[int], dropout: float):
        super().__init__()
        self.encoder = CatNumEncoder(cat_cardinals, n_num, emb_dropout=dropout * 0.5)
        dim = self.encoder.emb_out_dim + n_num
        self.cross = nn.ModuleList([CrossLayer(dim) for _ in range(cross_layers)])
        deep_layers: List[nn.Module] = []
        in_dim = dim
        for h in mlp_hidden:
            deep_layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        self.deep = nn.Sequential(*deep_layers)
        self.head = nn.Linear(dim + mlp_hidden[-1], 1)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        x0 = self.encoder(x_cat, x_num)
        xc = x0
        for layer in self.cross:
            xc = layer(x0, xc)
        xd = self.deep(x0)
        z = torch.cat([xc, xd], dim=1)
        return self.head(z).squeeze(1)


# ============================================================
# MODEL 5: TABULAR MIXTURE-OF-EXPERTS
# ============================================================
class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabMoE(nn.Module):
    def __init__(self, cat_cardinals: List[int], n_num: int, n_experts: int, expert_hidden: List[int], gate_hidden: int, dropout: float):
        super().__init__()
        self.encoder = CatNumEncoder(cat_cardinals, n_num, emb_dropout=dropout * 0.5)
        dim = self.encoder.emb_out_dim + n_num
        self.experts = nn.ModuleList([ExpertMLP(dim, expert_hidden, dropout) for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, n_experts),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x_cat, x_num)
        gate_logits = self.gate(z)
        gate_weights = torch.softmax(gate_logits, dim=1)
        expert_outs = torch.cat([expert(z) for expert in self.experts], dim=1)
        return (expert_outs * gate_weights).sum(dim=1)


# ============================================================
# MODEL FACTORY
# ============================================================
def build_model(model_name: str, cat_cardinals: List[int], n_num: int) -> nn.Module:
    cfg = MODEL_CONFIGS[model_name]
    if model_name == "emb_mlp_snap":
        return EmbMLPSnap(cat_cardinals, n_num, cfg["hidden"], cfg["emb_dropout"], cfg["dropout"])
    if model_name == "resnet_tabular":
        return ResNetTabular(cat_cardinals, n_num, cfg["width"], cfg["depth"], cfg["dropout"])
    if model_name == "ft_transformer":
        return FTTransformerTabular(cat_cardinals, n_num, cfg["d_token"], cfg["n_blocks"], cfg["n_heads"], cfg["ffn_mult"], cfg["dropout"])
    if model_name == "dcnv2_tabular":
        return DCNv2Tabular(cat_cardinals, n_num, cfg["cross_layers"], cfg["mlp_hidden"], cfg["dropout"])
    if model_name == "tab_moe":
        return TabMoE(cat_cardinals, n_num, cfg["n_experts"], cfg["expert_hidden"], cfg["gate_hidden"], cfg["dropout"])
    raise ValueError(f"Unknown model_name={model_name}")


# ============================================================
# TRAINING UTILITIES
# ============================================================
@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds = []
    for batch in loader:
        if len(batch) == 3:
            x_cat, x_num, _ = batch
        else:
            x_cat, x_num = batch
        x_cat = x_cat.to(DEVICE, non_blocking=True)
        x_num = x_num.to(DEVICE, non_blocking=True)
        logits = model(x_cat, x_num)
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(
    model_name: str,
    x_cat_tr: np.ndarray,
    x_num_tr: np.ndarray,
    y_tr: np.ndarray,
    x_cat_va: np.ndarray,
    x_num_va: np.ndarray,
    y_va: np.ndarray,
    cat_cardinals: List[int],
) -> Tuple[nn.Module, float]:
    cfg = MODEL_CONFIGS[model_name]
    model = build_model(model_name, cat_cardinals=cat_cardinals, n_num=x_num_tr.shape[1]).to(DEVICE)

    loss_fn = SmoothBCE(0.02)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.1)

    dl_tr = DataLoader(
        TabDataset(x_cat_tr, x_num_tr, y_tr),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    dl_va = DataLoader(
        TabDataset(x_cat_va, x_num_va, y_va),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    best_auc = -1.0
    best_state = None
    bad = 0

    print(f"  [{model_name}] train={len(x_cat_tr):,} valid={len(x_cat_va):,} batches={len(dl_tr):,}")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0
        for x_cat, x_num, yb in dl_tr:
            x_cat = x_cat.to(DEVICE, non_blocking=True)
            x_num = x_num.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x_cat, x_num)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            bs = x_cat.size(0)
            running_loss += loss.item() * bs
            seen += bs

        sched.step()
        train_loss = running_loss / max(1, seen)
        p_va = predict_proba(model, dl_va)
        auc = roc_auc_score(y_va, p_va)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"     epoch {epoch:02d} | lr {lr:.2e} | loss {train_loss:.5f} | val_auc {auc:.6f} | {dt:.1f}s")

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"     early stop @ epoch={epoch} | best_auc={best_auc:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc


# ============================================================
# PREPROCESSING
# ============================================================
def build_fold_arrays(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], List[str]]:
    tr_num_cat: Dict[str, np.ndarray] = {}
    va_num_cat: Dict[str, np.ndarray] = {}
    te_num_cat: Dict[str, np.ndarray] = {}
    tr_rare: Dict[str, np.ndarray] = {}
    va_rare: Dict[str, np.ndarray] = {}
    te_rare: Dict[str, np.ndarray] = {}

    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr_df[col])
        va_snap, va_israre = snapper(va_df[col])
        te_snap, te_israre = snapper(te_df[col])

        tr_num_cat[f"{col}__cat"] = tr_snap
        va_num_cat[f"{col}__cat"] = va_snap
        te_num_cat[f"{col}__cat"] = te_snap

        tr_rare[f"{col}__is_rare"] = tr_israre
        va_rare[f"{col}__is_rare"] = va_israre
        te_rare[f"{col}__is_rare"] = te_israre

    tr_cat_df = tr_df[CAT_FEATS].copy()
    va_cat_df = va_df[CAT_FEATS].copy()
    te_cat_df = te_df[CAT_FEATS].copy()

    for col in NUMS:
        tr_cat_df[f"{col}__cat"] = pd.Series(tr_num_cat[f"{col}__cat"]).astype(str).values
        va_cat_df[f"{col}__cat"] = pd.Series(va_num_cat[f"{col}__cat"]).astype(str).values
        te_cat_df[f"{col}__cat"] = pd.Series(te_num_cat[f"{col}__cat"]).astype(str).values

        tr_cat_df[f"{col}__is_rare"] = tr_rare[f"{col}__is_rare"]
        va_cat_df[f"{col}__is_rare"] = va_rare[f"{col}__is_rare"]
        te_cat_df[f"{col}__is_rare"] = te_rare[f"{col}__is_rare"]

    cat_all = list(tr_cat_df.columns)
    maps, sizes = make_vocab_maps(tr_cat_df, cat_all)
    cat_cardinals = [sizes[c] for c in cat_all]

    x_cat_tr = encode_with_maps(tr_cat_df, cat_all, maps)
    x_cat_va = encode_with_maps(va_cat_df, cat_all, maps)
    x_cat_te = encode_with_maps(te_cat_df, cat_all, maps)

    scaler = StandardScaler()
    x_num_tr = scaler.fit_transform(tr_df[NUMS].astype(np.float32).values).astype(np.float32)
    x_num_va = scaler.transform(va_df[NUMS].astype(np.float32).values).astype(np.float32)
    x_num_te = scaler.transform(te_df[NUMS].astype(np.float32).values).astype(np.float32)

    return x_cat_tr, x_num_tr, x_cat_va, x_num_va, x_cat_te, x_num_te, cat_cardinals, cat_all


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def run_model_cv(model_name: str, train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray) -> None:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train), dtype=np.float32)
    pred_test = np.zeros(len(test), dtype=np.float32)
    fold_aucs: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y), start=1):
        print("\n" + "=" * 80)
        print(f"MODEL={model_name} | FOLD {fold}/{N_SPLITS}")
        print("=" * 80)
        tr_df = train.iloc[tr_idx].reset_index(drop=True)
        va_df = train.iloc[va_idx].reset_index(drop=True)

        x_cat_tr, x_num_tr, x_cat_va, x_num_va, x_cat_te, x_num_te, cat_cardinals, _ = build_fold_arrays(tr_df, va_df, test)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model, best_auc = train_one_fold(
            model_name=model_name,
            x_cat_tr=x_cat_tr,
            x_num_tr=x_num_tr,
            y_tr=y_tr,
            x_cat_va=x_cat_va,
            x_num_va=x_num_va,
            y_va=y_va,
            cat_cardinals=cat_cardinals,
        )
        fold_aucs.append(best_auc)

        dl_va = DataLoader(TabDataset(x_cat_va, x_num_va, y_va), batch_size=MODEL_CONFIGS[model_name]["batch_size"], shuffle=False, num_workers=0)
        dl_te = DataLoader(TabDataset(x_cat_te, x_num_te, None), batch_size=MODEL_CONFIGS[model_name]["batch_size"], shuffle=False, num_workers=0)

        p_va = predict_proba(model, dl_va)
        p_te = predict_proba(model, dl_te)
        oof[va_idx] = p_va.astype(np.float32)
        pred_test += p_te.astype(np.float32) / N_SPLITS

        print(f"[fold {fold}] best_auc={best_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    print("\n" + "-" * 80)
    print(f"{model_name} fold aucs: {[round(a, 6) for a in fold_aucs]}")
    print(f"{model_name} OOF CV AUC: {cv_auc:.6f}")
    print("-" * 80)

    oof_path = NN_OUT_DIR / f"oof_{model_name}_v{VER}.npy"
    pred_path = NN_OUT_DIR / f"pred_{model_name}_v{VER}.npy"
    np.save(oof_path, oof)
    np.save(pred_path, pred_test)
    print(f"saved: {oof_path}")
    print(f"saved: {pred_path}")


# ============================================================
# ENTRYPOINT
# ============================================================
def main() -> None:
    seed_everything(SEED)
    NN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print(f"train={train.shape} | test={test.shape} | device={DEVICE}")

    y = train[TARGET_COL].values
    if y.dtype == object:
        y = pd.Series(y).map({"Yes": 1, "No": 0}).values
    y = y.astype(np.float32)

    for model_name in MODEL_CONFIGS:
        run_model_cv(model_name=model_name, train=train, test=test, y=y)

    print("\nDone. Generated 5 diverse tabular NN OOF/test prediction pairs.")


if __name__ == "__main__":
    main()
