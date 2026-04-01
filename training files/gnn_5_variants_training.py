import gc
import math
import os
import pathlib
import random
import resource
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dropout_edge

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

BASE_NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
DROP_COLS = ["customerID", "id"]
TARGET = "Churn"

DATA_DIR = pathlib.Path(__file__).resolve().parent
GNN_OUT_DIR = DATA_DIR / "gnn"


@dataclass
class VariantConfig:
    name: str
    graph_mode: str
    node_mode: str
    k: int = 12
    hidden_dim: int = 128
    dropout: float = 0.20
    lr: float = 1e-3
    weight_decay: float = 3e-4
    epochs: int = 220
    patience: int = 30
    edge_dropout: float = 0.0
    use_cross_features: bool = False
    graph_numeric_multiplier: float = 3.0
    rf_estimators: int = 120
    rf_depth: int = 7
    seed_offset: int = 0


@dataclass
class Config:
    train_csv: pathlib.Path = field(default_factory=lambda: DATA_DIR / "train.csv")
    test_csv: pathlib.Path = field(default_factory=lambda: DATA_DIR / "test.csv")
    seed: int = 42
    n_folds: int = 5
    variants: List[VariantConfig] = field(default_factory=list)
    # Memory / device (override via env: GNN_DEVICE, GNN_GPU_FRACTION, GNN_MAX_RAM_GB, GNN_MAX_GRAPH_EDGES, GNN_NO_AMP, GNN_NO_CHECKPOINT)
    device: str = ""
    force_cpu: bool = False
    gpu_memory_fraction: float = 0.88
    use_amp: bool = True
    gradient_checkpointing: bool = True
    max_graph_edges: Optional[int] = None
    max_process_ram_gb: Optional[float] = None
    max_compute_threads: Optional[int] = None
    set_cuda_expandable_segments: bool = True


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_env_float(key: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_truthy(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "y", "on")


def apply_env_to_config(cfg: Config) -> None:
    if _env_truthy("GNN_CPU") or _env_truthy("GNN_CPU_ONLY"):
        cfg.force_cpu = True
    if _env_truthy("GNN_NO_AMP"):
        cfg.use_amp = False
    if _env_truthy("GNN_NO_CHECKPOINT"):
        cfg.gradient_checkpointing = False
    gf = _parse_env_float("GNN_GPU_FRACTION")
    if gf is not None:
        cfg.gpu_memory_fraction = gf
    mr = _parse_env_float("GNN_MAX_RAM_GB")
    if mr is not None:
        cfg.max_process_ram_gb = mr
    me = _parse_env_int("GNN_MAX_GRAPH_EDGES")
    if me is not None:
        cfg.max_graph_edges = me
    mt = _parse_env_int("GNN_MAX_THREADS")
    if mt is not None:
        cfg.max_compute_threads = mt
    dev = os.environ.get("GNN_DEVICE", "").strip().lower()
    if dev in ("cpu", "cuda"):
        cfg.device = dev


def resolve_device(cfg: Config) -> str:
    if cfg.force_cpu:
        return "cpu"
    if cfg.device in ("cpu", "cuda"):
        return cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_runtime_memory(cfg: Config) -> str:
    if cfg.set_cuda_expandable_segments:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.max_compute_threads is not None and cfg.max_compute_threads > 0:
        torch.set_num_threads(int(cfg.max_compute_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(cfg.max_compute_threads))))
        except RuntimeError:
            pass
    if cfg.max_process_ram_gb is not None and cfg.max_process_ram_gb > 0 and hasattr(resource, "RLIMIT_AS"):
        limit = int(cfg.max_process_ram_gb * (1024**3))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):
            pass
    device = resolve_device(cfg)
    if device == "cuda" and torch.cuda.is_available():
        frac = float(cfg.gpu_memory_fraction)
        frac = max(0.05, min(1.0, frac))
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        torch.cuda.empty_cache()
    return device


def maybe_subsample_edges(edge_index: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    if max_edges <= 0 or edge_index.shape[1] <= max_edges:
        return edge_index
    rng = np.random.default_rng(seed)
    pick = rng.choice(edge_index.shape[1], size=max_edges, replace=False)
    pick.sort()
    return edge_index[:, pick].astype(np.int64)


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def emb_dim_from_card(card: int) -> int:
    return int(np.clip(round(1.8 * (max(card, 2) ** 0.25)), 4, 24))


# ============================================================
# Data preparation
# ============================================================

def clean_totalcharges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
        df["TotalCharges"] = df["TotalCharges"].replace("", np.nan)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def load_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(cfg.train_csv)
    test = pd.read_csv(cfg.test_csv)

    for col in DROP_COLS:
        if col in train.columns:
            train = train.drop(columns=col)
        if col in test.columns:
            test = test.drop(columns=col)

    train = clean_totalcharges(train)
    test = clean_totalcharges(test)

    y = (
        train[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(np.float32).values
    )
    return train, test, y


def preprocess_base(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train.copy(), test.copy()

    for df in (tr, te):
        for c in BASE_NUMS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        for c in BASE_CATS:
            df[c] = df[c].astype(str).fillna("missing").str.strip()

    for c in BASE_NUMS:
        med = float(np.nanmedian(tr[c].values.astype(np.float32)))
        if not np.isfinite(med):
            med = 0.0
        tr[c] = tr[c].fillna(med).astype(np.float32)
        te[c] = te[c].fillna(med).astype(np.float32)

    # Strong tabular features for node attributes
    for df in (tr, te):
        tenure_safe = np.maximum(df["tenure"].values.astype(np.float32), 0.0)
        monthly = np.maximum(df["MonthlyCharges"].values.astype(np.float32), 0.0)
        total = np.maximum(df["TotalCharges"].values.astype(np.float32), 0.0)

        df["avg_charge_per_month"] = total / np.maximum(tenure_safe, 1.0)
        df["charge_gap"] = total - monthly * tenure_safe
        df["has_internet"] = (df["InternetService"].astype(str) != "No").astype(np.int8)
        df["service_count"] = (
            (df["OnlineSecurity"].astype(str) != "No").astype(np.int8)
            + (df["OnlineBackup"].astype(str) != "No").astype(np.int8)
            + (df["DeviceProtection"].astype(str) != "No").astype(np.int8)
            + (df["TechSupport"].astype(str) != "No").astype(np.int8)
            + (df["StreamingTV"].astype(str) != "No").astype(np.int8)
            + (df["StreamingMovies"].astype(str) != "No").astype(np.int8)
        ).astype(np.int16)
        df["is_month_to_month"] = (df["Contract"].astype(str) == "Month-to-month").astype(np.int8)
        df["autopay"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False, regex=False).astype(np.int8)

    return tr, te


def add_freq_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_df.copy(), test_df.copy()
    for col in cat_cols:
        freq = tr[col].value_counts(dropna=False).to_dict()
        tr[f"{col}__freq"] = tr[col].map(freq).fillna(0).astype(np.float32)
        te[f"{col}__freq"] = te[col].map(freq).fillna(0).astype(np.float32)
    return tr, te


def add_cross_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr, te = train_df.copy(), test_df.copy()
    cross_pairs = [
        ("Contract", "InternetService"),
        ("PaymentMethod", "PaperlessBilling"),
        ("OnlineSecurity", "TechSupport"),
        ("StreamingTV", "StreamingMovies"),
    ]
    for a, b in cross_pairs:
        cname = f"{a}__X__{b}"
        tr[cname] = tr[a].astype(str) + "__" + tr[b].astype(str)
        te[cname] = te[a].astype(str) + "__" + te[b].astype(str)
    return tr, te


def encode_categories(train_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    tr_codes, te_codes, cards = [], [], []
    for col in cat_cols:
        all_vals = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0, ignore_index=True)
        uniq = all_vals.unique().tolist()
        mapping = {v: i for i, v in enumerate(uniq)}
        tr_codes.append(train_df[col].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        te_codes.append(test_df[col].astype(str).map(mapping).fillna(0).astype(np.int64).values)
        cards.append(len(mapping))
    xtr = np.stack(tr_codes, axis=1) if tr_codes else np.zeros((len(train_df), 0), dtype=np.int64)
    xte = np.stack(te_codes, axis=1) if te_codes else np.zeros((len(test_df), 0), dtype=np.int64)
    return xtr, xte, cards


# ============================================================
# Node feature views
# ============================================================

def build_node_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: VariantConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    base_num_cols = [
        "tenure", "MonthlyCharges", "TotalCharges",
        "avg_charge_per_month", "charge_gap", "service_count",
        "has_internet", "is_month_to_month", "autopay",
    ] + [f"{c}__freq" for c in BASE_CATS]

    if variant.use_cross_features:
        base_num_cols += [f"{c}__freq" for c in [
            "Contract__X__InternetService",
            "PaymentMethod__X__PaperlessBilling",
            "OnlineSecurity__X__TechSupport",
            "StreamingTV__X__StreamingMovies",
        ]]

    tr = train_df[base_num_cols].astype(np.float32).copy()
    te = test_df[base_num_cols].astype(np.float32).copy()

    # Different numerical views create diversity in node geometry.
    if variant.node_mode == "standard":
        scaler = StandardScaler()
        xtr = scaler.fit_transform(tr.values).astype(np.float32)
        xte = scaler.transform(te.values).astype(np.float32)

    elif variant.node_mode == "rankgauss":
        qt = QuantileTransformer(n_quantiles=min(256, len(tr)), output_distribution="normal", random_state=123)
        xtr = qt.fit_transform(tr.values).astype(np.float32)
        xte = qt.transform(te.values).astype(np.float32)

    elif variant.node_mode == "log_interact":
        tr2 = tr.copy()
        te2 = te.copy()
        for col in ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month"]:
            tr2[f"log1p_{col}"] = np.log1p(np.maximum(tr2[col].values, 0.0)).astype(np.float32)
            te2[f"log1p_{col}"] = np.log1p(np.maximum(te2[col].values, 0.0)).astype(np.float32)
        tr2["mc_x_tenure"] = (tr2["MonthlyCharges"] * tr2["tenure"]).astype(np.float32)
        te2["mc_x_tenure"] = (te2["MonthlyCharges"] * te2["tenure"]).astype(np.float32)
        tr2["mc_div_total"] = (tr2["MonthlyCharges"] / np.maximum(tr2["TotalCharges"], 1.0)).astype(np.float32)
        te2["mc_div_total"] = (te2["MonthlyCharges"] / np.maximum(te2["TotalCharges"], 1.0)).astype(np.float32)
        scaler = StandardScaler()
        xtr = scaler.fit_transform(tr2.values).astype(np.float32)
        xte = scaler.transform(te2.values).astype(np.float32)

    elif variant.node_mode == "quantile_bins":
        tr2 = tr.copy()
        te2 = te.copy()
        for col in ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month"]:
            q = pd.qcut(tr2[col].rank(method="first"), q=min(12, tr2.shape[0] // 100 + 2), labels=False, duplicates="drop")
            tr2[f"{col}__qbin"] = q.astype(np.float32)
            bins = np.quantile(tr2[col], np.linspace(0, 1, 9))
            bins = np.unique(bins)
            if len(bins) > 1:
                te2[f"{col}__qbin"] = np.digitize(te2[col], bins[1:-1], right=True).astype(np.float32)
            else:
                te2[f"{col}__qbin"] = 0.0
        scaler = StandardScaler()
        xtr = scaler.fit_transform(tr2.values).astype(np.float32)
        xte = scaler.transform(te2.values).astype(np.float32)

    else:
        raise ValueError(f"Unknown node_mode={variant.node_mode}")

    return xtr, xte


# ============================================================
# Graph construction
# ============================================================

def build_knn_edges(x: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    nnm = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nnm.fit(x)
    idx = nnm.kneighbors(x, return_distance=False)[:, 1:]

    src = np.repeat(np.arange(x.shape[0]), idx.shape[1])
    dst = idx.reshape(-1)
    edges = np.stack([src, dst], axis=0)
    rev = np.stack([dst, src], axis=0)
    edge_index = np.concatenate([edges, rev], axis=1)
    return unique_edges(edge_index)


def unique_edges(edge_index: np.ndarray) -> np.ndarray:
    key = edge_index[0].astype(np.int64) * (edge_index.shape[1] + 1_000_000) + edge_index[1].astype(np.int64)
    keep = np.unique(key, return_index=True)[1]
    keep = np.sort(keep)
    return edge_index[:, keep].astype(np.int64)


def build_cat_ohe_graph_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, num_weight: float) -> np.ndarray:
    ohe = make_ohe()
    all_cat = pd.concat([train_df[BASE_CATS].astype(str), test_df[BASE_CATS].astype(str)], axis=0, ignore_index=True)
    x_cat = ohe.fit_transform(all_cat).astype(np.float32)

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month", "service_count"]
    tr_num = train_df[num_cols].astype(np.float32).values
    te_num = test_df[num_cols].astype(np.float32).values
    scaler = StandardScaler()
    x_num = np.vstack([scaler.fit_transform(tr_num), scaler.transform(te_num)]).astype(np.float32)
    x_num *= num_weight
    return np.concatenate([x_cat, x_num], axis=1).astype(np.float32)


def build_freq_graph_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame, num_weight: float) -> np.ndarray:
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month", "charge_gap"] + [f"{c}__freq" for c in BASE_CATS]
    tr = train_df[cols].astype(np.float32).values
    te = test_df[cols].astype(np.float32).values
    qt = QuantileTransformer(n_quantiles=min(256, len(train_df)), output_distribution="normal", random_state=321)
    x = np.vstack([qt.fit_transform(tr), qt.transform(te)]).astype(np.float32)
    x[:, :5] *= num_weight
    return x


def build_quantile_ohe_graph_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    tr, te = train_df.copy(), test_df.copy()
    qcols = []
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month"]:
        qcol = f"{col}__q"
        qcols.append(qcol)
        try:
            tr[qcol] = pd.qcut(tr[col].rank(method="first"), q=min(10, max(3, len(tr) // 300)), duplicates="drop").astype(str)
        except ValueError:
            tr[qcol] = "0"
        bins = np.unique(np.quantile(tr[col], np.linspace(0, 1, 7)))
        te[qcol] = np.digitize(te[col], bins[1:-1], right=True).astype(str) if len(bins) > 1 else "0"

    ohe = make_ohe()
    all_df = pd.concat([
        tr[BASE_CATS + qcols].astype(str),
        te[BASE_CATS + qcols].astype(str),
    ], axis=0, ignore_index=True)
    return ohe.fit_transform(all_df).astype(np.float32)


def build_hybrid_union_edges(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int) -> np.ndarray:
    all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True).reset_index(drop=True)
    n = len(all_df)
    buckets: Dict[str, Dict[str, List[int]]] = {
        "Contract": {},
        "InternetService": {},
        "PaymentMethod": {},
    }
    for col in buckets:
        vals = all_df[col].astype(str).tolist()
        mp: Dict[str, List[int]] = {}
        for i, v in enumerate(vals):
            mp.setdefault(v, []).append(i)
        buckets[col] = mp

    src_list: List[int] = []
    dst_list: List[int] = []
    for i in range(n):
        nbrs = set()
        for col, mp in buckets.items():
            v = str(all_df.loc[i, col])
            cand = mp.get(v, [])
            if len(cand) > 1:
                picked = cand[: min(4, len(cand))]
                nbrs.update(p for p in picked if p != i)
        for j in nbrs:
            src_list.append(i)
            dst_list.append(j)
            src_list.append(j)
            dst_list.append(i)

    graph_x = build_freq_graph_matrix(train_df, test_df, num_weight=2.0)
    knn_edge = build_knn_edges(graph_x, k=k, metric="cosine")
    if src_list:
        extra = np.stack([np.array(src_list), np.array(dst_list)], axis=0)
        return unique_edges(np.concatenate([knn_edge, extra], axis=1))
    return knn_edge


def build_rf_proximity_edges(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: VariantConfig) -> np.ndarray:
    x = build_cat_ohe_graph_matrix(train_df, test_df, num_weight=variant.graph_numeric_multiplier)
    rte = RandomTreesEmbedding(
        n_estimators=variant.rf_estimators,
        max_depth=variant.rf_depth,
        random_state=777 + variant.seed_offset,
        sparse_output=True,
    )
    z = rte.fit_transform(x)
    nnm = NearestNeighbors(n_neighbors=variant.k + 1, metric="cosine")
    nnm.fit(z)
    idx = nnm.kneighbors(z, return_distance=False)[:, 1:]
    src = np.repeat(np.arange(z.shape[0]), idx.shape[1])
    dst = idx.reshape(-1)
    edge_index = np.concatenate([
        np.stack([src, dst], axis=0),
        np.stack([dst, src], axis=0),
    ], axis=1)
    return unique_edges(edge_index)


def build_graph(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: VariantConfig) -> np.ndarray:
    if variant.graph_mode == "base_knn_ohe":
        x = build_cat_ohe_graph_matrix(train_df, test_df, num_weight=variant.graph_numeric_multiplier)
        return build_knn_edges(x, k=variant.k, metric="euclidean")
    if variant.graph_mode == "freq_cosine":
        x = build_freq_graph_matrix(train_df, test_df, num_weight=variant.graph_numeric_multiplier)
        return build_knn_edges(x, k=variant.k, metric="cosine")
    if variant.graph_mode == "quantile_ohe":
        x = build_quantile_ohe_graph_matrix(train_df, test_df)
        return build_knn_edges(x, k=variant.k, metric="hamming")
    if variant.graph_mode == "hybrid_union":
        return build_hybrid_union_edges(train_df, test_df, k=variant.k)
    if variant.graph_mode == "rf_proximity":
        return build_rf_proximity_edges(train_df, test_df, variant)
    raise ValueError(f"Unknown graph_mode={variant.graph_mode}")


# ============================================================
# Model
# ============================================================

class CatEmbed(nn.Module):
    def __init__(self, cardinals: Sequence[int]):
        super().__init__()
        self.embs = nn.ModuleList()
        out_dim = 0
        for card in cardinals:
            dim = emb_dim_from_card(card)
            self.embs.append(nn.Embedding(max(2, int(card)), dim))
            out_dim += dim
        self.out_dim = out_dim
        for emb in self.embs:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.size(1) == 0:
            return torch.zeros((x_cat.size(0), 0), device=x_cat.device, dtype=torch.float32)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        num_in: int,
        cat_cards: Sequence[int],
        hidden_dim: int,
        dropout: float,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.cat = CatEmbed(cat_cards)
        in_dim = num_in + self.cat.out_dim
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data, edge_index: torch.Tensor = None) -> torch.Tensor:
        if edge_index is None:
            edge_index = data.edge_index
        zc = self.cat(data.x_cat)
        x = torch.cat([data.x_num, zc], dim=1)
        x = F.relu(self.lin0(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.use_checkpoint and self.training:
            def block1(x_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
                h = self.conv1(x_in, ei)
                h = F.relu(self.norm1(h))
                return F.dropout(h, p=self.dropout, training=True)

            h1 = torch.utils.checkpoint.checkpoint(block1, x, edge_index, use_reentrant=False)
        else:
            h1 = self.conv1(x, edge_index)
            h1 = F.relu(self.norm1(h1))
            h1 = F.dropout(h1, p=self.dropout, training=self.training)
        x = x + 0.5 * h1

        if self.use_checkpoint and self.training:
            def block2(x_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
                h = self.conv2(x_in, ei)
                h = F.relu(self.norm2(h))
                return F.dropout(h, p=self.dropout, training=True)

            h2 = torch.utils.checkpoint.checkpoint(block2, x, edge_index, use_reentrant=False)
        else:
            h2 = self.conv2(x, edge_index)
            h2 = F.relu(self.norm2(h2))
            h2 = F.dropout(h2, p=self.dropout, training=self.training)
        x = x + 0.5 * h2
        return self.head(x).squeeze(-1)


# ============================================================
# Training
# ============================================================

def make_masks(n_train: int, n_test: int, tr_idx: np.ndarray, va_idx: np.ndarray) -> Dict[str, torch.Tensor]:
    n_all = n_train + n_test
    train_mask = torch.zeros(n_all, dtype=torch.bool)
    val_mask = torch.zeros(n_all, dtype=torch.bool)
    test_mask = torch.zeros(n_all, dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[va_idx] = True
    test_mask[n_train:] = True
    return {"train": train_mask, "val": val_mask, "test": test_mask}


def build_pyg_data(x_num: np.ndarray, x_cat: np.ndarray, y_all: np.ndarray, edge_index: np.ndarray) -> Data:
    data = Data(
        x_num=torch.tensor(x_num, dtype=torch.float32),
        x_cat=torch.tensor(x_cat, dtype=torch.long),
        y=torch.tensor(y_all, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    return data


def train_one_fold(
    data: Data,
    masks: Dict[str, torch.Tensor],
    y_train: np.ndarray,
    variant: VariantConfig,
    cat_cards: Sequence[int],
    seed: int,
    device: str,
    use_amp: bool,
    gradient_checkpointing: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    set_seed(seed)
    amp_enabled = bool(use_amp and device == "cuda" and torch.cuda.is_available())
    data = data.to(device)
    masks = {k: v.to(device) for k, v in masks.items()}

    model = GraphSAGEClassifier(
        num_in=data.x_num.size(1),
        cat_cards=cat_cards,
        hidden_dim=variant.hidden_dim,
        dropout=variant.dropout,
        use_checkpoint=gradient_checkpointing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=variant.lr, weight_decay=variant.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0

    autocast_device = "cuda" if device == "cuda" else "cpu"

    for epoch in range(1, variant.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        edge_index = data.edge_index
        if variant.edge_dropout > 0:
            edge_index, _ = dropout_edge(edge_index, p=variant.edge_dropout, training=True)

        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            logits = model(data, edge_index=edge_index)
            loss = loss_fn(logits[masks["train"]], data.y[masks["train"]])
        scaler.scale(loss).backward()
        if amp_enabled:
            scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        model.eval()
        with torch.inference_mode():
            with torch.amp.autocast(autocast_device, enabled=amp_enabled):
                val_logits = model(data)
            val_prob = torch.sigmoid(val_logits[masks["val"]]).detach().cpu().numpy()
            val_idx = np.where(masks["val"].detach().cpu().numpy())[0]
            val_true = y_train[val_idx]
            auc = roc_auc_score(val_true, val_prob)

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= variant.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast(autocast_device, enabled=amp_enabled):
            logits = model(data)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    val_mask_np = masks["val"].detach().cpu().numpy()
    test_mask_np = masks["test"].detach().cpu().numpy()
    val_pred = probs[val_mask_np]
    test_pred = probs[test_mask_np]

    del model, optimizer, data, scaler
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_pred, test_pred, float(best_auc)


def prepare_variant_views(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    y: np.ndarray,
    variant: VariantConfig,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[int], np.ndarray]:
    tr, te = add_freq_features(train_base, test_base, BASE_CATS)
    if variant.use_cross_features:
        tr, te = add_cross_features(tr, te)
        tr, te = add_freq_features(tr, te, [
            "Contract__X__InternetService",
            "PaymentMethod__X__PaperlessBilling",
            "OnlineSecurity__X__TechSupport",
            "StreamingTV__X__StreamingMovies",
        ])

    x_num_tr, x_num_te = build_node_numeric_features(tr, te, variant)
    x_cat_tr, x_cat_te, cat_cards = encode_categories(tr, te, BASE_CATS)
    edge_index = build_graph(tr, te, variant)

    x_num_all = np.vstack([x_num_tr, x_num_te]).astype(np.float32)
    x_cat_all = np.vstack([x_cat_tr, x_cat_te]).astype(np.int64)
    return tr, te, x_num_all, x_cat_all, cat_cards, edge_index


def run_variant(
    cfg: Config,
    variant: VariantConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    print(f"\n{'=' * 100}")
    print(f"Variant: {variant.name}")
    print(f"graph_mode={variant.graph_mode} | node_mode={variant.node_mode} | cross_features={variant.use_cross_features}")
    print(f"{'=' * 100}")

    train_base, test_base = preprocess_base(train, test)
    _, _, x_num_all, x_cat_all, cat_cards, edge_index = prepare_variant_views(train_base, test_base, y, variant, cfg)
    if cfg.max_graph_edges is not None:
        edge_index = maybe_subsample_edges(edge_index, cfg.max_graph_edges, cfg.seed + variant.seed_offset)
        print(f"edge_index capped to {edge_index.shape[1]} edges (max_graph_edges={cfg.max_graph_edges})")

    n_train = len(train)
    n_test = len(test)
    y_all = np.concatenate([y.astype(np.float32), np.full(n_test, -1, dtype=np.float32)])

    oof = np.zeros(n_train, dtype=np.float32)
    pred_test = np.zeros(n_test, dtype=np.float32)

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed + variant.seed_offset)
    fold_scores: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y), start=1):
        print(f"[Fold {fold}/{cfg.n_folds}] training ...")
        masks = make_masks(n_train, n_test, tr_idx, va_idx)
        data = build_pyg_data(x_num_all, x_cat_all, y_all, edge_index)
        val_pred, test_pred, fold_auc = train_one_fold(
            data=data,
            masks=masks,
            y_train=y,
            variant=variant,
            cat_cards=cat_cards,
            seed=cfg.seed + variant.seed_offset + fold,
            device=device,
            use_amp=cfg.use_amp,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        oof[va_idx] = val_pred
        pred_test += test_pred / cfg.n_folds
        fold_scores.append(fold_auc)
        print(f"[Fold {fold}] best_auc={fold_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"Variant {variant.name} CV AUC: {cv_auc:.6f} | fold_mean={np.mean(fold_scores):.6f}")
    del x_num_all, x_cat_all, edge_index
    gc.collect()
    return oof.astype(np.float32), pred_test.astype(np.float32), float(cv_auc)


# ============================================================
# Main
# ============================================================

def make_default_variants() -> List[VariantConfig]:
    return [
        VariantConfig(
            name="gnn_sage_base_knn",
            graph_mode="base_knn_ohe",
            node_mode="standard",
            k=12,
            hidden_dim=128,
            dropout=0.20,
            graph_numeric_multiplier=3.0,
            edge_dropout=0.00,
            seed_offset=11,
        ),
        VariantConfig(
            name="gnn_sage_rankgauss_freq",
            graph_mode="freq_cosine",
            node_mode="rankgauss",
            k=14,
            hidden_dim=160,
            dropout=0.25,
            graph_numeric_multiplier=2.0,
            edge_dropout=0.05,
            seed_offset=23,
        ),
        VariantConfig(
            name="gnn_sage_crosscat_quantile",
            graph_mode="quantile_ohe",
            node_mode="quantile_bins",
            k=12,
            hidden_dim=128,
            dropout=0.15,
            use_cross_features=True,
            edge_dropout=0.03,
            seed_offset=37,
        ),
        VariantConfig(
            name="gnn_sage_hybrid_union",
            graph_mode="hybrid_union",
            node_mode="log_interact",
            k=10,
            hidden_dim=192,
            dropout=0.30,
            graph_numeric_multiplier=2.0,
            edge_dropout=0.08,
            seed_offset=41,
        ),
        VariantConfig(
            name="gnn_sage_rf_proximity",
            graph_mode="rf_proximity",
            node_mode="standard",
            k=16,
            hidden_dim=160,
            dropout=0.22,
            graph_numeric_multiplier=2.5,
            rf_estimators=160,
            rf_depth=8,
            edge_dropout=0.04,
            seed_offset=53,
        ),
    ]


def main() -> None:
    cfg = Config()
    apply_env_to_config(cfg)
    cfg.variants = make_default_variants()
    GNN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = setup_runtime_memory(cfg)
    print(
        f"Using device: {device} | amp={cfg.use_amp} | checkpoint={cfg.gradient_checkpointing} | "
        f"gpu_fraction={cfg.gpu_memory_fraction} | max_graph_edges={cfg.max_graph_edges}"
    )
    train, test, y = load_data(cfg)

    summary_rows = []
    for variant in cfg.variants:
        oof, pred_test, cv_auc = run_variant(cfg, variant, train, test, y, device)
        oof_path = GNN_OUT_DIR / f"oof_{variant.name}.npy"
        pred_path = GNN_OUT_DIR / f"pred_{variant.name}.npy"
        np.save(oof_path, oof)
        np.save(pred_path, pred_test)
        summary_rows.append({"variant": variant.name, "cv_auc": cv_auc})

    summary = pd.DataFrame(summary_rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    summary_path = GNN_OUT_DIR / "gnn_variant_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved files:")
    for variant in cfg.variants:
        print(f" - {GNN_OUT_DIR / f'oof_{variant.name}.npy'}")
        print(f" - {GNN_OUT_DIR / f'pred_{variant.name}.npy'}")
    print(f" - {summary_path}")
    print("\nCV summary:")
    print(summary)


if __name__ == "__main__":
    main()
