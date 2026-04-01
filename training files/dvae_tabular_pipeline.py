from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ.setdefault("PYTHONWARNINGS", "ignore")


TRAIN_PATH_CANDIDATES = [
    "train.csv",
    "/kaggle/input/playground-series-s6e3/train.csv",
    "/kaggle/input/competitions/playground-series-s6e3/train.csv",
]
TEST_PATH_CANDIDATES = [
    "test.csv",
    "/kaggle/input/playground-series-s6e3/test.csv",
    "/kaggle/input/competitions/playground-series-s6e3/test.csv",
]
OUTPUT_DIR = "dvae_outputs_optuna"

TARGET_COL = "Churn"
RUN_MODE = "train_best"
FAST_DEBUG = False

SEED = 42
N_SPLITS = 5
SEARCH_N_SPLITS = 3
SEARCH_SUBSAMPLE_FRAC = 0.40

SEARCH_BUDGET = "balanced"
SEARCH_TRIALS_RAW = 40 if not FAST_DEBUG else 3
SEARCH_TRIALS_DVAE = 50 if not FAST_DEBUG else 3
SEARCH_TRIALS_STACK = 15 if not FAST_DEBUG else 3

TOP_K_CONFIRM = 3
TOP_K_VARIANTS_FOR_STACK = 8
CORR_DROP_THRESHOLD = 0.998
SCORE_DROP_TOL = 0.0015

DVAE_SEEDS = [42, 123] if not FAST_DEBUG else [42]
ENABLE_SERVICE_VIEW = True
ENABLE_RICH_VARIANTS = False

GPU_BATCH_SIZE = 1024
CPU_BATCH_SIZE = 512
GPU_EVAL_BATCH_SIZE = 1024
CPU_EVAL_BATCH_SIZE = 256
MAX_GPU_MEMORY_GIB = 5.0
NUM_WORKERS = 0

DEFAULT_LATENT_DIM = 16
DEFAULT_MAX_EPOCHS = 30 if not FAST_DEBUG else 2
DEFAULT_PATIENCE = 6 if not FAST_DEBUG else 1
DEFAULT_DVAE_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_NOISE_STD = 0.05
DEFAULT_MASK_PROB = 0.10
DEFAULT_KL_BETA_MAX = 0.10
DEFAULT_KL_WARMUP_RATIO = 0.4
DEFAULT_DROPOUT = 0.10
DEFAULT_HIDDEN_DIMS = [256, 128]

XGB_N_ESTIMATORS = 2000 if not FAST_DEBUG else 200
XGB_EARLY_STOPPING_ROUNDS = 200 if not FAST_DEBUG else 20
DOWNSTREAM_VAL_SIZE = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = torch.cuda.is_available()


RAM_WARN_THRESHOLD_GIB = 1.5  # log a warning below this free RAM


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _PSUTIL_AVAILABLE:
        free_gib = _psutil.virtual_memory().available / (1024 ** 3)
        if free_gib < RAM_WARN_THRESHOLD_GIB:
            log(f"[MEM] Low RAM warning: {free_gib:.2f} GiB free — forcing extra gc pass")
            gc.collect()
            gc.collect()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_existing_path(candidates: list[str], label: str) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve {label} from candidates: {candidates}")


def map_target(target: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(target):
        unique = set(pd.Series(target).dropna().unique().tolist())
        if unique.issubset({0, 1}):
            return target.to_numpy(dtype=np.int64)

    mapped = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    )
    if mapped.isna().any():
        bad = target[mapped.isna()].astype(str).unique().tolist()
        raise ValueError(f"Unsupported target labels: {bad}")
    return mapped.to_numpy(dtype=np.int64)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def safe_string(series: pd.Series) -> pd.Series:
    return (
        series.fillna("__MISSING__")
        .astype(str)
        .str.strip()
        .replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    )


def build_engineered_frame(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    frame = df.copy()

    object_cols = frame.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in object_cols:
        frame[col] = safe_string(frame[col])

    numeric_candidates = [col for col in frame.columns if col not in object_cols and col != target_col]
    for col in numeric_candidates:
        frame[col] = safe_numeric(frame[col])

    if "tenure" in frame.columns:
        tenure = frame["tenure"].fillna(0.0).clip(lower=0.0)
    else:
        tenure = pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)

    if "MonthlyCharges" in frame.columns:
        monthly = frame["MonthlyCharges"].fillna(0.0)
    else:
        monthly = pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)

    if "TotalCharges" in frame.columns:
        total = frame["TotalCharges"].fillna(0.0)
    else:
        total = pd.Series(np.zeros(len(frame), dtype=np.float32), index=frame.index)

    denom = tenure.clip(lower=1.0)
    frame["avg_monthly_charge"] = (total / denom).astype(np.float32)
    frame["charge_gap"] = (total - tenure * monthly).astype(np.float32)
    frame["monthly_x_tenure"] = (monthly * tenure).astype(np.float32)
    frame["log_totalcharges"] = np.log1p(total.clip(lower=0.0)).astype(np.float32)
    frame["monthly_per_tenure_sqrt"] = (monthly / np.sqrt(denom)).astype(np.float32)

    service_cols = [
        col
        for col in [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PhoneService",
            "MultipleLines",
        ]
        if col in frame.columns
    ]
    if service_cols:
        yes_matrix = np.column_stack(
            [safe_string(frame[col]).str.lower().isin(["yes", "fiber optic"]).astype(np.float32) for col in service_cols]
        )
        frame["service_positive_count"] = yes_matrix.sum(axis=1).astype(np.float32)
        frame["service_positive_ratio"] = (yes_matrix.mean(axis=1)).astype(np.float32)

    if "InternetService" in frame.columns:
        internet = safe_string(frame["InternetService"])
        frame["has_internet"] = internet.ne("No").astype(np.float32)
    if "Contract" in frame.columns:
        contract = safe_string(frame["Contract"])
        frame["is_month_to_month"] = contract.eq("Month-to-month").astype(np.float32)
        frame["contract_x_payment"] = contract + "__" + safe_string(frame.get("PaymentMethod", pd.Series("", index=frame.index)))
    if "PaperlessBilling" in frame.columns and "PaymentMethod" in frame.columns:
        frame["paperless_x_payment"] = safe_string(frame["PaperlessBilling"]) + "__" + safe_string(frame["PaymentMethod"])
    if "InternetService" in frame.columns and "Contract" in frame.columns:
        frame["internet_x_contract"] = safe_string(frame["InternetService"]) + "__" + safe_string(frame["Contract"])

    tenure_bins = pd.cut(
        tenure,
        bins=[-0.1, 0.5, 6.5, 12.5, 24.5, 36.5, 48.5, 60.5, 72.5],
        labels=["0", "1-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"],
        include_lowest=True,
    )
    frame["tenure_bucket"] = tenure_bins.astype(str).replace({"nan": "0"})

    monthly_bins = pd.cut(
        monthly,
        bins=[-np.inf, 30.0, 50.0, 70.0, 90.0, np.inf],
        labels=["<30", "30-50", "50-70", "70-90", ">=90"],
        include_lowest=True,
    )
    frame["monthly_bucket"] = monthly_bins.astype(str).replace({"nan": "<30"})

    if target_col is not None and target_col in frame.columns:
        frame[target_col] = df[target_col]
    return frame


def detect_column_types(df: pd.DataFrame, target_col: str) -> dict[str, list[str]]:
    numeric_cols: list[str] = []
    binary_cols: list[str] = []
    categorical_cols: list[str] = []
    ignored_cols: list[str] = []
    row_count = len(df)

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        nunique = series.nunique(dropna=False)
        unique_ratio = nunique / max(row_count, 1)
        if "id" in col.lower() or unique_ratio >= 0.995:
            ignored_cols.append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            if nunique <= 2:
                binary_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            unique_values = safe_string(series).str.lower().unique().tolist()
            if len(unique_values) <= 2:
                binary_cols.append(col)
            else:
                categorical_cols.append(col)

    return {
        "numeric_cols": sorted(numeric_cols),
        "binary_cols": sorted(binary_cols),
        "categorical_cols": sorted(categorical_cols),
        "ignored_cols": sorted(ignored_cols),
    }


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def build_scaler(scaler_name: str) -> Any:
    if scaler_name == "robust":
        return RobustScaler()
    return StandardScaler()


def build_preprocessor(
    train_df: pd.DataFrame,
    numeric_cols: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
    scaler_name: str,
) -> ColumnTransformer:
    numeric_binary = [
        col for col in binary_cols if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col])
    ]
    categorical_binary = [
        col for col in binary_cols if col in train_df.columns and not pd.api.types.is_numeric_dtype(train_df[col])
    ]
    numeric_like = sorted(set(numeric_cols + numeric_binary))
    categorical_like = sorted(set(categorical_cols + categorical_binary))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_like:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", build_scaler(scaler_name)),
                    ]
                ),
                numeric_like,
            )
        )
    if categorical_like:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_like,
            )
        )
    if not transformers:
        raise ValueError("No usable features for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    preprocessor.fit(train_df)
    return preprocessor


def preprocess_fold_data(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
    scaler_name: str,
) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = build_preprocessor(
        train_df=train_df,
        numeric_cols=numeric_cols,
        binary_cols=binary_cols,
        categorical_cols=categorical_cols,
        scaler_name=scaler_name,
    )
    x_train = preprocessor.transform(train_df).astype(np.float32, copy=False)
    x_valid = preprocessor.transform(valid_df).astype(np.float32, copy=False)
    x_test = preprocessor.transform(test_df).astype(np.float32, copy=False)
    return preprocessor, x_train, x_valid, x_test


def derive_service_view_columns(feature_cols: list[str]) -> list[str]:
    keep_keywords = [
        "internet",
        "online",
        "stream",
        "techsupport",
        "device",
        "multiplelines",
        "phone",
        "service",
        "monthly",
        "total",
        "tenure",
        "contract",
        "payment",
        "paperless",
        "charge",
    ]
    service_cols = [col for col in feature_cols if any(key in col.lower() for key in keep_keywords)]
    return sorted(set(service_cols))


class TabularDVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = self._build_mlp(input_dim, hidden_dims, dropout)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.decoder = self._build_decoder(latent_dim, hidden_dims[::-1], input_dim, dropout)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(latent_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        dims = [latent_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu_head(hidden), self.logvar_head(hidden)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class DVAETrainer:
    def __init__(
        self,
        model: TabularDVAE,
        device: torch.device,
        max_epochs: int,
        patience: int,
        learning_rate: float,
        weight_decay: float,
        noise_std: float,
        mask_prob: float,
        beta_max: float,
        warmup_ratio: float,
        batch_size: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.beta_max = beta_max
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size or (GPU_BATCH_SIZE if device.type == "cuda" else CPU_BATCH_SIZE)

    def _make_loader(self, features: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.from_numpy(features)),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

    def _beta(self, epoch: int) -> float:
        warmup_epochs = max(1, int(math.ceil(self.max_epochs * self.warmup_ratio)))
        progress = min(epoch / warmup_epochs, 1.0)
        return self.beta_max * progress

    @staticmethod
    def _kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()

    def _corrupt(self, batch_x: torch.Tensor) -> torch.Tensor:
        corrupted = batch_x
        if self.noise_std > 0:
            corrupted = corrupted + torch.randn_like(corrupted) * self.noise_std
        if self.mask_prob > 0:
            mask = (torch.rand_like(corrupted) > self.mask_prob).float()
            corrupted = corrupted * mask
        return corrupted

    def fit(self, train_x: np.ndarray, valid_x: np.ndarray, verbose: bool) -> dict[str, Any]:
        train_loader = self._make_loader(train_x, shuffle=True)
        valid_loader = self._make_loader(valid_x, shuffle=False)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_valid = float("inf")
        best_epoch = 0
        wait = 0

        for epoch in range(1, self.max_epochs + 1):
            beta = self._beta(epoch)
            self.model.train()
            train_loss = 0.0
            train_count = 0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                recon, mu, logvar = self.model(self._corrupt(batch_x))
                recon_loss = torch.mean((recon - batch_x) ** 2, dim=1).mean()
                kl_loss = self._kl(mu, logvar)
                loss = recon_loss + beta * kl_loss
                loss.backward()
                optimizer.step()
                batch_size = batch_x.size(0)
                train_loss += loss.item() * batch_size
                train_count += batch_size

            self.model.eval()
            valid_loss = 0.0
            valid_count = 0
            with torch.no_grad():
                for (batch_x,) in valid_loader:
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    recon, mu, logvar = self.model(batch_x)
                    recon_loss = torch.mean((recon - batch_x) ** 2, dim=1).mean()
                    kl_loss = self._kl(mu, logvar)
                    loss = recon_loss + beta * kl_loss
                    batch_size = batch_x.size(0)
                    valid_loss += loss.item() * batch_size
                    valid_count += batch_size

            train_loss /= max(train_count, 1)
            valid_loss /= max(valid_count, 1)
            improved = valid_loss < best_valid - 1e-6
            if improved:
                best_valid = valid_loss
                best_epoch = epoch
                wait = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1

            if verbose and (epoch == 1 or epoch == self.max_epochs or improved or epoch % 2 == 0):
                log(
                    f"    DVAE epoch {epoch:02d}/{self.max_epochs} "
                    f"train={train_loss:.6f} valid={valid_loss:.6f} beta={beta:.4f}"
                )

            if wait >= self.patience:
                if verbose:
                    log(f"    DVAE early stop at epoch {epoch} best_epoch={best_epoch}")
                break

        self.model.load_state_dict(best_state)
        return {"best_epoch": best_epoch, "best_valid_loss": best_valid}


def make_dvae_features(
    model: TabularDVAE,
    features: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch_size = GPU_EVAL_BATCH_SIZE if device.type == "cuda" else CPU_EVAL_BATCH_SIZE
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    latent_parts: list[np.ndarray] = []
    logvar_parts: list[np.ndarray] = []
    recon_parts: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            mu, logvar = model.encode(batch_x)
            recon = model.decode(mu)
            latent_parts.append(mu.cpu().numpy().astype(np.float32, copy=False))
            logvar_parts.append(logvar.cpu().numpy().astype(np.float32, copy=False))
            recon_parts.append(recon.cpu().numpy().astype(np.float32, copy=False))
            del batch_x, mu, logvar, recon

    latent = np.concatenate(latent_parts, axis=0)
    logvar = np.concatenate(logvar_parts, axis=0)
    recon = np.concatenate(recon_parts, axis=0)
    residual = (features - recon).astype(np.float32, copy=False)
    abs_residual = np.abs(residual).astype(np.float32, copy=False)

    residual_summary = np.column_stack(
        [
            abs_residual.mean(axis=1),
            abs_residual.std(axis=1),
            np.median(abs_residual, axis=1),
            np.percentile(abs_residual, 90, axis=1),
            np.percentile(abs_residual, 95, axis=1),
            np.percentile(abs_residual, 99, axis=1),
            abs_residual.max(axis=1),
            abs_residual.sum(axis=1),
            np.sqrt((residual**2).sum(axis=1)),
            (residual**2).mean(axis=1),
            (abs_residual > 0.05).mean(axis=1),
            (abs_residual > 0.10).mean(axis=1),
            (abs_residual > 0.20).mean(axis=1),
        ]
    ).astype(np.float32, copy=False)

    top3 = np.sort(abs_residual, axis=1)[:, -3:].astype(np.float32, copy=False)
    latent_summary = np.column_stack(
        [
            latent.mean(axis=1),
            latent.std(axis=1),
            np.linalg.norm(latent, axis=1),
            logvar.mean(axis=1),
            logvar.std(axis=1),
        ]
    ).astype(np.float32, copy=False)

    return {
        "latent": latent,
        "logvar": logvar,
        "reconstruction": recon.astype(np.float32, copy=False),
        "abs_residual": abs_residual,
        "residual_summary": residual_summary,
        "top3": top3,
        "latent_summary": latent_summary,
    }


def build_xgb_params(overrides: dict[str, Any], seed: int) -> dict[str, Any]:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": XGB_N_ESTIMATORS,
        "early_stopping_rounds": XGB_EARLY_STOPPING_ROUNDS,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "random_state": seed,
        "n_jobs": -1,
        "tree_method": "hist",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbosity": 0,
    }
    params.update(overrides)
    return params


def fit_xgb_and_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_fit, x_eval, y_fit, y_eval = train_test_split(
        x_train,
        y_train,
        test_size=DOWNSTREAM_VAL_SIZE,
        stratify=y_train,
        random_state=seed,
    )
    model = xgb.XGBClassifier(**build_xgb_params(params, seed))
    model.fit(x_fit, y_fit, eval_set=[(x_eval, y_eval)], verbose=False)
    valid_pred = np.clip(model.predict_proba(x_valid)[:, 1], 0.0, 1.0).astype(np.float32)
    test_pred = np.clip(model.predict_proba(x_test)[:, 1], 0.0, 1.0).astype(np.float32)
    del model, x_fit, x_eval, y_fit, y_eval
    cleanup_memory()
    return valid_pred, test_pred


def get_search_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 0.999:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    keep_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        take = max(1, int(round(len(cls_idx) * frac)))
        chosen = rng.choice(cls_idx, size=take, replace=False)
        keep_parts.append(np.sort(chosen))
    return np.sort(np.concatenate(keep_parts))


def sample_raw_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "scaler_name": trial.suggest_categorical("raw_scaler_name", ["standard", "robust"]),
        "learning_rate": trial.suggest_float("raw_learning_rate", 0.015, 0.07, log=True),
        "max_depth": trial.suggest_int("raw_max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("raw_min_child_weight", 1.0, 12.0),
        "subsample": trial.suggest_float("raw_subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("raw_colsample_bytree", 0.60, 0.95),
        "gamma": trial.suggest_float("raw_gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("raw_reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("raw_reg_lambda", 0.5, 10.0, log=True),
        "max_bin": trial.suggest_categorical("raw_max_bin", [128, 256, 384, 512]),
    }


def sample_dvae_params(trial: optuna.Trial) -> dict[str, Any]:
    hidden_name = trial.suggest_categorical("dvae_hidden_name", ["192x96", "256x128", "384x192", "256x256"])
    hidden_map = {
        "192x96": [192, 96],
        "256x128": [256, 128],
        "384x192": [384, 192],
        "256x256": [256, 256],
    }
    return {
        "latent_dim": trial.suggest_categorical("dvae_latent_dim", [8, 12, 16, 24, 32, 40]),
        "hidden_dims": hidden_map[hidden_name],
        "dropout": trial.suggest_float("dvae_dropout", 0.0, 0.25),
        "learning_rate": trial.suggest_float("dvae_learning_rate", 3e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("dvae_weight_decay", 1e-6, 5e-4, log=True),
        "noise_std": trial.suggest_float("dvae_noise_std", 0.0, 0.12),
        "mask_prob": trial.suggest_float("dvae_mask_prob", 0.0, 0.20),
        "beta_max": trial.suggest_float("dvae_beta_max", 0.01, 0.15),
        "warmup_ratio": trial.suggest_float("dvae_warmup_ratio", 0.2, 0.8),
        "max_epochs": trial.suggest_int("dvae_max_epochs", 18 if not FAST_DEBUG else 2, 34 if not FAST_DEBUG else 2),
        "patience": trial.suggest_int("dvae_patience", 4 if not FAST_DEBUG else 1, 8 if not FAST_DEBUG else 1),
        "scaler_name": trial.suggest_categorical("dvae_scaler_name", ["standard", "robust"]),
    }


def sample_hybrid_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("hybrid_learning_rate", 0.015, 0.06, log=True),
        "max_depth": trial.suggest_int("hybrid_max_depth", 3, 7),
        "min_child_weight": trial.suggest_float("hybrid_min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("hybrid_subsample", 0.65, 0.95),
        "colsample_bytree": trial.suggest_float("hybrid_colsample_bytree", 0.60, 0.95),
        "gamma": trial.suggest_float("hybrid_gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("hybrid_reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("hybrid_reg_lambda", 0.5, 10.0, log=True),
        "max_bin": trial.suggest_categorical("hybrid_max_bin", [128, 256, 384, 512]),
    }


def build_variant_features(
    raw_train: np.ndarray,
    raw_valid: np.ndarray,
    raw_test: np.ndarray,
    all_train: dict[str, np.ndarray],
    all_valid: dict[str, np.ndarray],
    all_test: dict[str, np.ndarray],
    service_train: dict[str, np.ndarray] | None,
    service_valid: dict[str, np.ndarray] | None,
    service_test: dict[str, np.ndarray] | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    variants: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "latent": (all_train["latent"], all_valid["latent"], all_test["latent"]),
        "latent_resid": (
            np.concatenate([all_train["latent"], all_train["residual_summary"]], axis=1),
            np.concatenate([all_valid["latent"], all_valid["residual_summary"]], axis=1),
            np.concatenate([all_test["latent"], all_test["residual_summary"]], axis=1),
        ),
        "latent_recon": (
            np.concatenate([all_train["latent"], all_train["reconstruction"]], axis=1),
            np.concatenate([all_valid["latent"], all_valid["reconstruction"]], axis=1),
            np.concatenate([all_test["latent"], all_test["reconstruction"]], axis=1),
        ),
        "full": (
            np.concatenate(
                [
                    all_train["latent"],
                    all_train["reconstruction"],
                    all_train["residual_summary"],
                    all_train["latent_summary"],
                    all_train["top3"],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    all_valid["latent"],
                    all_valid["reconstruction"],
                    all_valid["residual_summary"],
                    all_valid["latent_summary"],
                    all_valid["top3"],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    all_test["latent"],
                    all_test["reconstruction"],
                    all_test["residual_summary"],
                    all_test["latent_summary"],
                    all_test["top3"],
                ],
                axis=1,
            ),
        ),
        "raw_xgb": (raw_train, raw_valid, raw_test),
        "raw_plus_latent": (
            np.concatenate([raw_train, all_train["latent"]], axis=1),
            np.concatenate([raw_valid, all_valid["latent"]], axis=1),
            np.concatenate([raw_test, all_test["latent"]], axis=1),
        ),
        "raw_plus_latent_resid": (
            np.concatenate([raw_train, all_train["latent"], all_train["residual_summary"], all_train["latent_summary"]], axis=1),
            np.concatenate([raw_valid, all_valid["latent"], all_valid["residual_summary"], all_valid["latent_summary"]], axis=1),
            np.concatenate([raw_test, all_test["latent"], all_test["residual_summary"], all_test["latent_summary"]], axis=1),
        ),
        "raw_plus_full": (
            np.concatenate(
                [raw_train, all_train["latent"], all_train["reconstruction"], all_train["residual_summary"], all_train["latent_summary"], all_train["top3"]],
                axis=1,
            ),
            np.concatenate(
                [raw_valid, all_valid["latent"], all_valid["reconstruction"], all_valid["residual_summary"], all_valid["latent_summary"], all_valid["top3"]],
                axis=1,
            ),
            np.concatenate(
                [raw_test, all_test["latent"], all_test["reconstruction"], all_test["residual_summary"], all_test["latent_summary"], all_test["top3"]],
                axis=1,
            ),
        ),
    }

    if ENABLE_RICH_VARIANTS:
        variants["resid_rich"] = (
            np.concatenate([all_train["residual_summary"], all_train["latent_summary"], all_train["top3"]], axis=1),
            np.concatenate([all_valid["residual_summary"], all_valid["latent_summary"], all_valid["top3"]], axis=1),
            np.concatenate([all_test["residual_summary"], all_test["latent_summary"], all_test["top3"]], axis=1),
        )

    if service_train is not None and service_valid is not None and service_test is not None:
        variants["service_full"] = (
            np.concatenate(
                [service_train["latent"], service_train["residual_summary"], service_train["latent_summary"], service_train["top3"]],
                axis=1,
            ),
            np.concatenate(
                [service_valid["latent"], service_valid["residual_summary"], service_valid["latent_summary"], service_valid["top3"]],
                axis=1,
            ),
            np.concatenate(
                [service_test["latent"], service_test["residual_summary"], service_test["latent_summary"], service_test["top3"]],
                axis=1,
            ),
        )
        variants["multiview_full"] = (
            np.concatenate(
                [
                    raw_train,
                    all_train["latent"],
                    all_train["residual_summary"],
                    service_train["latent"],
                    service_train["residual_summary"],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_valid,
                    all_valid["latent"],
                    all_valid["residual_summary"],
                    service_valid["latent"],
                    service_valid["residual_summary"],
                ],
                axis=1,
            ),
            np.concatenate(
                [
                    raw_test,
                    all_test["latent"],
                    all_test["residual_summary"],
                    service_test["latent"],
                    service_test["residual_summary"],
                ],
                axis=1,
            ),
        )

    return variants


def run_raw_objective(
    trial: optuna.Trial,
    features_df: pd.DataFrame,
    y: np.ndarray,
    type_info: dict[str, list[str]],
    test_df: pd.DataFrame,
) -> float:
    params = sample_raw_params(trial)
    idx = get_search_indices(y, SEARCH_SUBSAMPLE_FRAC if not FAST_DEBUG else 0.15, SEED + trial.number)
    x_df = features_df.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    test_proxy = test_df.iloc[: min(len(test_df), max(5000, len(test_df) // 4))].reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=SEARCH_N_SPLITS, shuffle=True, random_state=SEED)
    scores: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_df, y_sub), start=1):
        _, x_train, x_valid, x_test = preprocess_fold_data(
            train_df=x_df.iloc[tr_idx].reset_index(drop=True),
            valid_df=x_df.iloc[va_idx].reset_index(drop=True),
            test_df=test_proxy,
            numeric_cols=type_info["numeric_cols"],
            binary_cols=type_info["binary_cols"],
            categorical_cols=type_info["categorical_cols"],
            scaler_name=params["scaler_name"],
        )
        valid_pred, _ = fit_xgb_and_predict(
            x_train=x_train,
            y_train=y_sub[tr_idx],
            x_valid=x_valid,
            x_test=x_test,
            params={k.replace("raw_", ""): v for k, v in params.items() if k != "scaler_name"},
            seed=SEED + fold + trial.number,
        )
        score = roc_auc_score(y_sub[va_idx], valid_pred)
        scores.append(float(score))
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
        del x_train, x_valid, x_test, valid_pred
        cleanup_memory()
    return float(np.mean(scores))


def run_dvae_objective(
    trial: optuna.Trial,
    features_df: pd.DataFrame,
    y: np.ndarray,
    type_info: dict[str, list[str]],
    test_df: pd.DataFrame,
) -> float:
    dvae_params = sample_dvae_params(trial)
    hybrid_params = sample_hybrid_xgb_params(trial)

    idx = get_search_indices(y, SEARCH_SUBSAMPLE_FRAC if not FAST_DEBUG else 0.12, SEED + 1000 + trial.number)
    x_df = features_df.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    test_proxy = test_df.iloc[: min(len(test_df), max(5000, len(test_df) // 4))].reset_index(drop=True)

    splitter = StratifiedKFold(n_splits=SEARCH_N_SPLITS, shuffle=True, random_state=SEED + 7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: list[float] = []
    feature_cols = type_info["numeric_cols"] + type_info["binary_cols"] + type_info["categorical_cols"]
    service_cols = derive_service_view_columns(feature_cols)

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_df, y_sub), start=1):
        _, raw_train, raw_valid, raw_test = preprocess_fold_data(
            train_df=x_df.iloc[tr_idx].reset_index(drop=True),
            valid_df=x_df.iloc[va_idx].reset_index(drop=True),
            test_df=test_proxy,
            numeric_cols=type_info["numeric_cols"],
            binary_cols=type_info["binary_cols"],
            categorical_cols=type_info["categorical_cols"],
            scaler_name=dvae_params["scaler_name"],
        )

        model = TabularDVAE(
            input_dim=raw_train.shape[1],
            latent_dim=dvae_params["latent_dim"],
            hidden_dims=dvae_params["hidden_dims"],
            dropout=dvae_params["dropout"],
        )
        trainer = DVAETrainer(
            model=model,
            device=device,
            max_epochs=dvae_params["max_epochs"],
            patience=dvae_params["patience"],
            learning_rate=dvae_params["learning_rate"],
            weight_decay=dvae_params["weight_decay"],
            noise_std=dvae_params["noise_std"],
            mask_prob=dvae_params["mask_prob"],
            beta_max=dvae_params["beta_max"],
            warmup_ratio=dvae_params["warmup_ratio"],
        )
        trainer.fit(raw_train, raw_valid, verbose=False)
        train_feat = make_dvae_features(trainer.model, raw_train, device)
        valid_feat = make_dvae_features(trainer.model, raw_valid, device)
        test_feat = make_dvae_features(trainer.model, raw_test, device)

        service_train_feat = None
        service_valid_feat = None
        service_test_feat = None
        if ENABLE_SERVICE_VIEW and service_cols:
            service_type = detect_column_types(x_df[service_cols].copy(), target_col="__missing__")
            _, s_train, s_valid, s_test = preprocess_fold_data(
                train_df=x_df.iloc[tr_idx][service_cols].reset_index(drop=True),
                valid_df=x_df.iloc[va_idx][service_cols].reset_index(drop=True),
                test_df=test_proxy[service_cols].reset_index(drop=True),
                numeric_cols=service_type["numeric_cols"],
                binary_cols=service_type["binary_cols"],
                categorical_cols=service_type["categorical_cols"],
                scaler_name=dvae_params["scaler_name"],
            )
            service_model = TabularDVAE(
                input_dim=s_train.shape[1],
                latent_dim=max(4, dvae_params["latent_dim"] // 2),
                hidden_dims=dvae_params["hidden_dims"],
                dropout=dvae_params["dropout"],
            )
            service_trainer = DVAETrainer(
                model=service_model,
                device=device,
                max_epochs=max(8, int(dvae_params["max_epochs"] * 0.7)),
                patience=max(2, int(dvae_params["patience"] * 0.7)),
                learning_rate=dvae_params["learning_rate"],
                weight_decay=dvae_params["weight_decay"],
                noise_std=dvae_params["noise_std"],
                mask_prob=dvae_params["mask_prob"],
                beta_max=dvae_params["beta_max"],
                warmup_ratio=dvae_params["warmup_ratio"],
            )
            service_trainer.fit(s_train, s_valid, verbose=False)
            service_train_feat = make_dvae_features(service_trainer.model, s_train, device)
            service_valid_feat = make_dvae_features(service_trainer.model, s_valid, device)
            service_test_feat = make_dvae_features(service_trainer.model, s_test, device)
            del s_train, s_valid, s_test, service_model, service_trainer
            cleanup_memory()

        variants = build_variant_features(
            raw_train=raw_train,
            raw_valid=raw_valid,
            raw_test=raw_test,
            all_train=train_feat,
            all_valid=valid_feat,
            all_test=test_feat,
            service_train=service_train_feat,
            service_valid=service_valid_feat,
            service_test=service_test_feat,
        )
        proxy_name = "raw_plus_latent_resid" if "raw_plus_latent_resid" in variants else "full"
        valid_pred, _ = fit_xgb_and_predict(
            x_train=variants[proxy_name][0],
            y_train=y_sub[tr_idx],
            x_valid=variants[proxy_name][1],
            x_test=variants[proxy_name][2],
            params=hybrid_params,
            seed=SEED + 100 + fold + trial.number,
        )
        score = roc_auc_score(y_sub[va_idx], valid_pred)
        scores.append(float(score))
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        del raw_train, raw_valid, raw_test, model, trainer, train_feat, valid_feat, test_feat, variants, valid_pred
        cleanup_memory()

    return float(np.mean(scores))


def create_study(study_name: str, storage_path: Path) -> optuna.Study:
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
    )


def run_optuna_searches(
    train_features: pd.DataFrame,
    y: np.ndarray,
    type_info: dict[str, list[str]],
    test_features: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Starting Optuna raw-feature search")
    raw_storage = output_dir / "raw_xgb_optuna.db"
    raw_study = create_study("raw_xgb_search", raw_storage)
    raw_study.optimize(
        lambda trial: run_raw_objective(trial, train_features, y, type_info, test_features),
        n_trials=SEARCH_TRIALS_RAW,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    raw_best = {
        "score": float(raw_study.best_value),
        "params": raw_study.best_params,
    }
    log(f"Best raw-feature search score={raw_best['score']:.6f}")

    log("Starting Optuna DVAE+hybrid search")
    dvae_storage = output_dir / "dvae_hybrid_optuna.db"
    dvae_study = create_study("dvae_hybrid_search", dvae_storage)
    dvae_study.optimize(
        lambda trial: run_dvae_objective(trial, train_features, y, type_info, test_features),
        n_trials=SEARCH_TRIALS_DVAE,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    dvae_best = {
        "score": float(dvae_study.best_value),
        "params": dvae_study.best_params,
    }
    log(f"Best DVAE+hybrid search score={dvae_best['score']:.6f}")

    search_summary = {
        "raw_best": raw_best,
        "dvae_best": dvae_best,
        "raw_storage": str(raw_storage),
        "dvae_storage": str(dvae_storage),
    }
    (output_dir / "best_search_params.json").write_text(json.dumps(search_summary, indent=2))
    return search_summary


def extract_prefixed_params(best_params: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix_with_sep = f"{prefix}_"
    for key, value in best_params.items():
        if key.startswith(prefix_with_sep):
            out[key[len(prefix_with_sep) :]] = value
    return out


def get_default_search_summary() -> dict[str, Any]:
    return {
        "raw_best": {
            "score": None,
            "params": {
                "raw_scaler_name": "standard",
                "raw_learning_rate": 0.03,
                "raw_max_depth": 6,
                "raw_min_child_weight": 2.0,
                "raw_subsample": 0.85,
                "raw_colsample_bytree": 0.85,
                "raw_gamma": 0.0,
                "raw_reg_alpha": 0.1,
                "raw_reg_lambda": 1.0,
                "raw_max_bin": 256,
            },
        },
        "dvae_best": {
            "score": None,
            "params": {
                "dvae_latent_dim": DEFAULT_LATENT_DIM,
                "dvae_hidden_name": "256x128",
                "dvae_dropout": DEFAULT_DROPOUT,
                "dvae_learning_rate": DEFAULT_DVAE_LR,
                "dvae_weight_decay": DEFAULT_WEIGHT_DECAY,
                "dvae_noise_std": DEFAULT_NOISE_STD,
                "dvae_mask_prob": DEFAULT_MASK_PROB,
                "dvae_beta_max": DEFAULT_KL_BETA_MAX,
                "dvae_warmup_ratio": DEFAULT_KL_WARMUP_RATIO,
                "dvae_max_epochs": DEFAULT_MAX_EPOCHS,
                "dvae_patience": DEFAULT_PATIENCE,
                "dvae_scaler_name": "standard",
                "hybrid_learning_rate": 0.03,
                "hybrid_max_depth": 5,
                "hybrid_min_child_weight": 2.0,
                "hybrid_subsample": 0.85,
                "hybrid_colsample_bytree": 0.85,
                "hybrid_gamma": 0.0,
                "hybrid_reg_alpha": 0.1,
                "hybrid_reg_lambda": 1.0,
                "hybrid_max_bin": 256,
            },
        },
    }


def best_dvae_config_from_search(best_params: dict[str, Any]) -> dict[str, Any]:
    hidden_name = best_params.get("dvae_hidden_name", "256x128")
    hidden_map = {
        "192x96": [192, 96],
        "256x128": [256, 128],
        "384x192": [384, 192],
        "256x256": [256, 256],
    }
    return {
        "latent_dim": int(best_params.get("dvae_latent_dim", DEFAULT_LATENT_DIM)),
        "hidden_dims": hidden_map[hidden_name],
        "dropout": float(best_params.get("dvae_dropout", DEFAULT_DROPOUT)),
        "learning_rate": float(best_params.get("dvae_learning_rate", DEFAULT_DVAE_LR)),
        "weight_decay": float(best_params.get("dvae_weight_decay", DEFAULT_WEIGHT_DECAY)),
        "noise_std": float(best_params.get("dvae_noise_std", DEFAULT_NOISE_STD)),
        "mask_prob": float(best_params.get("dvae_mask_prob", DEFAULT_MASK_PROB)),
        "beta_max": float(best_params.get("dvae_beta_max", DEFAULT_KL_BETA_MAX)),
        "warmup_ratio": float(best_params.get("dvae_warmup_ratio", DEFAULT_KL_WARMUP_RATIO)),
        "max_epochs": int(best_params.get("dvae_max_epochs", DEFAULT_MAX_EPOCHS)),
        "patience": int(best_params.get("dvae_patience", DEFAULT_PATIENCE)),
        "scaler_name": str(best_params.get("dvae_scaler_name", "standard")),
    }


def fit_view_dvae(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    dvae_cfg: dict[str, Any],
    device: torch.device,
    verbose: bool,
) -> tuple[DVAETrainer, dict[str, Any]]:
    model = TabularDVAE(
        input_dim=x_train.shape[1],
        latent_dim=dvae_cfg["latent_dim"],
        hidden_dims=dvae_cfg["hidden_dims"],
        dropout=dvae_cfg["dropout"],
    )
    trainer = DVAETrainer(
        model=model,
        device=device,
        max_epochs=dvae_cfg["max_epochs"],
        patience=dvae_cfg["patience"],
        learning_rate=dvae_cfg["learning_rate"],
        weight_decay=dvae_cfg["weight_decay"],
        noise_std=dvae_cfg["noise_std"],
        mask_prob=dvae_cfg["mask_prob"],
        beta_max=dvae_cfg["beta_max"],
        warmup_ratio=dvae_cfg["warmup_ratio"],
    )
    fit_info = trainer.fit(x_train, x_valid, verbose=verbose)
    return trainer, fit_info


def select_variants_for_stack(
    scores: dict[str, float],
    oof_preds: dict[str, np.ndarray],
) -> list[str]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return []
    best_score = ranked[0][1]
    selected: list[str] = []
    selected_arrays: list[np.ndarray] = []
    for name, score in ranked:
        if score < best_score - SCORE_DROP_TOL:
            continue
        pred = oof_preds[name]
        duplicate = False
        for kept_pred in selected_arrays:
            corr = np.corrcoef(pred, kept_pred)[0, 1]
            if np.isfinite(corr) and corr >= CORR_DROP_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(name)
        selected_arrays.append(pred)
        if len(selected) >= TOP_K_VARIANTS_FOR_STACK:
            break
    return selected


def run_stacker_optuna(
    selected_variants: list[str],
    oof_predictions: dict[str, np.ndarray],
    y: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    if not selected_variants:
        raise ValueError("No variants available for stacker.")
    x_meta = np.column_stack([oof_predictions[name] for name in selected_variants]).astype(np.float32)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + 999)

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("ridge_alpha", 1e-3, 100.0, log=True)
        scores: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(x_meta, y), start=1):
            model = Ridge(alpha=alpha, random_state=SEED + fold)
            model.fit(x_meta[tr_idx], y[tr_idx])
            pred = np.clip(model.predict(x_meta[va_idx]), 0.0, 1.0)
            score = roc_auc_score(y[va_idx], pred)
            scores.append(float(score))
            trial.report(float(np.mean(scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    stack_storage = output_dir / "stack_optuna.db"
    study = create_study("stack_search", stack_storage)
    study.optimize(objective, n_trials=SEARCH_TRIALS_STACK, gc_after_trial=True, show_progress_bar=False)
    return {
        "variants": selected_variants,
        "alpha": float(study.best_params["ridge_alpha"]),
        "score": float(study.best_value),
        "storage": str(stack_storage),
    }


def fit_final_stacker(
    selected_variants: list[str],
    oof_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    alpha: float,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = np.column_stack([oof_predictions[name] for name in selected_variants]).astype(np.float32)
    x_test = np.column_stack([test_predictions[name] for name in selected_variants]).astype(np.float32)
    model = Ridge(alpha=alpha, random_state=SEED + 2025)
    model.fit(x_train, y)
    oof_stack = np.clip(model.predict(x_train), 0.0, 1.0).astype(np.float32)
    test_stack = np.clip(model.predict(x_test), 0.0, 1.0).astype(np.float32)
    return oof_stack, test_stack


def train_final_models(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    y: np.ndarray,
    type_info: dict[str, list[str]],
    search_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    raw_best = search_summary["raw_best"]["params"]
    dvae_best = search_summary["dvae_best"]["params"]

    raw_scaler = str(raw_best["raw_scaler_name"])
    raw_xgb_params = extract_prefixed_params(raw_best, "raw")
    raw_xgb_params.pop("scaler_name", None)

    dvae_cfg = best_dvae_config_from_search(dvae_best)
    hybrid_xgb_params = extract_prefixed_params(dvae_best, "hybrid")

    feature_cols = type_info["numeric_cols"] + type_info["binary_cols"] + type_info["categorical_cols"]
    service_cols = derive_service_view_columns(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        log(f"Using device=cuda name={torch.cuda.get_device_name(0)} total_vram={total_mem:.2f}GiB target_max={MAX_GPU_MEMORY_GIB:.2f}GiB")
    else:
        log("Using device=cpu")

    raw_oof = np.zeros(len(train_features), dtype=np.float32)
    raw_test = np.zeros(len(test_features), dtype=np.float32)

    aggregated_oof: dict[str, np.ndarray] = {}
    aggregated_test: dict[str, np.ndarray] = {}
    fold_scores: dict[str, list[float]] = {}

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_features, y), start=1):
        log(f"\nFold {fold}/{N_SPLITS}")
        fold_train_df = train_features.iloc[tr_idx].reset_index(drop=True)
        fold_valid_df = train_features.iloc[va_idx].reset_index(drop=True)
        y_train = y[tr_idx]
        y_valid = y[va_idx]

        _, raw_train, raw_valid, raw_test_view = preprocess_fold_data(
            train_df=fold_train_df,
            valid_df=fold_valid_df,
            test_df=test_features,
            numeric_cols=type_info["numeric_cols"],
            binary_cols=type_info["binary_cols"],
            categorical_cols=type_info["categorical_cols"],
            scaler_name=raw_scaler,
        )
        fold_raw_valid, fold_raw_test = fit_xgb_and_predict(
            x_train=raw_train,
            y_train=y_train,
            x_valid=raw_valid,
            x_test=raw_test_view,
            params=raw_xgb_params,
            seed=SEED + fold,
        )
        raw_oof[va_idx] = fold_raw_valid
        raw_test += fold_raw_test / N_SPLITS
        raw_score = roc_auc_score(y_valid, fold_raw_valid)
        log(f"  raw_xgb         ROC-AUC={raw_score:.6f}")
        fold_scores.setdefault("raw_xgb", []).append(float(raw_score))

        for seed_idx, dvae_seed in enumerate(DVAE_SEEDS, start=1):
            set_seed(dvae_seed + fold)
            log(f"  DVAE seed {dvae_seed} ({seed_idx}/{len(DVAE_SEEDS)})")
            _, all_train, all_valid, all_test = preprocess_fold_data(
                train_df=fold_train_df,
                valid_df=fold_valid_df,
                test_df=test_features,
                numeric_cols=type_info["numeric_cols"],
                binary_cols=type_info["binary_cols"],
                categorical_cols=type_info["categorical_cols"],
                scaler_name=dvae_cfg["scaler_name"],
            )
            all_trainer, fit_info = fit_view_dvae(all_train, all_valid, dvae_cfg, device, verbose=True)
            log(f"    all-view best_epoch={fit_info['best_epoch']} valid_loss={fit_info['best_valid_loss']:.6f}")
            all_train_feat = make_dvae_features(all_trainer.model, all_train, device)
            all_valid_feat = make_dvae_features(all_trainer.model, all_valid, device)
            all_test_feat = make_dvae_features(all_trainer.model, all_test, device)

            service_train_feat = None
            service_valid_feat = None
            service_test_feat = None
            if ENABLE_SERVICE_VIEW and service_cols:
                service_type = detect_column_types(fold_train_df[service_cols].copy(), target_col="__target__")
                _, sv_train, sv_valid, sv_test = preprocess_fold_data(
                    train_df=fold_train_df[service_cols].reset_index(drop=True),
                    valid_df=fold_valid_df[service_cols].reset_index(drop=True),
                    test_df=test_features[service_cols].reset_index(drop=True),
                    numeric_cols=service_type["numeric_cols"],
                    binary_cols=service_type["binary_cols"],
                    categorical_cols=service_type["categorical_cols"],
                    scaler_name=dvae_cfg["scaler_name"],
                )
                service_cfg = dict(dvae_cfg)
                service_cfg["latent_dim"] = max(4, int(math.ceil(dvae_cfg["latent_dim"] / 2)))
                service_cfg["max_epochs"] = max(8, int(round(dvae_cfg["max_epochs"] * 0.75)))
                service_cfg["patience"] = max(2, int(round(dvae_cfg["patience"] * 0.75)))
                service_trainer, service_fit = fit_view_dvae(sv_train, sv_valid, service_cfg, device, verbose=False)
                log(f"    service-view best_epoch={service_fit['best_epoch']} valid_loss={service_fit['best_valid_loss']:.6f}")
                service_train_feat = make_dvae_features(service_trainer.model, sv_train, device)
                service_valid_feat = make_dvae_features(service_trainer.model, sv_valid, device)
                service_test_feat = make_dvae_features(service_trainer.model, sv_test, device)
                del sv_train, sv_valid, sv_test, service_trainer
                cleanup_memory()

            variants = build_variant_features(
                raw_train=raw_train,
                raw_valid=raw_valid,
                raw_test=raw_test_view,
                all_train=all_train_feat,
                all_valid=all_valid_feat,
                all_test=all_test_feat,
                service_train=service_train_feat,
                service_valid=service_valid_feat,
                service_test=service_test_feat,
            )

            for variant_name, (x_tr, x_va, x_te) in variants.items():
                if variant_name == "raw_xgb":
                    continue
                valid_pred, test_pred = fit_xgb_and_predict(
                    x_train=x_tr,
                    y_train=y_train,
                    x_valid=x_va,
                    x_test=x_te,
                    params=hybrid_xgb_params,
                    seed=dvae_seed + fold,
                )
                agg_name = f"{variant_name}_ms"
                if agg_name not in aggregated_oof:
                    aggregated_oof[agg_name] = np.zeros(len(train_features), dtype=np.float32)
                    aggregated_test[agg_name] = np.zeros(len(test_features), dtype=np.float32)
                    fold_scores[agg_name] = []
                aggregated_oof[agg_name][va_idx] += valid_pred / len(DVAE_SEEDS)
                aggregated_test[agg_name] += test_pred / (len(DVAE_SEEDS) * N_SPLITS)

                seed_name = f"{variant_name}_seed{dvae_seed}"
                if seed_name not in aggregated_oof:
                    aggregated_oof[seed_name] = np.zeros(len(train_features), dtype=np.float32)
                    aggregated_test[seed_name] = np.zeros(len(test_features), dtype=np.float32)
                    fold_scores[seed_name] = []
                aggregated_oof[seed_name][va_idx] = valid_pred
                aggregated_test[seed_name] += test_pred / N_SPLITS

                seed_score = roc_auc_score(y_valid, valid_pred)
                fold_scores[seed_name].append(float(seed_score))
                log(f"    {seed_name:<22} ROC-AUC={seed_score:.6f}")

                del valid_pred, test_pred
                cleanup_memory()

            del all_train, all_valid, all_test, all_trainer, all_train_feat, all_valid_feat, all_test_feat, variants
            cleanup_memory()

        del raw_train, raw_valid, raw_test_view, fold_train_df, fold_valid_df, y_train, y_valid
        cleanup_memory()

    oof_predictions = {"raw_xgb": raw_oof}
    test_predictions = {"raw_xgb": raw_test}
    final_scores = {"raw_xgb": float(roc_auc_score(y, raw_oof))}

    for name, preds in aggregated_oof.items():
        oof_predictions[name] = preds.astype(np.float32, copy=False)
        test_predictions[name] = aggregated_test[name].astype(np.float32, copy=False)
        final_scores[name] = float(roc_auc_score(y, preds))

    selected_variants = select_variants_for_stack(final_scores, oof_predictions)
    log("\nVariant leaderboard")
    for name, score in sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True):
        log(f"  {name:<24} {score:.6f}")

    log("\nSelected stack variants")
    for name in selected_variants:
        log(f"  {name}")

    stack_search = run_stacker_optuna(selected_variants, oof_predictions, y, output_dir)
    oof_stack, test_stack = fit_final_stacker(
        selected_variants=selected_variants,
        oof_predictions=oof_predictions,
        test_predictions=test_predictions,
        alpha=stack_search["alpha"],
        y=y,
    )
    final_scores["stack"] = float(roc_auc_score(y, oof_stack))
    log(f"\nFinal stack ROC-AUC={final_scores['stack']:.6f}")

    return {
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
        "final_scores": final_scores,
        "fold_scores": fold_scores,
        "selected_variants": selected_variants,
        "stack_search": stack_search,
        "oof_stack": oof_stack,
        "test_stack": test_stack,
        "raw_params": raw_xgb_params,
        "dvae_cfg": dvae_cfg,
        "hybrid_params": hybrid_xgb_params,
    }


def save_outputs(
    output_dir: Path,
    results: dict[str, Any],
    search_summary: dict[str, Any],
    train_path: Path,
    test_path: Path,
    type_info: dict[str, list[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for name, arr in results["oof_predictions"].items():
        path = output_dir / f"oof_{name}.npy"
        np.save(path, np.asarray(arr, dtype=np.float32))
        saved_files.append(str(path))
        log(f"Saved {path}")
    for name, arr in results["test_predictions"].items():
        path = output_dir / f"test_{name}.npy"
        np.save(path, np.asarray(arr, dtype=np.float32))
        saved_files.append(str(path))
        log(f"Saved {path}")

    stack_oof_path = output_dir / "oof_stack.npy"
    stack_test_path = output_dir / "test_stack.npy"
    np.save(stack_oof_path, np.asarray(results["oof_stack"], dtype=np.float32))
    np.save(stack_test_path, np.asarray(results["test_stack"], dtype=np.float32))
    saved_files.extend([str(stack_oof_path), str(stack_test_path)])
    log(f"Saved {stack_oof_path}")
    log(f"Saved {stack_test_path}")

    metadata = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "output_dir": str(output_dir),
        "run_mode": RUN_MODE,
        "search_budget": SEARCH_BUDGET,
        "folds": N_SPLITS,
        "search_folds": SEARCH_N_SPLITS,
        "seed": SEED,
        "dvae_seeds": DVAE_SEEDS,
        "enable_service_view": ENABLE_SERVICE_VIEW,
        "enable_rich_variants": ENABLE_RICH_VARIANTS,
        "feature_counts": {
            "numeric": len(type_info["numeric_cols"]),
            "binary": len(type_info["binary_cols"]),
            "categorical": len(type_info["categorical_cols"]),
            "ignored": type_info["ignored_cols"],
        },
        "search_summary": search_summary,
        "final_scores": results["final_scores"],
        "selected_variants": results["selected_variants"],
        "stack_search": results["stack_search"],
        "raw_xgb_params": results["raw_params"],
        "dvae_cfg": results["dvae_cfg"],
        "hybrid_xgb_params": results["hybrid_params"],
        "saved_files": saved_files,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    log(f"Saved {metadata_path}")


def main() -> None:
    set_seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_existing_path(TRAIN_PATH_CANDIDATES, "train.csv")
    test_path = resolve_existing_path(TEST_PATH_CANDIDATES, "test.csv")
    log(f"Loading train from {train_path}")
    log(f"Loading test from {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = build_engineered_frame(train_df, target_col=TARGET_COL)
    test_df = build_engineered_frame(test_df)
    y = map_target(train_df[TARGET_COL])

    train_features = train_df.drop(columns=[TARGET_COL]).copy()
    test_features = test_df.copy()
    if list(train_features.columns) != list(test_features.columns):
        missing_in_test = [col for col in train_features.columns if col not in test_features.columns]
        extra_in_test = [col for col in test_features.columns if col not in train_features.columns]
        raise ValueError(f"Train/test feature mismatch. Missing in test={missing_in_test}, extra in test={extra_in_test}")

    type_info = detect_column_types(train_df, target_col=TARGET_COL)
    log(
        "Detected columns "
        + json.dumps(
            {
                "numeric": len(type_info["numeric_cols"]),
                "binary": len(type_info["binary_cols"]),
                "categorical": len(type_info["categorical_cols"]),
                "ignored": type_info["ignored_cols"],
            }
        )
    )

    keep_cols = type_info["numeric_cols"] + type_info["binary_cols"] + type_info["categorical_cols"]
    train_features = train_features[keep_cols].copy()
    test_features = test_features[keep_cols].copy()

    if RUN_MODE == "search":
        search_summary = run_optuna_searches(train_features, y, type_info, test_features, output_dir)
        log(json.dumps(search_summary, indent=2))
        return

    if RUN_MODE == "train_best" and (output_dir / "best_search_params.json").exists():
        search_summary = json.loads((output_dir / "best_search_params.json").read_text())
    elif RUN_MODE == "search_and_train":
        search_summary = run_optuna_searches(train_features, y, type_info, test_features, output_dir)
    else:
        search_summary = get_default_search_summary()

    results = train_final_models(
        train_features=train_features,
        test_features=test_features,
        y=y,
        type_info=type_info,
        search_summary=search_summary,
        output_dir=output_dir,
    )
    save_outputs(
        output_dir=output_dir,
        results=results,
        search_summary=search_summary,
        train_path=train_path,
        test_path=test_path,
        type_info=type_info,
    )


if __name__ == "__main__":
    main()
