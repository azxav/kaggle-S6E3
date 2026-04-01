from __future__ import annotations

import copy
import inspect
import json
import os
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


TARGET = "Churn"
ID_COL = "id"
ORIG_ID_COL = "customerID"
PROJECT_DIR = Path("/kaggle/input/competitions/playground-series-s6e3")
TRAIN_PATH = PROJECT_DIR / "train.csv"
TEST_PATH = PROJECT_DIR / "test.csv"
ORIG_PATH = "/kaggle/input/datasets/azizbekxasanov/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "/kaggle/working/"

BASE_CAT_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
BASE_NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
YES_NO_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
VARIANTS = [
    "base",
    "num_shape",
    "service_bundle",
    "cross_cats",
    "orig_targetenc",
]
TARGET_ENCODING_SPECS = {
    "Contract__InternetService__PaymentMethod": [
        "Contract",
        "InternetService",
        "PaymentMethod",
    ],
    "Contract__tenure_bucket": ["Contract", "tenure_bucket"],
    "InternetService__service_yes_count_bucket": [
        "InternetService",
        "service_yes_count_bucket",
    ],
    "PaperlessBilling__PaymentMethod": ["PaperlessBilling", "PaymentMethod"],
}


@dataclass
class TrainingConfig:
    train_path: Path = TRAIN_PATH
    test_path: Path = TEST_PATH
    orig_path: Path = ORIG_PATH
    output_dir: Path = OUTPUT_DIR
    n_splits: int = 5
    fold_seed: int = 42
    meta_seed: int = 314159
    target_encoding_seed: int = 2025
    max_epochs: int = 100
    patience: int = 20
    learning_rate: float = 0.002
    weight_decay: float = 0.0003
    batch_size_gpu: int = 16384
    batch_size_cpu: int = 1024
    eval_batch_size_gpu: int = 65536
    eval_batch_size_cpu: int = 4096
    num_workers_gpu: int = 4
    num_workers_cpu: int = 0
    num_embeddings_kind: str = "piecewise"
    num_embedding_bins: int = 48
    num_embedding_dim: int = 16
    tabm_k: int = 16
    ridge_alpha: float = 1.0
    target_encoding_smoothing: float = 50.0
    n_inner_target_encoding_splits: int = 5
    use_amp: bool = True
    use_data_parallel: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    log_every_epochs: int = 1
    device: str | None = None

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)
        self.orig_path = Path(self.orig_path)
        self.output_dir = Path(self.output_dir)


@dataclass
class VariantSpec:
    train: pd.DataFrame
    test: pd.DataFrame
    orig: pd.DataFrame
    cat_cols: list[str]
    num_cols: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: TrainingConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_num_workers(config: TrainingConfig, device: torch.device) -> int:
    return config.num_workers_gpu if device.type == "cuda" else config.num_workers_cpu


def get_amp_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def configure_runtime(device: torch.device) -> dict[str, object]:
    runtime: dict[str, object] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "kaggle_env": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")),
    }
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        runtime["gpu_names"] = [
            torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
        ]
    return runtime


def safe_string(series: pd.Series) -> pd.Series:
    return series.fillna("__MISSING__").astype(str).str.strip().replace({"": "__MISSING__"})


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32)


def tenure_bucket(series: pd.Series) -> pd.Series:
    bins = [-0.5, 0.5, 6.5, 12.5, 24.5, 36.5, 48.5, 60.5, 72.5]
    labels = ["0", "1-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    bucket = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return bucket.astype(str).replace({"nan": "0"})


def avg_monthly_bucket(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 30.0, 50.0, 70.0, 90.0, np.inf]
    labels = ["<30", "30-50", "50-70", "70-90", ">=90"]
    bucket = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return bucket.astype(str).replace({"nan": "<30"})


def join_features(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    cols = list(cols)
    return frame[cols].astype(str).agg("__".join, axis=1)


def build_common_features(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    out = df.copy()

    out[id_col] = out[id_col].astype(str)
    for col in BASE_CAT_COLS:
        out[col] = safe_string(out[col])
    for col in BASE_NUM_COLS:
        out[col] = safe_numeric(out[col])

    denom = out["tenure"].clip(lower=1.0)
    out["log_totalcharges"] = np.log1p(out["TotalCharges"]).astype(np.float32)
    out["avg_monthly_charge"] = (out["TotalCharges"] / denom).astype(np.float32)
    out["monthly_x_tenure"] = (out["MonthlyCharges"] * out["tenure"]).astype(np.float32)
    out["charge_gap"] = (
        out["TotalCharges"] - (out["MonthlyCharges"] * out["tenure"])
    ).astype(np.float32)
    out["tenure_bucket"] = tenure_bucket(out["tenure"])
    out["avg_monthly_bucket"] = avg_monthly_bucket(out["avg_monthly_charge"])

    out["has_internet"] = (out["InternetService"] != "No").astype(np.float32)
    out["has_phone"] = (out["PhoneService"] == "Yes").astype(np.float32)
    out["MultipleLines_yes"] = (out["MultipleLines"] == "Yes").astype(np.float32)
    for col in YES_NO_SERVICE_COLS:
        out[f"{col}_yes"] = (out[col] == "Yes").astype(np.float32)

    out["support_protection_count"] = (
        out["OnlineSecurity_yes"]
        + out["OnlineBackup_yes"]
        + out["DeviceProtection_yes"]
        + out["TechSupport_yes"]
    ).astype(np.float32)
    out["streaming_count"] = (
        out["StreamingTV_yes"] + out["StreamingMovies_yes"]
    ).astype(np.float32)
    out["internet_yes_count"] = (
        out["support_protection_count"] + out["streaming_count"]
    ).astype(np.float32)
    out["all_internet_addons_on"] = (
        (out["has_internet"] == 1.0) & (out["internet_yes_count"] == 6.0)
    ).astype(np.float32)
    out["no_internet_addons"] = (
        (out["has_internet"] == 1.0) & (out["internet_yes_count"] == 0.0)
    ).astype(np.float32)
    out["service_yes_count_bucket"] = np.where(
        out["has_internet"] == 0.0,
        "no_internet",
        out["internet_yes_count"].astype(int).astype(str),
    )

    out["Contract__InternetService"] = join_features(out, ["Contract", "InternetService"])
    out["Contract__PaymentMethod"] = join_features(out, ["Contract", "PaymentMethod"])
    out["InternetService__PaymentMethod"] = join_features(
        out, ["InternetService", "PaymentMethod"]
    )
    out["Contract__InternetService__PaymentMethod"] = join_features(
        out,
        ["Contract", "InternetService", "PaymentMethod"],
    )
    out["SeniorCitizen__Partner__Dependents"] = join_features(
        out, ["SeniorCitizen", "Partner", "Dependents"]
    )
    out["Contract__tenure_bucket"] = join_features(out, ["Contract", "tenure_bucket"])
    out["InternetService__service_yes_count_bucket"] = join_features(
        out, ["InternetService", "service_yes_count_bucket"]
    )

    return out


def build_variant_specs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: pd.DataFrame,
) -> dict[str, VariantSpec]:
    common_train = build_common_features(train_df, ID_COL)
    common_test = build_common_features(test_df, ID_COL)
    common_orig = build_common_features(orig_df, ORIG_ID_COL)

    specs: dict[str, VariantSpec] = {}
    specs["base"] = VariantSpec(
        train=common_train.copy(),
        test=common_test.copy(),
        orig=common_orig.copy(),
        cat_cols=BASE_CAT_COLS.copy(),
        num_cols=BASE_NUM_COLS.copy(),
    )
    specs["num_shape"] = VariantSpec(
        train=common_train.copy(),
        test=common_test.copy(),
        orig=common_orig.copy(),
        cat_cols=BASE_CAT_COLS + ["tenure_bucket", "avg_monthly_bucket"],
        num_cols=BASE_NUM_COLS
        + ["log_totalcharges", "avg_monthly_charge", "monthly_x_tenure", "charge_gap"],
    )
    specs["service_bundle"] = VariantSpec(
        train=common_train.copy(),
        test=common_test.copy(),
        orig=common_orig.copy(),
        cat_cols=BASE_CAT_COLS.copy(),
        num_cols=BASE_NUM_COLS
        + [
            "has_internet",
            "has_phone",
            "MultipleLines_yes",
            "OnlineSecurity_yes",
            "OnlineBackup_yes",
            "DeviceProtection_yes",
            "TechSupport_yes",
            "StreamingTV_yes",
            "StreamingMovies_yes",
            "internet_yes_count",
            "streaming_count",
            "support_protection_count",
            "all_internet_addons_on",
            "no_internet_addons",
        ],
    )
    specs["cross_cats"] = VariantSpec(
        train=common_train.copy(),
        test=common_test.copy(),
        orig=common_orig.copy(),
        cat_cols=BASE_CAT_COLS
        + [
            "Contract__InternetService",
            "Contract__PaymentMethod",
            "InternetService__PaymentMethod",
            "Contract__InternetService__PaymentMethod",
            "SeniorCitizen__Partner__Dependents",
            "Contract__tenure_bucket",
            "InternetService__service_yes_count_bucket",
        ],
        num_cols=BASE_NUM_COLS.copy(),
    )
    specs["orig_targetenc"] = VariantSpec(
        train=common_train.copy(),
        test=common_test.copy(),
        orig=common_orig.copy(),
        cat_cols=BASE_CAT_COLS.copy(),
        num_cols=BASE_NUM_COLS.copy(),
    )
    return specs


def make_target(series: pd.Series) -> np.ndarray:
    return series.map({"No": 0, "Yes": 1}).astype(np.float32).to_numpy()


def make_category_mappings(
    frames: list[pd.DataFrame],
    cat_cols: list[str],
) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        categories = sorted(
            pd.concat([safe_string(frame[col]) for frame in frames], ignore_index=True).unique()
        )
        mappings[col] = {value: idx for idx, value in enumerate(categories)}
    return mappings


def encode_categories(frame: pd.DataFrame, cat_cols: list[str], mappings: dict[str, dict[str, int]]) -> np.ndarray:
    arrays = []
    for col in cat_cols:
        mapped = safe_string(frame[col]).map(mappings[col]).astype(np.int64)
        arrays.append(mapped.to_numpy())
    return np.column_stack(arrays)


def compute_encoding_map(
    fit_keys: pd.Series,
    fit_target: np.ndarray,
    smoothing: float,
) -> tuple[pd.Series, pd.Series, float]:
    stats = pd.DataFrame({"key": fit_keys.to_numpy(), "target": fit_target})
    grouped = stats.groupby("key")["target"].agg(["sum", "count"])
    global_mean = float(stats["target"].mean())
    te = (grouped["sum"] + smoothing * global_mean) / (grouped["count"] + smoothing)
    freq = grouped["count"] / len(stats)
    return te, freq, global_mean


def add_target_encoded_features(
    train_frame: pd.DataFrame,
    train_target: np.ndarray,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    orig_frame: pd.DataFrame,
    orig_target: np.ndarray,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner_train = train_frame.copy()
    inner_val = val_frame.copy()
    inner_test = test_frame.copy()

    inner_cv = StratifiedKFold(
        n_splits=config.n_inner_target_encoding_splits,
        shuffle=True,
        random_state=config.target_encoding_seed,
    )

    for feature_name, cols in TARGET_ENCODING_SPECS.items():
        orig_keys = join_features(orig_frame, cols)
        train_keys = join_features(train_frame, cols)
        val_keys = join_features(val_frame, cols)
        test_keys = join_features(test_frame, cols)

        oof_te = np.zeros(len(train_frame), dtype=np.float32)
        oof_freq = np.zeros(len(train_frame), dtype=np.float32)

        for inner_fit_idx, inner_holdout_idx in inner_cv.split(train_frame, train_target):
            fit_keys = pd.concat(
                [train_keys.iloc[inner_fit_idx], orig_keys],
                ignore_index=True,
            )
            fit_target = np.concatenate([train_target[inner_fit_idx], orig_target])
            te_map, freq_map, global_mean = compute_encoding_map(
                fit_keys=fit_keys,
                fit_target=fit_target,
                smoothing=config.target_encoding_smoothing,
            )

            holdout_keys = train_keys.iloc[inner_holdout_idx]
            oof_te[inner_holdout_idx] = (
                holdout_keys.map(te_map).fillna(global_mean).astype(np.float32).to_numpy()
            )
            oof_freq[inner_holdout_idx] = (
                holdout_keys.map(freq_map).fillna(0.0).astype(np.float32).to_numpy()
            )

        full_fit_keys = pd.concat([train_keys, orig_keys], ignore_index=True)
        full_fit_target = np.concatenate([train_target, orig_target])
        full_te_map, full_freq_map, full_global_mean = compute_encoding_map(
            fit_keys=full_fit_keys,
            fit_target=full_fit_target,
            smoothing=config.target_encoding_smoothing,
        )

        inner_train[f"te_{feature_name}"] = oof_te
        inner_train[f"freq_{feature_name}"] = oof_freq
        inner_val[f"te_{feature_name}"] = (
            val_keys.map(full_te_map).fillna(full_global_mean).astype(np.float32)
        )
        inner_val[f"freq_{feature_name}"] = (
            val_keys.map(full_freq_map).fillna(0.0).astype(np.float32)
        )
        inner_test[f"te_{feature_name}"] = (
            test_keys.map(full_te_map).fillna(full_global_mean).astype(np.float32)
        )
        inner_test[f"freq_{feature_name}"] = (
            test_keys.map(full_freq_map).fillna(0.0).astype(np.float32)
        )

    return inner_train, inner_val, inner_test


def make_num_embeddings(
    kind: str,
    x_num_train: np.ndarray,
    config: TrainingConfig,
):
    from rtdl_num_embeddings import LinearReLUEmbeddings

    n_num_features = x_num_train.shape[1]
    if kind != "piecewise":
        return LinearReLUEmbeddings(n_num_features)

    try:
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

        bins_tensor = torch.as_tensor(x_num_train, dtype=torch.float32)
        try:
            bins = compute_bins(bins_tensor, n_bins=config.num_embedding_bins)
        except TypeError:
            bins = compute_bins(bins_tensor, config.num_embedding_bins)

        signature = inspect.signature(PiecewiseLinearEmbeddings)
        kwargs = {}
        if "n_features" in signature.parameters:
            kwargs["n_features"] = n_num_features
        if "n_num_features" in signature.parameters:
            kwargs["n_num_features"] = n_num_features
        if "bins" in signature.parameters:
            kwargs["bins"] = bins
        if "bin_edges" in signature.parameters:
            kwargs["bin_edges"] = bins
        if "d_embedding" in signature.parameters:
            kwargs["d_embedding"] = config.num_embedding_dim
        if "version" in signature.parameters:
            kwargs["version"] = "B"

        try:
            return PiecewiseLinearEmbeddings(**kwargs)
        except TypeError:
            pass

        positional_attempts = [
            (n_num_features, bins, config.num_embedding_dim),
            (bins, config.num_embedding_dim),
        ]
        for args in positional_attempts:
            try:
                return PiecewiseLinearEmbeddings(*args, version="B")
            except TypeError:
                continue
    except Exception as exc:  # pragma: no cover - runtime fallback
        print(f"Piecewise embeddings unavailable, falling back to LinearReLUEmbeddings: {exc}")

    return LinearReLUEmbeddings(n_num_features)


def build_model(
    x_num_train: np.ndarray,
    cat_cardinalities: list[int],
    config: TrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    from tabm import TabM

    num_embeddings = make_num_embeddings(
        kind=config.num_embeddings_kind,
        x_num_train=x_num_train,
        config=config,
    )
    model = TabM.make(
        n_num_features=x_num_train.shape[1],
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        num_embeddings=num_embeddings,
        arch_type="tabm",
        k=config.tabm_k,
    )
    model = model.to(device)

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if device.type == "cuda" and config.use_torch_compile and gpu_count <= 1:
        try:
            model = torch.compile(model, mode=config.compile_mode)
        except Exception as exc:  # pragma: no cover - runtime fallback
            print(f"torch.compile unavailable, continuing without it: {exc}")

    if device.type == "cuda" and config.use_data_parallel and gpu_count > 1:
        print(f"Using DataParallel across {gpu_count} GPUs")
        model = torch.nn.DataParallel(model)

    return model


def make_tensor_loader(
    x_num: np.ndarray,
    x_cat: np.ndarray,
    y: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    tensors = [
        torch.as_tensor(x_num, dtype=torch.float32),
        torch.as_tensor(x_cat, dtype=torch.long),
    ]
    if y is not None:
        tensors.append(torch.as_tensor(y, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict_proba(
    model: torch.nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    amp_enabled = device.type == "cuda"
    amp_dtype = get_amp_dtype()
    loader = make_tensor_loader(
        x_num,
        x_cat,
        None,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    preds: list[np.ndarray] = []
    for batch_num, batch_cat in loader:
        batch_num = batch_num.to(device, non_blocking=True)
        batch_cat = batch_cat.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(batch_num, batch_cat).squeeze(-1)
        probs = torch.sigmoid(logits).mean(dim=1)
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def train_fold(
    x_num_train: np.ndarray,
    x_cat_train: np.ndarray,
    cat_cardinalities: list[int],
    y_train: np.ndarray,
    x_num_val: np.ndarray,
    x_cat_val: np.ndarray,
    y_val: np.ndarray,
    x_num_test: np.ndarray,
    x_cat_test: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    variant_name: str,
    fold_index: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    model = build_model(x_num_train=x_num_train, cat_cardinalities=cat_cardinalities, config=config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    amp_enabled = config.use_amp and device.type == "cuda"
    amp_dtype = get_amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_batch_size = config.batch_size_gpu if device.type == "cuda" else config.batch_size_cpu
    eval_batch_size = (
        config.eval_batch_size_gpu if device.type == "cuda" else config.eval_batch_size_cpu
    )
    num_workers = get_num_workers(config, device)
    train_loader = make_tensor_loader(
        x_num_train,
        x_cat_train,
        y_train,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    best_auc = -np.inf
    best_state = None
    best_epoch = 0
    patience_counter = 0
    num_train_rows = len(y_train)
    num_val_rows = len(y_val)

    print(
        f"    [{variant_name}][fold {fold_index}] "
        f"train_rows={num_train_rows:,} val_rows={num_val_rows:,} "
        f"train_batch_size={train_batch_size:,} eval_batch_size={eval_batch_size:,} "
        f"k={config.tabm_k} max_epochs={config.max_epochs}"
    )

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_num, batch_cat, batch_target in train_loader:
            batch_num = batch_num.to(device, non_blocking=True)
            batch_cat = batch_cat.to(device, non_blocking=True)
            batch_target = batch_target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(batch_num, batch_cat).squeeze(-1)
                expanded_target = batch_target.unsqueeze(1).expand(-1, logits.shape[1])
                loss = F.binary_cross_entropy_with_logits(logits, expanded_target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            n_batches += 1

        val_pred = predict_proba(
            model=model,
            x_num=x_num_val,
            x_cat=x_cat_val,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        val_auc = roc_auc_score(y_val, val_pred)
        mean_loss = running_loss / max(n_batches, 1)

        if val_auc > best_auc + 1e-6:
            best_auc = float(val_auc)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            status = "improved"
        else:
            patience_counter += 1
            status = f"no_improve patience={patience_counter}/{config.patience}"
        if epoch == 1 or epoch % config.log_every_epochs == 0 or epoch == config.max_epochs:
            print(
                f"    [{variant_name}][fold {fold_index}] "
                f"epoch={epoch:03d}/{config.max_epochs} "
                f"loss={mean_loss:.6f} val_auc={val_auc:.6f} "
                f"best_auc={best_auc:.6f} best_epoch={best_epoch} {status}"
            )
        if patience_counter >= config.patience:
            print(
                f"    [{variant_name}][fold {fold_index}] "
                f"early_stopping at epoch={epoch} best_epoch={best_epoch} best_auc={best_auc:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    val_pred = predict_proba(
        model=model,
        x_num=x_num_val,
        x_cat=x_cat_val,
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    test_pred = predict_proba(
        model=model,
        x_num=x_num_test,
        x_cat=x_cat_test,
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    print(f"    best_epoch={best_epoch} best_val_auc={best_auc:.6f}")
    return val_pred, test_pred, float(best_auc)


def scale_numeric_features(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    num_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_num_train = scaler.fit_transform(train_frame[num_cols].astype(np.float32))
    x_num_val = scaler.transform(val_frame[num_cols].astype(np.float32))
    x_num_test = scaler.transform(test_frame[num_cols].astype(np.float32))
    return x_num_train.astype(np.float32), x_num_val.astype(np.float32), x_num_test.astype(np.float32)


def prepare_variant_fold_data(
    variant: str,
    spec: VariantSpec,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y_train_all: np.ndarray,
    y_orig: np.ndarray,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    train_frame = spec.train.iloc[train_idx].reset_index(drop=True).copy()
    val_frame = spec.train.iloc[val_idx].reset_index(drop=True).copy()
    test_frame = spec.test.reset_index(drop=True).copy()
    orig_frame = spec.orig.reset_index(drop=True).copy()
    train_target = y_train_all[train_idx]

    num_cols = spec.num_cols.copy()

    if variant == "orig_targetenc":
        train_frame, val_frame, test_frame = add_target_encoded_features(
            train_frame=train_frame,
            train_target=train_target,
            val_frame=val_frame,
            test_frame=test_frame,
            orig_frame=orig_frame,
            orig_target=y_orig,
            config=config,
        )
        added_num_cols = []
        for feature_name in TARGET_ENCODING_SPECS:
            added_num_cols.extend([f"te_{feature_name}", f"freq_{feature_name}"])
        num_cols = num_cols + added_num_cols

    mappings = make_category_mappings(
        frames=[train_frame, val_frame, test_frame, orig_frame],
        cat_cols=spec.cat_cols,
    )
    cat_cardinalities = [len(mappings[col]) for col in spec.cat_cols]
    x_cat_train = encode_categories(train_frame, spec.cat_cols, mappings)
    x_cat_val = encode_categories(val_frame, spec.cat_cols, mappings)
    x_cat_test = encode_categories(test_frame, spec.cat_cols, mappings)
    x_num_train, x_num_val, x_num_test = scale_numeric_features(
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        num_cols=num_cols,
    )
    return (
        x_num_train,
        x_cat_train,
        x_num_val,
        x_cat_val,
        x_num_test,
        x_cat_test,
        cat_cardinalities,
    )


def run_variant(
    variant: str,
    spec: VariantSpec,
    y_train: np.ndarray,
    y_orig: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros(len(spec.train), dtype=np.float32)
    test_fold_preds = []
    fold_scores = []
    splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.fold_seed,
    )

    for fold, (train_idx, val_idx) in enumerate(splitter.split(spec.train, y_train), start=1):
        print(f"[{variant}] fold {fold}/{config.n_splits}")
        (
            x_num_train,
            x_cat_train,
            x_num_val,
            x_cat_val,
            x_num_test,
            x_cat_test,
            cat_cardinalities,
        ) = prepare_variant_fold_data(
            variant=variant,
            spec=spec,
            train_idx=train_idx,
            val_idx=val_idx,
            y_train_all=y_train,
            y_orig=y_orig,
            config=config,
        )
        fold_oof, fold_test, fold_auc = train_fold(
            x_num_train=x_num_train,
            x_cat_train=x_cat_train,
            cat_cardinalities=cat_cardinalities,
            y_train=y_train[train_idx],
            x_num_val=x_num_val,
            x_cat_val=x_cat_val,
            y_val=y_train[val_idx],
            x_num_test=x_num_test,
            x_cat_test=x_cat_test,
            config=config,
            device=device,
            variant_name=variant,
            fold_index=fold,
        )
        oof[val_idx] = fold_oof
        test_fold_preds.append(fold_test)
        fold_scores.append(fold_auc)

    test_pred = np.mean(np.column_stack(test_fold_preds), axis=1).astype(np.float32)
    oof = np.clip(oof, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)
    np.save(output_dir / f"oof_tabm_{variant}.npy", oof)
    np.save(output_dir / f"test_tabm_{variant}.npy", test_pred)
    print(
        f"[{variant}] overall_oof_auc={roc_auc_score(y_train, oof):.6f} "
        f"mean_fold_auc={np.mean(fold_scores):.6f}"
    )
    return oof, test_pred, fold_scores


def meta_oof_predictions(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> np.ndarray:
    meta_splitter = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.meta_seed,
    )
    meta_oof = np.zeros(len(y), dtype=np.float32)
    for meta_train_idx, meta_val_idx in meta_splitter.split(x, y):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[meta_train_idx])
        x_val = scaler.transform(x[meta_val_idx])
        ridge = Ridge(alpha=config.ridge_alpha)
        ridge.fit(x_train, y[meta_train_idx])
        meta_oof[meta_val_idx] = ridge.predict(x_val)
    return np.clip(meta_oof, 0.0, 1.0)


def exhaustive_ridge_blend(
    oof_by_variant: dict[str, np.ndarray],
    test_by_variant: dict[str, np.ndarray],
    y_train: np.ndarray,
    config: TrainingConfig,
    output_dir: Path,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    variant_names = list(oof_by_variant)
    records = []
    best_subset: list[str] | None = None
    best_meta_oof = None
    best_score = -np.inf

    for subset_size in range(1, len(variant_names) + 1):
        for subset in combinations(variant_names, subset_size):
            x_subset = np.column_stack([oof_by_variant[name] for name in subset])
            subset_meta_oof = meta_oof_predictions(x_subset, y_train, config)
            subset_score = roc_auc_score(y_train, subset_meta_oof)
            records.append(
                {
                    "subset": "|".join(subset),
                    "n_models": len(subset),
                    "meta_oof_auc": float(subset_score),
                }
            )
            if subset_score > best_score:
                best_score = float(subset_score)
                best_subset = list(subset)
                best_meta_oof = subset_meta_oof

    if best_subset is None or best_meta_oof is None:
        raise RuntimeError("Subset search failed to produce a blend.")

    subset_matrix_oof = np.column_stack([oof_by_variant[name] for name in best_subset])
    subset_matrix_test = np.column_stack([test_by_variant[name] for name in best_subset])
    scaler = StandardScaler()
    x_oof_scaled = scaler.fit_transform(subset_matrix_oof)
    x_test_scaled = scaler.transform(subset_matrix_test)
    ridge = Ridge(alpha=config.ridge_alpha)
    ridge.fit(x_oof_scaled, y_train)
    final_test = np.clip(ridge.predict(x_test_scaled), 0.0, 1.0).astype(np.float32)

    np.save(output_dir / "oof_tabm_ridge.npy", best_meta_oof.astype(np.float32))
    np.save(output_dir / "test_tabm_ridge.npy", final_test)
    pd.DataFrame(records).sort_values(
        ["meta_oof_auc", "n_models"], ascending=[False, True]
    ).to_csv(output_dir / "blend_subset_scores.csv", index=False)
    print(f"[ridge] best_subset={best_subset} meta_oof_auc={best_score:.6f}")
    return best_subset, best_meta_oof.astype(np.float32), final_test


def run_pipeline(config: TrainingConfig) -> dict[str, object]:
    set_seed(config.fold_seed)
    device = resolve_device(config)
    runtime_info = configure_runtime(device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Runtime:", runtime_info)

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    orig_df = pd.read_csv(config.orig_path)

    if list(train_df.columns[:-1]) != list(test_df.columns):
        raise ValueError("train/test feature schemas do not align.")
    if set(orig_df.columns) - {ORIG_ID_COL} - set(train_df.columns):
        raise ValueError("Original data has unexpected columns.")

    y_train = make_target(train_df[TARGET])
    y_orig = make_target(orig_df[TARGET])
    specs = build_variant_specs(train_df=train_df, test_df=test_df, orig_df=orig_df)

    oof_by_variant: dict[str, np.ndarray] = {}
    test_by_variant: dict[str, np.ndarray] = {}
    variant_rows = []
    for variant in VARIANTS:
        print(f"\n=== training variant: {variant} ===")
        oof, test_pred, fold_scores = run_variant(
            variant=variant,
            spec=specs[variant],
            y_train=y_train,
            y_orig=y_orig,
            config=config,
            device=device,
            output_dir=config.output_dir,
        )
        oof_by_variant[variant] = oof
        test_by_variant[variant] = test_pred
        variant_rows.append(
            {
                "variant": variant,
                "oof_auc": float(roc_auc_score(y_train, oof)),
                "mean_fold_auc": float(np.mean(fold_scores)),
            }
        )

    pd.DataFrame(variant_rows).sort_values("oof_auc", ascending=False).to_csv(
        config.output_dir / "variant_scores.csv",
        index=False,
    )

    best_subset, meta_oof, final_test = exhaustive_ridge_blend(
        oof_by_variant=oof_by_variant,
        test_by_variant=test_by_variant,
        y_train=y_train,
        config=config,
        output_dir=config.output_dir,
    )
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: final_test})
    submission.to_csv(config.output_dir / "submission_tabm_ridge.csv", index=False)

    report = {
        "config": {
            **asdict(config),
            "train_path": str(config.train_path),
            "test_path": str(config.test_path),
            "orig_path": str(config.orig_path),
            "output_dir": str(config.output_dir),
            "resolved_device": str(device),
        },
        "runtime": runtime_info,
        "variant_scores": variant_rows,
        "best_subset": best_subset,
        "meta_oof_auc": float(roc_auc_score(y_train, meta_oof)),
    }
    (config.output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    print("\nArtifacts written to", config.output_dir)
    return report


def run_default_pipeline(device: str | None = None) -> dict[str, object]:
    config = TrainingConfig(device=device)
    return run_pipeline(config)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None
    except Exception:
        return False


def main() -> None:
    run_default_pipeline()


if __name__ == "__main__":
    if not is_notebook():
        main()
