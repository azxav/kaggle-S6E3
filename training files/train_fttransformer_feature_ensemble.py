from __future__ import annotations

import copy
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Config:
    train_path: Path = Path("/kaggle/input/competitions/playground-series-s6e3/train.csv")
    test_path: Path = Path("/kaggle/input/competitions/playground-series-s6e3/test.csv")
    original_path: Path | None = Path(
        "/kaggle/input/datasets/azizbekxasanov/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    output_dir: Path = Path("/kaggle/working/outputs/fttransformer")
    target_col: str = "Churn"
    id_col: str = "id"
    seed: int = 42
    run_mode: str = "submission"
    experiment_fraction: float = 0.1
    experiment_n_splits: int = 3
    submission_n_splits: int = 5
    save_parquet: bool = True
    use_external_stats: bool = True
    qcut_bins: int = 8
    cut_bins: int = 8
    rounded_bin_divisors: tuple[int, ...] = (5, 10)
    max_digit_numeric_cols: int = 8
    external_smoothing: float = 30.0
    external_min_count: int = 10
    channels: int = 32
    num_layers: int = 3
    batch_size: int = 1024
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    num_workers: int = 0
    use_amp: bool = True
    device: str | None = None
    selected_variants: list[str] | None = None
    list_variants_only: bool = False

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)
        self.output_dir = Path(self.output_dir)
        if self.original_path is not None:
            self.original_path = Path(self.original_path)
        if self.run_mode not in {"experiment", "submission"}:
            raise ValueError("run_mode must be 'experiment' or 'submission'")


@dataclass(frozen=True)
class FeatureVariantSpec:
    name: str
    feature_families: list[str]


@dataclass
class TorchFrameRuntimeConfig:
    seed: int
    channels: int = 32
    num_layers: int = 3
    batch_size: int = 1024
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    use_amp: bool = True
    num_workers: int = 0


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()
    except Exception:
        pass


def target_to_binary(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").fillna(0).astype(np.int8)
        uniq = set(values.unique().tolist())
        if not uniq.issubset({0, 1}):
            raise ValueError(f"Numeric target must be binary 0/1, got {sorted(uniq)}")
        return values.to_numpy(dtype=np.int8)
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    )
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported target labels: {bad_values[:10]}")
    return mapped.astype(np.int8).to_numpy()


def normalize_categorical(series: pd.Series) -> pd.Series:
    return (
        series.fillna("__MISSING__")
        .astype(str)
        .str.strip()
        .replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
    )


def normalize_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(np.float32)


def resolve_original_path(config: Config) -> Path | None:
    candidates: list[Path] = []
    if config.original_path is not None:
        candidates.append(config.original_path)
    candidates.extend(
        [
            Path("./original.csv"),
            Path("./orig-Telco-Customer-Churn.csv"),
            Path("./WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    external_df = None
    if config.use_external_stats:
        original_path = resolve_original_path(config)
        if original_path is not None:
            external_df = pd.read_csv(original_path)
            log(f"Using external data: {original_path}")
        else:
            log("External data not found. External-stat variants will be skipped.")
    return train_df, test_df, external_df


def detect_column_types(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    excluded = {target_col}
    id_like = {col for col in df.columns if col.lower() in {"id", "customerid"}}
    usable_cols = [col for col in df.columns if col not in excluded and col not in id_like]
    numeric_cols = [col for col in usable_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical_cols = [col for col in usable_cols if col not in numeric_cols]
    return numeric_cols, categorical_cols


def make_base_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    for col in numeric_cols:
        out_train[col] = normalize_numeric(train_df[col])
        out_valid[col] = normalize_numeric(valid_df[col])
        out_test[col] = normalize_numeric(test_df[col])
    for col in categorical_cols:
        out_train[col] = normalize_categorical(train_df[col])
        out_valid[col] = normalize_categorical(valid_df[col])
        out_test[col] = normalize_categorical(test_df[col])
    return out_train, out_valid, out_test


def build_binning_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    for col in numeric_cols:
        train_num = normalize_numeric(train_df[col])
        valid_num = normalize_numeric(valid_df[col])
        test_num = normalize_numeric(test_df[col])
        non_null = train_num.dropna()
        if non_null.nunique() < 3:
            continue

        quantiles = np.linspace(0.0, 1.0, config.qcut_bins + 1)
        q_edges = np.unique(non_null.quantile(quantiles).to_numpy(dtype=np.float64))
        if len(q_edges) >= 3:
            out_train[f"{col}__qbin"] = pd.cut(
                train_num, bins=q_edges, include_lowest=True, duplicates="drop"
            ).astype(str).replace({"nan": "__MISSING__"})
            out_valid[f"{col}__qbin"] = pd.cut(
                valid_num, bins=q_edges, include_lowest=True, duplicates="drop"
            ).astype(str).replace({"nan": "__MISSING__"})
            out_test[f"{col}__qbin"] = pd.cut(
                test_num, bins=q_edges, include_lowest=True, duplicates="drop"
            ).astype(str).replace({"nan": "__MISSING__"})

        low = float(non_null.min())
        high = float(non_null.max())
        if np.isfinite(low) and np.isfinite(high) and high > low:
            cut_edges = np.unique(np.linspace(low, high, config.cut_bins + 1).astype(np.float64))
            if len(cut_edges) >= 3:
                out_train[f"{col}__wbin"] = pd.cut(
                    train_num, bins=cut_edges, include_lowest=True, duplicates="drop"
                ).astype(str).replace({"nan": "__MISSING__"})
                out_valid[f"{col}__wbin"] = pd.cut(
                    valid_num, bins=cut_edges, include_lowest=True, duplicates="drop"
                ).astype(str).replace({"nan": "__MISSING__"})
                out_test[f"{col}__wbin"] = pd.cut(
                    test_num, bins=cut_edges, include_lowest=True, duplicates="drop"
                ).astype(str).replace({"nan": "__MISSING__"})

        for divisor in config.rounded_bin_divisors:
            if divisor <= 0:
                continue
            out_train[f"{col}__rounddiv_{divisor}"] = (
                np.floor(train_num / float(divisor)).fillna(-9999).astype(np.int32).astype(str)
            )
            out_valid[f"{col}__rounddiv_{divisor}"] = (
                np.floor(valid_num / float(divisor)).fillna(-9999).astype(np.int32).astype(str)
            )
            out_test[f"{col}__rounddiv_{divisor}"] = (
                np.floor(test_num / float(divisor)).fillna(-9999).astype(np.int32).astype(str)
            )
    return out_train, out_valid, out_test


def build_digit_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_cols = numeric_cols[: config.max_digit_numeric_cols]
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    for col in selected_cols:
        tr = normalize_numeric(train_df[col]).fillna(0.0)
        va = normalize_numeric(valid_df[col]).fillna(0.0)
        te = normalize_numeric(test_df[col]).fillna(0.0)

        def parts(series: pd.Series, prefix: str, out: pd.DataFrame) -> None:
            clipped = series.clip(-1_000_000, 1_000_000)
            abs_int = np.floor(np.abs(clipped)).astype(np.int64)
            out[f"{prefix}__sign"] = np.sign(clipped).astype(np.int8)
            out[f"{prefix}__units"] = (abs_int % 10).astype(np.int8)
            out[f"{prefix}__tens"] = ((abs_int // 10) % 10).astype(np.int8)
            out[f"{prefix}__hundreds"] = ((abs_int // 100) % 10).astype(np.int8)
            frac = np.floor((np.abs(clipped - np.floor(clipped))) * 100.0).astype(np.int64)
            out[f"{prefix}__d1"] = ((frac // 10) % 10).astype(np.int8)
            out[f"{prefix}__d2"] = (frac % 10).astype(np.int8)

        parts(tr, col, out_train)
        parts(va, col, out_valid)
        parts(te, col, out_test)
    return out_train, out_valid, out_test


def build_frequency_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    for col in cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            tr = normalize_numeric(train_df[col])
            va = normalize_numeric(valid_df[col])
            te = normalize_numeric(test_df[col])
        else:
            tr = normalize_categorical(train_df[col])
            va = normalize_categorical(valid_df[col])
            te = normalize_categorical(test_df[col])
        counts = tr.value_counts(dropna=False)
        total = max(1, len(tr))
        out_train[f"{col}__freq"] = tr.map(counts).fillna(0).astype(np.float32)
        out_valid[f"{col}__freq"] = va.map(counts).fillna(0).astype(np.float32)
        out_test[f"{col}__freq"] = te.map(counts).fillna(0).astype(np.float32)
        out_train[f"{col}__freq_pct"] = (out_train[f"{col}__freq"] / total).astype(np.float32)
        out_valid[f"{col}__freq_pct"] = (out_valid[f"{col}__freq"] / total).astype(np.float32)
        out_test[f"{col}__freq_pct"] = (out_test[f"{col}__freq"] / total).astype(np.float32)
    return out_train, out_valid, out_test


def build_external_stats_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    external_df: pd.DataFrame | None,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if external_df is None or config.target_col not in external_df.columns:
        return out_train, out_valid, out_test, False

    y_ext = target_to_binary(external_df[config.target_col])
    global_mean = float(np.mean(y_ext))
    global_pos = float(np.sum(y_ext))
    global_neg = float(len(y_ext) - global_pos)
    eps = 1e-6

    for col in cols:
        if col not in external_df.columns:
            continue
        ext_col = external_df[col]
        if pd.api.types.is_numeric_dtype(ext_col):
            ext_series = normalize_numeric(ext_col).round(3).astype(str)
            tr = normalize_numeric(train_df[col]).round(3).astype(str)
            va = normalize_numeric(valid_df[col]).round(3).astype(str)
            te = normalize_numeric(test_df[col]).round(3).astype(str)
        else:
            ext_series = normalize_categorical(ext_col)
            tr = normalize_categorical(train_df[col])
            va = normalize_categorical(valid_df[col])
            te = normalize_categorical(test_df[col])

        stats_df = pd.DataFrame({"key": ext_series, "target": y_ext})
        grouped = stats_df.groupby("key")["target"].agg(["sum", "count"])
        grouped = grouped[grouped["count"] >= config.external_min_count]
        if grouped.empty:
            continue

        mean_map = (grouped["sum"] / grouped["count"]).astype(np.float32)
        smooth_map = (
            (grouped["sum"] + config.external_smoothing * global_mean)
            / (grouped["count"] + config.external_smoothing)
        ).astype(np.float32)
        pos_rate = (grouped["sum"] + eps) / (grouped["count"] + 2 * eps)
        neg_rate = 1.0 - pos_rate
        entropy_map = (
            -(
                pos_rate * np.log2(np.clip(pos_rate, eps, 1.0))
                + neg_rate * np.log2(np.clip(neg_rate, eps, 1.0))
            )
        ).astype(np.float32)
        woe_map = (
            np.log((grouped["sum"] + eps) / (global_pos + eps))
            - np.log(((grouped["count"] - grouped["sum"]) + eps) / (global_neg + eps))
        ).astype(np.float32)

        maps = {
            f"{col}__ext_mean": mean_map,
            f"{col}__ext_smooth_mean": smooth_map,
            f"{col}__ext_woe": woe_map,
            f"{col}__ext_entropy": entropy_map,
        }
        for feat_name, feat_map in maps.items():
            out_train[feat_name] = tr.map(feat_map).fillna(global_mean).astype(np.float32)
            out_valid[feat_name] = va.map(feat_map).fillna(global_mean).astype(np.float32)
            out_test[feat_name] = te.map(feat_map).fillna(global_mean).astype(np.float32)
    return out_train, out_valid, out_test, True


def assemble_variant_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: FeatureVariantSpec,
    external_df: pd.DataFrame | None,
    config: Config,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    base_train, base_valid, base_test = make_base_frames(
        train_df, valid_df, test_df, numeric_cols, categorical_cols
    )
    final_train = pd.DataFrame(index=train_df.index)
    final_valid = pd.DataFrame(index=valid_df.index)
    final_test = pd.DataFrame(index=test_df.index)
    metadata: dict[str, Any] = {
        "status": "ok",
        "feature_families": list(variant.feature_families),
    }

    if "all_as_categorical" in variant.feature_families:
        all_cat_train = pd.concat(
            [base_train[numeric_cols].round(3).astype(str), base_train[categorical_cols]],
            axis=1,
        ).add_prefix("allcat__")
        all_cat_valid = pd.concat(
            [base_valid[numeric_cols].round(3).astype(str), base_valid[categorical_cols]],
            axis=1,
        ).add_prefix("allcat__")
        all_cat_test = pd.concat(
            [base_test[numeric_cols].round(3).astype(str), base_test[categorical_cols]],
            axis=1,
        ).add_prefix("allcat__")
        final_train = pd.concat([final_train, all_cat_train], axis=1)
        final_valid = pd.concat([final_valid, all_cat_valid], axis=1)
        final_test = pd.concat([final_test, all_cat_test], axis=1)
    else:
        final_train = pd.concat(
            [final_train, base_train[numeric_cols], base_train[categorical_cols]], axis=1
        )
        final_valid = pd.concat(
            [final_valid, base_valid[numeric_cols], base_valid[categorical_cols]], axis=1
        )
        final_test = pd.concat(
            [final_test, base_test[numeric_cols], base_test[categorical_cols]], axis=1
        )

    if "binning" in variant.feature_families:
        btr, bva, bte = build_binning_features(train_df, valid_df, test_df, numeric_cols, config)
        final_train = pd.concat([final_train, btr], axis=1)
        final_valid = pd.concat([final_valid, bva], axis=1)
        final_test = pd.concat([final_test, bte], axis=1)

    if "digit_features" in variant.feature_families:
        dtr, dva, dte = build_digit_features(train_df, valid_df, test_df, numeric_cols, config)
        final_train = pd.concat([final_train, dtr], axis=1)
        final_valid = pd.concat([final_valid, dva], axis=1)
        final_test = pd.concat([final_test, dte], axis=1)

    if "frequency_encoding" in variant.feature_families:
        ftr, fva, fte = build_frequency_features(
            train_df, valid_df, test_df, numeric_cols + categorical_cols
        )
        final_train = pd.concat([final_train, ftr], axis=1)
        final_valid = pd.concat([final_valid, fva], axis=1)
        final_test = pd.concat([final_test, fte], axis=1)

    if "external_stats" in variant.feature_families:
        etr, eva, ete, ok = build_external_stats_features(
            train_df, valid_df, test_df, numeric_cols + categorical_cols, external_df, config
        )
        if not ok:
            metadata["status"] = "skipped_missing_external"
            return None, None, None, metadata
        final_train = pd.concat([final_train, etr], axis=1)
        final_valid = pd.concat([final_valid, eva], axis=1)
        final_test = pd.concat([final_test, ete], axis=1)

    if final_train.empty:
        metadata["status"] = "skipped_empty_features"
        return None, None, None, metadata

    metadata["num_features_train"] = int(final_train.shape[1])
    return final_train, final_valid, final_test, metadata


def get_variants() -> list[FeatureVariantSpec]:
    return [
        FeatureVariantSpec("base_mixed", ["base"]),
        FeatureVariantSpec("base_plus_freq", ["base", "frequency_encoding"]),
        FeatureVariantSpec("base_plus_binning", ["base", "binning"]),
        FeatureVariantSpec("base_plus_digits", ["base", "digit_features"]),
        FeatureVariantSpec("all_as_categorical", ["all_as_categorical"]),
        FeatureVariantSpec("base_plus_external_stats", ["base", "external_stats"]),
        FeatureVariantSpec("hybrid_light", ["base", "binning", "frequency_encoding"]),
        FeatureVariantSpec(
            "hybrid_full_without_gp",
            ["base", "binning", "digit_features", "frequency_encoding", "external_stats"],
        ),
    ]


def select_variants(
    all_variants: list[FeatureVariantSpec],
    requested_names: list[str] | None,
) -> list[FeatureVariantSpec]:
    if not requested_names:
        return all_variants

    variant_map = {variant.name: variant for variant in all_variants}
    missing = [name for name in requested_names if name not in variant_map]
    if missing:
        available = ", ".join(sorted(variant_map))
        raise ValueError(f"Unknown variant(s): {missing}. Available variants: {available}")

    selected: list[FeatureVariantSpec] = []
    seen: set[str] = set()
    for name in requested_names:
        if name not in seen:
            selected.append(variant_map[name])
            seen.add(name)
    return selected


def save_manifest(df: pd.DataFrame, path: Path, save_parquet: bool) -> None:
    if save_parquet:
        try:
            df.to_parquet(path)
            return
        except Exception as exc:
            log(f"Parquet save failed for {path.name}, falling back to CSV: {exc}")
    df.to_csv(path.with_suffix(".csv"), index=False)


def make_experiment_subset(
    train_df: pd.DataFrame,
    y: np.ndarray,
    config: Config,
) -> tuple[pd.DataFrame, np.ndarray]:
    sample_size = max(1, int(len(train_df) * config.experiment_fraction))
    rng = np.random.default_rng(config.seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_target = max(1, int(round(sample_size * (len(pos_idx) / len(train_df)))))
    neg_target = max(1, sample_size - pos_target)
    chosen_pos = rng.choice(pos_idx, size=min(pos_target, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(neg_target, len(neg_idx)), replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    if len(chosen) < sample_size:
        remaining = np.setdiff1d(np.arange(len(train_df)), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=sample_size - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    chosen = np.sort(chosen)
    return train_df.iloc[chosen].reset_index(drop=True), y[chosen].copy()


def import_torch_frame() -> dict[str, Any]:
    try:
        import torch
        import torch_frame
        from torch_frame.data import DataLoader, Dataset
    except Exception as exc:
        raise ImportError(
            "train_fttransformer_feature_ensemble.py requires torch and pytorch-frame. "
            "Install them in the Kaggle notebook, for example: pip install pytorch-frame"
        ) from exc

    try:
        from torch_frame.nn.models import FTTransformer
    except Exception:
        from torch_frame.nn.models.ft_transformer import FTTransformer  # type: ignore

    return {
        "torch": torch,
        "torch_frame": torch_frame,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "FTTransformer": FTTransformer,
    }


def resolve_device(config: Config):
    torch = import_torch_frame()["torch"]
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_col_to_stype(df: pd.DataFrame):
    deps = import_torch_frame()
    stype = deps["torch_frame"].stype
    col_to_stype: dict[str, Any] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_to_stype[col] = stype.numerical
        else:
            col_to_stype[col] = stype.categorical
    return col_to_stype


def materialize_split_datasets(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
):
    deps = import_torch_frame()
    Dataset = deps["Dataset"]
    target_col = "__target__"

    train_df = X_train.copy()
    valid_df = X_valid.copy()
    test_df = X_test.copy()
    train_df[target_col] = y_train.astype(np.float32)
    valid_df[target_col] = y_valid.astype(np.float32)
    col_to_stype = infer_col_to_stype(X_train)

    train_dataset = Dataset(train_df, col_to_stype=col_to_stype, target_col=target_col).materialize()
    valid_dataset = Dataset(valid_df, col_to_stype=col_to_stype, target_col=target_col).materialize(
        col_stats=train_dataset.col_stats
    )
    test_dataset = Dataset(test_df, col_to_stype=col_to_stype, target_col=None).materialize(
        col_stats=train_dataset.col_stats
    )
    return train_dataset, valid_dataset, test_dataset


def build_runtime_config(config: Config) -> TorchFrameRuntimeConfig:
    return TorchFrameRuntimeConfig(
        seed=config.seed,
        channels=config.channels,
        num_layers=config.num_layers,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_epochs=config.max_epochs,
        patience=config.patience,
        use_amp=config.use_amp,
        num_workers=config.num_workers,
    )


def build_model(train_dataset, runtime_config: TorchFrameRuntimeConfig):
    FTTransformer = import_torch_frame()["FTTransformer"]
    return FTTransformer(
        channels=runtime_config.channels,
        out_channels=1,
        num_layers=runtime_config.num_layers,
        col_stats=train_dataset.col_stats,
        col_names_dict=train_dataset.tensor_frame.col_names_dict,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    DataLoader = import_torch_frame()["DataLoader"]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def predict_proba(model, dataset, batch_size: int, device, num_workers: int) -> np.ndarray:
    deps = import_torch_frame()
    torch = deps["torch"]
    loader = make_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch).view(-1)
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def extract_targets(dataset) -> np.ndarray:
    if hasattr(dataset, "y") and dataset.y is not None:
        return dataset.y.detach().cpu().numpy().astype(np.float32)
    if hasattr(dataset, "tensor_frame") and hasattr(dataset.tensor_frame, "y") and dataset.tensor_frame.y is not None:
        return dataset.tensor_frame.y.detach().cpu().numpy().astype(np.float32)
    raise ValueError("Validation targets were not found on the torch-frame dataset")


def train_one_fold(
    train_dataset,
    valid_dataset,
    test_dataset,
    runtime_config: TorchFrameRuntimeConfig,
    device,
) -> tuple[Any, np.ndarray, np.ndarray]:
    deps = import_torch_frame()
    torch = deps["torch"]
    model = build_model(train_dataset, runtime_config).to(device)
    train_loader = make_loader(
        train_dataset,
        batch_size=runtime_config.batch_size,
        shuffle=True,
        num_workers=runtime_config.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=runtime_config.learning_rate,
        weight_decay=runtime_config.weight_decay,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    amp_enabled = bool(runtime_config.use_amp and device.type == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_auc = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_left = runtime_config.patience
    y_valid = extract_targets(valid_dataset)

    for epoch in range(1, runtime_config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch).view(-1)
                y_true = batch.y.view(-1).float()
                loss = criterion(logits, y_true)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().item()) * len(y_true)
            seen += len(y_true)

        valid_pred = predict_proba(
            model,
            valid_dataset,
            batch_size=runtime_config.batch_size,
            device=device,
            num_workers=runtime_config.num_workers,
        )
        val_auc = float(roc_auc_score(y_valid, valid_pred))
        log(
            f"epoch={epoch:02d} loss={running_loss / max(1, seen):.5f} "
            f"val_auc={val_auc:.6f}"
        )
        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_left = runtime_config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    valid_pred = predict_proba(
        model,
        valid_dataset,
        batch_size=runtime_config.batch_size,
        device=device,
        num_workers=runtime_config.num_workers,
    )
    test_pred = predict_proba(
        model,
        test_dataset,
        batch_size=runtime_config.batch_size,
        device=device,
        num_workers=runtime_config.num_workers,
    )
    return model, valid_pred, test_pred


def mode_output_dir(config: Config) -> Path:
    return config.output_dir / config.run_mode


def mode_n_splits(config: Config) -> int:
    return config.experiment_n_splits if config.run_mode == "experiment" else config.submission_n_splits


def run_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_df: pd.DataFrame | None,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: FeatureVariantSpec,
    config: Config,
    device,
) -> dict[str, Any]:
    start = time.time()
    runtime_config = build_runtime_config(config)
    oof = np.zeros(len(train_df), dtype=np.float32)
    test_pred_sum = np.zeros(len(test_df), dtype=np.float64)
    fold_scores: list[float] = []
    feature_counts: list[int] = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_valid = train_df.iloc[va_idx].reset_index(drop=True)
        fold_test = test_df.reset_index(drop=True)
        y_train = y[tr_idx]
        y_valid = y[va_idx]

        X_train, X_valid, X_test, meta = assemble_variant_features(
            fold_train,
            fold_valid,
            fold_test,
            numeric_cols,
            categorical_cols,
            variant,
            external_df,
            config,
        )
        if meta["status"] != "ok":
            return {
                "variant_name": variant.name,
                "run_mode": config.run_mode,
                "rows_used": int(len(train_df)),
                "cv_auc": np.nan,
                "feature_family_list": "|".join(variant.feature_families),
                "num_features_train": 0,
                "fit_seconds": round(time.time() - start, 3),
                "status": meta["status"],
            }

        feature_counts.append(meta["num_features_train"])
        log(
            f"Variant={variant.name} fold={fold + 1}/{len(splits)} "
            f"shape_train={X_train.shape} shape_valid={X_valid.shape} device={device}"
        )

        train_dataset, valid_dataset, test_dataset = materialize_split_datasets(
            X_train, y_train, X_valid, y_valid, X_test
        )
        model, valid_pred, test_fold_pred = train_one_fold(
            train_dataset, valid_dataset, test_dataset, runtime_config, device
        )
        oof[va_idx] = valid_pred.astype(np.float32)
        fold_scores.append(float(roc_auc_score(y_valid, valid_pred)))
        if config.run_mode == "submission":
            test_pred_sum += test_fold_pred.astype(np.float64)

        del X_train, X_valid, X_test, train_dataset, valid_dataset, test_dataset, model
        gc.collect()

    cv_auc = float(roc_auc_score(y, oof))
    out_dir = mode_output_dir(config)
    with open(out_dir / f"{variant.name}_{config.run_mode}_fold_scores.json", "w", encoding="utf-8") as f:
        json.dump({"fold_auc": fold_scores, "variant": variant.name, "run_mode": config.run_mode}, f, indent=2)

    row = {
        "variant_name": variant.name,
        "run_mode": config.run_mode,
        "rows_used": int(len(train_df)),
        "cv_auc": cv_auc,
        "feature_family_list": "|".join(variant.feature_families),
        "num_features_train": int(np.median(feature_counts)) if feature_counts else 0,
        "fit_seconds": round(time.time() - start, 3),
        "status": "ok",
    }
    if config.run_mode == "submission":
        test_pred = (test_pred_sum / len(splits)).astype(np.float32)
        np.save(out_dir / f"oof_fttransformer_{variant.name}.npy", oof.astype(np.float32))
        np.save(out_dir / f"test_fttransformer_{variant.name}.npy", test_pred)
    return row


def main() -> None:
    config = Config()
    all_variants = get_variants()
    if config.list_variants_only:
        for variant in all_variants:
            print(variant.name)
        return

    out_dir = mode_output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = resolve_device(config)
    log(f"Config: {asdict(config)}")

    train_df, test_df, external_df = load_data(config)
    y_full = target_to_binary(train_df[config.target_col])
    if config.run_mode == "experiment":
        train_df, y = make_experiment_subset(train_df, y_full, config)
    else:
        y = y_full

    numeric_cols, categorical_cols = detect_column_types(train_df, config.target_col)
    log(
        f"Loaded train={train_df.shape} test={test_df.shape} "
        f"numeric_cols={len(numeric_cols)} categorical_cols={len(categorical_cols)} device={device}"
    )

    skf = StratifiedKFold(n_splits=mode_n_splits(config), shuffle=True, random_state=config.seed)
    splits = list(skf.split(train_df, y))
    variants = select_variants(all_variants, config.selected_variants)
    log(f"Selected variants: {[variant.name for variant in variants]}")

    summary_rows: list[dict[str, Any]] = []
    oof_manifest = pd.DataFrame(index=np.arange(len(train_df)))
    test_manifest = pd.DataFrame(index=np.arange(len(test_df)))

    for variant in variants:
        log(f"Starting variant: {variant.name}")
        try:
            row = run_variant(
                train_df=train_df,
                test_df=test_df,
                external_df=external_df,
                y=y,
                splits=splits,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                variant=variant,
                config=config,
                device=device,
            )
            summary_rows.append(row)
            if config.run_mode == "submission" and row["status"] == "ok":
                oof_manifest[variant.name] = np.load(out_dir / f"oof_fttransformer_{variant.name}.npy")
                test_manifest[variant.name] = np.load(out_dir / f"test_fttransformer_{variant.name}.npy")
            log(f"Finished variant={variant.name} status={row['status']} cv_auc={row['cv_auc']}")
        except Exception as exc:
            log(f"Variant failed: {variant.name} -> {type(exc).__name__}: {exc}")
            summary_rows.append(
                {
                    "variant_name": variant.name,
                    "run_mode": config.run_mode,
                    "rows_used": int(len(train_df)),
                    "cv_auc": np.nan,
                    "feature_family_list": "|".join(variant.feature_families),
                    "num_features_train": 0,
                    "fit_seconds": np.nan,
                    "status": f"failed_{type(exc).__name__}",
                }
            )
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / f"summary_fttransformer_{config.run_mode}.csv", index=False)
    if config.run_mode == "submission":
        save_manifest(
            oof_manifest.reset_index(drop=True),
            out_dir / "all_oof_fttransformer_submission.parquet",
            config.save_parquet,
        )
        save_manifest(
            test_manifest.reset_index(drop=True),
            out_dir / "all_test_fttransformer_submission.parquet",
            config.save_parquet,
        )
    log("FTTransformer feature ensemble run complete.")


if __name__ == "__main__":
    main()
