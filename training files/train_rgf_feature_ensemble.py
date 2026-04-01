# !pip install rgf-python
# !pip install gplearn

from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from rgf.sklearn import RGFClassifier
except Exception as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "train_rgf_feature_ensemble.py requires rgf-python. "
        "Install it in the notebook environment, for example: pip install rgf-python"
    ) from exc


@dataclass
class Config:
    train_path: Path = Path("./train.csv")
    test_path: Path = Path("./test.csv")
    original_path: Path | None = None
    output_dir: Path = Path("./outputs/rgf")
    target_col: str = "Churn"
    id_col: str = "id"
    seed: int = 42
    n_splits: int = 5
    n_jobs: int = -1
    use_external_stats: bool = True
    use_gp: bool = True
    save_parquet: bool = True
    gp_max_features: int = 4
    gp_numeric_cap: int = 6
    gp_generations: int = 4
    gp_population_size: int = 100
    max_digit_numeric_cols: int = 8
    qcut_bins: int = 8
    cut_bins: int = 8
    rounded_bin_divisors: tuple[int, ...] = (5, 10)
    external_smoothing: float = 30.0
    external_min_count: int = 10
    freq_include_numeric: bool = True
    sparse_one_hot_max_categories: int = 32
    fill_value_numeric: float = -9999.0
    selected_variants: list[str] | None = None
    list_variants_only: bool = False

    def __post_init__(self) -> None:
        self.train_path = Path(self.train_path)
        self.test_path = Path(self.test_path)
        self.output_dir = Path(self.output_dir)
        if self.original_path is not None:
            self.original_path = Path(self.original_path)


@dataclass
class VariantSpec:
    name: str
    feature_families: list[str]
    model_params: dict[str, Any] = field(default_factory=dict)
    rounded_only: bool = False
    use_sparse: bool = False


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
    candidates = []
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
    rounded_only: bool = False,
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
        if not rounded_only:
            quantiles = np.linspace(0.0, 1.0, config.qcut_bins + 1)
            q_edges = np.unique(non_null.quantile(quantiles).to_numpy(dtype=np.float64))
            if len(q_edges) >= 3:
                out_train[f"{col}__qbin"] = pd.cut(
                    train_num,
                    bins=q_edges,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str).replace({"nan": "__MISSING__"})
                out_valid[f"{col}__qbin"] = pd.cut(
                    valid_num,
                    bins=q_edges,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str).replace({"nan": "__MISSING__"})
                out_test[f"{col}__qbin"] = pd.cut(
                    test_num,
                    bins=q_edges,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(str).replace({"nan": "__MISSING__"})

            low = float(non_null.min())
            high = float(non_null.max())
            if np.isfinite(low) and np.isfinite(high) and high > low:
                cut_edges = np.linspace(low, high, config.cut_bins + 1)
                cut_edges = np.unique(cut_edges.astype(np.float64))
                if len(cut_edges) >= 3:
                    out_train[f"{col}__wbin"] = pd.cut(
                        train_num,
                        bins=cut_edges,
                        include_lowest=True,
                        duplicates="drop",
                    ).astype(str).replace({"nan": "__MISSING__"})
                    out_valid[f"{col}__wbin"] = pd.cut(
                        valid_num,
                        bins=cut_edges,
                        include_lowest=True,
                        duplicates="drop",
                    ).astype(str).replace({"nan": "__MISSING__"})
                    out_test[f"{col}__wbin"] = pd.cut(
                        test_num,
                        bins=cut_edges,
                        include_lowest=True,
                        duplicates="drop",
                    ).astype(str).replace({"nan": "__MISSING__"})

        for divisor in config.rounded_bin_divisors:
            scale = float(divisor)
            if scale <= 0:
                continue
            out_train[f"{col}__rounddiv_{divisor}"] = (
                np.floor(train_num / scale).fillna(-9999).astype(np.int32).astype(str)
            )
            out_valid[f"{col}__rounddiv_{divisor}"] = (
                np.floor(valid_num / scale).fillna(-9999).astype(np.int32).astype(str)
            )
            out_test[f"{col}__rounddiv_{divisor}"] = (
                np.floor(test_num / scale).fillna(-9999).astype(np.int32).astype(str)
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
        train_num = normalize_numeric(train_df[col]).fillna(0.0)
        valid_num = normalize_numeric(valid_df[col]).fillna(0.0)
        test_num = normalize_numeric(test_df[col]).fillna(0.0)

        def _parts(series: pd.Series, prefix: str, out: pd.DataFrame) -> None:
            clipped = series.clip(-1_000_000, 1_000_000)
            abs_int = np.floor(np.abs(clipped)).astype(np.int64)
            out[f"{prefix}__sign"] = np.sign(clipped).astype(np.int8)
            out[f"{prefix}__units"] = (abs_int % 10).astype(np.int8)
            out[f"{prefix}__tens"] = ((abs_int // 10) % 10).astype(np.int8)
            out[f"{prefix}__hundreds"] = ((abs_int // 100) % 10).astype(np.int8)
            scaled = np.floor((np.abs(clipped - np.floor(clipped))) * 100.0).astype(np.int64)
            out[f"{prefix}__d1"] = ((scaled // 10) % 10).astype(np.int8)
            out[f"{prefix}__d2"] = (scaled % 10).astype(np.int8)

        _parts(train_num, col, out_train)
        _parts(valid_num, col, out_valid)
        _parts(test_num, col, out_test)
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
        train_series = (
            normalize_numeric(train_df[col]).round(3).astype(str)
            if pd.api.types.is_numeric_dtype(train_df[col])
            else normalize_categorical(train_df[col])
        )
        valid_series = (
            normalize_numeric(valid_df[col]).round(3).astype(str)
            if pd.api.types.is_numeric_dtype(valid_df[col])
            else normalize_categorical(valid_df[col])
        )
        test_series = (
            normalize_numeric(test_df[col]).round(3).astype(str)
            if pd.api.types.is_numeric_dtype(test_df[col])
            else normalize_categorical(test_df[col])
        )
        freq_map = train_series.value_counts(dropna=False, normalize=True)
        out_train[f"{col}__freq"] = train_series.map(freq_map).fillna(0.0).astype(np.float32)
        out_valid[f"{col}__freq"] = valid_series.map(freq_map).fillna(0.0).astype(np.float32)
        out_test[f"{col}__freq"] = test_series.map(freq_map).fillna(0.0).astype(np.float32)
    return out_train, out_valid, out_test


def build_external_stats_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    external_df: pd.DataFrame | None,
    cols: list[str],
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if external_df is None or target_col not in external_df.columns:
        return out_train, out_valid, out_test, False

    ext_target = target_to_binary(external_df[target_col])
    train_target = target_to_binary(train_df[target_col])
    eps = 1e-6

    for col in cols:
        if col not in external_df.columns:
            continue
        if pd.api.types.is_numeric_dtype(train_df[col]):
            tr_key = normalize_numeric(train_df[col]).round(3).astype(str)
            va_key = normalize_numeric(valid_df[col]).round(3).astype(str)
            te_key = normalize_numeric(test_df[col]).round(3).astype(str)
            ex_key = normalize_numeric(external_df[col]).round(3).astype(str)
        else:
            tr_key = normalize_categorical(train_df[col])
            va_key = normalize_categorical(valid_df[col])
            te_key = normalize_categorical(test_df[col])
            ex_key = normalize_categorical(external_df[col])

        stats_frame = pd.DataFrame(
            {
                "key": pd.concat([tr_key, ex_key], axis=0, ignore_index=True),
                "target": np.concatenate([train_target, ext_target]),
            }
        )
        grouped = stats_frame.groupby("key")["target"].agg(["sum", "count"])
        grouped = grouped[grouped["count"] >= config.external_min_count]
        if grouped.empty:
            continue

        global_mean = float(stats_frame["target"].mean())
        global_pos = float(stats_frame["target"].sum())
        global_neg = float(len(stats_frame) - global_pos)
        pos_rate = (grouped["sum"] + eps) / (grouped["count"] + 2 * eps)
        neg_rate = 1.0 - pos_rate
        smoothed = (
            grouped["sum"] + config.external_smoothing * global_mean
        ) / (grouped["count"] + config.external_smoothing)
        entropy = -(
            pos_rate * np.log2(np.clip(pos_rate, eps, 1.0))
            + neg_rate * np.log2(np.clip(neg_rate, eps, 1.0))
        )
        woe = np.log((grouped["sum"] + eps) / (global_pos + eps)) - np.log(
            ((grouped["count"] - grouped["sum"]) + eps) / (global_neg + eps)
        )

        mappings = {
            f"{col}__ext_mean": (grouped["sum"] / grouped["count"]).astype(np.float32),
            f"{col}__ext_smooth_mean": smoothed.astype(np.float32),
            f"{col}__ext_woe": woe.astype(np.float32),
            f"{col}__ext_entropy": entropy.astype(np.float32),
            f"{col}__ext_count": grouped["count"].astype(np.float32),
        }
        for feat_name, feat_map in mappings.items():
            out_train[feat_name] = tr_key.map(feat_map).fillna(global_mean).astype(np.float32)
            out_valid[feat_name] = va_key.map(feat_map).fillna(global_mean).astype(np.float32)
            out_test[feat_name] = te_key.map(feat_map).fillna(global_mean).astype(np.float32)

    return out_train, out_valid, out_test, True


def build_gp_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    target: np.ndarray,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if not config.use_gp:
        return out_train, out_valid, out_test, False
    try:
        from gplearn.genetic import SymbolicTransformer
    except Exception as exc:
        warnings.warn(f"gplearn unavailable. GP features skipped: {exc}")
        return out_train, out_valid, out_test, False

    selected_cols = numeric_cols[: config.gp_numeric_cap]
    if not selected_cols:
        return out_train, out_valid, out_test, False

    X_train = (
        train_df[selected_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    X_valid = (
        valid_df[selected_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    X_test = (
        test_df[selected_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    try:
        transformer = SymbolicTransformer(
            generations=config.gp_generations,
            population_size=config.gp_population_size,
            hall_of_fame=config.gp_max_features,
            n_components=config.gp_max_features,
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs"),
            parsimony_coefficient=0.001,
            max_samples=0.7,
            random_state=config.seed,
            n_jobs=1,
            verbose=0,
        )
        transformer.fit(X_train, target)
        gp_train = transformer.transform(X_train)
        gp_valid = transformer.transform(X_valid)
        gp_test = transformer.transform(X_test)
        for idx in range(gp_train.shape[1]):
            name = f"gp_feature_{idx}"
            out_train[name] = gp_train[:, idx].astype(np.float32)
            out_valid[name] = gp_valid[:, idx].astype(np.float32)
            out_test[name] = gp_test[:, idx].astype(np.float32)
        return out_train, out_valid, out_test, True
    except Exception as exc:
        warnings.warn(f"GP feature generation failed. GP features skipped: {exc}")
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=valid_df.index), pd.DataFrame(index=test_df.index), False


def factorize_frame(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    for col in train_df.columns:
        tr = normalize_categorical(train_df[col])
        va = normalize_categorical(valid_df[col])
        te = normalize_categorical(test_df[col])
        uniq = pd.Index(tr.unique())
        mapping = {value: idx + 1 for idx, value in enumerate(uniq)}
        out_train[col] = tr.map(mapping).fillna(0).astype(np.int32)
        out_valid[col] = va.map(mapping).fillna(0).astype(np.int32)
        out_test[col] = te.map(mapping).fillna(0).astype(np.int32)
    return out_train, out_valid, out_test


def assemble_variant_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: VariantSpec,
    external_df: pd.DataFrame | None,
    config: Config,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    base_train, base_valid, base_test = make_base_frames(
        train_df, valid_df, test_df, numeric_cols, categorical_cols
    )
    numeric_parts_train = [base_train[numeric_cols].copy()]
    numeric_parts_valid = [base_valid[numeric_cols].copy()]
    numeric_parts_test = [base_test[numeric_cols].copy()]
    categorical_parts_train: list[pd.DataFrame] = []
    categorical_parts_valid: list[pd.DataFrame] = []
    categorical_parts_test: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {
        "status": "ok",
        "used_gp": False,
        "used_external": False,
        "feature_families": list(variant.feature_families),
    }

    if "all_as_categorical" in variant.feature_families:
        all_cat_train = pd.concat(
            [base_train[numeric_cols].round(3).astype(str), base_train[categorical_cols]],
            axis=1,
        )
        all_cat_valid = pd.concat(
            [base_valid[numeric_cols].round(3).astype(str), base_valid[categorical_cols]],
            axis=1,
        )
        all_cat_test = pd.concat(
            [base_test[numeric_cols].round(3).astype(str), base_test[categorical_cols]],
            axis=1,
        )
        categorical_parts_train.append(all_cat_train.add_prefix("allcat__"))
        categorical_parts_valid.append(all_cat_valid.add_prefix("allcat__"))
        categorical_parts_test.append(all_cat_test.add_prefix("allcat__"))

    if "binning" in variant.feature_families:
        btr, bva, bte = build_binning_features(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            numeric_cols=numeric_cols,
            config=config,
            rounded_only=variant.rounded_only,
        )
        categorical_parts_train.append(btr)
        categorical_parts_valid.append(bva)
        categorical_parts_test.append(bte)

    if "digit_features" in variant.feature_families:
        dtr, dva, dte = build_digit_features(train_df, valid_df, test_df, numeric_cols, config)
        numeric_parts_train.append(dtr)
        numeric_parts_valid.append(dva)
        numeric_parts_test.append(dte)

    if "frequency_encoding" in variant.feature_families:
        freq_cols = categorical_cols + (numeric_cols if config.freq_include_numeric else [])
        ftr, fva, fte = build_frequency_features(train_df, valid_df, test_df, freq_cols)
        numeric_parts_train.append(ftr)
        numeric_parts_valid.append(fva)
        numeric_parts_test.append(fte)

    if "external_stats" in variant.feature_families:
        etr, eva, ete, ok = build_external_stats_features(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            target_col=config.target_col,
            external_df=external_df,
            cols=numeric_cols + categorical_cols,
            config=config,
        )
        if not ok:
            metadata["status"] = "skipped_missing_external"
            return None, None, None, metadata
        metadata["used_external"] = True
        numeric_parts_train.append(etr)
        numeric_parts_valid.append(eva)
        numeric_parts_test.append(ete)

    if "gp_features" in variant.feature_families:
        gtr, gva, gte, ok = build_gp_features(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            numeric_cols=numeric_cols,
            target=y_train,
            config=config,
        )
        if not ok:
            metadata["status"] = "skipped_gp_unavailable"
            return None, None, None, metadata
        metadata["used_gp"] = True
        numeric_parts_train.append(gtr)
        numeric_parts_valid.append(gva)
        numeric_parts_test.append(gte)

    numeric_train = pd.concat(numeric_parts_train, axis=1)
    numeric_valid = pd.concat(numeric_parts_valid, axis=1)
    numeric_test = pd.concat(numeric_parts_test, axis=1)

    categorical_train = (
        pd.concat(categorical_parts_train, axis=1) if categorical_parts_train else pd.DataFrame(index=train_df.index)
    )
    categorical_valid = (
        pd.concat(categorical_parts_valid, axis=1) if categorical_parts_valid else pd.DataFrame(index=valid_df.index)
    )
    categorical_test = (
        pd.concat(categorical_parts_test, axis=1) if categorical_parts_test else pd.DataFrame(index=test_df.index)
    )

    cat_num_train = pd.DataFrame(index=train_df.index)
    cat_num_valid = pd.DataFrame(index=valid_df.index)
    cat_num_test = pd.DataFrame(index=test_df.index)
    if not categorical_train.empty:
        cat_num_train, cat_num_valid, cat_num_test = factorize_frame(
            categorical_train, categorical_valid, categorical_test
        )

    final_train = pd.concat([numeric_train, cat_num_train], axis=1)
    final_valid = pd.concat([numeric_valid, cat_num_valid], axis=1)
    final_test = pd.concat([numeric_test, cat_num_test], axis=1)
    if final_train.empty:
        metadata["status"] = "skipped_empty_features"
        return None, None, None, metadata

    final_train = final_train.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)
    final_valid = final_valid.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)
    final_test = final_test.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)
    metadata["num_features_train"] = int(final_train.shape[1])
    return final_train, final_valid, final_test, metadata


def maybe_to_sparse(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variant: VariantSpec,
) -> tuple[Any, Any, Any]:
    if not variant.use_sparse:
        return (
            train_df.to_numpy(dtype=np.float32),
            valid_df.to_numpy(dtype=np.float32),
            test_df.to_numpy(dtype=np.float32),
        )
    try:
        from scipy import sparse
        from sklearn.preprocessing import OneHotEncoder
    except Exception:
        return (
            train_df.to_numpy(dtype=np.float32),
            valid_df.to_numpy(dtype=np.float32),
            test_df.to_numpy(dtype=np.float32),
        )

    low_card_cols = [col for col in train_df.columns if train_df[col].nunique(dropna=False) <= 16]
    dense_cols = [col for col in train_df.columns if col not in low_card_cols]
    if not low_card_cols:
        return (
            train_df.to_numpy(dtype=np.float32),
            valid_df.to_numpy(dtype=np.float32),
            test_df.to_numpy(dtype=np.float32),
        )

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    tr_low = enc.fit_transform(train_df[low_card_cols].astype(np.int32))
    va_low = enc.transform(valid_df[low_card_cols].astype(np.int32))
    te_low = enc.transform(test_df[low_card_cols].astype(np.int32))
    tr_dense = sparse.csr_matrix(train_df[dense_cols].to_numpy(dtype=np.float32))
    va_dense = sparse.csr_matrix(valid_df[dense_cols].to_numpy(dtype=np.float32))
    te_dense = sparse.csr_matrix(test_df[dense_cols].to_numpy(dtype=np.float32))
    return sparse.hstack([tr_dense, tr_low]).tocsr(), sparse.hstack([va_dense, va_low]).tocsr(), sparse.hstack([te_dense, te_low]).tocsr()


def get_variants() -> list[VariantSpec]:
    return [
        VariantSpec("base_numeric", ["base"], {"max_leaf": 500, "algorithm": "RGF", "l2": 0.1, "sl2": 0.1}),
        VariantSpec("base_plus_freq", ["base", "frequency_encoding"], {"max_leaf": 700, "algorithm": "RGF", "l2": 0.05, "sl2": 0.05}),
        VariantSpec("base_plus_binning", ["base", "binning"], {"max_leaf": 900, "algorithm": "RGF", "l2": 0.1, "sl2": 1.0}),
        VariantSpec("base_plus_digits", ["base", "digit_features"], {"max_leaf": 700, "algorithm": "RGF_Sib", "l2": 0.2, "sl2": 0.1}),
        VariantSpec("base_plus_binning_freq", ["base", "binning", "frequency_encoding"], {"max_leaf": 1000, "algorithm": "RGF_Opt", "l2": 0.05, "sl2": 0.5}),
        VariantSpec("base_plus_external_stats", ["base", "external_stats"], {"max_leaf": 700, "algorithm": "RGF", "l2": 0.2, "sl2": 1.0}),
        VariantSpec("all_features_encoded", ["base", "binning", "digit_features", "frequency_encoding", "external_stats"], {"max_leaf": 1200, "algorithm": "RGF_Opt", "l2": 0.1, "sl2": 0.5}),
        VariantSpec("gp_plus_base", ["base", "gp_features"], {"max_leaf": 600, "algorithm": "RGF_Sib", "l2": 0.1, "sl2": 0.1}),
        VariantSpec("rounded_bins_plus_freq", ["base", "binning", "frequency_encoding"], {"max_leaf": 800, "algorithm": "RGF", "l2": 0.05, "sl2": 0.05}, rounded_only=True),
        VariantSpec("mixed_dense_sparse", ["base", "all_as_categorical", "frequency_encoding", "binning"], {"max_leaf": 1000, "algorithm": "RGF_Opt", "l2": 0.1, "sl2": 0.5}, use_sparse=True),
    ]


def select_variants(
    all_variants: list[VariantSpec],
    requested_names: list[str] | None,
) -> list[VariantSpec]:
    if not requested_names:
        return all_variants

    variant_map = {variant.name: variant for variant in all_variants}
    missing = [name for name in requested_names if name not in variant_map]
    if missing:
        available = ", ".join(sorted(variant_map))
        raise ValueError(
            f"Unknown variant(s): {missing}. Available variants: {available}"
        )

    selected: list[VariantSpec] = []
    seen: set[str] = set()
    for name in requested_names:
        if name not in seen:
            selected.append(variant_map[name])
            seen.add(name)
    return selected


def fit_rgf(
    X_train: Any,
    y_train: np.ndarray,
    X_valid: Any,
    params: dict[str, Any],
    config: Config,
) -> np.ndarray:
    model = RGFClassifier(
        max_leaf=params.get("max_leaf", 500),
        algorithm=params.get("algorithm", "RGF"),
        loss="Log",
        n_jobs=config.n_jobs,
        learning_rate=params.get("learning_rate", 0.5),
        min_samples_leaf=params.get("min_samples_leaf", 10),
        l2=params.get("l2", 0.1),
        sl2=params.get("sl2", 0.1),
        normalize=False,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_valid)[:, 1].astype(np.float32), model


def save_manifest(df: pd.DataFrame, path: Path, save_parquet: bool) -> None:
    if save_parquet:
        try:
            df.to_parquet(path)
            return
        except Exception as exc:
            log(f"Parquet save failed for {path.name}, falling back to CSV: {exc}")
    df.to_csv(path.with_suffix(".csv"), index=False)


def run_variant(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_df: pd.DataFrame | None,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: VariantSpec,
    config: Config,
) -> dict[str, Any]:
    start = time.time()
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

        X_train_df, X_valid_df, X_test_df, meta = assemble_variant_features(
            train_df=fold_train,
            valid_df=fold_valid,
            test_df=fold_test,
            y_train=y_train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            variant=variant,
            external_df=external_df,
            config=config,
        )
        if meta["status"] != "ok":
            return {
                "variant_name": variant.name,
                "cv_auc": np.nan,
                "feature_family_list": "|".join(variant.feature_families),
                "num_features_train": 0,
                "fit_seconds": round(time.time() - start, 3),
                "status": meta["status"],
            }

        feature_counts.append(meta["num_features_train"])
        log(
            f"Variant={variant.name} fold={fold + 1}/{config.n_splits} "
            f"shape_train={X_train_df.shape} shape_valid={X_valid_df.shape}"
        )

        X_train, X_valid, X_test = maybe_to_sparse(X_train_df, X_valid_df, X_test_df, variant)
        valid_pred, model = fit_rgf(X_train, y_train, X_valid, variant.model_params, config)
        test_fold_pred = model.predict_proba(X_test)[:, 1].astype(np.float32)

        oof[va_idx] = valid_pred
        test_pred_sum += test_fold_pred.astype(np.float64)
        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(float(fold_auc))

        del X_train_df, X_valid_df, X_test_df, X_train, X_valid, X_test, model
        gc.collect()

    cv_auc = float(roc_auc_score(y, oof))
    test_pred = (test_pred_sum / config.n_splits).astype(np.float32)
    np.save(config.output_dir / f"oof_rgf_{variant.name}.npy", oof.astype(np.float32))
    np.save(config.output_dir / f"test_rgf_{variant.name}.npy", test_pred.astype(np.float32))
    with open(config.output_dir / f"{variant.name}_fold_scores.json", "w", encoding="utf-8") as f:
        json.dump({"fold_auc": fold_scores, "variant": variant.name}, f, indent=2)
    return {
        "variant_name": variant.name,
        "cv_auc": cv_auc,
        "feature_family_list": "|".join(variant.feature_families),
        "num_features_train": int(np.median(feature_counts)) if feature_counts else 0,
        "fit_seconds": round(time.time() - start, 3),
        "status": "ok",
    }


def main() -> None:
    config = Config()
    all_variants = get_variants()
    if config.list_variants_only:
        for variant in all_variants:
            print(variant.name)
        return

    config.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    log(f"Config: {asdict(config)}")

    train_df, test_df, external_df = load_data(config)
    y = target_to_binary(train_df[config.target_col])
    numeric_cols, categorical_cols = detect_column_types(train_df, config.target_col)
    log(
        f"Loaded train={train_df.shape} test={test_df.shape} "
        f"numeric_cols={len(numeric_cols)} categorical_cols={len(categorical_cols)}"
    )

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
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
            )
            summary_rows.append(row)
            if row["status"] == "ok":
                oof_manifest[variant.name] = np.load(config.output_dir / f"oof_rgf_{variant.name}.npy")
                test_manifest[variant.name] = np.load(config.output_dir / f"test_rgf_{variant.name}.npy")
            log(
                f"Finished variant={variant.name} status={row['status']} "
                f"cv_auc={row['cv_auc']}"
            )
        except Exception as exc:
            log(f"Variant failed: {variant.name} -> {type(exc).__name__}: {exc}")
            summary_rows.append(
                {
                    "variant_name": variant.name,
                    "cv_auc": np.nan,
                    "feature_family_list": "|".join(variant.feature_families),
                    "num_features_train": 0,
                    "fit_seconds": np.nan,
                    "status": f"failed_{type(exc).__name__}",
                }
            )
        gc.collect()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.output_dir / "summary_rgf.csv", index=False)
    save_manifest(oof_manifest.reset_index(drop=True), config.output_dir / "all_oof_rgf.parquet", config.save_parquet)
    save_manifest(test_manifest.reset_index(drop=True), config.output_dir / "all_test_rgf.parquet", config.save_parquet)
    log("RGF feature ensemble run complete.")


if __name__ == "__main__":
    main()
