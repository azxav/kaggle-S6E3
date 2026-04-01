from __future__ import annotations

import gc
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class Config:
    train_path: Path = Path("./train.csv")
    test_path: Path = Path("./test.csv")
    original_path: Path | None = None
    output_dir: Path = Path("./outputs/catboost")
    target_col: str = "Churn"
    id_col: str = "id"
    seed: int = 42
    n_splits: int = 5
    n_jobs: int = -1
    use_external_stats: bool = True
    save_parquet: bool = True
    qcut_bins: int = 8
    cut_bins: int = 8
    rounded_bin_divisors: tuple[int, ...] = (5, 10)
    max_digit_numeric_cols: int = 8
    external_smoothing: float = 30.0
    external_min_count: int = 10
    freq_include_numeric: bool = True
    fill_value_numeric: float = -9999.0
    iterations: int = 1600
    early_stopping_rounds: int = 100
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
            cut_edges = np.unique(np.linspace(low, high, config.cut_bins + 1).astype(np.float64))
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


def build_all_categorical_frames(
    base_train: pd.DataFrame,
    base_valid: pd.DataFrame,
    base_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_train = pd.concat(
        [base_train[numeric_cols].round(3).astype(str), base_train[categorical_cols]],
        axis=1,
    ).add_prefix("allcat__")
    all_valid = pd.concat(
        [base_valid[numeric_cols].round(3).astype(str), base_valid[categorical_cols]],
        axis=1,
    ).add_prefix("allcat__")
    all_test = pd.concat(
        [base_test[numeric_cols].round(3).astype(str), base_test[categorical_cols]],
        axis=1,
    ).add_prefix("allcat__")
    return all_train, all_valid, all_test


def assemble_variant_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: VariantSpec,
    external_df: pd.DataFrame | None,
    config: Config,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    base_train, base_valid, base_test = make_base_frames(
        train_df, valid_df, test_df, numeric_cols, categorical_cols
    )
    metadata: dict[str, Any] = {
        "status": "ok",
        "used_external": False,
        "feature_families": list(variant.feature_families),
    }

    if "all_as_categorical" in variant.feature_families:
        numeric_train = pd.DataFrame(index=train_df.index)
        numeric_valid = pd.DataFrame(index=valid_df.index)
        numeric_test = pd.DataFrame(index=test_df.index)
        categorical_train, categorical_valid, categorical_test = build_all_categorical_frames(
            base_train, base_valid, base_test, numeric_cols, categorical_cols
        )
    else:
        numeric_train = base_train[numeric_cols].copy()
        numeric_valid = base_valid[numeric_cols].copy()
        numeric_test = base_test[numeric_cols].copy()
        categorical_train = base_train[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=train_df.index)
        categorical_valid = base_valid[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=valid_df.index)
        categorical_test = base_test[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=test_df.index)

    if "binning" in variant.feature_families:
        btr, bva, bte = build_binning_features(train_df, valid_df, test_df, numeric_cols, config)
        categorical_train = pd.concat([categorical_train, btr], axis=1)
        categorical_valid = pd.concat([categorical_valid, bva], axis=1)
        categorical_test = pd.concat([categorical_test, bte], axis=1)

    if "digit_features" in variant.feature_families:
        dtr, dva, dte = build_digit_features(train_df, valid_df, test_df, numeric_cols, config)
        numeric_train = pd.concat([numeric_train, dtr], axis=1)
        numeric_valid = pd.concat([numeric_valid, dva], axis=1)
        numeric_test = pd.concat([numeric_test, dte], axis=1)

    if "frequency_encoding" in variant.feature_families:
        freq_cols = categorical_cols + (numeric_cols if config.freq_include_numeric else [])
        ftr, fva, fte = build_frequency_features(train_df, valid_df, test_df, freq_cols)
        numeric_train = pd.concat([numeric_train, ftr], axis=1)
        numeric_valid = pd.concat([numeric_valid, fva], axis=1)
        numeric_test = pd.concat([numeric_test, fte], axis=1)

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
        numeric_train = pd.concat([numeric_train, etr], axis=1)
        numeric_valid = pd.concat([numeric_valid, eva], axis=1)
        numeric_test = pd.concat([numeric_test, ete], axis=1)

    numeric_train = numeric_train.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)
    numeric_valid = numeric_valid.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)
    numeric_test = numeric_test.apply(pd.to_numeric, errors="coerce").fillna(config.fill_value_numeric)

    for frame in (categorical_train, categorical_valid, categorical_test):
        for col in frame.columns:
            frame[col] = normalize_categorical(frame[col])

    final_train = pd.concat([numeric_train, categorical_train], axis=1)
    final_valid = pd.concat([numeric_valid, categorical_valid], axis=1)
    final_test = pd.concat([numeric_test, categorical_test], axis=1)
    if final_train.empty:
        metadata["status"] = "skipped_empty_features"
        return None, None, None, metadata

    metadata["categorical_cols"] = categorical_train.columns.tolist()
    metadata["num_features_train"] = int(final_train.shape[1])
    return final_train, final_valid, final_test, metadata


def get_variants() -> list[VariantSpec]:
    return [
        VariantSpec(
            "base_numeric",
            ["base"],
            {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 4.0, "random_strength": 0.5},
        ),
        VariantSpec(
            "base_plus_freq",
            ["base", "frequency_encoding"],
            {"learning_rate": 0.035, "depth": 6, "l2_leaf_reg": 5.0, "random_strength": 0.5},
        ),
        VariantSpec(
            "base_plus_binning",
            ["base", "binning"],
            {"learning_rate": 0.035, "depth": 7, "l2_leaf_reg": 5.0, "random_strength": 1.0},
        ),
        VariantSpec(
            "base_plus_digits",
            ["base", "digit_features"],
            {"learning_rate": 0.04, "depth": 6, "l2_leaf_reg": 4.0, "random_strength": 0.5},
        ),
        VariantSpec(
            "full_feature_stack",
            ["base", "binning", "digit_features", "frequency_encoding", "external_stats"],
            {"learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 6.0, "random_strength": 1.0},
        ),
        VariantSpec(
            "all_as_categorical",
            ["all_as_categorical"],
            {"learning_rate": 0.04, "depth": 8, "l2_leaf_reg": 5.0, "random_strength": 1.0},
        ),
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
        raise ValueError(f"Unknown variant(s): {missing}. Available variants: {available}")

    selected: list[VariantSpec] = []
    seen: set[str] = set()
    for name in requested_names:
        if name not in seen:
            selected.append(variant_map[name])
            seen.add(name)
    return selected


def fit_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    categorical_cols: list[str],
    params: dict[str, Any],
    config: Config,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise ImportError(
            "train_catboost_feature_ensemble.py requires catboost. "
            "Install it in the notebook environment, for example: pip install catboost"
        ) from exc

    cat_idx = [X_train.columns.get_loc(col) for col in categorical_cols]
    train_pool = Pool(X_train, label=y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, label=y_valid, cat_features=cat_idx)
    test_pool = Pool(X_test, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=params.get("iterations", config.iterations),
        learning_rate=params.get("learning_rate", 0.04),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 4.0),
        random_strength=params.get("random_strength", 1.0),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=config.seed,
        allow_writing_files=False,
        thread_count=config.n_jobs,
        verbose=False,
    )
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose=False,
    )
    valid_pred = model.predict_proba(valid_pool)[:, 1].astype(np.float32)
    test_pred = model.predict_proba(test_pool)[:, 1].astype(np.float32)
    return valid_pred, test_pred


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
            f"shape_train={X_train_df.shape} shape_valid={X_valid_df.shape} "
            f"cat_cols={len(meta['categorical_cols'])}"
        )

        valid_pred, test_fold_pred = fit_catboost(
            X_train=X_train_df,
            y_train=y_train,
            X_valid=X_valid_df,
            y_valid=y_valid,
            X_test=X_test_df,
            categorical_cols=meta["categorical_cols"],
            params=variant.model_params,
            config=config,
        )

        oof[va_idx] = valid_pred
        test_pred_sum += test_fold_pred.astype(np.float64)
        fold_scores.append(float(roc_auc_score(y_valid, valid_pred)))

        del X_train_df, X_valid_df, X_test_df
        gc.collect()

    cv_auc = float(roc_auc_score(y, oof))
    test_pred = (test_pred_sum / config.n_splits).astype(np.float32)
    np.save(config.output_dir / f"oof_cat_{variant.name}.npy", oof.astype(np.float32))
    np.save(config.output_dir / f"test_cat_{variant.name}.npy", test_pred.astype(np.float32))
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
                oof_manifest[variant.name] = np.load(config.output_dir / f"oof_cat_{variant.name}.npy")
                test_manifest[variant.name] = np.load(config.output_dir / f"test_cat_{variant.name}.npy")
            log(f"Finished variant={variant.name} status={row['status']} cv_auc={row['cv_auc']}")
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
    summary_df.to_csv(config.output_dir / "summary_catboost.csv", index=False)
    save_manifest(
        oof_manifest.reset_index(drop=True),
        config.output_dir / "all_oof_catboost.parquet",
        config.save_parquet,
    )
    save_manifest(
        test_manifest.reset_index(drop=True),
        config.output_dir / "all_test_catboost.parquet",
        config.save_parquet,
    )
    log("CatBoost feature ensemble run complete.")


if __name__ == "__main__":
    main()
