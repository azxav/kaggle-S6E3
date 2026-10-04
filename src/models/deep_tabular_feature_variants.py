from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureVariantSpec:
    name: str
    feature_families: list[str]


def log(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


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


def resolve_original_path(original_path: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if original_path is not None:
        candidates.append(Path(original_path))
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


def load_data(
    train_path: Path,
    test_path: Path,
    *,
    use_external_stats: bool,
    original_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    external_df = None
    if use_external_stats:
        resolved = resolve_original_path(original_path)
        if resolved is not None:
            external_df = pd.read_csv(resolved)
            log(f"Using external data: {resolved}")
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
    *,
    qcut_bins: int,
    cut_bins: int,
    rounded_bin_divisors: tuple[int, ...],
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

        quantiles = np.linspace(0.0, 1.0, qcut_bins + 1)
        q_edges = np.unique(non_null.quantile(quantiles).to_numpy(dtype=np.float64))
        if len(q_edges) >= 3:
            out_train[f"{col}__qbin"] = (
                pd.cut(train_num, bins=q_edges, include_lowest=True, duplicates="drop")
                .astype(str)
                .replace({"nan": "__MISSING__"})
            )
            out_valid[f"{col}__qbin"] = (
                pd.cut(valid_num, bins=q_edges, include_lowest=True, duplicates="drop")
                .astype(str)
                .replace({"nan": "__MISSING__"})
            )
            out_test[f"{col}__qbin"] = (
                pd.cut(test_num, bins=q_edges, include_lowest=True, duplicates="drop")
                .astype(str)
                .replace({"nan": "__MISSING__"})
            )

        low = float(non_null.min())
        high = float(non_null.max())
        if np.isfinite(low) and np.isfinite(high) and high > low:
            cut_edges = np.unique(np.linspace(low, high, cut_bins + 1).astype(np.float64))
            if len(cut_edges) >= 3:
                out_train[f"{col}__wbin"] = (
                    pd.cut(train_num, bins=cut_edges, include_lowest=True, duplicates="drop")
                    .astype(str)
                    .replace({"nan": "__MISSING__"})
                )
                out_valid[f"{col}__wbin"] = (
                    pd.cut(valid_num, bins=cut_edges, include_lowest=True, duplicates="drop")
                    .astype(str)
                    .replace({"nan": "__MISSING__"})
                )
                out_test[f"{col}__wbin"] = (
                    pd.cut(test_num, bins=cut_edges, include_lowest=True, duplicates="drop")
                    .astype(str)
                    .replace({"nan": "__MISSING__"})
                )

        for divisor in rounded_bin_divisors:
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
    *,
    max_digit_numeric_cols: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_cols = numeric_cols[:max_digit_numeric_cols]
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
    *,
    target_col: str,
    external_df: pd.DataFrame | None,
    cols: list[str],
    external_smoothing: float,
    external_min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    out_train = pd.DataFrame(index=train_df.index)
    out_valid = pd.DataFrame(index=valid_df.index)
    out_test = pd.DataFrame(index=test_df.index)
    if external_df is None or target_col not in external_df.columns:
        return out_train, out_valid, out_test, False

    y_ext = target_to_binary(external_df[target_col])
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
        grouped = grouped[grouped["count"] >= external_min_count]
        if grouped.empty:
            continue

        mean_map = (grouped["sum"] / grouped["count"]).astype(np.float32)
        smooth_map = (
            (grouped["sum"] + external_smoothing * global_mean)
            / (grouped["count"] + external_smoothing)
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


def get_default_feature_variants() -> list[FeatureVariantSpec]:
    return [
        FeatureVariantSpec("base_mixed", ["base"]),
        # FeatureVariantSpec("base_plus_freq", ["base", "frequency_encoding"]),
        # FeatureVariantSpec("base_plus_binning", ["base", "binning"]),
        # FeatureVariantSpec("base_plus_digits", ["base", "digit_features"]),
        # FeatureVariantSpec("all_as_categorical", ["all_as_categorical"]),
        # FeatureVariantSpec("base_plus_external_stats", ["base", "external_stats"]),
        # FeatureVariantSpec("hybrid_light", ["base", "binning", "frequency_encoding"]),
        # FeatureVariantSpec(
        #     "hybrid_full_without_gp",
        #     ["base", "binning", "digit_features", "frequency_encoding", "external_stats"],
        # ),
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


def assemble_variant_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    variant: FeatureVariantSpec,
    external_df: pd.DataFrame | None,
    target_col: str,
    qcut_bins: int,
    cut_bins: int,
    rounded_bin_divisors: tuple[int, ...],
    max_digit_numeric_cols: int,
    external_smoothing: float,
    external_min_count: int,
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
        btr, bva, bte = build_binning_features(
            train_df,
            valid_df,
            test_df,
            numeric_cols,
            qcut_bins=qcut_bins,
            cut_bins=cut_bins,
            rounded_bin_divisors=rounded_bin_divisors,
        )
        final_train = pd.concat([final_train, btr], axis=1)
        final_valid = pd.concat([final_valid, bva], axis=1)
        final_test = pd.concat([final_test, bte], axis=1)

    if "digit_features" in variant.feature_families:
        dtr, dva, dte = build_digit_features(
            train_df,
            valid_df,
            test_df,
            numeric_cols,
            max_digit_numeric_cols=max_digit_numeric_cols,
        )
        final_train = pd.concat([final_train, dtr], axis=1)
        final_valid = pd.concat([final_valid, dva], axis=1)
        final_test = pd.concat([final_test, dte], axis=1)

    if "frequency_encoding" in variant.feature_families:
        ftr, fva, fte = build_frequency_features(
            train_df,
            valid_df,
            test_df,
            numeric_cols + categorical_cols,
        )
        final_train = pd.concat([final_train, ftr], axis=1)
        final_valid = pd.concat([final_valid, fva], axis=1)
        final_test = pd.concat([final_test, fte], axis=1)

    if "external_stats" in variant.feature_families:
        etr, eva, ete, ok = build_external_stats_features(
            train_df,
            valid_df,
            test_df,
            target_col=target_col,
            external_df=external_df,
            cols=numeric_cols + categorical_cols,
            external_smoothing=external_smoothing,
            external_min_count=external_min_count,
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
    *,
    frac: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    if frac <= 0.0 or frac >= 1.0:
        return train_df.reset_index(drop=True), y.copy()
    sample_size = max(1, int(len(train_df) * frac))
    rng = np.random.default_rng(seed)
    yes_idx = np.where(y == 1)[0]
    no_idx = np.where(y == 0)[0]
    n_yes = max(1, int(round(sample_size * (len(yes_idx) / len(train_df)))))
    n_no = max(1, sample_size - n_yes)
    chosen_yes = rng.choice(yes_idx, size=min(n_yes, len(yes_idx)), replace=False)
    chosen_no = rng.choice(no_idx, size=min(n_no, len(no_idx)), replace=False)
    chosen = np.concatenate([chosen_yes, chosen_no])
    if len(chosen) < sample_size:
        remaining = np.setdiff1d(np.arange(len(train_df)), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=sample_size - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    rng.shuffle(chosen)
    subset_df = train_df.iloc[np.sort(chosen)].reset_index(drop=True)
    subset_y = y[np.sort(chosen)].copy()
    return subset_df, subset_y
