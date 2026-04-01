import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
PRED_DIR = DATA_DIR / "S6E3"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET = "Churn"
ID_COL = "id"

# Global blend between descending-rank blend and ascending-rank blend
DESC_BLEND_WEIGHT = 0.70
ASC_BLEND_WEIGHT = 0.30

# Spread thresholds for switching weight sets
LOW_SPREAD_MAX = 0.005
HIGH_SPREAD_MIN = 0.054
HIGH_SPREAD_MAX = 0.074

# ── Model prediction files ────────────────────────────────────────────────────
models = {
    "catboost": ("oof_cat.npy", "test_cat.npy"),
    "lgb": ("oof_lgb.npy", "test_lgb.npy"),
    "nn": ("oof_nn.npy", "test_nn.npy"),
    "xgb_v1": ("oof_xgb_v1.npy", "test_xgb_v1.npy"),
    "xgb_v2": ("oof_xgb_v2.npy", "test_xgb_v2.npy"),
    "ggn": ("oof_gnn_v1909.npy", "test_gnn_v1909.npy"),
    "mlp_all_cats": ("oof_MLP_ALL_CATS_s42.npy", "test_MLP_ALL_CATS_s42.npy"),
    "mlp_base": ("oof_MLP_BASE_s42.npy", "test_MLP_BASE_s42.npy"),
    "mlp_bin_digit": ("oof_MLP_BIN_DIGIT_s42.npy", "test_MLP_BIN_DIGIT_s42.npy"),
    "mlp_freq": ("oof_MLP_FREQ_s42.npy", "test_MLP_FREQ_s42.npy"),
    "mlp_full_fe": ("oof_MLP_FULL_FE_s42.npy", "test_MLP_FULL_FE_s42.npy"),
    "XGB_FULL_FE_s7": ("oof_XGB_FULL_FE_s7.npy", "test_XGB_FULL_FE_s7.npy"),
    "XGB_BASE_s7": ("oof_XGB_BASE_s7.npy", "test_XGB_BASE_s7.npy"),
    "XGB_BASE_s1234": ("oof_XGB_BASE_s1234.npy", "test_XGB_BASE_s1234.npy"),
    "XGB_BASE_s2025": ("oof_XGB_BASE_s2025.npy", "test_XGB_BASE_s2025.npy"),
    "XGB_BIN_DIGIT_s2025": ("oof_XGB_BIN_DIGIT_s2025.npy", "test_XGB_BIN_DIGIT_s2025.npy"),
    "MLP_ALL_CATS_s42": ("oof_MLP_ALL_CATS_s42.npy", "test_MLP_ALL_CATS_s42.npy"),
    "MLP_BASE_s42": ("oof_MLP_BASE_s42.npy", "test_MLP_BASE_s42.npy"),
    "XGB_BIN_DIGIT_s999": ("oof_XGB_BIN_DIGIT_s999.npy", "test_XGB_BIN_DIGIT_s999.npy"),
    "XGB_FULL_FE_s999": ("oof_XGB_FULL_FE_s999.npy", "test_XGB_FULL_FE_s999.npy"),
    "XGB_BASE_s999": ("oof_XGB_BASE_s999.npy", "test_XGB_BASE_s999.npy"),
    "XGB_ALL_CATS_s2025": ("oof_XGB_ALL_CATS_s2025.npy", "test_XGB_ALL_CATS_s2025.npy"),
    "XGB_FREQ_s42": ("oof_XGB_FREQ_s42.npy", "test_XGB_FREQ_s42.npy"),
    "XGB_ALL_CATS_s7": ("oof_XGB_ALL_CATS_s7.npy", "test_XGB_ALL_CATS_s7.npy"),
    "MLP_FULL_FE_s42": ("oof_MLP_FULL_FE_s42.npy", "test_MLP_FULL_FE_s42.npy"),
    "XGB_BIN_DIGIT_s42": ("oof_XGB_BIN_DIGIT_s42.npy", "test_XGB_BIN_DIGIT_s42.npy"),
    "XGB_FULL_FE_s2025": ("oof_XGB_FULL_FE_s2025.npy", "test_XGB_FULL_FE_s2025.npy"),
    "XGB_FREQ_s2025": ("oof_XGB_FREQ_s2025.npy", "test_XGB_FREQ_s2025.npy"),
    "XGB_FREQ_s7": ("oof_XGB_FREQ_s7.npy", "test_XGB_FREQ_s7.npy"),
    "XGB_BIN_DIGIT_s1234": ("oof_XGB_BIN_DIGIT_s1234.npy", "test_XGB_BIN_DIGIT_s1234.npy"),
    "XGB_FREQ_s999": ("oof_XGB_FREQ_s999.npy", "test_XGB_FREQ_s999.npy"),
    "MLP_FREQ_s42": ("oof_MLP_FREQ_s42.npy", "test_MLP_FREQ_s42.npy"),
    "XGB_ALL_CATS_s999": ("oof_XGB_ALL_CATS_s999.npy", "test_XGB_ALL_CATS_s999.npy"),
    "XGB_FREQ_s1234": ("oof_XGB_FREQ_s1234.npy", "test_XGB_FREQ_s1234.npy"),
    "XGB_ALL_CATS_s42": ("oof_XGB_ALL_CATS_s42.npy", "test_XGB_ALL_CATS_s42.npy"),
    "MLP_BIN_DIGIT_s42": ("oof_MLP_BIN_DIGIT_s42.npy", "test_MLP_BIN_DIGIT_s42.npy"),
    "XGB_BASE_s42": ("oof_XGB_BASE_s42.npy", "test_XGB_BASE_s42.npy"),
    "XGB_ALL_CATS_s1234": ("oof_XGB_ALL_CATS_s1234.npy", "test_XGB_ALL_CATS_s1234.npy"),
    "XGB_FULL_FE_s1234": ("oof_XGB_FULL_FE_s1234.npy", "test_XGB_FULL_FE_s1234.npy"),
    "XGB_FULL_FE_s42": ("oof_XGB_FULL_FE_s42.npy", "test_XGB_FULL_FE_s42.npy"),
    "XGB_BIN_DIGIT_s7": ("oof_XGB_BIN_DIGIT_s7.npy", "test_XGB_BIN_DIGIT_s7.npy"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_predictions(model_dict, pred_dir):
    oof = {}
    pred = {}

    for name, (oof_file, test_file) in model_dict.items():
        oof_path = pred_dir / oof_file
        test_path = pred_dir / test_file

        if not oof_path.exists():
            raise FileNotFoundError(f"Missing OOF file for {name}: {oof_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file for {name}: {test_path}")

        oof[name] = np.clip(np.load(oof_path), 0.0, 1.0).astype(np.float64)
        pred[name] = np.clip(np.load(test_path), 0.0, 1.0).astype(np.float64)

    return oof, pred


def validate_shapes(train_df, test_df, oof, pred):
    n_train = len(train_df)
    n_test = len(test_df)

    for name in oof:
        if len(oof[name]) != n_train:
            raise ValueError(
                f"OOF length mismatch for {name}: got {len(oof[name])}, expected {n_train}"
            )
        if len(pred[name]) != n_test:
            raise ValueError(
                f"Test prediction length mismatch for {name}: got {len(pred[name])}, expected {n_test}"
            )


def build_prediction_frame(ids, pred_dict, id_col):
    df = pd.DataFrame({id_col: ids})
    for name, values in pred_dict.items():
        df[name] = values
    return df


def default_base_weights(model_names):
    n = len(model_names)
    return np.full(n, 1.0 / n, dtype=np.float64)


def default_rank_adjustments(n_models, mode="desc"):
    """
    Rank adjustment by position.
    position 0 = highest-ranked model for this row in desc mode.
    position 0 = lowest-ranked model for this row in asc mode.

    Sum is approximately 0 so total scale stays more stable.
    """
    if n_models == 1:
        return np.array([0.0], dtype=np.float64)

    scale = 0.08
    linear = np.linspace(scale, -scale, n_models, dtype=np.float64)
    if mode == "asc":
        linear = linear[::-1]
    return linear


def rank_models(row_values, model_names, sorting_direction):
    reverse = sorting_direction == "desc"
    pairs = list(zip(model_names, row_values))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=reverse)
    return [name for name, _ in pairs_sorted]


def weighted_sum_with_rank(
    row_values_dict,
    model_names,
    ordered_names,
    base_weights,
    rank_adjustments,
):
    pos_map = {name: pos for pos, name in enumerate(ordered_names)}
    score = 0.0
    for j, name in enumerate(model_names):
        rank_pos = pos_map[name]
        weight = base_weights[j] + rank_adjustments[rank_pos]
        score += row_values_dict[name] * weight
    return score


def adaptive_row_blend(
    row_values_dict,
    model_names,
    sorting_direction,
    base_weights_1,
    rank_adjustments_1,
    base_weights_2=None,
    rank_adjustments_2=None,
    base_weights_3=None,
    rank_adjustments_3=None,
):
    values = np.array([row_values_dict[name] for name in model_names], dtype=np.float64)
    spread = float(values.max() - values.min())
    ordered_names = rank_models(values, model_names, sorting_direction)

    use_second = (
        base_weights_2 is not None
        and rank_adjustments_2 is not None
        and 0.0 < spread <= LOW_SPREAD_MAX
    )
    use_third = (
        base_weights_3 is not None
        and rank_adjustments_3 is not None
        and HIGH_SPREAD_MIN < spread <= HIGH_SPREAD_MAX
    )

    if use_third:
        return weighted_sum_with_rank(
            row_values_dict=row_values_dict,
            model_names=model_names,
            ordered_names=ordered_names,
            base_weights=base_weights_3,
            rank_adjustments=rank_adjustments_3,
        )

    if use_second:
        return weighted_sum_with_rank(
            row_values_dict=row_values_dict,
            model_names=model_names,
            ordered_names=ordered_names,
            base_weights=base_weights_2,
            rank_adjustments=rank_adjustments_2,
        )

    return weighted_sum_with_rank(
        row_values_dict=row_values_dict,
        model_names=model_names,
        ordered_names=ordered_names,
        base_weights=base_weights_1,
        rank_adjustments=rank_adjustments_1,
    )


def blend_frame(
    df_pred,
    model_names,
    id_col,
    target_col,
    sorting_direction,
    base_weights_1,
    rank_adjustments_1,
    base_weights_2=None,
    rank_adjustments_2=None,
    base_weights_3=None,
    rank_adjustments_3=None,
):
    outputs = np.zeros(len(df_pred), dtype=np.float64)

    for i, row in enumerate(df_pred.itertuples(index=False)):
        row_dict = {name: getattr(row, name) for name in model_names}
        outputs[i] = adaptive_row_blend(
            row_values_dict=row_dict,
            model_names=model_names,
            sorting_direction=sorting_direction,
            base_weights_1=base_weights_1,
            rank_adjustments_1=rank_adjustments_1,
            base_weights_2=base_weights_2,
            rank_adjustments_2=rank_adjustments_2,
            base_weights_3=base_weights_3,
            rank_adjustments_3=rank_adjustments_3,
        )

    outputs = np.clip(outputs, 0.0, 1.0)

    result = df_pred[[id_col]].copy()
    result[target_col] = outputs
    return result


def ensemble_dual_direction(
    df_pred,
    model_names,
    id_col,
    target_col,
    desc_weight,
    asc_weight,
    weights_cfg,
):
    df_desc = blend_frame(
        df_pred=df_pred,
        model_names=model_names,
        id_col=id_col,
        target_col=target_col,
        sorting_direction="desc",
        base_weights_1=weights_cfg["desc"]["base_1"],
        rank_adjustments_1=weights_cfg["desc"]["rank_1"],
        base_weights_2=weights_cfg["desc"].get("base_2"),
        rank_adjustments_2=weights_cfg["desc"].get("rank_2"),
        base_weights_3=weights_cfg["desc"].get("base_3"),
        rank_adjustments_3=weights_cfg["desc"].get("rank_3"),
    )

    df_asc = blend_frame(
        df_pred=df_pred,
        model_names=model_names,
        id_col=id_col,
        target_col=target_col,
        sorting_direction="asc",
        base_weights_1=weights_cfg["asc"]["base_1"],
        rank_adjustments_1=weights_cfg["asc"]["rank_1"],
        base_weights_2=weights_cfg["asc"].get("base_2"),
        rank_adjustments_2=weights_cfg["asc"].get("rank_2"),
        base_weights_3=weights_cfg["asc"].get("base_3"),
        rank_adjustments_3=weights_cfg["asc"].get("rank_3"),
    )

    out = df_pred[[id_col]].copy()
    out[target_col] = (
        desc_weight * df_desc[target_col].values
        + asc_weight * df_asc[target_col].values
    )
    out[target_col] = np.clip(out[target_col].values, 0.0, 1.0)
    return out


def build_default_weights_config(model_names):
    n_models = len(model_names)
    base = default_base_weights(model_names)

    # Main/default regime
    desc_rank_1 = default_rank_adjustments(n_models, mode="desc")
    asc_rank_1 = default_rank_adjustments(n_models, mode="asc")

    # Low spread regime: reduce rank effect because models agree
    desc_rank_2 = desc_rank_1 * 0.35
    asc_rank_2 = asc_rank_1 * 0.35

    # Mid-high spread regime: amplify rank effect because models disagree more
    desc_rank_3 = desc_rank_1 * 1.35
    asc_rank_3 = asc_rank_1 * 1.35

    return {
        "desc": {
            "base_1": base.copy(),
            "rank_1": desc_rank_1,
            "base_2": base.copy(),
            "rank_2": desc_rank_2,
            "base_3": base.copy(),
            "rank_3": desc_rank_3,
        },
        "asc": {
            "base_1": base.copy(),
            "rank_1": asc_rank_1,
            "base_2": base.copy(),
            "rank_2": asc_rank_2,
            "base_3": base.copy(),
            "rank_3": asc_rank_3,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    oof_dict, test_dict = load_predictions(models, PRED_DIR)
    validate_shapes(train, test, oof_dict, test_dict)

    y_true = train[TARGET].values
    model_names = list(models.keys())

    # print("Individual OOF AUCs:")
    # for name in model_names:
    #     auc = roc_auc_score(y_true, oof_dict[name])
    #     print(f"  {name:<20} {auc:.6f}")

    # Build row-wise prediction tables
    oof_df = build_prediction_frame(train[ID_COL].values, oof_dict, ID_COL)
    test_df = build_prediction_frame(test[ID_COL].values, test_dict, ID_COL)

    # Weight config adapted from the shared method:
    # base model weights + rank-position adjustments + spread-based switching
    weights_cfg = build_default_weights_config(model_names)

    # Optional manual weighting by model quality:
    # This is useful if you want stronger models to matter more even before rank correction.
    # Uncomment to use OOF AUC-based base weights.
    #
    # aucs = np.array([roc_auc_score(y_true, oof_dict[name]) for name in model_names], dtype=np.float64)
    # aucs = aucs - aucs.min() + 1e-6
    # auc_weights = aucs / aucs.sum()
    # for side in ["desc", "asc"]:
    #     weights_cfg[side]["base_1"] = auc_weights.copy()
    #     weights_cfg[side]["base_2"] = auc_weights.copy()
    #     weights_cfg[side]["base_3"] = auc_weights.copy()

    # OOF adaptive blend
    oof_blend = ensemble_dual_direction(
        df_pred=oof_df,
        model_names=model_names,
        id_col=ID_COL,
        target_col=TARGET,
        desc_weight=DESC_BLEND_WEIGHT,
        asc_weight=ASC_BLEND_WEIGHT,
        weights_cfg=weights_cfg,
    )

    # Test adaptive blend
    test_blend = ensemble_dual_direction(
        df_pred=test_df,
        model_names=model_names,
        id_col=ID_COL,
        target_col=TARGET,
        desc_weight=DESC_BLEND_WEIGHT,
        asc_weight=ASC_BLEND_WEIGHT,
        weights_cfg=weights_cfg,
    )

    oof_pred = np.clip(oof_blend[TARGET].values, 0.0, 1.0)
    test_pred = np.clip(test_blend[TARGET].values, 0.0, 1.0)

    auc = roc_auc_score(y_true, oof_pred)
    print(f"\nAdaptive h_blend-style OOF AUC: {auc:.6f}")

    # Save raw ensemble outputs
    np.save(DATA_DIR / "oof_hblend.npy", oof_pred)
    np.save(DATA_DIR / "test_hblend.npy", test_pred)

    print(f"Saved: {DATA_DIR / 'oof_hblend.npy'}")
    print(f"Saved: {DATA_DIR / 'test_hblend.npy'}")

    # Submission
    submission = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET: test_pred,
    })
    submission_path = DATA_DIR / "submission_hblend.csv"
    submission.to_csv(submission_path, index=False)

    print(f"Saved: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    main()