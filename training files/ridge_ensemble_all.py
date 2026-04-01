"""
Ridge meta-ensemble over all discovered OOF/test prediction stacks.

Same discovery as ``ridge.py`` but includes every model (no Optuna subset search).
Ridge ``alpha`` is chosen by grid search maximizing OOF AUC.

Run: ``.venv/bin/python ridge_ensemble_all.py``

Dependencies: ``pip install pandas numpy scikit-learn``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
PRED_DIR = DATA_DIR / "S6E3"
DVAE_OPTUNA_DIR = DATA_DIR / "dvae_outputs_optuna"
DVAE_LEGACY_DIR = DATA_DIR / "dvae_outputs"
NN_DIR = DATA_DIR / "nn"
GNN_DIR = DATA_DIR / "gnn"
OTHER_ML_DIR = DATA_DIR / "other_ML"

DISCOVER_DIRS = (
    PRED_DIR,
    DVAE_OPTUNA_DIR,
    DVAE_LEGACY_DIR,
    NN_DIR,
    GNN_DIR,
    OTHER_ML_DIR,
)

TARGET = "Churn"
ID_COL = "id"

RIDGE_ALPHA_GRID = np.logspace(np.log10(0.01), np.log10(100.0), 80)
RIDGE_RANDOM_STATE = 42


def _is_excluded_oof_path(path: Path) -> bool:
    name = path.name
    if name in {"oof_ridge.npy", "oof_ridge_all.npy"}:
        return True
    if "copy" in name.lower():
        return True
    return False


def discover_predictions(
    data_dir: Path,
    discover_dirs: tuple[Path, ...] = DISCOVER_DIRS,
) -> dict[str, tuple[Path, Path]]:
    """
    Scan directories for oof_*.npy files; pair with test_*.npy or pred_*.npy
    in the same folder.

    Keys are unique relative paths under data_dir, e.g. ``S6E3/oof_cat``.
    """
    models: dict[str, tuple[Path, Path]] = {}
    for base in discover_dirs:
        if not base.is_dir():
            continue
        try:
            rel_base = base.relative_to(data_dir)
        except ValueError:
            rel_base = Path(base.name)

        for oof_path in sorted(base.glob("oof_*.npy")):
            if _is_excluded_oof_path(oof_path):
                continue
            test_name = oof_path.name.replace("oof_", "test_", 1)
            test_path = oof_path.with_name(test_name)
            if not test_path.is_file():
                pred_name = oof_path.name.replace("oof_", "pred_", 1)
                pred_path = oof_path.with_name(pred_name)
                if pred_path.is_file():
                    test_path = pred_path
                else:
                    print(f"[SKIP] Missing test/pred: {test_path} | {pred_path}")
                    continue

            key = f"{rel_base.as_posix()}/{oof_path.stem}"
            if key in models:
                raise ValueError(f"Duplicate model key from discovery: {key}")
            models[key] = (oof_path, test_path)

    return models


def load_stacked_predictions(
    models: dict[str, tuple[Path, Path]],
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load OOF/test arrays into (n_train, M) and (n_test, M) matrices."""
    model_names = sorted(models.keys())
    oof_cols = []
    test_cols = []

    for name in model_names:
        oof_path, test_path = models[name]
        oof_arr = np.clip(np.load(oof_path), 0.0, 1.0).astype(np.float64)
        test_arr = np.clip(np.load(test_path), 0.0, 1.0).astype(np.float64)

        if len(oof_arr) != n_train:
            raise ValueError(
                f"OOF length mismatch for {name}: got {len(oof_arr)}, expected {n_train}"
            )
        if len(test_arr) != n_test:
            raise ValueError(
                f"Test prediction length mismatch for {name}: got {len(test_arr)}, expected {n_test}"
            )

        oof_cols.append(oof_arr)
        test_cols.append(test_arr)

    X_oof = np.column_stack(oof_cols)
    X_test = np.column_stack(test_cols)
    return X_oof, X_test, model_names


def main() -> None:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y_true = train[TARGET].map({"No": 0, "Yes": 1}).values.astype(np.float64)
    n_train = len(train)
    n_test = len(test)

    models = discover_predictions(DATA_DIR, DISCOVER_DIRS)
    if not models:
        raise RuntimeError("No OOF/test prediction pairs discovered.")

    X_oof, X_test, model_names = load_stacked_predictions(
        models, n_train=n_train, n_test=n_test
    )
    n_models = len(model_names)
    print(f"Discovered and loaded {n_models} OOF/test model pairs (all used).")
    for name in model_names:
        print(f"  {name}")

    scaler = StandardScaler()
    X_oof_scaled = scaler.fit_transform(X_oof)
    X_test_scaled = scaler.transform(X_test)

    cv_ridge = RidgeCV(alphas=RIDGE_ALPHA_GRID, fit_intercept=True)
    cv_ridge.fit(X_oof_scaled, y_true)
    best_alpha = float(cv_ridge.alpha_)
    print(f"\nRidgeCV best alpha: {best_alpha:.6f}")

    ridge = Ridge(
        alpha=best_alpha, fit_intercept=True, random_state=RIDGE_RANDOM_STATE
    )
    ridge.fit(X_oof_scaled, y_true)

    print("\nRidge coefficients (all models):")
    for name, coef in zip(model_names, ridge.coef_):
        print(f"  {name:<48} {coef:.6f}")
    print(f"  {'intercept':<48} {ridge.intercept_:.6f}")

    oof_ensemble = np.clip(ridge.predict(X_oof_scaled), 0.0, 1.0)
    pred_ensemble = np.clip(ridge.predict(X_test_scaled), 0.0, 1.0)

    ensemble_auc = roc_auc_score(y_true, oof_ensemble)
    print(f"\nFinal Ridge ensemble OOF AUC: {ensemble_auc:.6f}")

    out_oof = DATA_DIR / "oof_ridge_all.npy"
    out_test = DATA_DIR / "test_ridge_all.npy"
    np.save(out_oof, oof_ensemble)
    np.save(out_test, pred_ensemble)
    print(f"Saved OOF ensemble → {out_oof}")
    print(f"Saved test ensemble → {out_test}")

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_ensemble})
    out_path = DATA_DIR / "submission_ridge_all.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
