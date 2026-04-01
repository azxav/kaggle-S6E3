"""
Logistic-regression meta-ensemble over all discovered OOF/test prediction stacks.

Same discovery as ``ridge.py`` but includes every model (no Optuna subset search).
L2 LogisticRegression ``C`` is chosen by grid search maximizing OOF AUC.

Run: ``.venv/bin/python logistic_ensemble_all.py``

Dependencies: ``pip install pandas numpy scikit-learn``

Override grid density for smoke tests:
``LOGREG_GRID_POINTS=5 .venv/bin/python logistic_ensemble_all.py``
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
PRED_DIR = DATA_DIR / "S6E3"
DVAE_OPTUNA_DIR = DATA_DIR / "dvae_outputs_optuna"
DVAE_LEGACY_DIR = DATA_DIR / "dvae_outputs"
NN_DIR = DATA_DIR / "nn"
GNN_DIR = DATA_DIR / "gnn"

DISCOVER_DIRS = (
    PRED_DIR,
    DVAE_OPTUNA_DIR,
    DVAE_LEGACY_DIR,
    NN_DIR,
    GNN_DIR,
)

TARGET = "Churn"
ID_COL = "id"

LOGREG_C_GRID = np.logspace(
    np.log10(0.01),
    np.log10(100.0),
    int(os.environ.get("LOGREG_GRID_POINTS", "80")),
)
LOGREG_SOLVER = "lbfgs"
LOGREG_MAX_ITER = 5000


def _is_excluded_oof_path(path: Path) -> bool:
    name = path.name
    if name == "oof_ridge.npy":
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


def grid_search_logreg_c(
    X_oof: np.ndarray,
    y_true: np.ndarray,
    c_grid: np.ndarray,
) -> tuple[float, float]:
    """Return (best_c, best_oof_auc) using scaled features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_oof)
    best_c = float(c_grid[0])
    best_auc = -1.0
    for c_value in c_grid:
        logreg = LogisticRegression(
            C=float(c_value),
            solver=LOGREG_SOLVER,
            fit_intercept=True,
            max_iter=LOGREG_MAX_ITER,
        )
        logreg.fit(X_scaled, y_true)
        pred = logreg.predict_proba(X_scaled)[:, 1]
        auc = float(roc_auc_score(y_true, pred))
        if auc > best_auc:
            best_auc = auc
            best_c = float(c_value)
    return best_c, best_auc


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

    best_c, grid_auc = grid_search_logreg_c(X_oof, y_true, LOGREG_C_GRID)
    print(
        "\nGrid-search best LogisticRegression C: "
        f"{best_c:.6f} (OOF AUC on grid: {grid_auc:.6f})"
    )

    scaler = StandardScaler()
    X_oof_scaled = scaler.fit_transform(X_oof)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        C=best_c,
        solver=LOGREG_SOLVER,
        fit_intercept=True,
        max_iter=LOGREG_MAX_ITER,
    )
    logreg.fit(X_oof_scaled, y_true)

    print("\nLogisticRegression coefficients (all models):")
    for name, coef in zip(model_names, logreg.coef_[0]):
        print(f"  {name:<48} {coef:.6f}")
    print(f"  {'intercept':<48} {logreg.intercept_[0]:.6f}")

    oof_ensemble = logreg.predict_proba(X_oof_scaled)[:, 1]
    pred_ensemble = logreg.predict_proba(X_test_scaled)[:, 1]

    ensemble_auc = roc_auc_score(y_true, oof_ensemble)
    print(f"\nFinal LogisticRegression ensemble OOF AUC: {ensemble_auc:.6f}")

    out_oof = DATA_DIR / "oof_logreg_all.npy"
    out_test = DATA_DIR / "test_logreg_all.npy"
    np.save(out_oof, oof_ensemble)
    np.save(out_test, pred_ensemble)
    print(f"Saved OOF ensemble → {out_oof}")
    print(f"Saved test ensemble → {out_test}")

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_ensemble})
    out_path = DATA_DIR / "submission_logreg_all.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
