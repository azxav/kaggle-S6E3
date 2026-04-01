"""
OOF subset search with Optuna + Ridge meta-ensemble.

Auto-discovers oof_*.npy / test_*.npy pairs, runs Optuna to maximize OOF AUC
over binary inclusion masks and Ridge alpha, then fits the final Ridge on the
best subset.

Run with the project venv, e.g. ``.venv/bin/python ridge.py``. Dependencies:
``pip install optuna pandas numpy scikit-learn`` (bootstrap pip in a bare venv
with ``python -m ensurepip --upgrade`` if needed).

Override trial count for smoke tests: ``N_OPTUNA_TRIALS=50 .venv/bin/python ridge.py``.
Default is 2500.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge
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

DEFAULT_N_OPTUNA_TRIALS = 500
MIN_SELECTED_MODELS = 5
RIDGE_ALPHA_LOW = 0.01
RIDGE_ALPHA_HIGH = 100.0
OPTUNA_SEED = 42
RIDGE_RANDOM_STATE = 42


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
    in the same folder (``pred_`` matches outputs from diverse_tabular_nn_5models
    and gnn_5_variants_training).

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


def run_optuna_subset_ridge(
    X_oof: np.ndarray,
    y_true: np.ndarray,
    model_names: list[str],
    n_trials: int,
    selection_counts: np.ndarray,
) -> optuna.Study:
    """Maximize OOF AUC over per-model inclusion flags and Ridge alpha."""

    n_models = X_oof.shape[1]

    def objective(trial: optuna.Trial) -> float:
        mask = np.zeros(n_models, dtype=bool)
        for j in range(n_models):
            include = trial.suggest_categorical(f"inc_{j}", [False, True])
            mask[j] = bool(include)
            if include:
                selection_counts[j] += 1

        idx = np.flatnonzero(mask)
        if idx.size < MIN_SELECTED_MODELS:
            return 0.0

        alpha = trial.suggest_float(
            "alpha", RIDGE_ALPHA_LOW, RIDGE_ALPHA_HIGH, log=True
        )

        X_sub = X_oof[:, idx]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sub)
        ridge = Ridge(alpha=alpha, fit_intercept=True, random_state=RIDGE_RANDOM_STATE)
        ridge.fit(X_scaled, y_true)
        pred = np.clip(ridge.predict(X_scaled), 0.0, 1.0)
        return float(roc_auc_score(y_true, pred))

    sampler = TPESampler(seed=OPTUNA_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=sys.stderr.isatty(),
    )
    return study


def mask_from_best_trial(study: optuna.Study, n_models: int) -> np.ndarray:
    params = study.best_trial.params
    return np.array([bool(params[f"inc_{j}"]) for j in range(n_models)], dtype=bool)


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
    print(f"Discovered and loaded {n_models} OOF/test model pairs.")

    n_optuna_trials = int(
        os.environ.get("N_OPTUNA_TRIALS", str(DEFAULT_N_OPTUNA_TRIALS))
    )
    if n_optuna_trials < 1:
        raise ValueError("N_OPTUNA_TRIALS must be >= 1")
    print(f"Optuna trials: {n_optuna_trials}")

    selection_counts = np.zeros(n_models, dtype=np.int64)
    study = run_optuna_subset_ridge(
        X_oof,
        y_true,
        model_names,
        n_trials=n_optuna_trials,
        selection_counts=selection_counts,
    )

    trial_denom = max(len(study.trials), 1)
    # ── Selection frequency across all trials ─────────────────────────────────
    print("\nSelection frequency (fraction of trials where model was included):")
    freqs = selection_counts / float(trial_denom)
    order = np.argsort(-freqs)
    for j in order:
        print(f"  {freqs[j]:.4f}  {model_names[j]}")

    best_mask = mask_from_best_trial(study, n_models)
    best_alpha = float(study.best_trial.params["alpha"])
    selected_names = [model_names[j] for j in np.flatnonzero(best_mask)]

    print(f"\nBest Optuna trial OOF AUC: {study.best_value:.6f}")
    print(f"Best Ridge alpha: {best_alpha:.6f}")
    print(
        f"Best subset size: {best_mask.sum()} / {n_models} "
        f"({100.0 * best_mask.mean():.1f}%)"
    )
    print("Models in best subset:")
    for name in selected_names:
        print(f"  {name}")

    # ── Final Ridge on best subset ───────────────────────────────────────────
    idx = np.flatnonzero(best_mask)
    X_oof_sub = X_oof[:, idx]
    X_test_sub = X_test[:, idx]

    scaler = StandardScaler()
    X_oof_scaled = scaler.fit_transform(X_oof_sub)
    X_test_scaled = scaler.transform(X_test_sub)

    ridge = Ridge(
        alpha=best_alpha, fit_intercept=True, random_state=RIDGE_RANDOM_STATE
    )
    ridge.fit(X_oof_scaled, y_true)

    print("\nFinal Ridge coefficients (selected models only):")
    for name, coef in zip(selected_names, ridge.coef_):
        print(f"  {name:<48} {coef:.6f}")
    print(f"  {'intercept':<48} {ridge.intercept_:.6f}")

    oof_ensemble = np.clip(ridge.predict(X_oof_scaled), 0.0, 1.0)
    pred_ensemble = np.clip(ridge.predict(X_test_scaled), 0.0, 1.0)

    ensemble_auc = roc_auc_score(y_true, oof_ensemble)
    print(f"\nFinal Ridge ensemble OOF AUC: {ensemble_auc:.6f}")

    np.save(DATA_DIR / "oof_ridge.npy", oof_ensemble)
    np.save(DATA_DIR / "test_ridge.npy", pred_ensemble)
    print(f"Saved OOF ensemble → {DATA_DIR / 'oof_ridge.npy'}")
    print(f"Saved test ensemble → {DATA_DIR / 'test_ridge.npy'}")

    submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_ensemble})
    out_path = DATA_DIR / "submission_ridge.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nSubmission saved → {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()
