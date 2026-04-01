import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent
PRED_DIR = DATA_DIR / "S6E3"
TARGET   = "Churn"
ID_COL   = "id"

# ── Load data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# ── Load predictions and clip to valid probability range ──────────────────────
models = {
    "catboost": ("oof_cat.npy",   "test_cat.npy"),
    "lgb":      ("oof_lgb.npy",   "test_lgb.npy"),
    "nn":       ("oof_nn.npy",    "test_nn.npy"),
    "xgb_v1":   ("oof_xgb_v1.npy", "test_xgb_v1.npy"),
    "xgb_v2":   ("oof_xgb_v2.npy", "test_xgb_v2.npy"),
    "ggn":      ("oof_gnn_v1909.npy",    "test_gnn_v1909.npy"),
    "mlp_all_cats": (
        "oof_MLP_ALL_CATS_s42.npy",
        "test_MLP_ALL_CATS_s42.npy"
    ),
    "mlp_base": (
        "oof_MLP_BASE_s42.npy",
        "test_MLP_BASE_s42.npy"
    ),
    "mlp_bin_digit": (
        "oof_MLP_BIN_DIGIT_s42.npy",
        "test_MLP_BIN_DIGIT_s42.npy"
    ),
    "mlp_freq": (
        "oof_MLP_FREQ_s42.npy",
        "test_MLP_FREQ_s42.npy"
    ),
    "mlp_full_fe": (
        "oof_MLP_FULL_FE_s42.npy",
        "test_MLP_FULL_FE_s42.npy"
    ),
    'XGB_FULL_FE_s7': ('oof_XGB_FULL_FE_s7.npy', 'test_XGB_FULL_FE_s7.npy'), 'XGB_BASE_s7': ('oof_XGB_BASE_s7.npy', 'test_XGB_BASE_s7.npy'), 'XGB_BASE_s1234': ('oof_XGB_BASE_s1234.npy', 'test_XGB_BASE_s1234.npy'), 'XGB_BASE_s2025': ('oof_XGB_BASE_s2025.npy', 'test_XGB_BASE_s2025.npy'), 'XGB_BIN_DIGIT_s2025': ('oof_XGB_BIN_DIGIT_s2025.npy', 'test_XGB_BIN_DIGIT_s2025.npy'), 'MLP_ALL_CATS_s42': ('oof_MLP_ALL_CATS_s42.npy', 'test_MLP_ALL_CATS_s42.npy'), 'MLP_BASE_s42': ('oof_MLP_BASE_s42.npy', 'test_MLP_BASE_s42.npy'), 'XGB_BIN_DIGIT_s999': ('oof_XGB_BIN_DIGIT_s999.npy', 'test_XGB_BIN_DIGIT_s999.npy'), 'XGB_FULL_FE_s999': ('oof_XGB_FULL_FE_s999.npy', 'test_XGB_FULL_FE_s999.npy'), 'XGB_BASE_s999': ('oof_XGB_BASE_s999.npy', 'test_XGB_BASE_s999.npy'), 'XGB_ALL_CATS_s2025': ('oof_XGB_ALL_CATS_s2025.npy', 'test_XGB_ALL_CATS_s2025.npy'), 'XGB_FREQ_s42': ('oof_XGB_FREQ_s42.npy', 'test_XGB_FREQ_s42.npy'), 'XGB_ALL_CATS_s7': ('oof_XGB_ALL_CATS_s7.npy', 'test_XGB_ALL_CATS_s7.npy'), 'MLP_FULL_FE_s42': ('oof_MLP_FULL_FE_s42.npy', 'test_MLP_FULL_FE_s42.npy'), 'XGB_BIN_DIGIT_s42': ('oof_XGB_BIN_DIGIT_s42.npy', 'test_XGB_BIN_DIGIT_s42.npy'), 'XGB_FULL_FE_s2025': ('oof_XGB_FULL_FE_s2025.npy', 'test_XGB_FULL_FE_s2025.npy'), 'XGB_FREQ_s2025': ('oof_XGB_FREQ_s2025.npy', 'test_XGB_FREQ_s2025.npy'), 'XGB_FREQ_s7': ('oof_XGB_FREQ_s7.npy', 'test_XGB_FREQ_s7.npy'), 'XGB_BIN_DIGIT_s1234': ('oof_XGB_BIN_DIGIT_s1234.npy', 'test_XGB_BIN_DIGIT_s1234.npy'), 'XGB_FREQ_s999': ('oof_XGB_FREQ_s999.npy', 'test_XGB_FREQ_s999.npy'), 'MLP_FREQ_s42': ('oof_MLP_FREQ_s42.npy', 'test_MLP_FREQ_s42.npy'), 'XGB_ALL_CATS_s999': ('oof_XGB_ALL_CATS_s999.npy', 'test_XGB_ALL_CATS_s999.npy'), 'XGB_FREQ_s1234': ('oof_XGB_FREQ_s1234.npy', 'test_XGB_FREQ_s1234.npy'), 'XGB_ALL_CATS_s42': ('oof_XGB_ALL_CATS_s42.npy', 'test_XGB_ALL_CATS_s42.npy'), 'MLP_BIN_DIGIT_s42': ('oof_MLP_BIN_DIGIT_s42.npy', 'test_MLP_BIN_DIGIT_s42.npy'), 'XGB_BASE_s42': ('oof_XGB_BASE_s42.npy', 'test_XGB_BASE_s42.npy'), 'XGB_ALL_CATS_s1234': ('oof_XGB_ALL_CATS_s1234.npy', 'test_XGB_ALL_CATS_s1234.npy'), 'XGB_FULL_FE_s1234': ('oof_XGB_FULL_FE_s1234.npy', 'test_XGB_FULL_FE_s1234.npy'), 'XGB_FULL_FE_s42': ('oof_XGB_FULL_FE_s42.npy', 'test_XGB_FULL_FE_s42.npy'), 'XGB_BIN_DIGIT_s7': ('oof_XGB_BIN_DIGIT_s7.npy', 'test_XGB_BIN_DIGIT_s7.npy'),
}

oof  = {}
pred = {}
for name, (oof_file, test_file) in models.items():
    oof[name]  = np.clip(np.load(PRED_DIR / oof_file),  0.0, 1.0)
    pred[name] = np.clip(np.load(PRED_DIR / test_file), 0.0, 1.0)

y_true = train[TARGET].values

# ── Individual model AUCs ─────────────────────────────────────────────────────
print("Individual OOF AUCs:")
for name in models:
    auc = roc_auc_score(y_true, oof[name])
    print(f"  {name:<12} {auc:.6f}")

# ── Weight optimisation (COBYLA) ──────────────────────────────────────────────
n_models = len(models)
model_names = list(models.keys())

def objective(w):
    ensemble = sum(oof[m] * w[i] for i, m in enumerate(model_names))
    return -roc_auc_score(y_true, ensemble)

constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
bounds = Bounds(0, 1)
w0 = np.ones(n_models) / n_models

result = minimize(objective, w0, method="COBYLA", bounds=bounds, constraints=constraints)
w = result.x

print("\nOptimised weights:")
for name, wi in zip(model_names, w):
    print(f"  {name:<12} {wi:.6f}")

# ── Build ensemble ────────────────────────────────────────────────────────────
oof_ensemble  = sum(oof[m]  * w[i] for i, m in enumerate(model_names))
pred_ensemble = sum(pred[m] * w[i] for i, m in enumerate(model_names))

ensemble_auc = roc_auc_score(y_true, oof_ensemble)
print(f"\nEnsemble OOF AUC: {ensemble_auc:.6f}")

# ── Submission ────────────────────────────────────────────────────────────────
submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred_ensemble})
out_path = DATA_DIR / "submission.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission saved → {out_path}")
print(submission.head())
