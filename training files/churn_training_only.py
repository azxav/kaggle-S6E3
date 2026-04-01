import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.stats import rankdata
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

# =========================
# Paths
# =========================
TRAIN_PATH = Path("../input/competitions/playground-series-s6e3/train.csv")
TEST_PATH = Path("../input/competitions/playground-series-s6e3/test.csv")
SUB_PATH = Path("../input/competitions/playground-series-s6e3/sample_submission.csv")
ORIGINAL_PATH = Path("../input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_TRIALS = 15


# =========================
# Data loading + preprocessing
# =========================
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SUB_PATH)
    original = pd.read_csv(ORIGINAL_PATH)
    return train, test, sample_submission, original


def basic_cleanup(train: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    original = original.copy()

    train = train.drop(columns=["id"])
    original = original.drop(columns=["customerID"])

    original["TotalCharges"] = pd.to_numeric(original["TotalCharges"], errors="coerce").fillna(0)
    train["TotalCharges"] = pd.to_numeric(train["TotalCharges"], errors="coerce").fillna(0)
    test["TotalCharges"] = pd.to_numeric(test["TotalCharges"], errors="coerce").fillna(0)

    original["Churn"] = original["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    train["Churn"] = train["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    return train, test, original


def add_knn_anchor_features(train: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame):
    features = ["tenure", "MonthlyCharges", "TotalCharges"]
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(original[features])

    distances_tr, indices_tr = nn.kneighbors(train[features])
    train["original_churn_signal"] = (
        original.iloc[indices_tr.flatten()]["Churn"].to_numpy().reshape(-1, 5).mean(axis=1)
    )
    train["match_dist"] = distances_tr.mean(axis=1)

    distances_ts, indices_ts = nn.kneighbors(test[features])
    test["original_churn_signal"] = (
        original.iloc[indices_ts.flatten()]["Churn"].to_numpy().reshape(-1, 5).mean(axis=1)
    )
    test["match_dist"] = distances_ts.mean(axis=1)

    return train, test


def merge_with_original(train: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    original = original.copy()

    train["is_synthetic"] = 1
    test["is_synthetic"] = 1
    original["is_synthetic"] = 0

    train_full = pd.concat([train, original], axis=0).reset_index(drop=True)
    train_full["original_churn_signal"] = train_full["original_churn_signal"].fillna(train_full["Churn"])
    train_full["match_dist"] = train_full["match_dist"].fillna(0)

    return train_full, test


def collapse_service_labels(train_full: pd.DataFrame, test: pd.DataFrame):
    internet_addons = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for df in [train_full, test]:
        for col in internet_addons:
            df[col] = df[col].replace("No internet service", "No")
        df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

    return train_full, test


def add_pre_encoding_features(train_full: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame):
    for df in [train_full, test]:
        df["contract_x_internet"] = df["Contract"].astype(str) + "_" + df["InternetService"].astype(str)

    orig_cat_cols = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
        "Partner",
        "Dependents",
        "MultipleLines",
    ]

    for col in orig_cat_cols:
        prob_map = original.groupby(col)["Churn"].mean()
        train_full[f"orig_proba_{col}"] = train_full[col].map(prob_map).fillna(0.5)
        test[f"orig_proba_{col}"] = test[col].map(prob_map).fillna(0.5)

    for df in [train_full, test]:
        df["no_protection_score"] = (
            df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]].eq("No").sum(axis=1)
        )
        df["mtm_fiber"] = (
            (df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic")
        ).astype(int)
        df["autopay_flag"] = (df["PaymentMethod"] != "Electronic check").astype(int)
        df["has_any_addon"] = (
            df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]]
            .eq("Yes")
            .any(axis=1)
            .astype(int)
        )

    return train_full, test


def encode_and_engineer(train_full: pd.DataFrame, test: pd.DataFrame):
    train_full = train_full.copy()
    test = test.copy()

    binary_cols = []
    onehot_cols = []

    obj_cols = train_full.select_dtypes(include="object").columns
    for col in obj_cols:
        n_unique = train_full[col].nunique()
        if n_unique == 2:
            binary_cols.append(col)
        else:
            onehot_cols.append(col)

    for df in [train_full, test]:
        for col in binary_cols:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals <= {"Yes", "No"}:
                df[col] = df[col].map({"Yes": 1, "No": 0})
            elif unique_vals <= {"Male", "Female"}:
                df[col] = df[col].map({"Male": 1, "Female": 0})

    train_full = pd.get_dummies(train_full, columns=onehot_cols, drop_first=True)
    test = pd.get_dummies(test, columns=onehot_cols, drop_first=True)

    for df in [train_full, test]:
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        df["service_count"] = df[service_cols].sum(axis=1)
        df["charges_per_month"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0)
        df["charges_deviation"] = df["TotalCharges"] - (df["tenure"] * df["MonthlyCharges"])
        df["monthly_to_total_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
        df["price_per_service"] = df["MonthlyCharges"] / df["service_count"].clip(lower=1)

        tenure_bins = pd.cut(df["tenure"], bins=[-1, 6, 24, 48, 72], labels=[0, 1, 2, 3]).astype(int)
        df["tenure_bin_x_monthly"] = tenure_bins * df["MonthlyCharges"]
        df["early_life_flag"] = (df["tenure"] <= 6).astype(int)
        df["tenure_squared"] = df["tenure"] ** 2

    return train_full, test, binary_cols, onehot_cols


def build_matrices(train_full: pd.DataFrame, test: pd.DataFrame):
    y = train_full["Churn"].copy()
    X = train_full.drop(columns=["Churn"]).copy()
    X_test = test.drop(columns=["id"]).copy()
    X_test = X_test.reindex(columns=X.columns, fill_value=0)
    return X, y, X_test


# =========================
# Training helpers
# =========================
def cv_baseline_report(X_train: pd.DataFrame, y_train: pd.Series):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    models = {
        "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=SEED,
            eval_metric="auc",
            verbosity=0,
        ),
    }

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
        results[name] = {"Mean AUC": cv_scores.mean(), "Std": cv_scores.std()}
        print(f"{name}: AUC = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return results


def tune_models(X: pd.DataFrame, y: pd.Series):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    def objective_xgb(trial):
        model = XGBClassifier(
            n_estimators=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=SEED,
            early_stopping_rounds=50,
            eval_metric="auc",
            verbosity=0,
        )
        model.fit(X_tune, y_tune, eval_set=[(X_eval, y_eval)], verbose=False)
        return model.best_score

    def objective_lgbm(trial):
        model = LGBMClassifier(
            n_estimators=2000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=SEED,
            verbose=-1,
        )
        model.fit(
            X_tune,
            y_tune,
            eval_set=[(X_eval, y_eval)],
            callbacks=[early_stopping(50), log_evaluation(0)],
        )
        return model.best_score_["valid_0"]["binary_logloss"]

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)
    print(f"XGBoost best AUC: {study_xgb.best_value:.4f}")
    print(f"XGBoost best params: {study_xgb.best_params}")

    study_lgbm = optuna.create_study(direction="minimize")
    study_lgbm.optimize(objective_lgbm, n_trials=N_TRIALS)
    print(f"LightGBM best logloss: {study_lgbm.best_value:.4f}")
    print(f"LightGBM best params: {study_lgbm.best_params}")

    return study_xgb, study_lgbm


def generate_oof_and_test_predictions(model_name: str, base_model, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X), dtype=np.float32)
    test_preds_folds = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[tr_idx]
        y_tr = y.iloc[tr_idx]
        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        model = base_model()

        if model_name.lower().startswith("xgb"):
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif model_name.lower().startswith("lgb"):
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[log_evaluation(0)])
        else:
            model.fit(X_tr, y_tr)

        va_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(X_test)[:, 1]

        oof[va_idx] = va_pred
        test_preds_folds.append(te_pred)
        fold_auc = roc_auc_score(y_va, va_pred)
        fold_scores.append(fold_auc)
        print(f"{model_name} fold {fold}: AUC = {fold_auc:.5f}")

    test_pred = np.mean(np.column_stack(test_preds_folds), axis=1)
    full_auc = roc_auc_score(y, oof)
    print(f"{model_name} OOF AUC = {full_auc:.5f}")
    print(f"{model_name} fold AUC mean/std = {np.mean(fold_scores):.5f}/{np.std(fold_scores):.5f}")
    return oof, test_pred, fold_scores, full_auc


# =========================
# Main
# =========================
def main():
    train, test, sample_submission, original = load_data()
    test_ids = test["id"].copy()

    train, test, original = basic_cleanup(train, test, original)
    train, test = add_knn_anchor_features(train, test, original)
    train_full, test = merge_with_original(train, test, original)
    train_full, test = collapse_service_labels(train_full, test)
    train_full, test = add_pre_encoding_features(train_full, test, original)
    train_full, test, binary_cols, onehot_cols = encode_and_engineer(train_full, test)
    X, y, X_test = build_matrices(train_full, test)

    # Validation split kept to mirror notebook outputs for quick checks
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"train_full shape: {train_full.shape}")
    print(f"test shape:       {test.shape}")
    print(f"X shape:          {X.shape}")
    print(f"X_test shape:     {X_test.shape}")
    print(f"Binary cols:      {len(binary_cols)}")
    print(f"One-hot cols:     {len(onehot_cols)}")

    baseline_results = cv_baseline_report(X_train, y_train)
    pd.DataFrame(baseline_results).T.to_csv(OUTPUT_DIR / "baseline_cv_results.csv")

    study_xgb, study_lgbm = tune_models(X, y)

    def make_xgb():
        return XGBClassifier(
            **study_xgb.best_params,
            n_estimators=2000,
            random_state=SEED,
            eval_metric="auc",
            verbosity=0,
        )

    def make_lgbm():
        return LGBMClassifier(
            **study_lgbm.best_params,
            n_estimators=2000,
            random_state=SEED,
            verbose=-1,
        )

    # Per-model OOF and test predictions
    oof_xgb, test_xgb, xgb_fold_scores, xgb_oof_auc = generate_oof_and_test_predictions(
        "xgb", make_xgb, X, y, X_test
    )
    oof_lgbm, test_lgbm, lgbm_fold_scores, lgbm_oof_auc = generate_oof_and_test_predictions(
        "lgbm", make_lgbm, X, y, X_test
    )

    np.save(OUTPUT_DIR / "oof_xgb.npy", oof_xgb)
    np.save(OUTPUT_DIR / "test_xgb.npy", test_xgb)
    np.save(OUTPUT_DIR / "oof_lgbm.npy", oof_lgbm)
    np.save(OUTPUT_DIR / "test_lgbm.npy", test_lgbm)

    pd.DataFrame({
        "id": test_ids,
        "xgb_pred": test_xgb,
        "lgbm_pred": test_lgbm,
    }).to_csv(OUTPUT_DIR / "base_model_test_predictions.csv", index=False)

    oof_df = pd.DataFrame({
        "target": y,
        "xgb_oof": oof_xgb,
        "lgbm_oof": oof_lgbm,
    })
    oof_df.to_csv(OUTPUT_DIR / "base_model_oof_predictions.csv", index=False)

    metrics_df = pd.DataFrame([
        {"model": "xgb", "oof_auc": xgb_oof_auc, "fold_mean_auc": np.mean(xgb_fold_scores), "fold_std_auc": np.std(xgb_fold_scores)},
        {"model": "lgbm", "oof_auc": lgbm_oof_auc, "fold_mean_auc": np.mean(lgbm_fold_scores), "fold_std_auc": np.std(lgbm_fold_scores)},
    ])
    metrics_df.to_csv(OUTPUT_DIR / "base_model_metrics.csv", index=False)

    # Stacking ensemble
    estimators = [
        ("xgb", make_xgb()),
        ("lgbm", make_lgbm()),
    ]
    final_stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method="predict_proba",
    )
    final_stack.fit(X, y)
    ensemble_preds = final_stack.predict_proba(X_test)[:, 1]
    val_preds = final_stack.predict_proba(X_val)[:, 1]
    stack_val_auc = roc_auc_score(y_val, val_preds)
    print(f"Stacking Ensemble Validation AUC: {stack_val_auc:.4f}")

    submission = sample_submission.copy()
    submission["Churn"] = ensemble_preds
    submission.to_csv(OUTPUT_DIR / "submission_ensemble.csv", index=False)

    # Rank-average ensemble based on CV-generated model predictions
    ensemble_preds_rank = (rankdata(test_lgbm) + rankdata(test_xgb)) / (2 * len(test_lgbm))
    oof_rank = (rankdata(oof_lgbm) + rankdata(oof_xgb)) / (2 * len(oof_lgbm))
    rank_oof_auc = roc_auc_score(y, oof_rank)
    print(f"Rank Avg OOF AUC: {rank_oof_auc:.4f}")

    submission_rank = sample_submission.copy()
    submission_rank["Churn"] = ensemble_preds_rank
    submission_rank.to_csv(OUTPUT_DIR / "submission_rank_avg.csv", index=False)

    np.save(OUTPUT_DIR / "oof_rank_avg.npy", oof_rank)
    np.save(OUTPUT_DIR / "test_rank_avg.npy", ensemble_preds_rank)
    np.save(OUTPUT_DIR / "test_stacking.npy", ensemble_preds)

    summary = {
        "stack_val_auc": stack_val_auc,
        "rank_oof_auc": rank_oof_auc,
        "xgb_oof_auc": xgb_oof_auc,
        "lgbm_oof_auc": lgbm_oof_auc,
        "n_features": X.shape[1],
        "n_train_rows": X.shape[0],
        "n_test_rows": X_test.shape[0],
    }
    pd.Series(summary).to_csv(OUTPUT_DIR / "run_summary.csv", header=False)

    print("\nSaved files:")
    for name in [
        "baseline_cv_results.csv",
        "base_model_metrics.csv",
        "base_model_oof_predictions.csv",
        "base_model_test_predictions.csv",
        "oof_xgb.npy",
        "test_xgb.npy",
        "oof_lgbm.npy",
        "test_lgbm.npy",
        "oof_rank_avg.npy",
        "test_rank_avg.npy",
        "test_stacking.npy",
        "submission_ensemble.csv",
        "submission_rank_avg.csv",
        "run_summary.csv",
    ]:
        print(f" - {name}")


if __name__ == "__main__":
    main()
