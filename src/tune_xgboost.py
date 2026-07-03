from pathlib import Path
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load processed training and testing datasets."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["TARGET"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["TARGET"]
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_data()
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 2, 5],
    }
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )
    print("\nStarting hyperparameter tuning...\n")
    search.fit(X_train, y_train)
    print("\nTuning completed.\n")
    print("Best ROC-AUC:")
    print(search.best_score_)
    print("\nBest Parameters:")
    print(search.best_params_)
    best_model = search.best_estimator_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest ROC-AUC: {roc_auc:.4f}")
    joblib.dump(
        best_model,
        MODELS_DIR / "xgboost_tuned.joblib"
    )
    with open(MODELS_DIR / "best_xgb_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=4)
    pd.DataFrame(search.cv_results_).to_csv(
        REPORTS_DIR / "xgboost_tuning_results.csv",
        index=False,
    )
    print("\nSaved:")
    print("- models/xgboost_tuned.joblib")
    print("- models/best_xgb_params.json")
    print("- reports/xgboost_tuning_results.csv")


if __name__ == "__main__":
    main()
