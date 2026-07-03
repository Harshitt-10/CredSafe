from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def main():

    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["TARGET"]
    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}\n")
    thresholds = np.arange(0.10, 0.91, 0.05)
    results = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )
        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )
        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            }
        )

    results = pd.DataFrame(results)
    print(results)
    best = results.loc[results["F1 Score"].idxmax()]
    print("\nBest Threshold")
    print(best)
    results.to_csv(
        REPORTS_DIR / "threshold_results.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
