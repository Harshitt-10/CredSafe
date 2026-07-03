from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    '''Loading datasets from processed directory'''
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train.parquet")["TARGET"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["TARGET"]
    return X_train, X_test, y_train, y_test


def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, use_smote=False):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(
        model_name,
        y_test,
        y_pred,
        y_prob,
    )
    filename = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    joblib.dump(model, MODELS_DIR / f"{filename}.joblib")
    print(f"{model_name} saved successfully.")
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training data : {X_train.shape}")
    print(f"Testing data  : {X_test.shape}")
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")
    MODELS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),

        "XGBoost (Balanced)": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        ),

        "XGBoost (Custom)": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []
    for model_name, model in MODELS.items():
        metrics = train_and_evaluate(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(comparison_df)
    comparison_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )
    comparison_df.to_markdown(
        REPORTS_DIR / "model_comparison.md",
        index=False,
    )
    print("\nModel comparison saved to reports/model_comparison.csv")


if __name__ == "__main__":
    main()
