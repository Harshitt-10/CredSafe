from pathlib import Path
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load tuned model and test dataset."""
    model = joblib.load(MODELS_DIR / "xgboost_tuned.joblib")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")

    return model, X_test


def main():
    model, X_test = load_data()
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    # Use only first 1000 samples for faster explanations
    X_sample = X_test.iloc[:1000]
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    print("Generating summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generating bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generating waterfall plot...")
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_sample.iloc[0],
        feature_names=X_sample.columns,
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )
    plt.savefig(
        REPORTS_DIR / "shap_waterfall.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("\nSHAP reports saved successfully.")
    print("\nGenerated files:")
    print("- reports/shap_summary.png")
    print("- reports/shap_feature_importance.png")
    print("- reports/shap_waterfall.png")


if __name__ == "__main__":
    main()
