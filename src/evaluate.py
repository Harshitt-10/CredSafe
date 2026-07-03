from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def evaluate_model(model_name, y_true, y_pred, y_prob):
    """
    Evaluate a classification model and save a confusion matrix.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }
    print(f"\n{'=' * 50}")
    print(model_name)
    print(f"{'=' * 50}")
    for metric, value in metrics.items():
        print(f"{metric:<12}: {value:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(REPORTS_DIR / f"{model_name.lower().replace(' ', '_')}_cm.png")
    plt.close()

    return {
        "Model": model_name,
        **metrics,
    }
