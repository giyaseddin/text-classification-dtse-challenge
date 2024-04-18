from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
from itertools import cycle
import seaborn as sns
import numpy as np


# visualize(file) -> reporting (first parent) -> src -> root
PROJECT_ROOT_PATH: Path = Path(__file__).parent.parent.parent


def ensure_numpy_array(data):
    """Ensure the input data is a numpy array."""
    if isinstance(data, pd.DataFrame):
        return data.values
    return data


def save_experiment_results(y_true_valid, y_pred_valid, label_names, experiment_name, y_pred_test=None):
    base_dir = Path(PROJECT_ROOT_PATH)
    reports_dir = base_dir / f"reports" / experiment_name
    results_dir = base_dir / f"results" / experiment_name

    # Convert input data to numpy arrays if they are DataFrames
    y_true_valid = ensure_numpy_array(y_true_valid)
    y_pred_valid = ensure_numpy_array(y_pred_valid)
    y_pred_test = ensure_numpy_array(y_pred_test)

    # Create directories if they do not exist
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save validation predictions
    df_valid = pd.DataFrame(np.hstack([y_true_valid, y_pred_valid]), columns=[f'true_{label}' for label in label_names] + [f'pred_{label}' for label in label_names])
    df_valid.to_csv(results_dir / "valid_predictions.csv", index=False)

    if y_pred_test:
        # Save test predictions
        df_test = pd.DataFrame(y_pred_test, columns=[f'pred_{label}' for label in label_names])
        df_test.to_csv(results_dir / "test_predictions.csv", index=False)

    # Generate and save classification reports
    for i, label in enumerate(label_names):
        with open(reports_dir / f'clf_report_{label}.json', 'w') as f:
            json.dump(classification_report(y_true_valid[:, i], y_pred_valid[:, i], output_dict=True), f, indent=4)
        print(f"Classification Report for {label}:")
        print(classification_report(y_true_valid[:, i], y_pred_valid[:, i]))

    # Classification Report for validation set
    report = classification_report(y_true_valid, y_pred_valid, target_names=label_names, output_dict=True)
    print(classification_report(y_true_valid, y_pred_valid, target_names=label_names))
    with open(reports_dir / 'clf_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # Plot confusion matrices for each label
    for i, label in enumerate(label_names):
        cm = confusion_matrix(y_true_valid[:, i], y_pred_valid[:, i])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'Not_{label}', label], yticklabels=[f'Not_{label}', label])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {label}')
        plt.savefig(reports_dir / f"{label}_confusion_matrix.png")
        plt.close()

    # Plot ROC-AUC curves for each label
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, label in enumerate(label_names):
        fpr[i], tpr[i], _ = roc_curve(y_true_valid[:, i], y_pred_valid[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(label_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {label_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(reports_dir / "roc_auc_curve.png")
    plt.close()

    # Save ROC-AUC scores
    with open(reports_dir / "roc_auc_scores.json", 'w') as f:
        json.dump(roc_auc, f, indent=4)

    print("All reports and plots have been generated and saved successfully.")
