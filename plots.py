from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import config
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
)

plots_dir = config.PLOTS_DIR
pred_dir  = config.PRED_DIR

def plot_confusion_matrix(y_true, y_pred, out_dir=pred_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Healthy (0)", "Faulty (1)"],
                yticklabels=["Healthy (0)", "Faulty (1)"])
    ax.set_title(f"Confusion Matrix ({config.DATASET_MODE})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    out_file = Path(out_dir) / f"{config.DATASET_MODE}_confusion_matrix.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_roc_pr_curves(y_true, y_prob, out_dir=plots_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0,1],[0,1],"k--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(out_dir) / f"{config.DATASET_MODE}_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label="PR curve")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.savefig(Path(out_dir) / f"{config.DATASET_MODE}_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

def compare_selected_features(real_df, synth_df, out_dir=plots_dir):
    selected_features = ["temperature", "vibration", "pressure"]
    for feature in selected_features:
        combined = pd.concat([
            real_df.assign(source="Real"),
            synth_df.assign(source="Synthetic")
        ])
        fig_html = px.histogram(combined, x=feature, color="source", barmode="overlay",
                                title=f"Comparison of {feature} (Real vs Synthetic)")
        fig_html.write_html(Path(out_dir) / f"compare_{feature}.html")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=combined, x=feature, hue="source", element="step",
                     stat="density", common_norm=False, ax=ax)
        ax.set_title(f"Comparison of {feature} (Real vs Synthetic)")
        fig.savefig(Path(out_dir) / f"compare_{feature}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

def export_metrics_summary(y_true, y_pred, threshold=None, out_dir=plots_dir):
    """
    Export evaluation metrics (Accuracy, F1, MCC, Precision, Recall, Threshold) to a CSV file.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }
    if threshold is not None:
        metrics["Threshold"] = threshold
    df = pd.DataFrame([metrics])
    out_file = Path(out_dir) / f"{config.DATASET_MODE}_metrics_summary.csv"
    df.to_csv(out_file, index=False)
    print(f"Metrics summary saved to {out_file}")

def main():
    if config.DATASET_MODE == "real":
        data = np.load(config.REAL_PRED_PATH)
        real_df = pd.read_csv(config.REAL_DATA_PATH)
        synth_df = pd.read_csv(config.SYNTH_DATA_PATH)
    else:
        data = np.load(config.SYNTH_PRED_PATH)
        real_df = pd.read_csv(config.REAL_DATA_PATH)
        synth_df = pd.read_csv(config.SYNTH_DATA_PATH)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]
    threshold = data.get("threshold") if "threshold" in data.files else None

    plot_confusion_matrix(y_true, y_pred)
    plot_roc_pr_curves(y_true, y_prob)
    compare_selected_features(real_df, synth_df)
    export_metrics_summary(y_true, y_pred, threshold=threshold, out_dir=plots_dir)
    print("Plots and metrics summary saved to", plots_dir)

if __name__ == "__main__":
    main()
