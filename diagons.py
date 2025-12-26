import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score, matthews_corrcoef

def sweep_thresholds(pred_file):
    # Load saved predictions
    data = np.load(pred_file)
    y_true, y_prob = data["y_true"], data["y_prob"].ravel()

    # Sweep thresholds
    thresholds = np.linspace(0, 1, 101)
    accs, mccs = [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        accs.append(accuracy_score(y_true, y_pred))
        mccs.append(matthews_corrcoef(y_true, y_pred))

    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    # Plot everything
    plt.figure(figsize=(12,8))

    plt.subplot(2,2,1)
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(rec, prec, label="PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(thresholds, accs, label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(thresholds, mccs, label="MCC")
    plt.xlabel("Threshold")
    plt.ylabel("MCC")
    plt.title("MCC vs Threshold")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Report best thresholds
    best_acc_t = thresholds[np.argmax(accs)]
    best_mcc_t = thresholds[np.argmax(mccs)]
    print("Best accuracy threshold:", best_acc_t, "Accuracy:", max(accs))
    print("Best MCC threshold:", best_mcc_t, "MCC:", max(mccs))

def main():
    # Adjust path depending on dataset mode
    pred_file = "artifacts/predictions/real_predictions.npz"
    # Or: pred_file = "artifacts/predictions/synth_predictions.npz"
    sweep_thresholds(pred_file)

if __name__ == "__main__":
    main()
