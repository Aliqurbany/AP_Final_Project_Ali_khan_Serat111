import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import logging
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, matthews_corrcoef, precision_score, recall_score
)
from trainmodel import TemporalAttention
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")

def main():
    # Paths
    if config.DATASET_MODE == "real":
        SEQ_PATH   = config.REAL_SEQ_PATH
        MODEL_PATH = config.REAL_MODEL_KERAS_PATH
        PRED_PATH  = config.REAL_PRED_PATH
    elif config.DATASET_MODE == "synthetic":
        SEQ_PATH   = config.SYNTH_SEQ_PATH
        MODEL_PATH = config.SYNTH_MODEL_KERAS_PATH
        PRED_PATH  = config.SYNTH_PRED_PATH
    else:
        raise ValueError("Unknown dataset mode")

    # Load data
    data = np.load(SEQ_PATH, allow_pickle=True)
    X_test, y_test = data["X_test"], data["y_test"].astype(int).ravel()
    X_test_last    = data["X_test_last"]
    sid_test       = data["sensor_test"]
    sid_test_idx   = sid_test - sid_test.min()  # zero-based sensor IDs

    # Load model
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False,
        custom_objects={"TemporalAttention": TemporalAttention}
    )

    # Predict probabilities
    y_prob = model.predict([X_test, X_test_last, sid_test_idx], batch_size=config.BATCH_SIZE).ravel()

    # Sweep thresholds to find best MCC
    thresholds = np.linspace(0, 1, 101)
    mccs = []
    for t in thresholds:
        y_pred_tmp = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_test, y_pred_tmp))
    best_mcc_t = thresholds[np.argmax(mccs)]

    # Final predictions
    y_pred = (y_prob >= best_mcc_t).astype(int)

    # Report metrics
    logger.info("Best threshold (MCC): %.2f", best_mcc_t)
    logger.info("Test ROC-AUC=%.3f PR-AUC=%.3f", roc_auc_score(y_test, y_prob), average_precision_score(y_test, y_prob))
    logger.info("Accuracy=%.3f F1=%.3f MCC=%.3f", accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), matthews_corrcoef(y_test, y_pred))
    logger.info("Precision=%.3f Recall=%.3f", precision_score(y_test, y_pred), recall_score(y_test, y_pred))

    # ✅ Save predictions (overwrite existing file)
    np.savez_compressed(
        PRED_PATH,
        y_true=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        sensor_ids=sid_test,
        threshold=best_mcc_t
    )
    logger.info("Predictions overwritten at %s", PRED_PATH)

if __name__ == "__main__":
    main()
