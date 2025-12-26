from pathlib import Path

# ---- Paths ----
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
ARTIFACTS  = ROOT / "artifacts"
MODEL_DIR  = ARTIFACTS / "model"
PLOTS_DIR  = ARTIFACTS / "plots"
PRED_DIR   = ARTIFACTS / "predictions"
CALIB_DIR  = ARTIFACTS / "calibrators"

for d in (DATA_DIR, ARTIFACTS, MODEL_DIR, PLOTS_DIR, PRED_DIR, CALIB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Data sources ----
DATASET_MODE   = "synthetic"   # "real" or "synthetic"
REAL_DATA_FILE = "iot_real_sensor_dataset.csv"
SYNTH_DATA_FILE= "synthetic_iot_dataset.csv"
REAL_DATA_PATH = DATA_DIR / REAL_DATA_FILE
SYNTH_DATA_PATH= DATA_DIR / SYNTH_DATA_FILE

# ---- Canonical columns ----
TIMESTAMP_COL  = "timestamp"
SENSOR_ID_COL  = "sensor_id"
TARGET_COL     = "fault_status"

FEATURE_COLS   = [
    "temperature", "vibration", "pressure", "voltage", "current",
    "fft_feature1", "fft_feature2"
]

EXCLUDE_COLS   = [
    "anomaly_score",
    "normalized_temp", "normalized_vibration", "normalized_pressure",
    "normalized_voltage", "normalized_current"
]

# ---- Sequence + label policy ----
SEQ_LEN        = 6
SEQ_STRIDE     = 6
LABEL_POLICY   = "last"   # "last" | "any" | "majority" | "recent"
RECENT_STEPS   = 3  

# ---- Splits ----
TEST_SPLIT     = 0.20
VAL_SPLIT      = 0.20
GROUP_SPLIT_COL= SENSOR_ID_COL
SEED           = 42

# ---- Scaling ----
SCALER_TYPE    = "robust"
SCALE_ON       = "train_only"

# ---- Training ----
BATCH_SIZE     = 64
EPOCHS         = 20
LR             = 2e-4

CALIBRATION_METHOD = "isotonic"  # "platt" | "isotonic" | None
CALIBRATION_PER_SENSOR = True

# ---- Artifacts ----
# Real dataset artifacts
REAL_SEQ_PATH         = ARTIFACTS / "real_sequences.npz"
REAL_SCALER_PATH      = ARTIFACTS / "real_scaler.joblib"
REAL_HISTORY_PATH     = ARTIFACTS / "real_history.joblib"
REAL_MODEL_KERAS_PATH = MODEL_DIR / "real_model.keras"
REAL_PRED_PATH        = PRED_DIR / "real_predictions.npz"
REAL_CALIBRATOR_GENERAL= CALIB_DIR / "real_calibrator_general.joblib"
REAL_CALIB_PER_SENSOR = CALIB_DIR/ "real_per_sensor_calib.joblib"
REAL_MODEL_ONNX_PATH = MODEL_DIR/ "real_model.onnx"

# Synthetic dataset artifacts
SYNTH_SEQ_PATH      = ARTIFACTS / "synth_sequences.npz"
SYNTH_SCALER_PATH   = ARTIFACTS / "synth_scaler.joblib"
SYNTH_HISTORY_PATH  = ARTIFACTS / "synth_history.joblib"
SYNTH_MODEL_KERAS_PATH= MODEL_DIR / "synth_model.keras"
SYNTH_PRED_PATH       = PRED_DIR / "synth_predictions.npz"
SYNTH_CALIBRATOR_GENERAL= CALIB_DIR / "synth_calibrator_general.joblib"
SYNTH_CALIB_PER_SENSOR = CALIB_DIR/ "synth_per_sensor_calib.joblib"
SYNTH_MODEL_ONNX_PATH = MODEL_DIR/ "synth_model.onnx"

# ---- Synthetic defaults ----
SYNTH_ROWS     = 20_000
SENSOR_COUNT   = 10
