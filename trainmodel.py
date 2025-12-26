import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os, logging, joblib, numpy as np, tensorflow as tf
from keras import layers, models, optimizers, regularizers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import config



logger = logging.getLogger("trainmodel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---- Attention Layer ----
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)   # handles trainable, dtype, name, etc.
        self.units = units
        self.dense_tanh = tf.keras.layers.Dense(units, activation="tanh")
        self.dense_score = tf.keras.layers.Dense(1)

    def call(self, inputs):
        e = self.dense_tanh(inputs)
        w = self.dense_score(e)
        a = tf.nn.softmax(w, axis=1)
        return tf.reduce_sum(inputs * a, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# ---- Encoder ----
def build_encoder(seq_len, n_seq_feats, n_last_feats, n_sensors, lr=2e-4, emb_dim=16):
    seq_inp   = layers.Input((seq_len, n_seq_feats), name="seq_inp")
    last_inp  = layers.Input((n_last_feats,), name="last_inp")
    sensor_in = layers.Input((), dtype=tf.int32, name="sensor_id")

    sid_vec = layers.Flatten()(layers.Embedding(n_sensors, emb_dim, name="sid_emb")(sensor_in))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(seq_inp)
    ctx = TemporalAttention()(x)
    fused = layers.Concatenate(name="fused_concat")([ctx, last_inp, sid_vec])
    fused = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fused_dense")(fused)
    out = layers.Dense(1, activation="sigmoid", name="global_head")(fused)

    model = models.Model([seq_inp, last_inp, sensor_in], out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 metrics.AUC(name="auc_roc", curve="ROC"),
                 metrics.AUC(name="auc_pr", curve="PR")]
    )
    # Feature extractor for ONNX export
    feat_model = models.Model([seq_inp, last_inp, sensor_in], fused)
    return model, feat_model

# ---- ONNX Export ----
def export_encoder_onnx(feat_model, onnx_path, seq_len, n_seq_feats, n_last_feats):
    import tf2onnx
    specs = (
        tf.TensorSpec([None, seq_len, n_seq_feats], dtype=tf.float32, name="seq_inp"),
        tf.TensorSpec([None, n_last_feats], dtype=tf.float32, name="last_inp"),
        tf.TensorSpec([None], dtype=tf.int32, name="sensor_id"),
    )
    tf2onnx.convert.from_keras(feat_model, input_signature=specs, opset=15, output_path=onnx_path)
    logger.info("ONNX encoder exported to %s", onnx_path)

# ---- Main ----
def main():
    # Dataset-aware paths
    if config.DATASET_MODE == "real":
        SEQ_PATH     = config.REAL_SEQ_PATH
        BEST_PATH    = config.REAL_MODEL_KERAS_PATH
        HISTORY_PATH = config.REAL_HISTORY_PATH
        PRED_PATH    = config.REAL_PRED_PATH
        ONNX_PATH    = config.REAL_MODEL_ONNX_PATH
    elif config.DATASET_MODE == "synthetic":
        SEQ_PATH     = config.SYNTH_SEQ_PATH
        BEST_PATH    = config.SYNTH_MODEL_KERAS_PATH
        HISTORY_PATH = config.SYNTH_HISTORY_PATH
        PRED_PATH    = config.SYNTH_PRED_PATH
        ONNX_PATH    = config.SYNTH_MODEL_ONNX_PATH
    else:
        raise ValueError(f"Unknown dataset mode: {config.DATASET_MODE}")

    # Load sequences
    data = np.load(SEQ_PATH, allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"].astype(int).ravel()
    X_val,   y_val   = data["X_val"],   data["y_val"].astype(int).ravel()
    X_test,  y_test  = data["X_test"],  data["y_test"].astype(int).ravel()

    X_train_last, X_val_last, X_test_last = data["X_train_last"], data["X_val_last"], data["X_test_last"]
    sid_train, sid_val, sid_test = data["sensor_train"], data["sensor_val"], data["sensor_test"]

    # Zero-base sensor IDs
    min_sid = sid_train.min()
    sid_train = sid_train - min_sid
    sid_val   = sid_val   - min_sid
    sid_test  = sid_test  - min_sid
    n_sensors = int(np.max([sid_train.max(), sid_val.max(), sid_test.max()])) + 1

    # Build model
    seq_len, n_seq_feats, n_last_feats = X_train.shape[1], X_train.shape[2], X_train_last.shape[1]
    model, feat_model = build_encoder(seq_len, n_seq_feats, n_last_feats, n_sensors, lr=config.LR)
    logger.info("Model built: seq_len=%d n_seq_feats=%d n_last_feats=%d", seq_len, n_seq_feats, n_last_feats)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_auc_pr", mode="max", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(BEST_PATH, monitor="val_auc_pr", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_auc_pr", mode="max", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    # Train
    history = model.fit(
        [X_train, X_train_last, sid_train], y_train,
        validation_data=([X_val, X_val_last, sid_val], y_val),
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    # Save model and history
    model.save(BEST_PATH)
    joblib.dump(history.history, HISTORY_PATH)
    logger.info("Saved model to %s and history to %s", BEST_PATH, HISTORY_PATH)

    # Evaluate on test
    y_prob = model.predict([X_test, X_test_last, sid_test], batch_size=config.BATCH_SIZE)
    y_pred = (y_prob >= 0.5).astype(int)
    logger.info("Test ROC-AUC=%.3f PR-AUC=%.3f F1=%.3f Acc=%.3f MCC=%.3f",
                roc_auc_score(y_test, y_prob),
                average_precision_score(y_test, y_prob),
                f1_score(y_test, y_pred),
                (y_pred == y_test).mean(),
                matthews_corrcoef(y_test, y_pred))

    # Save predictions
    np.savez(PRED_PATH, y_true=y_test, y_prob=y_prob)
    logger.info("Saved predictions to %s", PRED_PATH)

    # Export encoder to ONNX
    export_encoder_onnx(feat_model, ONNX_PATH, seq_len, n_seq_feats, n_last_feats)

if __name__ == "__main__":
    main()
