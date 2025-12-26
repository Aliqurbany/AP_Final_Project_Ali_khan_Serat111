import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from joblib import dump
import config
from collect_data import clean, load_dataset

def split_time_grouped(df):
    train_parts, val_parts, test_parts = [], [], []
    for sid, g in df.groupby(config.SENSOR_ID_COL):
        g = g.sort_values(config.TIMESTAMP_COL)
        n = len(g)
        n_test = int(n * config.TEST_SPLIT)
        n_trainval = n - n_test
        n_val = int(n_trainval * config.VAL_SPLIT)
        test_parts.append(g.iloc[-n_test:])
        val_parts.append(g.iloc[n_trainval - n_val:n_trainval])
        train_parts.append(g.iloc[:n_trainval - n_val])
    return (pd.concat(train_parts).reset_index(drop=True),
            pd.concat(val_parts).reset_index(drop=True),
            pd.concat(test_parts).reset_index(drop=True))

def add_features(g: pd.DataFrame, win_std=5, win_ma=10):
    g = g.copy()
    for c in config.FEATURE_COLS:
        g[f"{c}_delta"] = g[c].diff()
        g[f"{c}_rstd{win_std}"] = g[c].rolling(win_std, min_periods=1).std()
        g[f"{c}_ma{win_ma}"] = g[c].rolling(win_ma, min_periods=1).mean()
    g.fillna(0.0, inplace=True)
    return g

def scale_global(train_df, val_df, test_df):
    sc = RobustScaler()
    sc.fit(train_df[config.FEATURE_COLS])
    for df in (train_df, val_df, test_df):
        df.loc[:, config.FEATURE_COLS] = sc.transform(df[config.FEATURE_COLS])
    if config.DATASET_MODE=="real":
        dump(sc, config.REAL_SCALER_PATH)
    else:
        dump(sc, config.SYNTH_SCALER_PATH)
    return train_df, val_df, test_df

def make_sequences(df, L=12, S=12):
    X_seq, y, X_last, SIDs = [], [], [], []
    for sid, g in df.groupby(config.SENSOR_ID_COL):
        g = g.sort_values(config.TIMESTAMP_COL)
        g = add_features(g)
        seq_cols = config.FEATURE_COLS + [f"{c}_delta" for c in config.FEATURE_COLS] + [f"{c}_rstd5" for c in config.FEATURE_COLS]
        F = g[seq_cols].values
        T = g[config.TARGET_COL].values
        for i in range(0, len(g) - L + 1, S):
            j = i + L
            X_seq.append(F[i:j])
            X_last.append(F[j-1])
            label = int(T[j-1])  # simplified: last policy
            y.append(label)
            SIDs.append(sid)
    return np.array(X_seq), np.array(y), np.array(X_last), np.array(SIDs)

def main():
    df = clean(load_dataset())
    tr, va, te = split_time_grouped(df)
    tr_s, va_s, te_s = scale_global(tr, va, te)

    X_tr, y_tr, X_tr_last, sid_tr = make_sequences(tr_s)
    X_va, y_va, X_va_last, sid_va = make_sequences(va_s)
    X_te, y_te, X_te_last, sid_te = make_sequences(te_s)

    SEQ_PATH = config.REAL_SEQ_PATH if config.DATASET_MODE=="real" else config.SYNTH_SEQ_PATH
    np.savez_compressed(
        SEQ_PATH,
        X_train=X_tr, y_train=y_tr, sensor_train=sid_tr,
        X_val=X_va,   y_val=y_va,   sensor_val=sid_va,
        X_test=X_te,  y_test=y_te,  sensor_test=sid_te,
        X_train_last=X_tr_last, X_val_last=X_va_last, X_test_last=X_te_last,
    )

    print("Unique values of y_train:", np.unique(y_tr))
    print("Unique values of y_val:", np.unique(y_va))
    print("Unique values of y_test:", np.unique(y_te))
    print("Shapes:", y_tr.shape, y_va.shape, y_te.shape)
    print("train:", X_tr.shape, "pos_rate=", float(y_tr.mean()),
          "| val:", X_va.shape, "pos_rate=", float(y_va.mean()),
          "| test:", X_te.shape, "pos_rate=", float(y_te.mean()))

if __name__ == "__main__":
    main()
