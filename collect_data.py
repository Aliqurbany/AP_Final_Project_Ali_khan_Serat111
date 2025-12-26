"""
collectdata.py
- Loads either the real or synthetic dataset depending on config.DATASET_MODE.
- If synthetic mode and file not found, generates a synthetic dataset with config.SYNTH_ROWS and config.SENSOR_COUNT.
- Drops excluded columns, enforces canonical schema, saves cleaned CSV back to DATA_DIR.
"""

import pandas as pd
import numpy as np
import config

def generate_synthetic(rows: int= config.SYNTH_ROWS, sensors: int = config.SENSOR_COUNT, fault_prob: float = 0.05) -> pd.DataFrame:
    """Generate a synthetic IoT dataset with canonical columns and structured fault episodes."""
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="T")
    sensor_ids = np.random.choice(range(1, sensors + 1), size=rows)

    # Step 1: Generate initial fault labels (Bernoulli)
    fault = np.random.binomial(1, fault_prob, size=rows)

    # Step 2: Stitch contiguous fault episodes (5–20 steps long)
    i = 0
    while i < rows:
        if fault[i] == 1:
            run = np.random.randint(5, 20)
            fault[i:i+run] = 1
            i += run
        else:
            i += 1

    # Step 3: Build base dataset
    data = {
        config.TIMESTAMP_COL: timestamps,
        config.SENSOR_ID_COL: sensor_ids,
        config.TARGET_COL: fault,
    }

    # Step 4: Generate features with drift + fault-dependent shifts
    for feat in config.FEATURE_COLS:
        # Base signal with mild sensor-specific drift
        drift = (sensor_ids / (sensors + 1)) * np.random.uniform(0.05, 0.2)
        base = np.random.normal(loc=drift, scale=1.0, size=rows)

        # Fault episodes: mean shift + variance spike
        mean_shift = np.where(fault == 1, np.random.uniform(0.8, 1.5), 0.0)
        var_scale  = np.where(fault == 1, np.random.uniform(1.5, 3.0), 1.0)

        data[feat] = base * var_scale + mean_shift

    return pd.DataFrame(data)


def load_dataset() -> pd.DataFrame:
    if config.DATASET_MODE == "real":
        path = config.REAL_DATA_PATH
        df = pd.read_csv(path)
    else:
        path = config.SYNTH_DATA_PATH
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            df = generate_synthetic(config.SYNTH_ROWS, config.SENSOR_COUNT)
            df.to_csv(path, index=False)
            print(f"Synthetic dataset generated and saved to {path}")
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop excluded columns if present
    for col in getattr(config, "EXCLUDE_COLS", []):
        if col in df.columns:
            df = df.drop(columns=col)

    # Ensure canonical columns exist
    keep = [config.TIMESTAMP_COL, config.SENSOR_ID_COL,
             config.TARGET_COL] + config.FEATURE_COLS
    df = df[keep].copy()
    
    """df[config.TARGET_COL]= (
        df[config.TARGET_COL].astype(str) # Convert everthing to string for uniformity
        .str.strip() # remove space
        .replace({"1":1, "1.0":1, "0": 0, "0.0": 0}) #Expilict mapping  

    )"""
    print("Dataset shape:", df.shape)
    print("Target value counts:\n", df[config.TARGET_COL].value_counts())

    raw = pd.read_csv(config.REAL_DATA_PATH) 
    print("Raw target unique values (pre-clean):", pd.Series(raw[config.TARGET_COL]).astype(str).str.strip().unique())
    
    # Per-sensor distribution to catch skew
    per_sensor = ( df.groupby(config.SENSOR_ID_COL)[config.TARGET_COL] .value_counts(normalize=True) .rename("rate") .reset_index() )
    print("Per-sensor positive rates (head):") 
    print(per_sensor[per_sensor[config.TARGET_COL] == 1].head(10))




    # Convert types
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce")
    df[config.TARGET_COL] = pd.to_numeric(df[config.TARGET_COL], 
                                          errors="coerce").fillna(0).astype(int)

    # Sanitize features
    for c in config.FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sanity check: verify only 0/1 present after clean
    print("Cleaned target unique values:", df[config.TARGET_COL].unique())    

    # Sort chronologically per sensor
    return df.sort_values([config.TIMESTAMP_COL, config.SENSOR_ID_COL]).reset_index(drop=True)

def main():
    df = load_dataset()
    df = clean(df)
    
    print("Dataset Shape After cleaning:  ",df.shape)
    print("Target columns counts After clean:  ")
    print("After clean",df[config.TARGET_COL].value_counts())

    invalid_count = df[config.TARGET_COL].isna().sum()
    print("The number of invalid count After clean :  ", invalid_count)
    

    out_path = config.REAL_DATA_PATH if config.DATASET_MODE == "real" else config.SYNTH_DATA_PATH
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")

if __name__ == "__main__":
    main()
