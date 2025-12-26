import pandas as pd
import numpy as np
import config
from collect_data import load_dataset, clean

def inject_fault_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fault_mask = df[config.TARGET_COL] == 1

    # Inject synthetic signal changes when fault_status == 1
    df.loc[fault_mask, "vibration"] += np.random.normal(2.0, 0.5, fault_mask.sum())
    df.loc[fault_mask, "pressure"] *= 0.7
    df.loc[fault_mask, "voltage"] += np.linspace(0, -1, fault_mask.sum())
    df.loc[fault_mask, "current"] *= 1.2

    # Optional: perturb FFT features if present
    if "fft_feature1" in df.columns:
        df.loc[fault_mask, "fft_feature1"] *= 1.5
    if "fft_feature2" in df.columns:
        df.loc[fault_mask, "fft_feature2"] *= 0.5

    return df

def main():
    # Load and clean original dataset
    raw = clean(load_dataset())
    print("Original dataset shape:", raw.shape)

    # Inject signals aligned with fault_status
    enriched = inject_fault_signals(raw)

    # Overwrite the original dataset file path
    dataset_path = config.REAL_DATA_PATH  # path used by your pipeline
    enriched.to_csv(dataset_path, index=False)

    print(f"Replaced original dataset with enriched dataset at: {dataset_path}")
    print("Sample after enrichment:")
    print(enriched.head())

if __name__ == "__main__":
    main()
