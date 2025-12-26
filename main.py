from collect_data import load_dataset, clean
from preprocessing import main as preprocess_main
from trainmodel import main as train_main
from evaluate import main as evaluate_main
from plots import main as plots_main   
import config


def main():
    # 1) Load and clean dataset (real or synthetic)
    df = load_dataset()
    df = clean(df)
    out_path = config.REAL_DATA_PATH if config.DATASET_MODE == "real" else config.SYNTH_DATA_PATH
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path} | shape={df.shape}")

    # 2) Preprocess (grouped splits, per-sensor scaling, feature engineering, sequences)
    #    Writes sequences to config.SEQ_PATH (npz)
    preprocess_main()

    # 3) Train (BiLSTM + attention encoder, per-sensor logistic calibrators, ONNX export)
    train_main()

    # 4) Evaluate (ROC-AUC, PR-AUC, F1, per-sensor PR-AUC)
    evaluate_main()

    # 5) Plot results (distributions, metrics trends, predictions, dataset comparisons)
    plots_main()


if __name__ == "__main__":
    main()
