"""Run ClinVar preprocessing, Random Forest training, and additional evaluation."""
from src.config import RAW_DIR, RAW_FILENAME
from src.data_preprocessing import load_raw_clinvar, prepare_dataset, save_processed_data
from src.train import train_and_evaluate
from src.evaluate import run_additional_evaluations


def main():
    raw_path = RAW_DIR / RAW_FILENAME
    raw_df = load_raw_clinvar(raw_path)
    print(f"Loaded {len(raw_df):,} raw ClinVar records.")

    model_df, funnel, label_audit = prepare_dataset(raw_df)
    print("\nDataset filtering funnel:")
    print(funnel.to_string(index=False))
    print("\nFinal class distribution:")
    print(model_df["Label"].map({0: "Benign", 1: "Pathogenic"}).value_counts())

    labeled_path, lookup_path = save_processed_data(model_df)
    print(f"\nSaved processed dataset: {labeled_path}")
    print(f"Saved variant lookup: {lookup_path}")

    train_and_evaluate(labeled_path)
    run_additional_evaluations(labeled_path)


if __name__ == "__main__":
    main()
