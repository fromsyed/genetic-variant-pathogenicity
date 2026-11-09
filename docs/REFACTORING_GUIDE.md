# Notebook-to-Python Refactoring Guide

## What changed

The notebook's reusable workflow has been separated into Python modules:

| File | Responsibility |
|---|---|
| `src/config.py` | Project paths, selected genes, features and model settings |
| `src/data_preprocessing.py` | Load ClinVar, map labels, extract gene/protein features, validate and save datasets |
| `src/model.py` | Build the preprocessing and Random Forest pipeline |
| `src/train.py` | Train the model, evaluate the held-out test split, save the model and test results |
| `src/evaluate.py` | Protein-grouped cross-validation and leave-one-gene-out evaluation |
| `run_pipeline.py` | Run the complete preprocessing, training and evaluation workflow |

## What remains in the notebook

Keep `notebooks/01_dataset_exploration_final_reviewed.ipynb` as the research record for:
- exploratory data analysis and plots;
- SHAP global and local explanations;
- narrative interpretation and research notes.

The notebook is not deleted or replaced. The Python modules make the repeatable pipeline easier to run and maintain.

## How to install and run

Run commands from the project root (the directory containing `app.py` and `run_pipeline.py`).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

The raw ClinVar export is expected at:

```text
data/raw/clinvar_download_20261107_103417.tsv
```

If your local filename differs, update `RAW_FILENAME` in `src/config.py`.

## Generated outputs

The pipeline writes:

```text
data/processed/clinvar_labeled.csv
data/processed/variant_lookup.csv
models/random_forest_clinvar.pkl
results/test_metrics.json
results/confusion_matrix.csv
results/test_predictions.csv
results/protein_grouped_cv.csv
results/leave_one_gene_out.csv
results/additional_validation.json
```

## Important compatibility note

The saved model keeps the same feature names, preprocessing structure and Random Forest configuration as the notebook. The Streamlit app should continue to load `models/random_forest_clinvar.pkl`.

Before replacing your existing model, keep a backup of the current `.pkl` file. Run the pipeline and test all three app tabs after retraining.

## Reproducibility notes

- Random seed: `42`.
- Held-out test size: `20%`, stratified by target label.
- Random Forest: 300 trees, balanced class weights.
- Categorical values: one-hot encoded with unknown categories ignored.
- Protein position: standardized inside the scikit-learn pipeline.
- The preprocessing transformer is fitted only on each training fold.
- Protein-grouped validation groups by gene, protein position, original amino acid and new amino acid.
- Leave-one-gene-out evaluation tests transfer to a gene absent from the training fold.

## Limitations to preserve in reports

- The project uses only six selected genes and a small set of six input features.
- ClinVar records and classifications can change over time; record the download date and query.
- Unsupported, uncertain and conflicting classifications are excluded from binary modelling.
- A random split may be optimistic when biologically related variants occur in both partitions.
- Gene-held-out performance was below the majority-class baseline in the current experiment.
- Model probabilities are not calibrated clinical risks.
- This project is for research and education, not clinical diagnosis or medical decisions.
