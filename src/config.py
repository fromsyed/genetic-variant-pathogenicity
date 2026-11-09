"""Shared project paths, feature definitions, and model configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# The notebook used this exact ClinVar export filename.
RAW_FILENAME = "clinvar_download_20261107_103417.tsv"

TARGET_GENES = ["CFTR", "BRCA1", "TP53", "HBB", "LDLR", "PAH"]

FEATURES = [
    "Gene",
    "Variant type",
    "Molecular consequence",
    "Protein_Position",
    "Original_AA",
    "New_AA",
]
CATEGORICAL_FEATURES = [
    "Gene",
    "Variant type",
    "Molecular consequence",
    "Original_AA",
    "New_AA",
]
NUMERIC_FEATURES = ["Protein_Position"]
TARGET = "Label"

FORBIDDEN_INPUTS = [
    "Germline classification",
    "Condition(s)",
    "VariationID",
    "Accession",
    "AlleleID(s)",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
RF_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MODEL_PATH = MODELS_DIR / "random_forest_clinvar.pkl"
LABELED_CSV = PROCESSED_DIR / "clinvar_labeled.csv"
LOOKUP_CSV = PROCESSED_DIR / "variant_lookup.csv"
