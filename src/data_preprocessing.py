"""ClinVar loading, binary labeling, protein-feature extraction, and validation."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import (
    LABELED_CSV,
    LOOKUP_CSV,
    PROCESSED_DIR,
    RAW_DIR,
    RAW_FILENAME,
    TARGET_GENES,
)

EXPECTED_COLUMNS = [
    "Name", "Gene(s)", "Protein change", "Condition(s)", "Accession",
    "GRCh37Chromosome", "GRCh37Location", "GRCh38Chromosome", "GRCh38Location",
    "VariationID", "AlleleID(s)", "dbSNP ID", "Canonical SPDI",
    "Variant type", "Molecular consequence", "Germline classification",
    "Germline date last evaluated", "Germline review status",
    "Somatic clinical impact", "Somatic clinical impact date last evaluated",
    "Somatic clinical impact review status", "Oncogenicity classification",
    "Oncogenicity date last evaluated", "Oncogenicity review status",
]

LABEL_MAP = {
    "Pathogenic": 1,
    "Likely pathogenic": 1,
    "Pathogenic/Likely pathogenic": 1,
    "Benign": 0,
    "Likely benign": 0,
    "Benign/Likely benign": 0,
}

AA_3_TO_1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
SUBSTITUTION_PATTERN = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

LOOKUP_COLUMNS = [
    "VariationID", "Accession", "Gene", "Name", "Protein change",
    "Germline classification", "Label", "Variant type", "Molecular consequence",
    "Protein_Position", "Original_AA", "New_AA",
]


def load_raw_clinvar(raw_path: str | Path | None = None) -> pd.DataFrame:
    """Load the expected columns from a tab-separated ClinVar export."""
    path = Path(raw_path) if raw_path else RAW_DIR / RAW_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"ClinVar TSV not found: {path}\n"
            "Place the downloaded file in data/raw/ or pass raw_path explicitly."
        )

    header = pd.read_csv(path, sep="\t", nrows=0).columns
    missing = sorted(set(EXPECTED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"ClinVar file is missing expected columns: {missing}")

    return pd.read_csv(
        path, sep="\t", usecols=EXPECTED_COLUMNS, low_memory=False
    )


def extract_target_gene(gene_field):
    """Accept a record only when it names exactly one selected target gene."""
    if pd.isna(gene_field) or not str(gene_field).strip():
        return None, "Gene(s) is missing"
    genes = [g.strip() for g in str(gene_field).split("|") if g.strip()]
    selected = sorted({g for g in genes if g in TARGET_GENES})
    if len(selected) == 1:
        return selected[0], None
    if len(selected) == 0:
        return None, "No selected target gene in Gene(s)"
    return None, f"Multiple selected target genes in Gene(s) ({len(selected)})"


def parse_single_substitution(name):
    """Parse exactly one protein substitution from a ClinVar Name field."""
    if not isinstance(name, str):
        return None, None, None, "Name is missing"
    matches = SUBSTITUTION_PATTERN.findall(name)
    if len(matches) == 0:
        return None, None, None, "Zero amino acid substitutions found in Name"
    if len(matches) > 1:
        return None, None, None, f"Multiple amino acid substitutions found in Name ({len(matches)})"

    original_code, position_text, new_code = matches[0]
    original_aa = AA_3_TO_1.get(original_code)
    new_aa = AA_3_TO_1.get(new_code)
    if original_aa is None or new_aa is None:
        return None, None, None, "Unrecognised amino acid code"
    return int(position_text), original_aa, new_aa, None


def prepare_dataset(raw_df: pd.DataFrame):
    """
    Convert raw ClinVar rows into the final binary-labeled modelling dataframe.

    Returns:
        model_df, funnel_df, label_audit
    """
    missing = sorted(set(EXPECTED_COLUMNS) - set(raw_df.columns))
    if missing:
        raise ValueError(f"Input dataframe is missing expected columns: {missing}")

    classification_label = raw_df["Germline classification"].map(LABEL_MAP)
    is_labelled = classification_label.notna()

    label_audit = (
        raw_df.assign(Included=is_labelled)
        .assign(Classification=raw_df["Germline classification"].fillna("(missing)"))
        .groupby("Classification")
        .agg(Records=("VariationID", "size"), Included=("Included", "sum"))
    )
    label_audit["Excluded"] = label_audit["Records"] - label_audit["Included"]

    labelled_df = raw_df.loc[is_labelled].copy()
    labelled_df["Label"] = classification_label.loc[is_labelled].astype(int)

    gene_values = labelled_df["Gene(s)"].apply(extract_target_gene).tolist()
    gene_df = pd.DataFrame(
        gene_values, index=labelled_df.index, columns=["Gene", "Gene_Issue"]
    )
    labelled_df[gene_df.columns] = gene_df

    parsed_values = labelled_df["Name"].apply(parse_single_substitution).tolist()
    parsed_df = pd.DataFrame(
        parsed_values,
        index=labelled_df.index,
        columns=["Protein_Position", "Original_AA", "New_AA", "Parse_Issue"],
    )
    labelled_df[parsed_df.columns] = parsed_df

    accepted = labelled_df["Gene_Issue"].isna() & labelled_df["Parse_Issue"].isna()
    model_df = labelled_df.loc[accepted].copy()
    model_df["Protein_Position"] = model_df["Protein_Position"].astype("int64")
    model_df = model_df.drop(columns=["Gene_Issue", "Parse_Issue"])

    gene_excluded = int(labelled_df["Gene_Issue"].notna().sum())
    protein_excluded = int(
        (labelled_df["Gene_Issue"].isna() & labelled_df["Parse_Issue"].notna()).sum()
    )
    funnel_df = pd.DataFrame(
        [
            ("Raw ClinVar records", len(raw_df)),
            ("Excluded: unsupported or missing classification", int((~is_labelled).sum())),
            ("Excluded: missing, unrecognised, or multiple selected genes", gene_excluded),
            ("Excluded: zero or multiple amino acid substitutions in Name", protein_excluded),
            ("Final modelling records", len(model_df)),
        ],
        columns=["Step", "Records"],
    )

    validate_dataset(raw_df, labelled_df, model_df, funnel_df)
    return model_df, funnel_df, label_audit


def validate_dataset(raw_df, labelled_df, model_df, funnel_df):
    """Run the same key integrity checks used in the research notebook."""
    assert model_df["VariationID"].is_unique, "Duplicate VariationID in modelling data"
    assert set(model_df["Label"].unique()) <= {0, 1}, "Unexpected label values"
    assert set(model_df["Gene"].unique()) <= set(TARGET_GENES), "Unexpected gene name"
    required = ["Gene", "Protein_Position", "Original_AA", "New_AA"]
    assert model_df[required].notna().all().all(), "Missing model feature values"
    assert model_df["Original_AA"].isin(VALID_AA).all(), "Invalid original amino acid"
    assert model_df["New_AA"].isin(VALID_AA).all(), "Invalid substituted amino acid"
    assert (model_df["Protein_Position"] > 0).all(), "Protein positions must be positive"

    selected_gene_counts = model_df["Gene(s)"].apply(
        lambda s: len({g.strip() for g in str(s).split("|") if g.strip() in TARGET_GENES})
    )
    assert (selected_gene_counts == 1).all(), "A record does not name exactly one selected gene"

    substitution_counts = model_df["Name"].apply(
        lambda s: len(SUBSTITUTION_PATTERN.findall(str(s)))
    )
    assert (substitution_counts == 1).all(), "A record does not have exactly one amino acid substitution"

    assert funnel_df.iloc[0]["Records"] == len(raw_df)
    assert funnel_df.iloc[-1]["Records"] == len(model_df)
    assert funnel_df["Records"].iloc[:-1].sum() >= len(model_df)


def save_processed_data(model_df: pd.DataFrame):
    """Save the labelled modelling data and the app's unique-VariationID lookup."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(LABELED_CSV, index=False)

    missing = sorted(set(LOOKUP_COLUMNS) - set(model_df.columns))
    if missing:
        raise ValueError(f"Cannot create lookup table; missing columns: {missing}")

    variant_lookup = model_df[LOOKUP_COLUMNS].drop_duplicates(subset=["VariationID"])
    variant_lookup.to_csv(LOOKUP_CSV, index=False)
    return LABELED_CSV, LOOKUP_CSV
