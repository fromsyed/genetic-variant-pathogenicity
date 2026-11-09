"""Additional validation: protein-grouped CV and leave-one-gene-out testing."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .config import (
    CATEGORICAL_FEATURES, FEATURES, LABELED_CSV, NUMERIC_FEATURES,
    RANDOM_STATE, RESULTS_DIR, TARGET,
)
from .model import build_model
from .train import calculate_metrics


def run_additional_evaluations(dataset_path=LABELED_CSV):
    """Run grouped validation experiments and save their fold-level results."""
    dataset_path = __import__("pathlib").Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. Run preprocessing first."
        )

    df = pd.read_csv(dataset_path)
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Protein-change-grouped cross-validation: the same gene-position-AA-change
    # combination cannot appear in both train and validation portions of a fold.
    group_columns = ["Gene", "Protein_Position", "Original_AA", "New_AA"]
    protein_groups = df[group_columns].astype(str).agg("|".join, axis=1)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    protein_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=protein_groups), start=1
    ):
        train_groups = set(protein_groups.iloc[train_idx])
        test_groups = set(protein_groups.iloc[test_idx])
        if not train_groups.isdisjoint(test_groups):
            raise AssertionError(f"Protein-group overlap detected in fold {fold}")

        model = build_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        protein_rows.append({
            "fold": fold,
            "test_records": len(test_idx),
            **calculate_metrics(y.iloc[test_idx], pred),
        })

    protein_results = pd.DataFrame(protein_rows)
    protein_results.to_csv(RESULTS_DIR / "protein_grouped_cv.csv", index=False)

    protein_summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]:
        protein_summary[metric] = {
            "mean": float(protein_results[metric].mean()),
            "std": float(protein_results[metric].std(ddof=1)),
        }

    # Leave-one-gene-out: train on five genes and test on the sixth.
    logo = LeaveOneGroupOut()
    gene_rows = []
    for train_idx, test_idx in logo.split(X, y, groups=df["Gene"]):
        held_out_gene = df.iloc[test_idx]["Gene"].iloc[0]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = build_model()
        model.fit(X.iloc[train_idx], y_train)
        pred = model.predict(X.iloc[test_idx])

        majority_label = int(y_train.mode().iloc[0])
        majority_pred = np.full(len(test_idx), majority_label)

        gene_rows.append({
            "held_out_gene": held_out_gene,
            "test_records": len(test_idx),
            "benign_records": int((y_test == 0).sum()),
            "pathogenic_records": int((y_test == 1).sum()),
            "rf_accuracy": float(accuracy_score(y_test, pred)),
            "majority_accuracy": float(accuracy_score(y_test, majority_pred)),
            "rf_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        })

    gene_results = pd.DataFrame(gene_rows)
    gene_results.to_csv(RESULTS_DIR / "leave_one_gene_out.csv", index=False)

    gene_summary = {
        "mean_rf_accuracy": float(gene_results["rf_accuracy"].mean()),
        "mean_majority_accuracy": float(gene_results["majority_accuracy"].mean()),
        "mean_rf_balanced_accuracy": float(gene_results["rf_balanced_accuracy"].mean()),
    }

    summary = {
        "protein_grouped_cv": protein_summary,
        "leave_one_gene_out": gene_summary,
        "interpretation_note": (
            "Leave-one-gene-out evaluates transfer to a gene absent from training. "
            "Do not claim generalization to unseen genes if performance is below baseline."
        ),
    }
    with open(RESULTS_DIR / "additional_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Protein-grouped CV:")
    print(protein_results.round(4).to_string(index=False))
    print("\nLeave-one-gene-out:")
    print(gene_results.round(4).to_string(index=False))
    print(f"\nSaved additional validation results in {RESULTS_DIR}")
    return protein_results, gene_results, summary


if __name__ == "__main__":
    run_additional_evaluations()
