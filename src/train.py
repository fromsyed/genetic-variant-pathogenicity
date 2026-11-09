"""Train the baseline Random Forest and save model plus test-set results."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from .config import (
    FEATURES, FORBIDDEN_INPUTS, LABELED_CSV, MODEL_PATH, MODELS_DIR,
    NUMERIC_FEATURES, PROCESSED_DIR, RANDOM_STATE, RESULTS_DIR, TARGET,
    TEST_SIZE,
)
from .model import build_model


def calculate_metrics(y_true, y_pred):
    """Return headline metrics with Pathogenic (label 1) as the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def train_and_evaluate(dataset_path=LABELED_CSV):
    """Train, evaluate, persist the model and write reproducible result files."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. Run preprocessing first."
        )

    model_df = pd.read_csv(dataset_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(model_df.columns))
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {missing}")

    overlap = set(FEATURES) & set(FORBIDDEN_INPUTS)
    if overlap:
        raise ValueError(f"Forbidden columns found in feature list: {sorted(overlap)}")

    X = model_df[FEATURES]
    y = model_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    metrics = {
        "random_forest": calculate_metrics(y_test, predictions),
        "majority_class_baseline": calculate_metrics(y_test, baseline_predictions),
        "dataset_records": int(len(model_df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": FEATURES,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["Actual_Benign", "Actual_Pathogenic"],
        columns=["Predicted_Benign", "Predicted_Pathogenic"],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    prediction_output = model_df.loc[X_test.index, [
        "VariationID", "Gene", "Name", "Germline classification",
        "Protein_Position", "Original_AA", "New_AA", TARGET,
    ]].copy()
    prediction_output["Predicted_Label"] = np.asarray(predictions)
    prediction_output["Predicted_Class"] = prediction_output["Predicted_Label"].map(
        {0: "Benign", 1: "Pathogenic"}
    )
    prediction_output["Actual_Class"] = prediction_output[TARGET].map(
        {0: "Benign", 1: "Pathogenic"}
    )
    prediction_output["Correct"] = prediction_output[TARGET] == prediction_output["Predicted_Label"]
    prediction_output.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    # Verify that the serialized pipeline reproduces its predictions.
    reloaded = joblib.load(MODEL_PATH)
    assert np.array_equal(reloaded.predict(X_test.head(5)), model.predict(X_test.head(5)))
    assert np.allclose(
        reloaded.predict_proba(X_test.head(5)), model.predict_proba(X_test.head(5))
    )

    print("Training completed.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {RESULTS_DIR / 'test_metrics.json'}")
    print(f"Saved confusion matrix: {RESULTS_DIR / 'confusion_matrix.csv'}")
    print("\nRandom Forest metrics:")
    for key, value in metrics["random_forest"].items():
        print(f"  {key}: {value:.4f}")
    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
