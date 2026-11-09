"""Scikit-learn preprocessing and Random Forest model construction."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RF_PARAMS


def build_preprocessor(categorical=None, numeric=None):
    """One-hot encode categorical features and standardize numeric features."""
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    numeric = NUMERIC_FEATURES if numeric is None else numeric

    transformers = [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
    if numeric:
        transformers.append(("numerical", StandardScaler(), numeric))
    return ColumnTransformer(transformers=transformers)


def build_model(categorical=None, numeric=None):
    """Return preprocessing and Random Forest as one leakage-safe pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(categorical, numeric)),
            ("classifier", RandomForestClassifier(**RF_PARAMS)),
        ]
    )
