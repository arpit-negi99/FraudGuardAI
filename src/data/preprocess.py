from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


DEFAULT_EXCLUDED_COLUMNS = ("isFraud", "TransactionID", "TransactionDT")
CATEGORICAL_MISSING_VALUE = "__MISSING__"


class PreprocessingError(ValueError):
    """Raised when preprocessing is used with invalid state or input."""


class FraudDataPreprocessor:
    """Train-fitted, leakage-safe preprocessing for FraudGuard AI tabular data."""

    def __init__(
        self,
        missing_threshold: float = 0.95,
        excluded_columns: tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS,
    ) -> None:
        self.missing_threshold = missing_threshold
        self.excluded_columns = tuple(excluded_columns)
        self.dropped_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.feature_columns: list[str] = []
        self.numeric_medians: dict[str, float] = {}
        self.categorical_encoder: OrdinalEncoder | None = None
        self.final_feature_count: int = 0
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "FraudDataPreprocessor":
        """Fit missingness, medians, categorical vocabulary, and feature order on train only."""
        if X_train.empty:
            raise PreprocessingError("Cannot fit preprocessor on empty training features.")
        if not 0 <= self.missing_threshold < 1:
            raise PreprocessingError("missing_threshold must be in the range [0, 1).")

        train_features = X_train.drop(
            columns=[col for col in self.excluded_columns if col in X_train.columns],
            errors="ignore",
        )
        missing_rates = train_features.isna().mean()
        self.dropped_columns = sorted(missing_rates[missing_rates > self.missing_threshold].index.tolist())
        kept_features = train_features.drop(columns=self.dropped_columns, errors="ignore")

        self.numeric_columns = kept_features.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_columns = [col for col in kept_features.columns if col not in self.numeric_columns]
        self.feature_columns = self.numeric_columns + self.categorical_columns

        numeric_medians = kept_features[self.numeric_columns].median(numeric_only=True)
        self.numeric_medians = {
            col: (0.0 if pd.isna(value) else float(value)) for col, value in numeric_medians.items()
        }

        self.categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )
        if self.categorical_columns:
            categorical_train = self._prepare_categorical_frame(kept_features, self.categorical_columns)
            self.categorical_encoder.fit(categorical_train)

        self.final_feature_count = len(self.feature_columns)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using only state learned during fit."""
        if not self.is_fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform.")

        aligned = self._align_features(X)
        arrays: list[np.ndarray] = []

        if self.numeric_columns:
            numeric_df = aligned[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
            numeric_df = numeric_df.fillna(self.numeric_medians)
            arrays.append(numeric_df.to_numpy(dtype=np.float32, copy=False))

        if self.categorical_columns:
            if self.categorical_encoder is None:
                raise PreprocessingError("Categorical encoder is missing from fitted preprocessor.")
            categorical_df = self._prepare_categorical_frame(aligned, self.categorical_columns)
            arrays.append(self.categorical_encoder.transform(categorical_df))

        if not arrays:
            return np.empty((len(X), 0), dtype=np.float32)
        if len(arrays) == 1:
            return arrays[0].astype(np.float32, copy=False)
        return np.concatenate(arrays, axis=1).astype(np.float32, copy=False)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on training features and return transformed training features."""
        return self.fit(X_train).transform(X_train)

    def get_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable preprocessing metadata."""
        return {
            "dropped_columns": self.dropped_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "final_feature_count": self.final_feature_count,
            "missing_threshold": self.missing_threshold,
            "excluded_columns": list(self.excluded_columns),
            "categorical_encoding": {
                "encoder": "sklearn.preprocessing.OrdinalEncoder",
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
                "missing_value": CATEGORICAL_MISSING_VALUE,
            },
            "numeric_missing_strategy": "median fitted on training data only",
        }

    def save(self, preprocessor_path: str | Path, metadata_path: str | Path | None = None) -> None:
        """Save fitted preprocessor and optional metadata."""
        if not self.is_fitted:
            raise PreprocessingError("Only a fitted preprocessor can be saved.")

        preprocessor_path = Path(preprocessor_path)
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, preprocessor_path)

        if metadata_path is not None:
            save_metadata(self.get_metadata(), metadata_path)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        raw_features = X.drop(
            columns=[col for col in self.excluded_columns if col in X.columns],
            errors="ignore",
        )
        kept_features = raw_features.drop(columns=self.dropped_columns, errors="ignore")
        return kept_features.reindex(columns=self.feature_columns)

    @staticmethod
    def _prepare_categorical_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        categorical_df = df[columns].copy()
        return categorical_df.fillna(CATEGORICAL_MISSING_VALUE).astype(str)


def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """Write preprocessing metadata as indented JSON."""
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def load_preprocessor(path: str | Path) -> FraudDataPreprocessor:
    """Load a fitted FraudDataPreprocessor from disk."""
    preprocessor = joblib.load(path)
    if not isinstance(preprocessor, FraudDataPreprocessor):
        raise PreprocessingError("Loaded object is not a FraudDataPreprocessor.")
    return preprocessor
