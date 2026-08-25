from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load import read_config
from src.data.preprocess import FraudDataPreprocessor, load_preprocessor
from src.explainability.shap_explainer import (
    create_tree_explainer,
    explain_transaction,
    get_feature_names,
)
from src.models.xgboost_model import load_xgboost_model, predict_fraud_probabilities


DEFAULT_MODEL_PATH = Path("artifacts/models/xgboost_model.json")
DEFAULT_PREPROCESSOR_PATH = Path("artifacts/preprocessors/preprocessor.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/preprocessors/preprocessing_metadata.json")
DEFAULT_CONFIG_PATH = Path("configs/config.yaml")
DEFAULT_THRESHOLD = 0.60
METADATA_COLUMNS = ("TransactionID", "TransactionDT")
TARGET_COLUMN = "isFraud"


class InferenceError(ValueError):
    """Raised when FraudGuard inference cannot score a transaction."""


class ArtifactLoadError(InferenceError):
    """Raised when required frozen inference artifacts are missing or invalid."""


class InferenceInputError(InferenceError):
    """Raised when inference input is invalid."""


class FraudPredictor:
    """Reusable frozen inference pipeline for single and batch fraud-risk scoring."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        threshold: float | None = None,
        model: Any | None = None,
        preprocessor: FraudDataPreprocessor | Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)
        self.model = model if model is not None else load_model_artifact(self.model_path)
        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else load_preprocessor_artifact(self.preprocessor_path)
        )
        self.metadata = (
            metadata
            if metadata is not None
            else load_preprocessing_metadata(self.metadata_path)
        )
        self.threshold = float(threshold if threshold is not None else load_default_threshold(config_path))
        _validate_threshold(self.threshold)
        self._explainer = None

    def predict_transaction(
        self,
        transaction: dict[str, Any] | pd.Series | pd.DataFrame,
        threshold: float | None = None,
        include_explanation: bool = False,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Score one transaction and return a JSON-serializable result."""
        raw_df = _coerce_single_transaction(transaction)
        result = self._predict_dataframe(raw_df, threshold=threshold, include_explanations=False)[0]
        if include_explanation:
            try:
                scoring_df, _ = self._prepare_input(raw_df)
                transformed = self.preprocessor.transform(scoring_df)
                result["explanation"] = self._explain_transformed_row(
                    transformed[0],
                    threshold=float(result["threshold"]),
                    top_n=top_n,
                )
            except Exception as exc:  # pragma: no cover - exact SHAP failure varies by runtime
                result["explanation"] = {
                    "top_risk_factors": [],
                    "top_protective_factors": [],
                }
                result["warnings"].append(f"SHAP explanation unavailable: {exc}")
        return _json_safe(result)

    def predict_batch(
        self,
        transactions: pd.DataFrame,
        threshold: float | None = None,
        include_explanations: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Score a batch of transactions without requiring labels."""
        if include_explanations:
            rows = self._predict_dataframe(
                transactions,
                threshold=threshold,
                include_explanations=True,
                top_n=top_n,
            )
            return pd.DataFrame(rows)
        rows = self._predict_dataframe(transactions, threshold=threshold, include_explanations=False)
        return pd.DataFrame(rows)

    def summarize_batch(self, predictions: pd.DataFrame | list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize inference outputs without using labels or evaluation metrics."""
        prediction_df = pd.DataFrame(predictions)
        if prediction_df.empty:
            raise InferenceInputError("Cannot summarize empty prediction results.")
        if "risk_score" not in prediction_df.columns or "decision" not in prediction_df.columns:
            raise InferenceInputError("Predictions must contain risk_score and decision columns.")

        scores = pd.to_numeric(prediction_df["risk_score"], errors="coerce")
        review_count = int((prediction_df["decision"] == "REVIEW").sum())
        allow_count = int((prediction_df["decision"] == "ALLOW").sum())
        total = int(len(prediction_df))
        return {
            "transactions_scored": total,
            "review_count": review_count,
            "allow_count": allow_count,
            "review_rate": float(review_count / total),
            "average_risk_score": float(scores.mean()),
            "median_risk_score": float(scores.median()),
            "maximum_risk_score": float(scores.max()),
        }

    def _predict_dataframe(
        self,
        transactions: pd.DataFrame,
        threshold: float | None = None,
        include_explanations: bool = False,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        threshold_value = float(self.threshold if threshold is None else threshold)
        _validate_threshold(threshold_value)
        raw_df = _coerce_batch(transactions)
        scoring_df, warnings = self._prepare_input(raw_df)
        transformed = self.preprocessor.transform(scoring_df)
        scores = predict_fraud_probabilities(self.model, transformed)
        scores = np.asarray(scores, dtype=float)
        if len(scores) != len(raw_df):
            raise InferenceError(
                f"Model returned {len(scores)} scores for {len(raw_df)} input rows."
            )

        rows: list[dict[str, Any]] = []
        for index, score in enumerate(scores):
            risk_score = float(np.clip(score, 0.0, 1.0))
            row_warnings = list(warnings)
            result = {
                "transaction_id": _maybe_python_scalar(raw_df.iloc[index].get("TransactionID")),
                "risk_score": risk_score,
                "threshold": threshold_value,
                "decision": "REVIEW" if risk_score >= threshold_value else "ALLOW",
                "risk_band": _risk_band(risk_score, threshold_value),
                "warnings": row_warnings,
            }
            if include_explanations:
                try:
                    result["explanation"] = self._explain_transformed_row(
                        transformed[index],
                        threshold=threshold_value,
                        top_n=top_n,
                    )
                except Exception as exc:  # pragma: no cover - exact SHAP failure varies by runtime
                    result["explanation"] = {
                        "top_risk_factors": [],
                        "top_protective_factors": [],
                    }
                    result["warnings"].append(f"SHAP explanation unavailable: {exc}")
            rows.append(_json_safe(result))
        return rows

    def _prepare_input(self, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        expected_features = list(getattr(self.preprocessor, "feature_columns", []))
        if not expected_features:
            raise ArtifactLoadError("Preprocessor does not expose fitted feature_columns.")

        warnings: list[str] = []
        missing_features = [col for col in expected_features if col not in raw_df.columns]
        if missing_features:
            warnings.append(
                f"Inserted missing values for {len(missing_features)} absent expected feature columns."
            )

        allowed_columns = set(expected_features) | set(METADATA_COLUMNS) | {TARGET_COLUMN}
        extra_columns = [col for col in raw_df.columns if col not in allowed_columns]
        if extra_columns:
            warnings.append(f"Ignored {len(extra_columns)} extra input columns not used by the model.")

        scoring_columns = [column for column in METADATA_COLUMNS if column in raw_df.columns]
        scoring_columns.extend(expected_features)
        scoring_df = raw_df.reindex(columns=scoring_columns).copy()

        malformed_numeric = []
        for column in getattr(self.preprocessor, "numeric_columns", []):
            if column in raw_df.columns:
                values = raw_df[column]
                coerced = pd.to_numeric(values, errors="coerce")
                malformed = values.notna() & coerced.isna()
                if malformed.any():
                    malformed_numeric.append(column)
        if malformed_numeric:
            warnings.append(
                "Malformed numeric values were coerced to missing for columns: "
                + ", ".join(malformed_numeric[:10])
                + ("..." if len(malformed_numeric) > 10 else "")
            )

        if TARGET_COLUMN in raw_df.columns:
            warnings.append("Ignored isFraud label column during inference.")
        return scoring_df, warnings

    def _explain_transformed_row(
        self,
        transformed_row: np.ndarray,
        threshold: float,
        top_n: int,
    ) -> dict[str, Any]:
        if self._explainer is None:
            self._explainer = create_tree_explainer(self.model)
        explanation = explain_transaction(
            self.model,
            self._explainer,
            transformed_row,
            get_feature_names(self.preprocessor),
            threshold=threshold,
            top_n=top_n,
        )
        return {
            "top_risk_factors": explanation["top_risk_factors"],
            "top_protective_factors": explanation["top_protective_factors"],
        }


def predict_transaction(
    transaction: dict[str, Any] | pd.Series | pd.DataFrame,
    threshold: float | None = None,
    include_explanation: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper that loads frozen artifacts and scores one transaction."""
    predictor = FraudPredictor(threshold=threshold)
    return predictor.predict_transaction(transaction, include_explanation=include_explanation)


def predict_batch(
    transactions: pd.DataFrame,
    threshold: float | None = None,
    include_explanations: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper that loads frozen artifacts and scores a transaction batch."""
    predictor = FraudPredictor(threshold=threshold)
    return predictor.predict_batch(transactions, include_explanations=include_explanations)


def load_model_artifact(path: str | Path) -> Any:
    """Load the frozen XGBoost model or raise a clear artifact error."""
    model_path = Path(path)
    if not model_path.exists():
        raise ArtifactLoadError(f"Missing XGBoost model artifact: {model_path}")
    try:
        return load_xgboost_model(model_path)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not load XGBoost model artifact {model_path}: {exc}") from exc


def load_preprocessor_artifact(path: str | Path) -> FraudDataPreprocessor:
    """Load the frozen fitted preprocessor or raise a clear artifact error."""
    preprocessor_path = Path(path)
    if not preprocessor_path.exists():
        raise ArtifactLoadError(f"Missing preprocessor artifact: {preprocessor_path}")
    try:
        preprocessor = load_preprocessor(preprocessor_path)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Could not load preprocessor artifact {preprocessor_path}: {exc}"
        ) from exc
    if not getattr(preprocessor, "is_fitted", False):
        raise ArtifactLoadError("Loaded preprocessor is not fitted.")
    return preprocessor


def load_preprocessing_metadata(path: str | Path) -> dict[str, Any]:
    """Load preprocessing metadata required by inference."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise ArtifactLoadError(f"Missing preprocessing metadata artifact: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception as exc:
        raise ArtifactLoadError(
            f"Could not load preprocessing metadata artifact {metadata_path}: {exc}"
        ) from exc
    required = {"dropped_columns", "numeric_columns", "categorical_columns", "final_feature_count"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ArtifactLoadError(f"Preprocessing metadata is missing required keys: {missing}")
    return metadata


def load_default_threshold(config_path: str | Path = DEFAULT_CONFIG_PATH) -> float:
    """Read the demo/default inference threshold from config when available."""
    try:
        config = read_config(config_path)
        return float(config.get("inference", {}).get("default_threshold", DEFAULT_THRESHOLD))
    except FileNotFoundError:
        return DEFAULT_THRESHOLD


def _coerce_single_transaction(transaction: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(transaction, dict):
        return pd.DataFrame([transaction])
    if isinstance(transaction, pd.Series):
        return transaction.to_frame().T
    if isinstance(transaction, pd.DataFrame):
        if len(transaction) != 1:
            raise InferenceInputError("Single transaction input must contain exactly one row.")
        return transaction.copy()
    raise InferenceInputError(
        "Single transaction input must be a dict, pandas Series, or one-row DataFrame."
    )


def _coerce_batch(transactions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(transactions, pd.DataFrame):
        raise InferenceInputError("Batch inference input must be a pandas DataFrame.")
    if transactions.empty:
        raise InferenceInputError("Batch inference input is empty.")
    return transactions.copy()


def _validate_threshold(threshold: float) -> None:
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise InferenceInputError("Threshold must be a finite value between 0.0 and 1.0.")


def _risk_band(risk_score: float, threshold: float) -> str:
    if risk_score >= threshold:
        return "HIGH"
    if risk_score < 0.30:
        return "LOW"
    return "MEDIUM"


def _maybe_python_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value
