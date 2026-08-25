from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.inference.predict import (
    ArtifactLoadError,
    FraudPredictor,
    InferenceInputError,
    load_model_artifact,
    load_preprocessor_artifact,
)


class FakeModel:
    def predict_proba(self, X):
        scores = np.clip(np.asarray(X)[:, 0] / 100.0, 0.0, 1.0)
        return np.column_stack([1.0 - scores, scores])


class FakeExplainer:
    def shap_values(self, X):
        return np.tile(np.array([[0.5, -0.2, 0.1]], dtype=float), (len(X), 1))


class FakePreprocessor:
    is_fitted = True
    feature_columns = ["amount", "category", "optional"]
    numeric_columns = ["amount", "optional"]
    categorical_columns = ["category"]
    final_feature_count = 3

    def __init__(self) -> None:
        self.last_columns = []

    def transform(self, X):
        self.last_columns = list(X.columns)
        amount = pd.to_numeric(X["amount"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
        category = X["category"].fillna("__MISSING__").astype(str)
        category_encoded = np.where(category == "risky", 1.0, 0.0)
        optional = pd.to_numeric(X["optional"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.column_stack([amount, category_encoded, optional]).astype(np.float32)


def make_predictor() -> FraudPredictor:
    return FraudPredictor(
        model=FakeModel(),
        preprocessor=FakePreprocessor(),
        metadata={
            "dropped_columns": [],
            "numeric_columns": ["amount", "optional"],
            "categorical_columns": ["category"],
            "final_feature_count": 3,
        },
        threshold=0.60,
    )


def test_single_transaction_prediction_returns_valid_risk_score() -> None:
    result = make_predictor().predict_transaction({"TransactionID": 1, "amount": 70, "category": "risky"})

    assert result["transaction_id"] == 1
    assert 0.0 <= result["risk_score"] <= 1.0


def test_threshold_decision_logic_uses_review_at_or_above_threshold() -> None:
    predictor = make_predictor()

    review = predictor.predict_transaction({"amount": 60, "category": "a"})
    allow = predictor.predict_transaction({"amount": 59, "category": "a"})

    assert review["decision"] == "REVIEW"
    assert allow["decision"] == "ALLOW"


def test_identifier_time_and_label_are_not_passed_as_model_features() -> None:
    predictor = make_predictor()

    predictor.predict_transaction(
        {
            "TransactionID": 1,
            "TransactionDT": 10,
            "isFraud": 1,
            "amount": 70,
            "category": "a",
        }
    )

    assert "TransactionID" in predictor.preprocessor.last_columns
    assert "TransactionDT" in predictor.preprocessor.last_columns
    assert "isFraud" not in predictor.preprocessor.last_columns


def test_is_fraud_is_not_required_for_inference() -> None:
    result = make_predictor().predict_transaction({"amount": 70, "category": "a"})

    assert result["decision"] == "REVIEW"


def test_unknown_categories_do_not_crash_inference() -> None:
    result = make_predictor().predict_transaction({"amount": 10, "category": "brand_new"})

    assert result["decision"] == "ALLOW"


def test_missing_optional_fields_are_inserted_with_warning() -> None:
    result = make_predictor().predict_transaction({"amount": 10})

    assert any("Inserted missing values" in warning for warning in result["warnings"])


def test_extra_input_fields_are_ignored_with_warning() -> None:
    result = make_predictor().predict_transaction({"amount": 10, "category": "a", "extra": "x"})

    assert any("extra input columns" in warning for warning in result["warnings"])


def test_malformed_numeric_values_are_handled_with_warning() -> None:
    result = make_predictor().predict_transaction({"amount": "bad", "category": "a"})

    assert any("Malformed numeric values" in warning for warning in result["warnings"])
    assert 0.0 <= result["risk_score"] <= 1.0


def test_batch_inference_preserves_row_count_and_required_columns() -> None:
    batch = pd.DataFrame({"TransactionID": [1, 2], "amount": [10, 80], "category": ["a", "risky"]})

    predictions = make_predictor().predict_batch(batch)

    assert len(predictions) == 2
    assert {"risk_score", "decision"}.issubset(predictions.columns)


def test_output_is_json_serializable() -> None:
    result = make_predictor().predict_transaction({"TransactionID": np.int64(1), "amount": 70, "category": "a"})

    json.dumps(result)


def test_shap_explanation_can_be_disabled() -> None:
    result = make_predictor().predict_transaction({"amount": 70, "category": "a"}, include_explanation=False)

    assert "explanation" not in result


def test_shap_failure_does_not_destroy_successful_prediction(monkeypatch) -> None:
    predictor = make_predictor()

    def fail(_model):
        raise RuntimeError("broken shap")

    monkeypatch.setattr("src.inference.predict.create_tree_explainer", fail)

    result = predictor.predict_transaction({"amount": 70, "category": "a"}, include_explanation=True)

    assert result["decision"] == "REVIEW"
    assert any("SHAP explanation unavailable" in warning for warning in result["warnings"])


def test_batch_summary_uses_inference_outputs_only() -> None:
    predictions = make_predictor().predict_batch(
        pd.DataFrame({"amount": [10, 80], "category": ["a", "risky"]})
    )

    summary = make_predictor().summarize_batch(predictions)

    assert summary["transactions_scored"] == 2
    assert summary["review_count"] == 1
    assert summary["allow_count"] == 1


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(InferenceInputError, match="empty"):
        make_predictor().predict_batch(pd.DataFrame())


def test_missing_model_artifact_raises_clear_error(tmp_path) -> None:
    with pytest.raises(ArtifactLoadError, match="Missing XGBoost model artifact"):
        load_model_artifact(tmp_path / "missing.json")


def test_missing_preprocessor_artifact_raises_clear_error(tmp_path) -> None:
    with pytest.raises(ArtifactLoadError, match="Missing preprocessor artifact"):
        load_preprocessor_artifact(tmp_path / "missing.joblib")
