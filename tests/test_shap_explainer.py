from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from src.explainability.shap_explainer import (
    calculate_shap_values,
    explain_transaction,
    format_explanation,
    global_shap_importance,
    top_contributors,
    validate_feature_name_mapping,
)


class FakeExplainer:
    def shap_values(self, X):
        return np.tile(np.array([[0.4, -0.2, 0.1]], dtype=float), (len(X), 1))


class FakeModel:
    def predict_proba(self, X):
        return np.tile(np.array([[0.25, 0.75]], dtype=float), (len(X), 1))


def test_feature_name_mapping_length_matches_transformed_feature_count() -> None:
    validate_feature_name_mapping(["TransactionAmt", "V258", "DeviceType"], 3)


def test_shap_output_length_matches_feature_count() -> None:
    values = calculate_shap_values(FakeExplainer(), np.array([[1.0, 2.0, 3.0]]))

    assert values.shape[1] == 3


def test_top_risk_contributors_are_sorted_descending_by_positive_contribution() -> None:
    risk, _ = top_contributors(
        np.array([0.2, 0.7, -0.5, 0.1]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        ["a", "b", "c", "d"],
        top_n=3,
    )

    assert [item["feature"] for item in risk] == ["b", "a", "d"]


def test_protective_contributors_are_sorted_correctly() -> None:
    _, protective = top_contributors(
        np.array([-0.2, 0.7, -0.5, -0.1]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        ["a", "b", "c", "d"],
        top_n=3,
    )

    assert [item["feature"] for item in protective] == ["c", "a", "d"]


def test_explanation_output_is_json_serializable() -> None:
    explanation = explain_transaction(
        FakeModel(),
        FakeExplainer(),
        np.array([1.0, 2.0, 3.0]),
        ["TransactionAmt", "V258", "DeviceType"],
    )

    json.dumps(explanation)


def test_probability_remains_between_zero_and_one() -> None:
    explanation = explain_transaction(
        FakeModel(),
        FakeExplainer(),
        np.array([1.0, 2.0, 3.0]),
        ["TransactionAmt", "V258", "DeviceType"],
    )

    assert 0 <= explanation["fraud_probability"] <= 1


def test_no_causal_wording_is_generated_by_formatter() -> None:
    explanation = explain_transaction(
        FakeModel(),
        FakeExplainer(),
        np.array([1.0, 2.0, 3.0]),
        ["TransactionAmt", "V258", "DeviceType"],
    )
    formatted = format_explanation(explanation).lower()

    assert "caused fraud" not in formatted
    assert "proved" not in formatted
    assert "contributed to a higher model score" in formatted


def test_generate_explanations_loads_model_rather_than_retraining() -> None:
    tree = ast.parse(Path("scripts/generate_explanations.py").read_text(encoding="utf-8"))
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "fit" not in call_names


def test_held_out_test_set_is_not_evaluated() -> None:
    tree = ast.parse(Path("scripts/generate_explanations.py").read_text(encoding="utf-8"))
    forbidden_names = {"X_test_transformed", "test_probabilities", "test_metrics"}
    observed_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert forbidden_names.isdisjoint(observed_names)


def test_anonymized_feature_names_are_not_given_invented_meanings() -> None:
    explanation = {
        "fraud_probability": 0.75,
        "policy_threshold": 0.5,
        "policy_decision": "REVIEW",
        "top_risk_factors": [{"feature": "V258", "value": 1.0, "shap_value": 0.3}],
        "top_protective_factors": [{"feature": "C13", "value": 2.0, "shap_value": -0.2}],
    }
    formatted = format_explanation(explanation)

    assert "V258" in formatted
    assert "IP" not in formatted
    assert "device fingerprint" not in formatted.lower()


def test_global_importance_returns_ranked_feature_names() -> None:
    table = global_shap_importance(
        np.array([[1.0, -2.0], [3.0, 0.0]]),
        ["TransactionAmt", "V258"],
    )

    assert table.iloc[0]["feature"] == "TransactionAmt"
    assert table.iloc[0]["rank"] == 1
