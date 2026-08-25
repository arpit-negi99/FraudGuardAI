from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


NON_CAUSAL_FORBIDDEN_TERMS = ("caused fraud", "proved", "guaranteed")


class ShapExplanationError(ValueError):
    """Raised when SHAP explanation inputs are inconsistent."""


def get_feature_names(preprocessor: Any) -> list[str]:
    """Return transformed feature names from the fitted FraudGuard preprocessor."""
    feature_names = list(getattr(preprocessor, "feature_columns", []))
    final_feature_count = getattr(preprocessor, "final_feature_count", None)
    if not feature_names:
        raise ShapExplanationError("Preprocessor does not expose feature_columns.")
    if final_feature_count is not None and len(feature_names) != final_feature_count:
        raise ShapExplanationError(
            "Feature-name count does not match preprocessor final_feature_count: "
            f"{len(feature_names)} != {final_feature_count}."
        )
    return feature_names


def validate_feature_name_mapping(feature_names: list[str], transformed_feature_count: int) -> None:
    """Validate one readable feature name per transformed feature."""
    if len(feature_names) != transformed_feature_count:
        raise ShapExplanationError(
            f"Feature-name count {len(feature_names)} does not match transformed "
            f"feature count {transformed_feature_count}."
        )


def create_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the trained XGBoost model."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    return shap.TreeExplainer(booster)


def calculate_shap_values(explainer: shap.TreeExplainer, X: np.ndarray) -> np.ndarray:
    """Calculate SHAP values and normalize the output to a 2D array."""
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim != 2:
        raise ShapExplanationError(f"Expected 2D SHAP values, got shape {values.shape}.")
    return values


def top_contributors(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    """Return sorted positive and negative SHAP contributors for one transaction."""
    values = np.asarray(shap_values, dtype=float)
    features = np.asarray(feature_values, dtype=float)
    if len(values) != len(feature_names) or len(features) != len(feature_names):
        raise ShapExplanationError("SHAP values, feature values, and feature names must align.")

    positive_indices = np.where(values > 0)[0]
    negative_indices = np.where(values < 0)[0]
    risk_indices = positive_indices[np.argsort(values[positive_indices])[::-1]][:top_n]
    protective_indices = negative_indices[np.argsort(values[negative_indices])][:top_n]

    return (
        [_contributor(index, values, features, feature_names) for index in risk_indices],
        [_contributor(index, values, features, feature_names) for index in protective_indices],
    )


def explain_transaction(
    model: Any,
    explainer: shap.TreeExplainer,
    transformed_transaction: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.5,
    top_n: int = 5,
) -> dict[str, Any]:
    """Explain one transformed transaction with top positive and negative SHAP contributors."""
    transaction = np.asarray(transformed_transaction, dtype=float)
    if transaction.ndim == 1:
        transaction = transaction.reshape(1, -1)
    validate_feature_name_mapping(feature_names, transaction.shape[1])

    fraud_probability = float(model.predict_proba(transaction)[0, 1])
    shap_values = calculate_shap_values(explainer, transaction)[0]
    top_risk_factors, top_protective_factors = top_contributors(
        shap_values,
        transaction[0],
        feature_names,
        top_n=top_n,
    )

    return {
        "fraud_probability": fraud_probability,
        "policy_threshold": float(threshold),
        "policy_decision": "REVIEW" if fraud_probability >= threshold else "ALLOW",
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,
    }


def format_explanation(explanation: dict[str, Any]) -> str:
    """Create a non-causal human-readable explanation."""
    lines = [
        f"Fraud risk score: {explanation['fraud_probability']:.1%}",
        f"Policy threshold: {explanation['policy_threshold']:.2f}",
        f"Policy decision: {explanation['policy_decision']}",
        "",
        "Main factors that contributed to a higher model score:",
    ]
    lines.extend(f"- {item['feature']}" for item in explanation["top_risk_factors"])
    lines.append("")
    lines.append("Factors that contributed to a lower model score:")
    lines.extend(f"- {item['feature']}" for item in explanation["top_protective_factors"])
    text = "\n".join(lines)
    lowered = text.lower()
    if any(term in lowered for term in NON_CAUSAL_FORBIDDEN_TERMS):
        raise ShapExplanationError("Generated explanation contains causal or overclaiming wording.")
    return text


def global_shap_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Return mean absolute SHAP importance by transformed feature."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2:
        raise ShapExplanationError("Global SHAP importance expects a 2D SHAP value matrix.")
    validate_feature_name_mapping(feature_names, values.shape[1])
    importance = np.abs(values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap_value": importance,
        }
    ).sort_values("mean_absolute_shap_value", ascending=False, kind="mergesort")
    table["rank"] = np.arange(1, len(table) + 1)
    return table[["feature", "mean_absolute_shap_value", "rank"]].reset_index(drop=True)


def _contributor(
    index: int,
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float | str]:
    return {
        "feature": feature_names[index],
        "value": float(feature_values[index]),
        "shap_value": float(shap_values[index]),
    }
