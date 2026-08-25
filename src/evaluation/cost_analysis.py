from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_binary_classifier


class CostAnalysisError(ValueError):
    """Raised when business-cost analysis inputs are invalid."""


def validate_cost_assumptions(
    fraud_loss_multiplier: float,
    review_cost_per_transaction: float,
) -> None:
    """Reject negative scenario assumptions."""
    if fraud_loss_multiplier < 0:
        raise CostAnalysisError("fraud_loss_multiplier must be non-negative.")
    if review_cost_per_transaction < 0:
        raise CostAnalysisError("review_cost_per_transaction must be non-negative.")


def prepare_transaction_amounts(amounts: Iterable[object], expected_length: int) -> np.ndarray:
    """Validate and return transaction amounts aligned with labels/probabilities."""
    amount_series = pd.to_numeric(pd.Series(amounts), errors="coerce")
    invalid_count = int(amount_series.isna().sum())
    negative_count = int((amount_series < 0).sum())
    if len(amount_series) != expected_length:
        raise CostAnalysisError(
            f"TransactionAmt length {len(amount_series)} does not match expected {expected_length}."
        )
    if invalid_count or negative_count:
        affected = invalid_count + negative_count
        raise CostAnalysisError(
            "TransactionAmt contains invalid, missing, or negative values; "
            f"affected transactions: {affected}."
        )
    return amount_series.to_numpy(dtype=float)


def calculate_transaction_costs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transaction_amounts: np.ndarray,
    fraud_loss_multiplier: float,
    review_cost_per_transaction: float,
) -> dict[str, float]:
    """Calculate threshold-specific modeled costs and fraud amount exposure."""
    validate_cost_assumptions(fraud_loss_multiplier, review_cost_per_transaction)
    labels = np.asarray(y_true).astype(int)
    predictions = np.asarray(y_pred).astype(int)
    amounts = prepare_transaction_amounts(transaction_amounts, expected_length=len(labels))

    if len(predictions) != len(labels):
        raise CostAnalysisError(
            f"Prediction length {len(predictions)} does not match label length {len(labels)}."
        )

    false_positive_mask = (labels == 0) & (predictions == 1)
    false_negative_mask = (labels == 1) & (predictions == 0)
    true_positive_mask = (labels == 1) & (predictions == 1)

    fraud_amount_detected = float(amounts[true_positive_mask].sum())
    fraud_amount_missed = float(amounts[false_negative_mask].sum())
    false_positive_count = int(false_positive_mask.sum())
    missed_fraud_cost = fraud_amount_missed * fraud_loss_multiplier
    false_positive_cost = false_positive_count * review_cost_per_transaction

    return {
        "fraud_amount_detected": fraud_amount_detected,
        "fraud_amount_missed": fraud_amount_missed,
        "missed_fraud_cost": float(missed_fraud_cost),
        "false_positive_cost": float(false_positive_cost),
        "total_estimated_cost": float(missed_fraud_cost + false_positive_cost),
    }


def evaluate_cost_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    transaction_amounts: Iterable[object],
    thresholds: Iterable[float],
    scenario_name: str,
    fraud_loss_multiplier: float,
    review_cost_per_transaction: float,
) -> pd.DataFrame:
    """Evaluate threshold metrics and scenario costs for one configured scenario."""
    validate_cost_assumptions(fraud_loss_multiplier, review_cost_per_transaction)
    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob, dtype=float)
    amounts = prepare_transaction_amounts(transaction_amounts, expected_length=len(labels))
    if len(probabilities) != len(labels):
        raise CostAnalysisError(
            f"Probability length {len(probabilities)} does not match label length {len(labels)}."
        )

    rows = []
    for threshold in thresholds:
        predictions = (probabilities >= float(threshold)).astype(int)
        metrics = evaluate_binary_classifier(labels, probabilities, threshold=float(threshold))
        costs = calculate_transaction_costs(
            labels,
            predictions,
            amounts,
            fraud_loss_multiplier=fraud_loss_multiplier,
            review_cost_per_transaction=review_cost_per_transaction,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "true_negative": metrics["true_negative"],
                "false_negative": metrics["false_negative"],
                "review_rate": metrics["review_rate"],
                "fraud_amount_detected": costs["fraud_amount_detected"],
                "fraud_amount_missed": costs["fraud_amount_missed"],
                "missed_fraud_cost": costs["missed_fraud_cost"],
                "false_positive_cost": costs["false_positive_cost"],
                "total_estimated_cost": costs["total_estimated_cost"],
                "fraud_loss_multiplier": fraud_loss_multiplier,
                "review_cost_per_transaction": review_cost_per_transaction,
            }
        )
    return pd.DataFrame(rows)


def calculate_allow_all_cost(
    y_true: np.ndarray,
    transaction_amounts: Iterable[object],
    fraud_loss_multiplier: float,
    review_cost_per_transaction: float,
    scenario_name: str = "allow_all",
) -> dict[str, float | int | str]:
    """Calculate the policy cost when every validation transaction is allowed."""
    labels = np.asarray(y_true).astype(int)
    predictions = np.zeros_like(labels)
    costs = calculate_transaction_costs(
        labels,
        predictions,
        prepare_transaction_amounts(transaction_amounts, expected_length=len(labels)),
        fraud_loss_multiplier=fraud_loss_multiplier,
        review_cost_per_transaction=review_cost_per_transaction,
    )
    return {
        "scenario": scenario_name,
        "threshold": None,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": int((labels == 0).sum()),
        "false_negative": int((labels == 1).sum()),
        "review_rate": 0.0,
        **costs,
    }


def find_minimum_cost_threshold(cost_table: pd.DataFrame) -> dict[str, float | int | str]:
    """Return the lowest-cost threshold row, breaking ties by higher recall then lower review rate."""
    return _row_to_dict(_sort_cost_candidates(cost_table).iloc[0])


def find_constrained_cost_threshold(
    cost_table: pd.DataFrame,
    min_recall: float | None = None,
    max_review_rate: float | None = None,
) -> dict[str, float | int | str] | None:
    """Return lowest-cost threshold row satisfying optional recall/review constraints."""
    candidates = cost_table
    if min_recall is not None:
        candidates = candidates[candidates["recall"] >= min_recall]
    if max_review_rate is not None:
        candidates = candidates[candidates["review_rate"] <= max_review_rate]
    if candidates.empty:
        return None
    return _row_to_dict(_sort_cost_candidates(candidates).iloc[0])


def _sort_cost_candidates(cost_table: pd.DataFrame) -> pd.DataFrame:
    return cost_table.sort_values(
        ["total_estimated_cost", "recall", "review_rate", "threshold"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )


def _row_to_dict(row: pd.Series) -> dict[str, float | int | str]:
    integer_fields = {"true_positive", "false_positive", "true_negative", "false_negative"}
    result: dict[str, float | int | str] = {}
    for key, value in row.to_dict().items():
        if key in integer_fields:
            result[key] = int(value)
        elif isinstance(value, (float, int, np.floating, np.integer)):
            result[key] = float(value)
        else:
            result[key] = value
    return result
