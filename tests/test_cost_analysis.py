from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.cost_analysis import (
    CostAnalysisError,
    calculate_allow_all_cost,
    calculate_transaction_costs,
    evaluate_cost_thresholds,
    find_constrained_cost_threshold,
    find_minimum_cost_threshold,
    prepare_transaction_amounts,
)


def test_false_negative_cost_calculation() -> None:
    costs = calculate_transaction_costs(
        np.array([1, 1]),
        np.array([1, 0]),
        np.array([100.0, 50.0]),
        fraud_loss_multiplier=2.0,
        review_cost_per_transaction=5.0,
    )

    assert costs["fraud_amount_missed"] == 50.0
    assert costs["missed_fraud_cost"] == 100.0


def test_false_positive_cost_calculation() -> None:
    costs = calculate_transaction_costs(
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
        np.array([10.0, 20.0, 30.0]),
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=7.0,
    )

    assert costs["false_positive_cost"] == 14.0


def test_total_cost_calculation() -> None:
    costs = calculate_transaction_costs(
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([10.0, 30.0]),
        fraud_loss_multiplier=1.5,
        review_cost_per_transaction=2.0,
    )

    assert costs["total_estimated_cost"] == 47.0


def test_transaction_amounts_are_aligned_with_labels_and_probabilities() -> None:
    with pytest.raises(CostAnalysisError, match="does not match expected"):
        prepare_transaction_amounts([10.0, 20.0], expected_length=3)


def test_allow_all_baseline_cost() -> None:
    baseline = calculate_allow_all_cost(
        np.array([0, 1, 1]),
        np.array([10.0, 20.0, 30.0]),
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=5.0,
    )

    assert baseline["false_positive"] == 0
    assert baseline["false_negative"] == 2
    assert baseline["fraud_amount_missed"] == 50.0
    assert baseline["total_estimated_cost"] == 50.0


def test_threshold_cost_calculations() -> None:
    table = evaluate_cost_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.9, 0.8, 0.1]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        thresholds=[0.5],
        scenario_name="medium",
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=5.0,
    )

    row = table.iloc[0]
    assert row["false_positive"] == 1
    assert row["false_negative"] == 1
    assert row["false_positive_cost"] == 5.0
    assert row["missed_fraud_cost"] == 40.0


def test_minimum_cost_threshold_selection() -> None:
    table = evaluate_cost_thresholds(
        np.array([0, 1]),
        np.array([0.4, 0.6]),
        np.array([10.0, 100.0]),
        thresholds=[0.5, 0.7],
        scenario_name="low",
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=1.0,
    )

    candidate = find_minimum_cost_threshold(table)

    assert candidate["threshold"] == 0.5


def test_recall_constrained_selection() -> None:
    table = evaluate_cost_thresholds(
        np.array([0, 1, 1]),
        np.array([0.2, 0.8, 0.4]),
        np.array([10.0, 100.0, 50.0]),
        thresholds=[0.5, 0.9],
        scenario_name="low",
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=1.0,
    )

    candidate = find_constrained_cost_threshold(table, min_recall=0.5)

    assert candidate is not None
    assert candidate["threshold"] == 0.5


def test_review_rate_constrained_selection() -> None:
    table = evaluate_cost_thresholds(
        np.array([0, 1, 1]),
        np.array([0.2, 0.8, 0.4]),
        np.array([10.0, 100.0, 50.0]),
        thresholds=[0.3, 0.9],
        scenario_name="low",
        fraud_loss_multiplier=1.0,
        review_cost_per_transaction=1.0,
    )

    candidate = find_constrained_cost_threshold(table, max_review_rate=0.1)

    assert candidate is not None
    assert candidate["review_rate"] == 0.0


def test_invalid_negative_cost_assumptions_are_rejected() -> None:
    with pytest.raises(CostAnalysisError, match="non-negative"):
        calculate_transaction_costs(
            np.array([1]),
            np.array([0]),
            np.array([10.0]),
            fraud_loss_multiplier=-1.0,
            review_cost_per_transaction=1.0,
        )


def test_missing_transaction_amounts_are_handled_explicitly() -> None:
    with pytest.raises(CostAnalysisError, match="affected transactions: 1"):
        prepare_transaction_amounts([10.0, None], expected_length=2)


def test_no_held_out_test_evaluation_occurs() -> None:
    script_path = Path("scripts/analyze_costs.py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    forbidden_names = {"X_test_transformed", "test_probabilities", "test_metrics"}
    observed_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert forbidden_names.isdisjoint(observed_names)
