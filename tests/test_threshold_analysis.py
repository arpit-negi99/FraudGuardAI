from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.evaluation.threshold_analysis import (
    evaluate_thresholds,
    highest_precision_with_min_recall,
    row_with_highest_f1,
)


def test_threshold_grid_evaluation_returns_one_row_per_threshold() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )

    assert len(table) == 3


def test_increasing_threshold_does_not_increase_predicted_positive_count() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )
    predicted_positive = table["true_positive"] + table["false_positive"]

    assert predicted_positive.is_monotonic_decreasing


def test_precision_and_recall_values_remain_within_zero_and_one() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )

    assert table["precision"].between(0, 1).all()
    assert table["recall"].between(0, 1).all()


def test_review_rate_is_correct() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.5],
    )

    assert table.iloc[0]["review_rate"] == 0.5


def test_confusion_counts_equal_sample_count() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )

    totals = (
        table["true_positive"]
        + table["false_positive"]
        + table["true_negative"]
        + table["false_negative"]
    )
    assert (totals == 4).all()


def test_highest_f1_selection_works_correctly() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )

    candidate = row_with_highest_f1(table)

    assert candidate["threshold"] == 0.5
    assert candidate["f1"] == 1.0


def test_constrained_threshold_selection_works_correctly() -> None:
    table = evaluate_thresholds(
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.8, 0.4, 0.7]),
        thresholds=[0.25, 0.5, 0.75],
    )

    candidate = highest_precision_with_min_recall(table, min_recall=0.5)

    assert candidate is not None
    assert candidate["precision"] == 1.0


def test_no_held_out_test_evaluation_occurs() -> None:
    script_path = Path("scripts/analyze_thresholds.py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    forbidden_names = {"X_test_transformed", "test_probabilities", "test_metrics"}
    observed_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert forbidden_names.isdisjoint(observed_names)
