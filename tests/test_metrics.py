from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.evaluation.metrics import (
    binary_predictions,
    evaluate_binary_classifier,
    evaluate_majority_legitimate_baseline,
)


def test_thresholding_works_correctly() -> None:
    probabilities = np.array([0.1, 0.5, 0.9])

    predictions = binary_predictions(probabilities, threshold=0.5)

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1]))


def test_confusion_matrix_values_are_correct() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.2])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1


def test_precision_recall_and_f1_are_correct() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.2])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_pr_auc_and_roc_auc_run_correctly() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.9])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["pr_auc"] is not None
    assert metrics["roc_auc"] is not None


def test_review_rate_is_calculated_correctly() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.7, 0.8, 0.2])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["review_rate"] == 0.5


def test_evaluation_handles_zero_predicted_positive_cases_without_crashing() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.9)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["review_rate"] == 0.0


def test_majority_baseline_predicts_legitimate_for_every_row() -> None:
    y_true = np.array([0, 0, 1, 1])

    metrics = evaluate_majority_legitimate_baseline(y_true)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_train_baseline_does_not_evaluate_held_out_test_split() -> None:
    script_path = Path("scripts/train_baseline.py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    forbidden_names = {"X_test_transformed", "X_test_scaled", "test_probabilities"}
    observed_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    constant_strings = {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert forbidden_names.isdisjoint(observed_names)
    assert "test" not in {
        value for value in constant_strings if value == "test"
    }
