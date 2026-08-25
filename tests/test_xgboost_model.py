from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from scripts.train_xgboost import build_model_comparison
from src.evaluation.metrics import evaluate_binary_classifier
from src.models.xgboost_model import (
    calculate_scale_pos_weight,
    create_xgboost_classifier,
    load_xgboost_model,
    predict_fraud_probabilities,
    save_xgboost_model,
)


def test_xgboost_model_builder_returns_xgb_classifier() -> None:
    model = create_xgboost_classifier(scale_pos_weight=2.0, early_stopping_rounds=None)

    assert isinstance(model, XGBClassifier)
    assert model.get_params()["scale_pos_weight"] == 2.0


def test_scale_pos_weight_is_calculated_correctly() -> None:
    y_train = np.array([0, 0, 0, 1, 1])

    assert calculate_scale_pos_weight(y_train) == 1.5


def test_training_script_does_not_evaluate_held_out_test_split() -> None:
    script_path = Path("scripts/train_xgboost.py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    forbidden_names = {"X_test_transformed", "test_probabilities", "test_metrics"}
    observed_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert forbidden_names.isdisjoint(observed_names)


def test_validation_probabilities_are_between_zero_and_one() -> None:
    X_train = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0], [0.9, 0.8], [0.1, 0.2]],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 1, 0])
    model = create_xgboost_classifier(
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        early_stopping_rounds=None,
        n_estimators=5,
    )
    model.fit(X_train, y_train, verbose=False)

    probabilities = predict_fraud_probabilities(model, X_train)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_existing_evaluation_functions_work_with_xgboost_outputs() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.8, 0.3, 0.7])

    metrics = evaluate_binary_classifier(y_true, y_prob, threshold=0.5)

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0


def test_model_comparison_logic_reads_logistic_regression_metrics() -> None:
    logistic_results = {
        "precision": 0.1,
        "recall": 0.2,
        "f1": 0.3,
        "pr_auc": 0.4,
        "roc_auc": 0.5,
        "review_rate": 0.6,
        "majority_baseline": {
            "accuracy": 0.9,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    }
    xgboost_results = {
        "precision": 0.7,
        "recall": 0.8,
        "f1": 0.75,
        "pr_auc": 0.85,
        "roc_auc": 0.9,
        "review_rate": 0.25,
    }

    comparison = build_model_comparison(logistic_results, xgboost_results)

    assert comparison["logistic_regression"]["pr_auc"] == 0.4
    assert comparison["xgboost"]["pr_auc"] == 0.85
    assert comparison["majority_baseline"]["accuracy"] == 0.9


def test_training_script_does_not_use_logistic_regression_scaler() -> None:
    script_text = Path("scripts/train_xgboost.py").read_text(encoding="utf-8")

    assert "StandardScaler" not in script_text
    assert "baseline_scaler" not in script_text


def test_xgboost_model_can_be_saved_and_reloaded(tmp_path) -> None:
    X_train = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0], [0.9, 0.8], [0.1, 0.2]],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1, 1, 0])
    model = create_xgboost_classifier(
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        early_stopping_rounds=None,
        n_estimators=5,
    )
    model.fit(X_train, y_train, verbose=False)
    model_path = tmp_path / "xgboost_model.json"

    save_xgboost_model(model, model_path)
    loaded = load_xgboost_model(model_path)

    assert model_path.exists()
    assert isinstance(loaded, XGBClassifier)
    assert json.loads(model_path.read_text(encoding="utf-8"))
