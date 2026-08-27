from __future__ import annotations

import ast
from pathlib import Path


SCRIPT_PATH = Path("scripts/evaluate_final_test.py")


def _script_tree() -> ast.Module:
    return ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def _called_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_final_evaluation_uses_frozen_threshold_and_expected_test_rows() -> None:
    tree = _script_tree()
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }

    assert constants["FROZEN_THRESHOLD"] == 0.60
    assert constants["EXPECTED_TEST_ROWS"] == 88581
    assert constants["FROZEN_FEATURE_COUNT"] == 422


def test_final_evaluation_loads_frozen_artifacts() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "load_preprocessor(PREPROCESSOR_PATH)" in script_text
    assert "load_xgboost_model(MODEL_PATH)" in script_text
    assert "xgboost_model.json" in script_text
    assert "preprocessor.joblib" in script_text


def test_final_evaluation_does_not_retrain_or_refit() -> None:
    forbidden_calls = {
        "fit",
        "fit_transform",
        "fit_xgboost_classifier",
        "create_xgboost_classifier",
        "save_xgboost_model",
    }
    called_names = _called_names(_script_tree())

    assert forbidden_calls.isdisjoint(called_names)


def test_final_evaluation_does_not_optimize_or_tune_on_test() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden_terms = [
        "GridSearchCV",
        "RandomizedSearchCV",
        "Optuna",
        "study.optimize",
        "find_minimum_cost_threshold",
        "find_constrained_cost_threshold",
        "threshold_grid",
        "highest_f1",
        "minimum_cost_threshold",
    ]

    for term in forbidden_terms:
        assert term not in script_text


def test_final_evaluation_writes_required_final_artifacts() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "final_test_metrics.json" in script_text
    assert "final_validation_vs_test.json" in script_text
    assert "final_validation_vs_test.csv" in script_text
    assert "final_test_cost_simulation.json" in script_text
    assert "final_confusion_matrix.png" in script_text
    assert "final_validation_vs_test.png" in script_text
