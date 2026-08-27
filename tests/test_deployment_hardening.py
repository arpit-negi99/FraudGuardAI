from __future__ import annotations

import ast
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from app import DEFAULT_THRESHOLD, missing_deployment_artifacts
from src.inference.predict import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSOR_PATH,
    FraudPredictor,
    load_model_artifact,
    load_preprocessor_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DEMO_TRANSACTIONS_PATH = ROOT / "artifacts" / "demo" / "demo_transactions.csv"
DEMO_LABELS_PATH = ROOT / "artifacts" / "demo" / "demo_labels.csv"


def test_required_deployment_artifacts_exist() -> None:
    assert missing_deployment_artifacts(ROOT) == []


def test_frozen_model_and_preprocessor_paths_remain_unchanged() -> None:
    assert DEFAULT_MODEL_PATH == Path("artifacts/models/xgboost_model.json")
    assert DEFAULT_PREPROCESSOR_PATH == Path("artifacts/preprocessors/preprocessor.joblib")
    assert DEFAULT_THRESHOLD == 0.60


def test_preprocessor_loads_without_sklearn_version_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        preprocessor = load_preprocessor_artifact(DEFAULT_PREPROCESSOR_PATH)

    assert preprocessor.final_feature_count == 422
    assert not any(isinstance(item.message, InconsistentVersionWarning) for item in caught)


def test_xgboost_model_loads_from_frozen_artifact() -> None:
    model = load_model_artifact(DEFAULT_MODEL_PATH)

    assert hasattr(model, "predict_proba")


def test_demo_sample_exists_without_inference_labels() -> None:
    demo_transactions = pd.read_csv(DEMO_TRANSACTIONS_PATH)
    demo_labels = pd.read_csv(DEMO_LABELS_PATH)

    assert not demo_transactions.empty
    assert not demo_labels.empty
    assert "isFraud" not in demo_transactions.columns
    assert {"TransactionID", "isFraud"}.issubset(demo_labels.columns)


def test_standard_inference_does_not_require_raw_training_data() -> None:
    app_text = Path("app.py").read_text(encoding="utf-8")
    demo_script_text = Path("scripts/demo_inference.py").read_text(encoding="utf-8")

    assert "load_labeled_data" not in app_text
    assert "chronological_split" not in app_text
    assert "load_labeled_data" not in demo_script_text
    assert "chronological_split" not in demo_script_text


def test_app_helper_paths_are_relative() -> None:
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    path_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]

    assert not any("D:\\fraudguard-ai" in value for value in path_strings)
    assert not any("C:\\Users\\" in value for value in path_strings)


def test_no_external_model_api_key_is_required() -> None:
    requirement_text = Path("requirements-lock.txt").read_text(encoding="utf-8").lower()
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [Path("app.py"), Path("src/inference/predict.py")]
    ).lower()

    assert "openai" not in requirement_text
    assert "anthropic" not in requirement_text
    assert "gemini" not in requirement_text
    assert "api_key" not in source_text


def test_known_demo_prediction_matches_recorded_behavior() -> None:
    demo_transactions = pd.read_csv(DEMO_TRANSACTIONS_PATH)
    row = demo_transactions[demo_transactions["TransactionID"] == 3400378].iloc[[0]]
    result = FraudPredictor().predict_transaction(row, include_explanation=False)

    assert result["decision"] == "ALLOW"
    assert abs(result["risk_score"] - 0.163838) < 0.0005
