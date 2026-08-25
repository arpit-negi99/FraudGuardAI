from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import FraudDataPreprocessor
from src.data.split import chronological_split, separate_features_target
from src.evaluation.metrics import evaluate_binary_classifier
from src.models.xgboost_model import (
    calculate_scale_pos_weight,
    fit_xgboost_classifier,
    predict_fraud_probabilities,
    save_xgboost_model,
)


THRESHOLD = 0.5


def train_xgboost(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Train and evaluate the first XGBoost model on validation only."""
    config = read_config(config_path)
    merged_df, transaction_rows, identity_rows = load_labeled_data(config_path)

    split_config = config["split"]
    train_df, validation_df, test_df = chronological_split(
        merged_df,
        train_ratio=split_config["train_ratio"],
        validation_ratio=split_config["validation_ratio"],
        test_ratio=split_config["test_ratio"],
    )

    target_column = config["target"]["column"]
    X_train, y_train, X_validation, y_validation, _, _ = separate_features_target(
        train_df, validation_df, test_df, target_column=target_column
    )

    preprocessor = FraudDataPreprocessor(
        missing_threshold=config["preprocessing"]["missing_column_threshold"]
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)

    y_train_array = y_train.to_numpy()
    y_validation_array = y_validation.to_numpy()
    scale_pos_weight = calculate_scale_pos_weight(y_train_array)

    training_start = time.perf_counter()
    model = fit_xgboost_classifier(
        X_train_transformed,
        y_train_array,
        X_validation_transformed,
        y_validation_array,
        scale_pos_weight=scale_pos_weight,
    )
    training_time_seconds = time.perf_counter() - training_start

    validation_probabilities = predict_fraud_probabilities(model, X_validation_transformed)
    metrics = evaluate_binary_classifier(
        y_validation_array, validation_probabilities, threshold=THRESHOLD
    )

    best_iteration = getattr(model, "best_iteration", None)
    estimators_used = (best_iteration + 1) if best_iteration is not None else model.n_estimators

    results = {
        "model": "XGBoost",
        "evaluation_split": "validation",
        "threshold": THRESHOLD,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "accuracy": metrics["accuracy"],
        "true_positive": metrics["true_positive"],
        "false_positive": metrics["false_positive"],
        "true_negative": metrics["true_negative"],
        "false_negative": metrics["false_negative"],
        "review_rate": metrics["review_rate"],
        "scale_pos_weight": scale_pos_weight,
        "training_time_seconds": training_time_seconds,
        "best_iteration": best_iteration,
        "estimators_used": estimators_used,
        "early_stopping_rounds": model.get_params().get("early_stopping_rounds"),
        "data": {
            "transaction_rows": transaction_rows,
            "identity_rows": identity_rows,
            "merged_rows": len(merged_df),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "held_out_test_rows_not_evaluated": len(test_df),
            "train_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_validation.mean()),
        },
        "preprocessing": preprocessor.get_metadata(),
        "model_parameters": model.get_params(),
        "artifact_format": "native XGBoost JSON",
    }

    _save_artifacts(model, results)
    return results


def build_model_comparison(
    logistic_results: dict[str, Any],
    xgboost_results: dict[str, Any],
) -> dict[str, Any]:
    """Create a validation comparison across majority, Logistic Regression, and XGBoost."""
    majority = logistic_results["majority_baseline"]
    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc", "review_rate"]
    return {
        "evaluation_split": "validation",
        "threshold": THRESHOLD,
        "majority_baseline": {key: majority.get(key) for key in ("precision", "recall", "f1", "accuracy")},
        "logistic_regression": {key: logistic_results.get(key) for key in metrics},
        "xgboost": {key: xgboost_results.get(key) for key in metrics},
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_artifacts(model: Any, results: dict[str, Any]) -> None:
    models_dir = ROOT / "artifacts" / "models"
    results_dir = ROOT / "artifacts" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    save_xgboost_model(model, models_dir / "xgboost_model.json")

    xgboost_metrics_path = results_dir / "xgboost_validation_metrics.json"
    with xgboost_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    logistic_results = _load_json(results_dir / "logistic_baseline_metrics.json")
    comparison = build_model_comparison(logistic_results, results)
    with (results_dir / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)


def main() -> None:
    print("FraudGuard AI - XGBoost Validation Model")
    print()
    results = train_xgboost()

    print("XGBoost validation results")
    print(f"Threshold: {results['threshold']:.2f}")
    print(f"Precision: {results['precision']:.6f}")
    print(f"Recall: {results['recall']:.6f}")
    print(f"F1: {results['f1']:.6f}")
    print(f"PR-AUC: {results['pr_auc']:.6f}")
    print(f"ROC-AUC: {results['roc_auc']:.6f}")
    print(f"Accuracy: {results['accuracy']:.6f}")
    print(f"True positives: {results['true_positive']}")
    print(f"False positives: {results['false_positive']}")
    print(f"True negatives: {results['true_negative']}")
    print(f"False negatives: {results['false_negative']}")
    print(f"Review rate: {results['review_rate']:.6f}")
    print(f"scale_pos_weight: {results['scale_pos_weight']:.6f}")
    print(f"Best iteration: {results['best_iteration']}")
    print(f"Estimators used: {results['estimators_used']}")
    print(f"Training time: {results['training_time_seconds']:.2f} seconds")
    print()
    print("Saved XGBoost model, validation metrics, and model comparison.")


if __name__ == "__main__":
    main()
