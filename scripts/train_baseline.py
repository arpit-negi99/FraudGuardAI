from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import FraudDataPreprocessor
from src.data.split import chronological_split, separate_features_target
from src.evaluation.metrics import (
    evaluate_binary_classifier,
    evaluate_majority_legitimate_baseline,
)
from src.models.baseline import fit_logistic_regression_baseline, predict_fraud_probabilities


THRESHOLD = 0.5


def train_baseline(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Train and evaluate the Logistic Regression baseline on validation only."""
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_validation_scaled = scaler.transform(X_validation_transformed)

    training_start = time.perf_counter()
    model = fit_logistic_regression_baseline(X_train_scaled, y_train.to_numpy())
    training_time_seconds = time.perf_counter() - training_start

    validation_probabilities = predict_fraud_probabilities(model, X_validation_scaled)
    logistic_metrics = evaluate_binary_classifier(
        y_validation.to_numpy(), validation_probabilities, threshold=THRESHOLD
    )
    logistic_metrics.update(
        {
            "model": "LogisticRegression",
            "evaluation_split": "validation",
            "threshold_note": "Fixed conventional 0.50 threshold; not optimized.",
            "training_time_seconds": training_time_seconds,
        }
    )

    majority_metrics = evaluate_majority_legitimate_baseline(y_validation.to_numpy())
    majority_metrics["evaluation_split"] = "validation"

    results = {
        "model": "LogisticRegression",
        "evaluation_split": "validation",
        "threshold": THRESHOLD,
        "precision": logistic_metrics["precision"],
        "recall": logistic_metrics["recall"],
        "f1": logistic_metrics["f1"],
        "pr_auc": logistic_metrics["pr_auc"],
        "roc_auc": logistic_metrics["roc_auc"],
        "accuracy": logistic_metrics["accuracy"],
        "true_positive": logistic_metrics["true_positive"],
        "false_positive": logistic_metrics["false_positive"],
        "true_negative": logistic_metrics["true_negative"],
        "false_negative": logistic_metrics["false_negative"],
        "review_rate": logistic_metrics["review_rate"],
        "training_time_seconds": training_time_seconds,
        "threshold_note": logistic_metrics["threshold_note"],
        "majority_baseline": majority_metrics,
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
        "scaling": {
            "scaler": "sklearn.preprocessing.StandardScaler",
            "fit_split": "train",
            "with_mean": scaler.with_mean,
            "with_std": scaler.with_std,
        },
        "model_parameters": model.get_params(),
        "solver_note": (
            "lbfgs used because liblinear and saga did not complete promptly on "
            "the full dense chronological training matrix."
        ),
    }

    _save_artifacts(model, scaler, results)
    return results


def _save_artifacts(model: Any, scaler: StandardScaler, results: dict[str, Any]) -> None:
    models_dir = ROOT / "artifacts" / "models"
    results_dir = ROOT / "artifacts" / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "logistic_regression.joblib")
    joblib.dump(scaler, models_dir / "baseline_scaler.joblib")

    with (results_dir / "logistic_baseline_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def main() -> None:
    print("FraudGuard AI - Logistic Regression Baseline")
    print()
    results = train_baseline()
    majority = results["majority_baseline"]

    print("Majority baseline: predict every transaction as legitimate")
    print(f"Accuracy: {majority['accuracy']:.6f}")
    print(f"Precision: {majority['precision']:.6f}")
    print(f"Recall: {majority['recall']:.6f}")
    print(f"F1: {majority['f1']:.6f}")
    print()

    print("Logistic Regression validation results")
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
    print(f"Training time: {results['training_time_seconds']:.2f} seconds")
    print()
    print("Saved logistic baseline artifacts and validation metrics.")


if __name__ == "__main__":
    main()
