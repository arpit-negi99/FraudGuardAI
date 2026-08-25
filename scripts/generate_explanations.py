from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import load_preprocessor
from src.data.split import chronological_split, separate_features_target
from src.explainability.shap_explainer import (
    calculate_shap_values,
    create_tree_explainer,
    explain_transaction,
    format_explanation,
    get_feature_names,
    global_shap_importance,
    validate_feature_name_mapping,
)
from src.models.xgboost_model import load_xgboost_model, predict_fraud_probabilities


THRESHOLD = 0.5
GLOBAL_SAMPLE_SIZE = 3000
RANDOM_SEED = 42


def generate_explanations(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Generate validation-only SHAP global and example explanations."""
    started = time.perf_counter()
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
    _, _, X_validation, y_validation, _, _ = separate_features_target(
        train_df,
        validation_df,
        test_df,
        target_column=target_column,
    )

    preprocessor = load_preprocessor(ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib")
    X_validation_transformed = preprocessor.transform(X_validation)
    feature_names = get_feature_names(preprocessor)
    validate_feature_name_mapping(feature_names, X_validation_transformed.shape[1])

    model = load_xgboost_model(ROOT / "artifacts" / "models" / "xgboost_model.json")
    validation_probabilities = predict_fraud_probabilities(model, X_validation_transformed)
    predictions = (validation_probabilities >= THRESHOLD).astype(int)
    labels = y_validation.to_numpy().astype(int)

    explainer = create_tree_explainer(model)
    sample_indices = _fixed_sample_indices(len(validation_df), GLOBAL_SAMPLE_SIZE)
    X_sample = X_validation_transformed[sample_indices]
    shap_values_sample = calculate_shap_values(explainer, X_sample)
    importance_table = global_shap_importance(shap_values_sample, feature_names)

    example_indices = _select_example_indices(labels, predictions, validation_probabilities)
    example_explanations = {}
    for name, index in example_indices.items():
        explanation = explain_transaction(
            model,
            explainer,
            X_validation_transformed[index],
            feature_names,
            threshold=THRESHOLD,
            top_n=5,
        )
        example_explanations[name] = {
            "TransactionID": int(validation_df.iloc[index]["TransactionID"]),
            "actual_label": int(labels[index]),
            "fraud_probability": explanation["fraud_probability"],
            "prediction_at_threshold_0_50": int(predictions[index]),
            "policy_decision_at_threshold_0_50": explanation["policy_decision"],
            "top_positive_shap_contributors": explanation["top_risk_factors"],
            "top_negative_shap_contributors": explanation["top_protective_factors"],
            "formatted_explanation": format_explanation(explanation),
        }

    runtime_seconds = time.perf_counter() - started
    summary = {
        "model": "XGBoost",
        "explainer": "shap.TreeExplainer",
        "evaluation_split": "validation",
        "threshold_for_example_categories": THRESHOLD,
        "transformed_feature_count": len(feature_names),
        "global_explanation_sample_size": int(len(sample_indices)),
        "runtime_seconds": runtime_seconds,
        "data": {
            "transaction_rows": transaction_rows,
            "identity_rows": identity_rows,
            "merged_rows": len(merged_df),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "held_out_test_rows_not_evaluated": len(test_df),
        },
        "limitations": [
            "SHAP values describe model attribution, not causality.",
            "Anonymized IEEE-CIS features are reported by feature name only; no semantic meanings are invented.",
            "Global SHAP importance uses a fixed validation sample for runtime control.",
            "Threshold 0.50 is used only to select example confusion-matrix categories.",
        ],
        "examples": example_explanations,
    }

    _save_outputs(importance_table, X_sample, shap_values_sample, feature_names, summary)
    return summary


def _fixed_sample_indices(row_count: int, sample_size: int) -> np.ndarray:
    sample_size = min(sample_size, row_count)
    rng = np.random.default_rng(RANDOM_SEED)
    return np.sort(rng.choice(row_count, size=sample_size, replace=False))


def _select_example_indices(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, int]:
    masks = {
        "high_risk_true_positive": (labels == 1) & (predictions == 1),
        "high_risk_false_positive": (labels == 0) & (predictions == 1),
        "missed_fraud_false_negative": (labels == 1) & (predictions == 0),
        "clearly_legitimate_true_negative": (labels == 0) & (predictions == 0),
    }
    selected: dict[str, int] = {}
    for name, mask in masks.items():
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue
        if name == "missed_fraud_false_negative":
            selected[name] = int(indices[np.argmin(probabilities[indices])])
        elif name == "clearly_legitimate_true_negative":
            selected[name] = int(indices[np.argmin(probabilities[indices])])
        else:
            selected[name] = int(indices[np.argmax(probabilities[indices])])
    return selected


def _save_outputs(
    importance_table,
    X_sample: np.ndarray,
    shap_values_sample: np.ndarray,
    feature_names: list[str],
    summary: dict[str, Any],
) -> None:
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    importance_table.to_csv(results_dir / "shap_global_importance.csv", index=False)
    with (results_dir / "example_explanations.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    _plot_global_importance(importance_table, results_dir / "shap_global_importance.png")
    _plot_summary(shap_values_sample, X_sample, feature_names, results_dir / "shap_summary.png")


def _plot_global_importance(importance_table, path: Path, top_n: int = 20) -> None:
    top_table = importance_table.head(top_n).sort_values(
        "mean_absolute_shap_value", ascending=True
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_table["feature"], top_table["mean_absolute_shap_value"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global SHAP Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_summary(
    shap_values_sample: np.ndarray,
    X_sample: np.ndarray,
    feature_names: list[str],
    path: Path,
) -> None:
    shap.summary_plot(
        shap_values_sample,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("FraudGuard AI - SHAP Explainability")
    print()
    summary = generate_explanations()
    print(f"Explainer: {summary['explainer']}")
    print(f"Transformed features: {summary['transformed_feature_count']}")
    print(f"Global sample size: {summary['global_explanation_sample_size']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
    print()
    print("Generated examples:")
    for name, example in summary["examples"].items():
        contributors = example["top_positive_shap_contributors"]
        top_feature = contributors[0]["feature"] if contributors else "none"
        print(
            f"{name}: TransactionID {example['TransactionID']}, "
            f"probability {example['fraud_probability']:.6f}, "
            f"top positive contributor {top_feature}"
        )
    print()
    print("Saved SHAP global importance, summary plot, and example explanations.")


if __name__ == "__main__":
    main()
