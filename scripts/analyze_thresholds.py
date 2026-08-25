from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import load_preprocessor
from src.data.split import chronological_split, separate_features_target
from src.evaluation.threshold_analysis import (
    DEFAULT_THRESHOLDS,
    evaluate_thresholds,
    summarize_thresholds,
)
from src.models.xgboost_model import load_xgboost_model, predict_fraud_probabilities


def analyze_thresholds(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Run XGBoost threshold analysis on the chronological validation split only."""
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
        train_df, validation_df, test_df, target_column=target_column
    )

    preprocessor = load_preprocessor(ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib")
    X_validation_transformed = preprocessor.transform(X_validation)

    model = load_xgboost_model(ROOT / "artifacts" / "models" / "xgboost_model.json")
    validation_probabilities = predict_fraud_probabilities(model, X_validation_transformed)

    threshold_table = evaluate_thresholds(
        y_validation.to_numpy(),
        validation_probabilities,
        thresholds=DEFAULT_THRESHOLDS,
    )
    summary = summarize_thresholds(threshold_table)
    summary.update(
        {
            "model": "XGBoost",
            "probability_source": "Recomputed from saved XGBoost model and train-fitted preprocessor.",
            "data": {
                "transaction_rows": transaction_rows,
                "identity_rows": identity_rows,
                "merged_rows": len(merged_df),
                "train_rows": len(train_df),
                "validation_rows": len(validation_df),
                "held_out_test_rows_not_evaluated": len(test_df),
            },
            "notes": [
                "Threshold analysis uses validation labels only.",
                "No final production threshold selected.",
                "No business-cost optimization calculated.",
                "Held-out test split was not evaluated.",
            ],
        }
    )

    _save_outputs(threshold_table, summary)
    return summary


def _save_outputs(threshold_table, summary: dict[str, Any]) -> None:
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    threshold_table.to_csv(results_dir / "xgboost_threshold_analysis.csv", index=False)
    with (results_dir / "xgboost_threshold_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    _plot_precision_recall(threshold_table, results_dir / "precision_recall_vs_threshold.png")
    _plot_f1(threshold_table, results_dir / "f1_vs_threshold.png")
    _plot_review_rate(threshold_table, results_dir / "review_rate_vs_threshold.png")
    _plot_errors(threshold_table, results_dir / "false_positives_false_negatives_vs_threshold.png")


def _plot_precision_recall(threshold_table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision")
    ax.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Precision and Recall vs Threshold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_f1(threshold_table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["f1"], color="tab:green")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs Threshold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_review_rate(threshold_table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["review_rate"], color="tab:orange")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Review rate")
    ax.set_title("Review Rate vs Threshold")
    ax.set_ylim(0, max(0.01, float(threshold_table["review_rate"].max()) * 1.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_errors(threshold_table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["false_positive"], label="False positives")
    ax.plot(threshold_table["threshold"], threshold_table["false_negative"], label="False negatives")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.set_title("False Positives and False Negatives vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _format_candidate(candidate: dict[str, Any] | None) -> str:
    if candidate is None:
        return "No valid threshold found."
    return (
        f"Threshold: {candidate['threshold']:.2f}, "
        f"Precision: {candidate['precision']:.6f}, "
        f"Recall: {candidate['recall']:.6f}, "
        f"F1: {candidate['f1']:.6f}, "
        f"Review rate: {candidate['review_rate']:.6f}"
    )


def main() -> None:
    print("FraudGuard AI - XGBoost Threshold Analysis")
    print()
    summary = analyze_thresholds()

    print("Threshold 0.50")
    print(_format_candidate(summary["metrics_at_threshold_0_50"]))
    print()
    print("Highest-F1 candidate")
    print(_format_candidate(summary["highest_f1"]))
    print()
    print("Best precision with recall >= 0.60")
    print(_format_candidate(summary["highest_precision_recall_at_least_0_60"]))
    print()
    print("Best precision with recall >= 0.70")
    print(_format_candidate(summary["highest_precision_recall_at_least_0_70"]))
    print()
    print("Lowest review rate with recall >= 0.60")
    print(_format_candidate(summary["lowest_review_rate_recall_at_least_0_60"]))
    print()
    print("Saved threshold table, summary JSON, and plots.")


if __name__ == "__main__":
    main()
