from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import load_preprocessor
from src.data.split import chronological_split, separate_features_target
from src.evaluation.cost_analysis import (
    calculate_allow_all_cost,
    evaluate_cost_thresholds,
    find_constrained_cost_threshold,
    find_minimum_cost_threshold,
    prepare_transaction_amounts,
)
from src.evaluation.threshold_analysis import DEFAULT_THRESHOLDS
from src.models.xgboost_model import load_xgboost_model, predict_fraud_probabilities


def analyze_costs(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Run validation-only business-cost scenario analysis for saved XGBoost scores."""
    config = read_config(config_path)
    business_config = config["business_cost"]
    fraud_loss_multiplier = float(business_config["fraud_loss_multiplier"])
    review_cost_scenarios = {
        name: float(value)
        for name, value in business_config["review_cost_scenarios"].items()
    }

    merged_df, transaction_rows, identity_rows = load_labeled_data(config_path)
    split_config = config["split"]
    train_df, validation_df, test_df = chronological_split(
        merged_df,
        train_ratio=split_config["train_ratio"],
        validation_ratio=split_config["validation_ratio"],
        test_ratio=split_config["test_ratio"],
    )

    if "TransactionAmt" not in validation_df.columns:
        raise ValueError("Validation split is missing required TransactionAmt column.")
    validation_amounts = prepare_transaction_amounts(
        validation_df["TransactionAmt"],
        expected_length=len(validation_df),
    )

    target_column = config["target"]["column"]
    _, _, X_validation, y_validation, _, _ = separate_features_target(
        train_df, validation_df, test_df, target_column=target_column
    )

    preprocessor = load_preprocessor(ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib")
    X_validation_transformed = preprocessor.transform(X_validation)

    model = load_xgboost_model(ROOT / "artifacts" / "models" / "xgboost_model.json")
    validation_probabilities = predict_fraud_probabilities(model, X_validation_transformed)

    scenario_tables = []
    scenario_summaries: dict[str, Any] = {}
    y_validation_array = y_validation.to_numpy()

    for scenario_name, review_cost in review_cost_scenarios.items():
        scenario_table = evaluate_cost_thresholds(
            y_validation_array,
            validation_probabilities,
            validation_amounts,
            thresholds=DEFAULT_THRESHOLDS,
            scenario_name=scenario_name,
            fraud_loss_multiplier=fraud_loss_multiplier,
            review_cost_per_transaction=review_cost,
        )
        scenario_tables.append(scenario_table)
        threshold_0_50 = _row_at_threshold(scenario_table, 0.50)
        allow_all = calculate_allow_all_cost(
            y_validation_array,
            validation_amounts,
            fraud_loss_multiplier=fraud_loss_multiplier,
            review_cost_per_transaction=review_cost,
            scenario_name=scenario_name,
        )
        minimum_cost = find_minimum_cost_threshold(scenario_table)
        scenario_summaries[scenario_name] = {
            "review_cost_per_transaction": review_cost,
            "fraud_loss_multiplier": fraud_loss_multiplier,
            "allow_all": allow_all,
            "threshold_0_50": threshold_0_50,
            "minimum_cost_threshold": minimum_cost,
            "minimum_cost_recall_at_least_0_60": find_constrained_cost_threshold(
                scenario_table, min_recall=0.60
            ),
            "minimum_cost_recall_at_least_0_70": find_constrained_cost_threshold(
                scenario_table, min_recall=0.70
            ),
            "minimum_cost_review_rate_at_most_0_05": find_constrained_cost_threshold(
                scenario_table, max_review_rate=0.05
            ),
            "simulated_cost_reduction_vs_allow_all": (
                allow_all["total_estimated_cost"] - minimum_cost["total_estimated_cost"]
            ),
            "simulated_cost_reduction_vs_threshold_0_50": (
                threshold_0_50["total_estimated_cost"] - minimum_cost["total_estimated_cost"]
            ),
        }

    cost_table = pd.concat(scenario_tables, ignore_index=True)
    summary = {
        "model": "XGBoost",
        "evaluation_split": "validation",
        "business_assumptions": {
            "fraud_loss_multiplier": fraud_loss_multiplier,
            "review_cost_scenarios": review_cost_scenarios,
            "currency_note": "Costs are scenario cost units, not IEEE-CIS-provided merchant currency.",
            "source_note": (
                "TransactionAmt is dataset-derived. Review cost and fraud loss multiplier "
                "are user-configured assumptions, not IEEE-CIS ground truth."
            ),
        },
        "validation_performance": {
            "threshold_grid": {
                "min": float(min(DEFAULT_THRESHOLDS)),
                "max": float(max(DEFAULT_THRESHOLDS)),
                "count": len(DEFAULT_THRESHOLDS),
            },
            "transaction_amount_invalid_count": 0,
        },
        "recommended_candidate_thresholds": scenario_summaries,
        "data": {
            "transaction_rows": transaction_rows,
            "identity_rows": identity_rows,
            "merged_rows": len(merged_df),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "held_out_test_rows_not_evaluated": len(test_df),
        },
        "notes": [
            "Validation-set business impact is modeled from configurable assumptions.",
            "No final production threshold selected.",
            "No held-out test evaluation performed.",
        ],
    }

    _save_outputs(cost_table, summary)
    return summary


def _row_at_threshold(cost_table: pd.DataFrame, threshold: float) -> dict[str, Any]:
    index = (cost_table["threshold"] - threshold).abs().idxmin()
    return _json_safe_dict(cost_table.loc[index].to_dict())


def _json_safe_dict(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    integer_fields = {"true_positive", "false_positive", "true_negative", "false_negative"}
    for key, value in row.items():
        if key in integer_fields:
            result[key] = int(value)
        elif isinstance(value, str) or value is None:
            result[key] = value
        else:
            result[key] = float(value)
    return result


def _save_outputs(cost_table: pd.DataFrame, summary: dict[str, Any]) -> None:
    results_dir = ROOT / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    cost_table.to_csv(results_dir / "xgboost_cost_analysis.csv", index=False)
    with (results_dir / "xgboost_cost_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    _plot_total_cost(cost_table, results_dir / "total_cost_vs_threshold.png")
    _plot_cost_components(cost_table, results_dir / "cost_components_vs_threshold.png")


def _plot_total_cost(cost_table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for scenario_name, scenario_table in cost_table.groupby("scenario"):
        ax.plot(
            scenario_table["threshold"],
            scenario_table["total_estimated_cost"],
            label=f"{scenario_name} review cost",
        )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Estimated cost units")
    ax.set_title("Total Estimated Cost vs Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cost_components(cost_table: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for scenario_name, scenario_table in cost_table.groupby("scenario"):
        axes[0].plot(
            scenario_table["threshold"],
            scenario_table["missed_fraud_cost"],
            label=f"{scenario_name} missed fraud",
        )
        axes[1].plot(
            scenario_table["threshold"],
            scenario_table["false_positive_cost"],
            label=f"{scenario_name} false positive",
        )
    axes[0].set_title("Missed-Fraud Cost")
    axes[1].set_title("False-Positive Cost")
    for ax in axes:
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Estimated cost units")
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
        f"Review rate: {candidate['review_rate']:.6f}, "
        f"FP: {candidate['false_positive']}, "
        f"FN: {candidate['false_negative']}, "
        f"Fraud detected: {candidate['fraud_amount_detected']:.2f}, "
        f"Fraud missed: {candidate['fraud_amount_missed']:.2f}, "
        f"FP cost: {candidate['false_positive_cost']:.2f}, "
        f"Missed-fraud cost: {candidate['missed_fraud_cost']:.2f}, "
        f"Total cost: {candidate['total_estimated_cost']:.2f}"
    )


def main() -> None:
    print("FraudGuard AI - XGBoost Cost Analysis")
    print()
    summary = analyze_costs()
    assumptions = summary["business_assumptions"]
    print(f"Fraud loss multiplier: {assumptions['fraud_loss_multiplier']:.2f}")
    print(f"Review cost scenarios: {assumptions['review_cost_scenarios']}")
    print()

    for scenario_name, scenario in summary["recommended_candidate_thresholds"].items():
        print(f"Scenario: {scenario_name}")
        print(f"Review cost per false positive: {scenario['review_cost_per_transaction']:.2f}")
        print(f"Fraud loss multiplier: {scenario['fraud_loss_multiplier']:.2f}")
        print(f"Allow-all total cost: {scenario['allow_all']['total_estimated_cost']:.2f}")
        print(f"Threshold 0.50: {_format_candidate(scenario['threshold_0_50'])}")
        print(f"Minimum-cost threshold: {_format_candidate(scenario['minimum_cost_threshold'])}")
        print(
            "Minimum cost with recall >= 0.60: "
            f"{_format_candidate(scenario['minimum_cost_recall_at_least_0_60'])}"
        )
        print(
            "Minimum cost with recall >= 0.70: "
            f"{_format_candidate(scenario['minimum_cost_recall_at_least_0_70'])}"
        )
        print(
            "Minimum cost with review rate <= 0.05: "
            f"{_format_candidate(scenario['minimum_cost_review_rate_at_most_0_05'])}"
        )
        print(
            "Simulated cost reduction vs allow-all: "
            f"{scenario['simulated_cost_reduction_vs_allow_all']:.2f}"
        )
        print(
            "Simulated cost reduction vs threshold 0.50: "
            f"{scenario['simulated_cost_reduction_vs_threshold_0_50']:.2f}"
        )
        print()

    print("Saved cost table, summary JSON, and plots.")


if __name__ == "__main__":
    main()
