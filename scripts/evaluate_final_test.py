from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load import load_labeled_data, read_config
from src.data.preprocess import load_preprocessor
from src.data.split import chronological_split, separate_features_target
from src.evaluation.cost_analysis import calculate_allow_all_cost, calculate_transaction_costs
from src.evaluation.metrics import (
    binary_predictions,
    evaluate_binary_classifier,
    evaluate_majority_legitimate_baseline,
)
from src.models.xgboost_model import load_xgboost_model, predict_fraud_probabilities

FROZEN_THRESHOLD = 0.60
EXPECTED_TEST_ROWS = 88581
FROZEN_FEATURE_COUNT = 422

MODEL_PATH = ROOT / "artifacts" / "models" / "xgboost_model.json"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessors" / "preprocessor.joblib"
VALIDATION_METRICS_PATH = ROOT / "artifacts" / "results" / "xgboost_validation_metrics.json"
VALIDATION_COST_SUMMARY_PATH = ROOT / "artifacts" / "results" / "xgboost_cost_summary.json"
LOGISTIC_MODEL_PATH = ROOT / "artifacts" / "models" / "logistic_regression.joblib"
LOGISTIC_SCALER_PATH = ROOT / "artifacts" / "models" / "baseline_scaler.joblib"

RESULTS_DIR = ROOT / "artifacts" / "results"
FINAL_TEST_METRICS_PATH = RESULTS_DIR / "final_test_metrics.json"
FINAL_VALIDATION_VS_TEST_JSON_PATH = RESULTS_DIR / "final_validation_vs_test.json"
FINAL_VALIDATION_VS_TEST_CSV_PATH = RESULTS_DIR / "final_validation_vs_test.csv"
FINAL_COST_SIMULATION_PATH = RESULTS_DIR / "final_test_cost_simulation.json"
FINAL_CONFUSION_MATRIX_PATH = RESULTS_DIR / "final_confusion_matrix.png"
FINAL_VALIDATION_VS_TEST_PLOT_PATH = RESULTS_DIR / "final_validation_vs_test.png"
FINAL_PR_CURVE_PATH = RESULTS_DIR / "final_precision_recall_curve.png"


def evaluate_final_test(config_path: str | Path = ROOT / "configs" / "config.yaml") -> dict[str, Any]:
    """Evaluate the frozen XGBoost policy once on the chronological held-out test split."""
    _assert_frozen_inputs_exist()
    config = read_config(config_path)

    merged_df, transaction_rows, identity_rows = load_labeled_data(config_path)
    train_df, validation_df, test_df = chronological_split(
        merged_df,
        train_ratio=config["split"]["train_ratio"],
        validation_ratio=config["split"]["validation_ratio"],
        test_ratio=config["split"]["test_ratio"],
    )
    if len(test_df) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Held-out test row count mismatch: expected {EXPECTED_TEST_ROWS}, got {len(test_df)}."
        )

    _, _, _, _, X_test, y_test = separate_features_target(
        train_df,
        validation_df,
        test_df,
        target_column=config["target"]["column"],
    )

    preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    if preprocessor.final_feature_count != FROZEN_FEATURE_COUNT:
        raise RuntimeError(
            "Frozen preprocessor feature count mismatch: "
            f"expected {FROZEN_FEATURE_COUNT}, got {preprocessor.final_feature_count}."
        )
    if "TransactionID" in X_test.columns or "TransactionDT" in X_test.columns or "isFraud" in X_test.columns:
        raise RuntimeError("Leakage column found in held-out test features.")

    X_test_transformed = preprocessor.transform(X_test)
    if X_test_transformed.shape[1] != FROZEN_FEATURE_COUNT:
        raise RuntimeError(
            f"Transformed test feature count mismatch: {X_test_transformed.shape[1]}."
        )

    model = load_xgboost_model(MODEL_PATH)
    y_prob = predict_fraud_probabilities(model, X_test_transformed)
    test_metrics = evaluate_binary_classifier(y_test.to_numpy(), y_prob, threshold=FROZEN_THRESHOLD)
    majority_metrics = evaluate_majority_legitimate_baseline(y_test.to_numpy())
    logistic_metrics = _evaluate_logistic_if_available(X_test_transformed, y_test.to_numpy())
    validation_metrics = _load_validation_threshold_metrics()
    validation_vs_test = _build_validation_comparison(validation_metrics, test_metrics)
    cost_simulation = _build_cost_simulation(
        y_test=y_test.to_numpy(),
        y_prob=y_prob,
        transaction_amounts=test_df["TransactionAmt"],
        config=config,
    )
    error_analysis = _build_error_analysis(test_df, y_test.to_numpy(), y_prob, FROZEN_THRESHOLD)

    summary: dict[str, Any] = {
        "model": "XGBoost",
        "model_artifact": str(MODEL_PATH.relative_to(ROOT)),
        "preprocessor_artifact": str(PREPROCESSOR_PATH.relative_to(ROOT)),
        "evaluation_split": "chronological_held_out_test",
        "frozen_threshold": FROZEN_THRESHOLD,
        "decision_rule": "risk_score >= 0.60 => REVIEW; otherwise ALLOW",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "transaction_rows": transaction_rows,
            "identity_rows": identity_rows,
            "merged_rows": len(merged_df),
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_fraud_rate": float(train_df["isFraud"].mean()),
            "validation_fraud_rate": float(validation_df["isFraud"].mean()),
            "test_fraud_rate": float(y_test.mean()),
        },
        "preprocessing": {
            "frozen_feature_count": preprocessor.final_feature_count,
            "numeric_columns": len(preprocessor.numeric_columns),
            "categorical_columns": len(preprocessor.categorical_columns),
            "dropped_high_missing_columns": len(preprocessor.dropped_columns),
            "excluded_columns": list(preprocessor.excluded_columns),
        },
        "metrics": test_metrics,
        "majority_baseline": majority_metrics,
        "logistic_regression_comparison": logistic_metrics,
        "validation_at_threshold_0_60": validation_metrics,
        "validation_vs_test": validation_vs_test,
        "error_analysis": error_analysis,
        "leakage_protection": {
            "model_loaded_from_artifact": True,
            "preprocessor_loaded_from_artifact": True,
            "threshold_fixed_before_test": True,
            "threshold_optimization_on_test": False,
            "hyperparameter_tuning_on_test": False,
            "feature_selection_on_test": False,
            "kaggle_unlabeled_test_used": False,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(FINAL_TEST_METRICS_PATH, summary)
    _write_json(FINAL_VALIDATION_VS_TEST_JSON_PATH, validation_vs_test)
    pd.DataFrame(validation_vs_test["rows"]).to_csv(FINAL_VALIDATION_VS_TEST_CSV_PATH, index=False)
    _write_json(FINAL_COST_SIMULATION_PATH, cost_simulation)
    _plot_confusion_matrix(test_metrics, FINAL_CONFUSION_MATRIX_PATH)
    _plot_validation_vs_test(validation_vs_test["rows"], FINAL_VALIDATION_VS_TEST_PLOT_PATH)
    _plot_precision_recall_curve(y_test.to_numpy(), y_prob, FINAL_PR_CURVE_PATH)
    _update_project_status(summary, cost_simulation)

    return {
        "summary": summary,
        "cost_simulation": cost_simulation,
        "validation_vs_test": validation_vs_test,
    }


def _assert_frozen_inputs_exist() -> None:
    for path in (MODEL_PATH, PREPROCESSOR_PATH, VALIDATION_METRICS_PATH, VALIDATION_COST_SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact is missing: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, indent=2)


def _load_validation_threshold_metrics() -> dict[str, Any]:
    validation_metrics = _load_json(VALIDATION_METRICS_PATH)
    cost_summary = _load_json(VALIDATION_COST_SUMMARY_PATH)
    threshold_metrics = cost_summary["recommended_candidate_thresholds"]["medium"][
        "minimum_cost_review_rate_at_most_0_05"
    ]
    if abs(float(threshold_metrics["threshold"]) - FROZEN_THRESHOLD) > 1e-12:
        raise RuntimeError("Stored validation operating threshold is not the frozen 0.60 policy.")
    return {
        "threshold": FROZEN_THRESHOLD,
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "pr_auc": validation_metrics["pr_auc"],
        "roc_auc": validation_metrics["roc_auc"],
        "review_rate": threshold_metrics["review_rate"],
        "true_positive": threshold_metrics["true_positive"],
        "false_positive": threshold_metrics["false_positive"],
        "true_negative": threshold_metrics["true_negative"],
        "false_negative": threshold_metrics["false_negative"],
    }


def _build_validation_comparison(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for metric in ("precision", "recall", "f1", "pr_auc", "roc_auc", "review_rate"):
        validation_value = validation_metrics[metric]
        test_value = test_metrics[metric]
        rows.append(
            {
                "metric": metric,
                "validation": validation_value,
                "held_out_test": test_value,
                "difference_test_minus_validation": (
                    None if validation_value is None or test_value is None else test_value - validation_value
                ),
            }
        )
    return {
        "model": "XGBoost",
        "threshold": FROZEN_THRESHOLD,
        "rows": rows,
        "note": "Validation threshold metrics were selected before held-out test evaluation.",
    }


def _build_cost_simulation(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    transaction_amounts: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    assumptions = config["business_cost"]
    fraud_loss_multiplier = float(assumptions["fraud_loss_multiplier"])
    predictions = binary_predictions(y_prob, threshold=FROZEN_THRESHOLD)

    scenarios = {}
    for scenario_name, review_cost in assumptions["review_cost_scenarios"].items():
        review_cost_value = float(review_cost)
        allow_all = calculate_allow_all_cost(
            y_test,
            transaction_amounts,
            fraud_loss_multiplier=fraud_loss_multiplier,
            review_cost_per_transaction=review_cost_value,
            scenario_name=scenario_name,
        )
        frozen_policy_cost = calculate_transaction_costs(
            y_test,
            predictions,
            transaction_amounts.to_numpy(),
            fraud_loss_multiplier=fraud_loss_multiplier,
            review_cost_per_transaction=review_cost_value,
        )
        scenarios[scenario_name] = {
            "review_cost_per_transaction": review_cost_value,
            "fraud_loss_multiplier": fraud_loss_multiplier,
            "allow_all": allow_all,
            "frozen_threshold_0_60": {
                "threshold": FROZEN_THRESHOLD,
                **frozen_policy_cost,
            },
            "modeled_cost_change_vs_allow_all": (
                frozen_policy_cost["total_estimated_cost"] - allow_all["total_estimated_cost"]
            ),
        }

    return {
        "model": "XGBoost",
        "evaluation_split": "chronological_held_out_test",
        "threshold": FROZEN_THRESHOLD,
        "assumption_note": "Held-out modeled cost simulation only; threshold was not optimized on test costs.",
        "scenarios": scenarios,
    }


def _evaluate_logistic_if_available(
    X_test_transformed: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any] | None:
    if not LOGISTIC_MODEL_PATH.exists() or not LOGISTIC_SCALER_PATH.exists():
        return None
    model = joblib.load(LOGISTIC_MODEL_PATH)
    scaler = joblib.load(LOGISTIC_SCALER_PATH)
    X_scaled = scaler.transform(X_test_transformed)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    metrics = evaluate_binary_classifier(y_test, probabilities, threshold=0.50)
    return {"model": "Logistic Regression", "threshold": 0.50, **metrics}


def _build_error_analysis(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = binary_predictions(y_prob, threshold=threshold)
    analysis_df = test_df[
        [
            col
            for col in (
                "TransactionID",
                "TransactionAmt",
                "ProductCD",
                "card4",
                "card6",
                "DeviceType",
            )
            if col in test_df.columns
        ]
    ].copy()
    analysis_df["isFraud"] = y_test
    analysis_df["risk_score"] = y_prob
    analysis_df["prediction"] = predictions

    false_positives = analysis_df[(analysis_df["isFraud"] == 0) & (analysis_df["prediction"] == 1)]
    false_negatives = analysis_df[(analysis_df["isFraud"] == 1) & (analysis_df["prediction"] == 0)]
    highest_risk_false_positives = false_positives.nlargest(50, "risk_score")
    lowest_risk_false_negatives = false_negatives.nsmallest(50, "risk_score")

    return {
        "note": "Descriptive counts only; anonymized IEEE-CIS feature values are not interpreted.",
        "highest_risk_false_positive_count_reviewed": int(len(highest_risk_false_positives)),
        "lowest_risk_false_negative_count_reviewed": int(len(lowest_risk_false_negatives)),
        "highest_risk_false_positive_examples": _examples(highest_risk_false_positives),
        "lowest_risk_false_negative_examples": _examples(lowest_risk_false_negatives),
        "highest_risk_false_positive_patterns": _pattern_counts(highest_risk_false_positives),
        "lowest_risk_false_negative_patterns": _pattern_counts(lowest_risk_false_negatives),
    }


def _examples(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [col for col in ("TransactionID", "TransactionAmt", "risk_score") if col in df.columns]
    return df[columns].head(10).to_dict(orient="records")


def _pattern_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    patterns: dict[str, dict[str, int]] = {}
    for column in ("ProductCD", "card4", "card6", "DeviceType"):
        if column in df.columns:
            patterns[column] = {
                str(key): int(value)
                for key, value in df[column].fillna("__MISSING__").value_counts().head(5).items()
            }
    return patterns


def _plot_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred ALLOW", "Pred REVIEW"])
    ax.set_yticks([0, 1], labels=["Actual Legit", "Actual Fraud"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center", color="black")
    ax.set_title("Held-Out Test Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_validation_vs_test(rows: list[dict[str, Any]], path: Path) -> None:
    plot_df = pd.DataFrame(rows)
    x = np.arange(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, plot_df["validation"], width, label="Validation")
    ax.bar(x + width / 2, plot_df["held_out_test"], width, label="Held-out test")
    ax.set_xticks(x, labels=plot_df["metric"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Validation vs Held-Out Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_precision_recall_curve(y_test: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Held-Out Test Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _update_project_status(summary: dict[str, Any], cost_simulation: dict[str, Any]) -> None:
    status_path = ROOT / "PROJECT-STATUS.md"
    text = status_path.read_text(encoding="utf-8")
    text = text.replace(
        "Demo Polished â€” Ready to Freeze Policy",
        "Final Held-Out Evaluation Complete - Ready for Deployment and Presentation",
    )
    text = text.replace(
        "Demo Polished — Ready to Freeze Policy",
        "Final Held-Out Evaluation Complete - Ready for Deployment and Presentation",
    )
    marker = "## Final Frozen System"
    next_action_marker = "## Next Action"
    if marker in text and next_action_marker in text:
        start = text.index(marker)
        end = text.index(next_action_marker)
        text = text[:start].rstrip() + "\n\n" + text[end:]

    metrics = summary["metrics"]
    data = summary["data"]
    preprocessing = summary["preprocessing"]
    comparison_rows = summary["validation_vs_test"]["rows"]
    cost_rows = cost_simulation["scenarios"]
    final_block = f"""## Final Frozen System

Model: XGBoost

Model artifact: `artifacts/models/xgboost_model.json`

Preprocessor artifact: `artifacts/preprocessors/preprocessor.joblib`

Features: {preprocessing["frozen_feature_count"]}

Threshold: {summary["frozen_threshold"]:.2f}

Decision: ALLOW / REVIEW

## Final Held-Out Test Metrics

```text
Test rows: {data["test_rows"]}
Test fraud rate: {data["test_fraud_rate"]:.6f}

Precision: {metrics["precision"]:.6f}
Recall: {metrics["recall"]:.6f}
F1: {metrics["f1"]:.6f}
PR-AUC: {metrics["pr_auc"]:.6f}
ROC-AUC: {metrics["roc_auc"]:.6f}
Accuracy: {metrics["accuracy"]:.6f}
TP: {metrics["true_positive"]}
FP: {metrics["false_positive"]}
TN: {metrics["true_negative"]}
FN: {metrics["false_negative"]}
Review rate: {metrics["review_rate"]:.6f}
```

## Generalization

Validation vs held-out test at frozen threshold 0.60:

```text
{_format_comparison_rows(comparison_rows)}
```

## Final Cost Simulation

Held-out modeled cost simulation only. Threshold 0.60 was not optimized on test costs.

```text
Low review cost total: {cost_rows["low"]["frozen_threshold_0_60"]["total_estimated_cost"]:.2f}
Medium review cost total: {cost_rows["medium"]["frozen_threshold_0_60"]["total_estimated_cost"]:.2f}
High review cost total: {cost_rows["high"]["frozen_threshold_0_60"]["total_estimated_cost"]:.2f}
```

Final held-out evaluation artifacts produced:

* `artifacts/results/final_test_metrics.json`
* `artifacts/results/final_validation_vs_test.json`
* `artifacts/results/final_validation_vs_test.csv`
* `artifacts/results/final_test_cost_simulation.json`
* `artifacts/results/final_confusion_matrix.png`
* `artifacts/results/final_validation_vs_test.png`
* `artifacts/results/final_precision_recall_curve.png`

"""
    if next_action_marker in text:
        text = text.replace(next_action_marker, final_block + next_action_marker, 1)
    text = text.replace(
        "Freeze the existing XGBoost model, preprocessing, features, and threshold 0.60, then perform final held-out evaluation once.",
        "Finalize reproducible dependencies, deployment configuration, README results, and demo presentation.",
    )
    status_path.write_text(text, encoding="utf-8")


def _format_comparison_rows(rows: list[dict[str, Any]]) -> str:
    lines = ["Metric | Validation | Held-out test | Test - validation"]
    for row in rows:
        lines.append(
            f"{row['metric']}: {row['validation']:.6f} | "
            f"{row['held_out_test']:.6f} | {row['difference_test_minus_validation']:.6f}"
        )
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float):
        return None if np.isnan(value) else value
    return value


def main() -> None:
    result = evaluate_final_test()
    summary = result["summary"]
    metrics = summary["metrics"]
    data = summary["data"]
    preprocessing = summary["preprocessing"]

    print("FraudGuard AI - Final Held-Out Evaluation")
    print()
    print(f"Frozen model: XGBoost ({MODEL_PATH.relative_to(ROOT)})")
    print(f"Frozen threshold: {FROZEN_THRESHOLD:.2f}")
    print(f"Frozen feature count: {preprocessing['frozen_feature_count']}")
    print()
    print(f"Transaction rows: {data['transaction_rows']}")
    print(f"Identity rows: {data['identity_rows']}")
    print(f"Merged rows: {data['merged_rows']}")
    print(f"Held-out test rows: {data['test_rows']}")
    print()
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Review rate: {metrics['review_rate']:.6f}")
    print()
    print("Final artifacts saved successfully.")


if __name__ == "__main__":
    main()
