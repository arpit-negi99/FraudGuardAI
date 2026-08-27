from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_THRESHOLD = 0.60


def priority_band(risk_score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return UI priority bands without changing the frozen decision threshold."""
    if risk_score >= 0.90:
        return "Critical"
    if risk_score >= 0.75:
        return "High"
    if risk_score >= threshold:
        return "Review"
    if risk_score >= 0.30:
        return "Medium"
    return "Low"


def risk_status(review_rate: float, highest_risk: float) -> str:
    """Summarize current scored activity using transparent presentation rules."""
    if highest_risk >= 0.90 or review_rate >= 0.15:
        return "High"
    if highest_risk >= DEFAULT_THRESHOLD or review_rate >= 0.05:
        return "Elevated"
    return "Normal"


def risk_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["Risk band"] = rows["risk_score"].astype(float).map(priority_band)
    order = ["Low", "Medium", "Review", "High", "Critical"]
    counts = rows["Risk band"].value_counts().reindex(order, fill_value=0)
    return counts.rename_axis("risk_band").reset_index(name="transactions")


def enrich_predictions(predictions: pd.DataFrame, source_rows: pd.DataFrame) -> pd.DataFrame:
    """Add product display fields while preserving original model outputs."""
    display = predictions.copy()
    if "TransactionAmt" in source_rows.columns:
        amounts = source_rows[["TransactionID", "TransactionAmt"]].rename(
            columns={"TransactionID": "transaction_id", "TransactionAmt": "amount"}
        )
        display = display.merge(amounts, how="left", on="transaction_id")
    display["priority"] = display["risk_score"].astype(float).map(priority_band)
    return display


def review_queue(predictions: pd.DataFrame) -> pd.DataFrame:
    queue = predictions[predictions["decision"] == "REVIEW"].copy()
    return queue.sort_values("risk_score", ascending=False, kind="mergesort")


def filter_review_queue(
    queue: pd.DataFrame,
    priority_filter: str,
    minimum_risk: float,
) -> pd.DataFrame:
    filtered = queue[queue["risk_score"] >= minimum_risk].copy()
    if priority_filter != "All":
        filtered = filtered[filtered["priority"] == priority_filter]
    return filtered.sort_values("risk_score", ascending=False, kind="mergesort")


def build_policy_presets(threshold_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build validation-derived strategy presets used by API and fallback UI."""
    return [
        {
            "name": "Fraud First",
            "key": "fraud_first",
            "description": "Catch more suspicious activity",
            "tradeoff": "Higher review workload",
            "threshold": float(
                threshold_summary["highest_precision_recall_at_least_0_70"]["threshold"]
            ),
        },
        {
            "name": "Balanced",
            "key": "balanced",
            "description": "Balance fraud capture and analyst workload",
            "tradeoff": "Frozen default policy",
            "threshold": DEFAULT_THRESHOLD,
        },
        {
            "name": "Low Friction",
            "key": "low_friction",
            "description": "Reduce manual review workload",
            "tradeoff": "Lower customer friction",
            "threshold": float(threshold_summary["highest_f1"]["threshold"]),
        },
    ]


def nearest_threshold_metrics(threshold_table: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if threshold_table.empty:
        raise ValueError("Threshold table is empty.")
    index = (threshold_table["threshold"] - threshold).abs().idxmin()
    return _json_safe(threshold_table.loc[index].to_dict())


def parse_cost_scenarios(cost_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_name, details in cost_summary.get("recommended_candidate_thresholds", {}).items():
        rows.append(
            {
                "scenario": scenario_name,
                "review_cost_per_transaction": details["review_cost_per_transaction"],
                "fraud_loss_multiplier": details["fraud_loss_multiplier"],
                "candidate": details["minimum_cost_threshold"],
                "allow_all": details["allow_all"],
                "threshold_0_50": details["threshold_0_50"],
                "capacity_constrained": details.get("minimum_cost_review_rate_at_most_0_05"),
            }
        )
    return rows


def simulate_policy_cost(
    cost_summary: dict[str, Any],
    review_cost: float,
    fraud_loss_multiplier: float,
) -> dict[str, float | str]:
    """Scale existing validation cost assumptions without re-optimizing thresholds."""
    if review_cost < 0 or fraud_loss_multiplier < 0:
        raise ValueError("Cost simulation inputs must be non-negative.")
    scenario = _nearest_cost_scenario(cost_summary, review_cost)
    base = scenario["capacity_constrained"] or scenario["candidate"]
    false_positive = float(base["false_positive"])
    fraud_amount_missed = float(base["fraud_amount_missed"])
    modeled_review_cost = false_positive * review_cost
    modeled_missed_fraud_cost = fraud_amount_missed * fraud_loss_multiplier
    return {
        "scenario": str(scenario["scenario"]),
        "threshold": float(base["threshold"]),
        "modeled_review_cost": modeled_review_cost,
        "modeled_missed_fraud_cost": modeled_missed_fraud_cost,
        "total_modeled_cost": modeled_review_cost + modeled_missed_fraud_cost,
        "note": "Scenario simulation only - not actual merchant savings.",
    }


def spike_status(predictions: pd.DataFrame, window_size: int = 5) -> dict[str, Any]:
    """Lightweight rolling review-rate detector over current scored transaction order."""
    if predictions.empty:
        return {
            "status": "Normal",
            "baseline_review_rate": 0.0,
            "latest_window_review_rate": 0.0,
            "z_score": 0.0,
            "window_size": window_size,
            "method": "Rolling review-rate z-score over current scored transaction order.",
        }
    ordered = predictions.reset_index(drop=True).copy()
    decisions = (ordered["decision"] == "REVIEW").astype(float)
    window = max(2, min(window_size, len(ordered)))
    rolling = decisions.rolling(window=window, min_periods=1).mean()
    baseline = float(decisions.mean())
    latest = float(rolling.iloc[-1])
    std = float(rolling.std(ddof=0))
    z_score = 0.0 if std == 0 else float((latest - baseline) / std)
    if latest >= 0.40 or z_score >= 2.0:
        status = "High"
    elif latest >= 0.20 or z_score >= 1.0:
        status = "Elevated"
    else:
        status = "Normal"
    return {
        "status": status,
        "baseline_review_rate": baseline,
        "latest_window_review_rate": latest,
        "z_score": z_score,
        "window_size": window,
        "method": "Rolling review-rate z-score over current scored transaction order.",
        "thresholds": "Elevated at latest window >= 20% or z >= 1.0; High at latest window >= 40% or z >= 2.0.",
    }


def historical_outcome(label: int) -> str:
    return "Fraud" if int(label) == 1 else "Legitimate"


def demo_outcome_message(label: int, decision: str) -> str | None:
    if label == 1 and decision == "ALLOW":
        return "Model miss"
    if label == 0 and decision == "REVIEW":
        return "False alert"
    return None


def _nearest_cost_scenario(cost_summary: dict[str, Any], review_cost: float) -> dict[str, Any]:
    scenarios = parse_cost_scenarios(cost_summary)
    if not scenarios:
        raise ValueError("No cost scenarios are available.")
    return min(
        scenarios,
        key=lambda item: abs(float(item["review_cost_per_transaction"]) - float(review_cost)),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if pd.isna(value):
        return None
    return value
