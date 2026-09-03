from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.baseline import build_historical_baseline
from src.monitoring.schema import (
    DEFAULT_BASELINE_WINDOWS,
    DEFAULT_EWMA_ALPHA,
    MONITORED_METRICS,
    MonitoringStatus,
)


ACTION_GUIDANCE = {
    "REVIEW_RATE": "Inspect the highest-risk transactions and analyst review queue.",
    "VERY_HIGH_RISK_RATE": "Inspect the highest-risk transactions and analyst review queue.",
    "MEAN_FRAUD_RISK": "Review the recent transaction fraud-risk distribution.",
    "PAYMENT_INCIDENT_RATE": "Inspect recent payment lifecycle incidents and reconciliation states.",
    "CRITICAL_HIGH_INCIDENT_RATE": "Prioritize high and critical payment incident response.",
    "DEBIT_SERVICE_MISMATCH_RATE": "Check payment callback and reconciliation flow for delayed or failed status updates.",
    "COMPLAINT_ESCALATION_RATE": "Review unresolved customer complaints and refund status.",
    "RETRY_RISK_RATE": "Inspect retry patterns and unresolved earlier payment attempts.",
    "CAPTURED_UNFULFILLED_RATE": "Check orders captured without fulfilment and reconcile service delivery.",
}


def evaluate_monitoring_windows(
    windows: pd.DataFrame,
    baseline_windows: int = DEFAULT_BASELINE_WINDOWS,
    ewma_alpha: float = DEFAULT_EWMA_ALPHA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if windows.empty:
        raise ValueError("Cannot evaluate empty monitoring windows.")
    baseline = build_historical_baseline(windows, baseline_windows)
    ordered = windows.sort_values("window_start").reset_index(drop=True).copy()
    metric_results = []
    ewma_state = {metric: baseline[metric]["mean"] for metric in MONITORED_METRICS}
    for index, row in ordered.iterrows():
        metrics = {}
        highest = MonitoringStatus.NORMAL
        primary_driver = "NONE"
        primary_z = 0.0
        for metric in MONITORED_METRICS:
            current = float(row[metric])
            base = baseline[metric]
            std = _safe_std(base["std"], base["mean"], metric)
            z_score = max(0.0, (current - base["mean"]) / std)
            ewma_state[metric] = ewma_alpha * current + (1 - ewma_alpha) * ewma_state[metric]
            ewma_z = max(0.0, (ewma_state[metric] - base["mean"]) / std)
            status = _status_from_z(max(z_score, ewma_z))
            driver = metric.upper()
            metrics[metric] = {
                "current": current,
                "baseline": base["mean"],
                "baseline_std": base["std"],
                "z_score": z_score,
                "ewma": ewma_state[metric],
                "ewma_z_score": ewma_z,
                "relative_change": relative_change(current, base["mean"]),
                "status": status.value,
            }
            if _status_rank(status) > _status_rank(highest) or (
                status == highest and max(z_score, ewma_z) > primary_z
            ):
                highest = status
                primary_driver = driver
                primary_z = max(z_score, ewma_z)
        if index < baseline_windows:
            highest = MonitoringStatus.NORMAL
            primary_driver = "NONE"
            for values in metrics.values():
                values["status"] = MonitoringStatus.NORMAL.value
        secondary = [
            metric.upper()
            for metric, values in metrics.items()
            if values["status"] != MonitoringStatus.NORMAL.value and metric.upper() != primary_driver
        ]
        metric_results.append(
            {
                **row.to_dict(),
                "status": highest.value,
                "primary_driver": primary_driver,
                "secondary_drivers": secondary,
                "recommended_action": recommended_action(primary_driver),
                "metrics": metrics,
            }
        )
    return pd.DataFrame(metric_results), baseline


def latest_monitoring_result(evaluated_windows: pd.DataFrame) -> dict[str, Any]:
    if evaluated_windows.empty:
        raise ValueError("No monitoring windows are available.")
    row = evaluated_windows.iloc[-1].to_dict()
    metrics = row["metrics"]
    return {
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "scenario_type": row["scenario_type"],
        "status": row["status"],
        "primary_driver": row["primary_driver"],
        "current_review_rate": metrics["review_rate"]["current"],
        "baseline_review_rate": metrics["review_rate"]["baseline"],
        "current_incident_rate": metrics["payment_incident_rate"]["current"],
        "baseline_incident_rate": metrics["payment_incident_rate"]["baseline"],
        "recommended_action": row["recommended_action"],
        "metrics": metrics,
    }


def alert_rows(evaluated_windows: pd.DataFrame) -> list[dict[str, Any]]:
    alerts = []
    for index, row in evaluated_windows.iterrows():
        if row["status"] == MonitoringStatus.NORMAL.value:
            continue
        metric_key = row["primary_driver"].lower()
        values = row["metrics"].get(metric_key, {})
        alerts.append(
            {
                "alert_id": f"alert_{index:04d}",
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "severity": row["status"],
                "primary_driver": row["primary_driver"],
                "current_value": values.get("current"),
                "baseline_value": values.get("baseline"),
                "z_score": values.get("z_score"),
                "relative_change": values.get("relative_change"),
                "recommended_action": row["recommended_action"],
                "scenario_type": row["scenario_type"],
            }
        )
    return alerts


def evaluate_binary_spike_detection(evaluated_windows: pd.DataFrame) -> dict[str, Any]:
    expected = evaluated_windows["expected_spike"].astype(bool)
    predicted = evaluated_windows["status"] != MonitoringStatus.NORMAL.value
    tp = int((expected & predicted).sum())
    fp = int((~expected & predicted).sum())
    tn = int((~expected & ~predicted).sum())
    fn = int((expected & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normal_windows = int((~expected).sum())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_alert_rate": fp / normal_windows if normal_windows else 0.0,
    }


def detection_delay(evaluated_windows: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for scenario, group in evaluated_windows.groupby("scenario_type", sort=False):
        if not bool(group["expected_spike"].any()):
            continue
        detected = group[group["status"] != MonitoringStatus.NORMAL.value]
        if detected.empty:
            continue
        delay = int(detected.index[0] - group.index.min())
        rows.append({"scenario_type": scenario, "delay_windows": delay, "delay_minutes": delay * 15})
    delays = [item["delay_minutes"] for item in rows]
    return {
        "by_scenario": rows,
        "mean_detection_delay_minutes": float(np.mean(delays)) if delays else None,
        "median_detection_delay_minutes": float(np.median(delays)) if delays else None,
        "max_detection_delay_minutes": int(max(delays)) if delays else None,
    }


def scenario_performance(evaluated_windows: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for scenario, group in evaluated_windows.groupby("scenario_type", sort=False):
        expected = bool(group["expected_spike"].any())
        detected = bool((group["status"] != MonitoringStatus.NORMAL.value).any())
        rows.append(
            {
                "scenario_type": scenario,
                "expected_spike": expected,
                "detected": detected,
                "alert_windows": int((group["status"] != MonitoringStatus.NORMAL.value).sum()),
                "total_windows": int(len(group)),
                "max_status": max(group["status"], key=lambda item: _status_rank(MonitoringStatus(item))),
            }
        )
    return rows


def relative_change(current: float, baseline: float) -> float:
    if abs(baseline) < 1e-9:
        return 0.0 if abs(current) < 1e-9 else 10.0
    return (current - baseline) / baseline


def recommended_action(primary_driver: str) -> str:
    return ACTION_GUIDANCE.get(primary_driver, "Continue monitoring current operating conditions.")


def _safe_std(std: float, mean: float, metric: str) -> float:
    if std > 1e-9:
        return std
    if metric.endswith("_rate"):
        return max(mean * 0.25, 0.01)
    return max(abs(mean) * 0.25, 0.01)


def _status_from_z(z_score: float) -> MonitoringStatus:
    if z_score >= 4.0:
        return MonitoringStatus.CRITICAL
    if z_score >= 3.0:
        return MonitoringStatus.HIGH
    if z_score >= 2.0:
        return MonitoringStatus.ELEVATED
    return MonitoringStatus.NORMAL


def _status_rank(status: MonitoringStatus) -> int:
    return {
        MonitoringStatus.NORMAL: 0,
        MonitoringStatus.ELEVATED: 1,
        MonitoringStatus.HIGH: 2,
        MonitoringStatus.CRITICAL: 3,
    }[status]
