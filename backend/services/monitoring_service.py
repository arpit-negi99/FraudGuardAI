from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.monitoring.schema import DEFAULT_BASELINE_WINDOWS, DEFAULT_WINDOW_MINUTES, MonitoringScenario
from src.monitoring.simulator import generate_monitoring_stream
from src.monitoring.spike import (
    alert_rows,
    detection_delay,
    evaluate_binary_spike_detection,
    evaluate_monitoring_windows,
    latest_monitoring_result,
    scenario_performance,
)
from src.monitoring.windows import aggregate_windows


ROOT = Path(__file__).resolve().parents[2]
STREAM_PATH = ROOT / "data/synthetic/monitoring_stream.csv"


@lru_cache(maxsize=1)
def get_stream() -> pd.DataFrame:
    if STREAM_PATH.exists():
        return pd.read_csv(STREAM_PATH)
    return generate_monitoring_stream()


@lru_cache(maxsize=1)
def get_monitoring_bundle() -> dict[str, Any]:
    windows = aggregate_windows(get_stream(), window_minutes=DEFAULT_WINDOW_MINUTES)
    evaluated, baseline = evaluate_monitoring_windows(
        windows,
        baseline_windows=DEFAULT_BASELINE_WINDOWS,
    )
    return {
        "windows": evaluated,
        "baseline": baseline,
        "metrics": evaluate_binary_spike_detection(evaluated),
        "delays": detection_delay(evaluated),
        "scenario_metrics": scenario_performance(evaluated),
    }


def summary() -> dict[str, Any]:
    bundle = get_monitoring_bundle()
    windows = bundle["windows"]
    metrics = bundle["metrics"]
    return {
        "stream_rows": int(len(get_stream())),
        "window_count": int(len(windows)),
        "window_size_minutes": DEFAULT_WINDOW_MINUTES,
        "baseline_windows": DEFAULT_BASELINE_WINDOWS,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "false_alert_rate": metrics["false_alert_rate"],
        "latest": latest_monitoring_result(windows),
        "detection_delay": bundle["delays"],
    }


def current(scenario_type: str | None = None) -> dict[str, Any]:
    windows = _filtered_windows(scenario_type)
    return latest_monitoring_result(windows)


def windows(scenario_type: str | None = None, limit: int = 120) -> dict[str, Any]:
    rows = _filtered_windows(scenario_type)
    page = rows.tail(min(max(limit, 1), 500))
    return {
        "window_count": int(len(rows)),
        "windows": [_window_row(row) for row in page.to_dict(orient="records")],
    }


def alerts(scenario_type: str | None = None, limit: int = 50) -> dict[str, Any]:
    rows = alert_rows(_filtered_windows(scenario_type))
    return {"alerts": rows[-min(max(limit, 1), 200) :]}


def scenarios() -> dict[str, Any]:
    bundle = get_monitoring_bundle()
    return {
        "scenarios": [item.value for item in MonitoringScenario],
        "scenario_metrics": bundle["scenario_metrics"],
    }


def evaluate_custom(records: list[dict[str, Any]], window_minutes: int = DEFAULT_WINDOW_MINUTES) -> dict[str, Any]:
    data = pd.DataFrame(records)
    windows = aggregate_windows(data, window_minutes=window_minutes)
    evaluated, _ = evaluate_monitoring_windows(windows)
    return {
        "summary": latest_monitoring_result(evaluated),
        "metrics": evaluate_binary_spike_detection(evaluated),
    }


def _filtered_windows(scenario_type: str | None) -> pd.DataFrame:
    windows = get_monitoring_bundle()["windows"]
    if not scenario_type:
        return windows
    valid = {item.value for item in MonitoringScenario}
    if scenario_type not in valid:
        raise ValueError(f"Unknown monitoring scenario: {scenario_type}")
    filtered = windows[windows["scenario_type"] == scenario_type]
    if filtered.empty:
        raise ValueError(f"No monitoring windows found for scenario: {scenario_type}")
    return filtered


def _window_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    return {
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "scenario_type": row["scenario_type"],
        "expected_spike": bool(row["expected_spike"]),
        "status": row["status"],
        "primary_driver": row["primary_driver"],
        "review_rate": row["review_rate"],
        "payment_incident_rate": row["payment_incident_rate"],
        "critical_high_incident_rate": row["critical_high_incident_rate"],
        "debit_service_mismatch_rate": row["debit_service_mismatch_rate"],
        "complaint_escalation_rate": row["complaint_escalation_rate"],
        "retry_risk_rate": row["retry_risk_rate"],
        "recommended_action": row["recommended_action"],
        "primary_metric": metrics.get(str(row["primary_driver"]).lower(), {}),
    }
