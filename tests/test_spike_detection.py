from __future__ import annotations

import pandas as pd

from src.monitoring.schema import MONITORED_METRICS, MonitoringStatus
from src.monitoring.simulator import generate_monitoring_stream
from src.monitoring.spike import (
    detection_delay,
    evaluate_binary_spike_detection,
    evaluate_monitoring_windows,
    recommended_action,
    scenario_performance,
)
from src.monitoring.windows import aggregate_windows


def test_stable_normal_stream_does_not_alert_excessively() -> None:
    records = generate_monitoring_stream(events_per_window=60, random_seed=314)
    normal = records[records["scenario_type"] == "NORMAL"]
    windows = aggregate_windows(normal)
    evaluated, _ = evaluate_monitoring_windows(windows, baseline_windows=50)
    metrics = evaluate_binary_spike_detection(evaluated)

    assert metrics["false_alert_rate"] <= 0.10


def test_fraud_spike_detected() -> None:
    assert _scenario_detected("FRAUD_RISK_SPIKE")


def test_payment_incident_spike_detected() -> None:
    assert _scenario_detected("PAYMENT_INCIDENT_SPIKE")


def test_mismatch_spike_detected() -> None:
    assert _scenario_detected("DEBIT_SERVICE_MISMATCH_SPIKE")


def test_complaint_spike_detected() -> None:
    assert _scenario_detected("COMPLAINT_SPIKE")


def test_retry_spike_detected() -> None:
    assert _scenario_detected("RETRY_SPIKE")


def test_mixed_spike_detected() -> None:
    assert _scenario_detected("MIXED_RISK_SPIKE")


def test_recovery_returns_to_normal_latest_window() -> None:
    evaluated = _evaluated()
    recovery = evaluated[evaluated["scenario_type"] == "RECOVERY"]

    assert recovery.iloc[-1]["status"] == MonitoringStatus.NORMAL.value


def test_highest_anomaly_determines_overall_status_and_driver() -> None:
    row = {metric: 0.0 for metric in MONITORED_METRICS}
    rows = []
    for index in range(52):
        rows.append({**row, "window_start": index, "window_end": index + 1, "scenario_type": "NORMAL", "expected_spike": False})
    rows.append({**row, "review_rate": 0.5, "window_start": 53, "window_end": 54, "scenario_type": "FRAUD_RISK_SPIKE", "expected_spike": True})
    evaluated, _ = evaluate_monitoring_windows(pd.DataFrame(rows), baseline_windows=50)

    assert evaluated.iloc[-1]["status"] in {"HIGH", "CRITICAL"}
    assert evaluated.iloc[-1]["primary_driver"] == "REVIEW_RATE"


def test_primary_driver_and_action_are_business_readable() -> None:
    assert "reconciliation" in recommended_action("DEBIT_SERVICE_MISMATCH_RATE")


def test_detection_delay_reported() -> None:
    delays = detection_delay(_evaluated())

    assert delays["mean_detection_delay_minutes"] == 0.0


def test_scenario_performance_contains_expected_scenarios() -> None:
    scenarios = {row["scenario_type"]: row for row in scenario_performance(_evaluated())}

    assert scenarios["NORMAL"]["expected_spike"] is False
    assert scenarios["MIXED_RISK_SPIKE"]["detected"] is True


def _evaluated() -> pd.DataFrame:
    windows = aggregate_windows(generate_monitoring_stream())
    evaluated, _ = evaluate_monitoring_windows(windows)
    return evaluated


def _scenario_detected(scenario: str) -> bool:
    evaluated = _evaluated()
    rows = evaluated[evaluated["scenario_type"] == scenario]
    return bool((rows["status"] != MonitoringStatus.NORMAL.value).any())
