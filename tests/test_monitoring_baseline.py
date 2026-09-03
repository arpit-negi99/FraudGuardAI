from __future__ import annotations

import pandas as pd
import pytest

from src.monitoring.baseline import build_historical_baseline
from src.monitoring.schema import MONITORED_METRICS
from src.monitoring.spike import relative_change
from src.monitoring.windows import aggregate_windows


def test_window_aggregation_calculates_core_metrics() -> None:
    records = pd.DataFrame(
        [
            _record("2026-08-27T10:00:00", 0.95, "REVIEW", True, "DEBIT_SERVICE_MISMATCH", "HIGH"),
            _record("2026-08-27T10:04:00", 0.10, "ALLOW", False, "NORMAL_PAYMENT", "NONE"),
        ]
    )

    windows = aggregate_windows(records)

    assert len(windows) == 1
    assert windows.iloc[0]["transaction_count"] == 2
    assert windows.iloc[0]["review_rate"] == 0.5
    assert windows.iloc[0]["payment_incident_rate"] == 0.5
    assert windows.iloc[0]["debit_service_mismatch_count"] == 1


def test_empty_monitoring_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_windows(pd.DataFrame())


def test_baseline_uses_initial_windows_only_without_future_leakage() -> None:
    windows = pd.DataFrame([{**{metric: 0.1 for metric in MONITORED_METRICS}, "window_start": i} for i in range(4)])
    windows.loc[3, "review_rate"] = 0.9

    baseline = build_historical_baseline(windows, baseline_windows=3)

    assert baseline["review_rate"]["mean"] == pytest.approx(0.1)


def test_std_zero_baseline_is_valid_input_for_spike_logic() -> None:
    windows = pd.DataFrame([{**{metric: 0.0 for metric in MONITORED_METRICS}, "window_start": i} for i in range(3)])

    baseline = build_historical_baseline(windows, baseline_windows=3)

    assert baseline["review_rate"]["std"] == 0.0


def test_relative_change_handles_zero_baseline() -> None:
    assert relative_change(0.0, 0.0) == 0.0
    assert relative_change(0.2, 0.0) == 10.0


def _record(time: str, risk: float, decision: str, incident: bool, incident_type: str, severity: str) -> dict:
    return {
        "event_id": time,
        "event_time": time,
        "transaction_id": 1,
        "payment_id": "pay",
        "merchant_id": "merchant_001",
        "fraud_risk_score": risk,
        "fraud_decision": decision,
        "payment_incident_detected": incident,
        "payment_incident_type": incident_type,
        "payment_incident_severity": severity,
        "amount": 10.0,
        "payment_method": "upi",
        "scenario_type": "NORMAL",
        "expected_spike": False,
    }
