from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.monitoring.schema import MonitoringScenario, VERY_HIGH_RISK_THRESHOLD


DEFAULT_MONITORING_SEED = 314
DEFAULT_EVENTS_PER_WINDOW = 80
SYNTHETIC_MONITORING_NOTE = "Synthetic monitoring time; not live merchant or payment-provider traffic."
SCENARIO_WINDOWS = {
    MonitoringScenario.NORMAL.value: 80,
    MonitoringScenario.FRAUD_RISK_SPIKE.value: 35,
    MonitoringScenario.PAYMENT_INCIDENT_SPIKE.value: 35,
    MonitoringScenario.DEBIT_SERVICE_MISMATCH_SPIKE.value: 30,
    MonitoringScenario.COMPLAINT_SPIKE.value: 25,
    MonitoringScenario.RETRY_SPIKE.value: 25,
    MonitoringScenario.MIXED_RISK_SPIKE.value: 40,
    MonitoringScenario.RECOVERY.value: 40,
}


def generate_monitoring_stream(
    events_per_window: int = DEFAULT_EVENTS_PER_WINDOW,
    random_seed: int = DEFAULT_MONITORING_SEED,
) -> pd.DataFrame:
    """Generate scenario-first synthetic monitoring records."""
    if events_per_window <= 0:
        raise ValueError("events_per_window must be positive.")
    rng = np.random.default_rng(random_seed)
    start = datetime(2026, 8, 27, 10, 0, 0)
    rows = []
    event_index = 1
    window_index = 0
    for scenario, window_count in SCENARIO_WINDOWS.items():
        for _ in range(window_count):
            window_start = start + timedelta(minutes=15 * window_index)
            for item in range(events_per_window):
                event_time = window_start + timedelta(seconds=int(item * 900 / events_per_window))
                rows.append(_record(event_index, event_time, scenario, rng))
                event_index += 1
            window_index += 1
    return pd.DataFrame(rows)


def _record(index: int, event_time: datetime, scenario: str, rng: np.random.Generator) -> dict:
    rates = _scenario_rates(scenario)
    fraud_risk = _fraud_score(rates["fraud_level"], rng)
    review = fraud_risk >= 0.60 or rng.random() < rates["extra_review_rate"]
    incident_type = _incident_type(scenario, rng)
    incident_detected = incident_type != "NORMAL_PAYMENT"
    severity = _severity_for_incident(incident_type, rng)
    return {
        "event_id": f"mon_evt_{index:07d}",
        "event_time": event_time.isoformat(),
        "transaction_id": 10_000_000 + index,
        "payment_id": f"pay_mon_{index:07d}",
        "merchant_id": f"merchant_{int(rng.integers(1, 121)):03d}",
        "fraud_risk_score": round(float(fraud_risk), 6),
        "fraud_decision": "REVIEW" if review else "ALLOW",
        "payment_incident_detected": incident_detected,
        "payment_incident_type": incident_type,
        "payment_incident_severity": severity,
        "amount": round(float(rng.lognormal(mean=4.1, sigma=0.85)), 2),
        "payment_method": str(rng.choice(["card", "upi", "netbanking", "wallet"])),
        "scenario_type": scenario,
        "expected_spike": scenario not in {MonitoringScenario.NORMAL.value, MonitoringScenario.RECOVERY.value},
        "synthetic_monitoring_note": SYNTHETIC_MONITORING_NOTE,
    }


def _scenario_rates(scenario: str) -> dict[str, float]:
    if scenario == MonitoringScenario.NORMAL.value:
        return {"fraud_level": 0.11, "extra_review_rate": 0.01}
    if scenario == MonitoringScenario.RECOVERY.value:
        return {"fraud_level": 0.12, "extra_review_rate": 0.01}
    if scenario == MonitoringScenario.FRAUD_RISK_SPIKE.value:
        return {"fraud_level": 0.34, "extra_review_rate": 0.03}
    if scenario == MonitoringScenario.MIXED_RISK_SPIKE.value:
        return {"fraud_level": 0.31, "extra_review_rate": 0.03}
    return {"fraud_level": 0.13, "extra_review_rate": 0.01}


def _fraud_score(level: float, rng: np.random.Generator) -> float:
    if rng.random() < level:
        return rng.beta(7.0, 1.6)
    return rng.beta(1.2, 10.5)


def _incident_type(scenario: str, rng: np.random.Generator) -> str:
    distributions = {
        MonitoringScenario.NORMAL.value: [
            ("NORMAL_PAYMENT", 0.95),
            ("DEBIT_SERVICE_MISMATCH", 0.02),
            ("CAPTURED_BUT_UNFULFILLED", 0.015),
            ("RETRY_RELATED_PAYMENT_RISK", 0.01),
            ("COMPLAINT_ESCALATION_RISK", 0.005),
        ],
        MonitoringScenario.RECOVERY.value: [
            ("NORMAL_PAYMENT", 0.94),
            ("DEBIT_SERVICE_MISMATCH", 0.025),
            ("CAPTURED_BUT_UNFULFILLED", 0.02),
            ("RETRY_RELATED_PAYMENT_RISK", 0.01),
            ("COMPLAINT_ESCALATION_RISK", 0.005),
        ],
        MonitoringScenario.PAYMENT_INCIDENT_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.78),
            ("DEBIT_SERVICE_MISMATCH", 0.07),
            ("CAPTURED_BUT_UNFULFILLED", 0.06),
            ("RETRY_RELATED_PAYMENT_RISK", 0.04),
            ("COMPLAINT_ESCALATION_RISK", 0.05),
        ],
        MonitoringScenario.DEBIT_SERVICE_MISMATCH_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.82),
            ("DEBIT_SERVICE_MISMATCH", 0.15),
            ("CAPTURED_BUT_UNFULFILLED", 0.015),
            ("RETRY_RELATED_PAYMENT_RISK", 0.01),
            ("COMPLAINT_ESCALATION_RISK", 0.005),
        ],
        MonitoringScenario.COMPLAINT_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.84),
            ("DEBIT_SERVICE_MISMATCH", 0.03),
            ("CAPTURED_BUT_UNFULFILLED", 0.02),
            ("RETRY_RELATED_PAYMENT_RISK", 0.02),
            ("COMPLAINT_ESCALATION_RISK", 0.09),
        ],
        MonitoringScenario.RETRY_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.84),
            ("DEBIT_SERVICE_MISMATCH", 0.025),
            ("CAPTURED_BUT_UNFULFILLED", 0.02),
            ("RETRY_RELATED_PAYMENT_RISK", 0.11),
            ("COMPLAINT_ESCALATION_RISK", 0.005),
        ],
        MonitoringScenario.FRAUD_RISK_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.94),
            ("DEBIT_SERVICE_MISMATCH", 0.025),
            ("CAPTURED_BUT_UNFULFILLED", 0.015),
            ("RETRY_RELATED_PAYMENT_RISK", 0.01),
            ("COMPLAINT_ESCALATION_RISK", 0.01),
        ],
        MonitoringScenario.MIXED_RISK_SPIKE.value: [
            ("NORMAL_PAYMENT", 0.74),
            ("DEBIT_SERVICE_MISMATCH", 0.11),
            ("CAPTURED_BUT_UNFULFILLED", 0.05),
            ("RETRY_RELATED_PAYMENT_RISK", 0.05),
            ("COMPLAINT_ESCALATION_RISK", 0.05),
        ],
    }
    options = distributions[scenario]
    names = np.array([item[0] for item in options])
    probabilities = np.array([item[1] for item in options], dtype=float)
    return str(rng.choice(names, p=probabilities / probabilities.sum()))


def _severity_for_incident(incident_type: str, rng: np.random.Generator) -> str:
    if incident_type == "NORMAL_PAYMENT":
        return "NONE"
    if incident_type == "COMPLAINT_ESCALATION_RISK":
        return "CRITICAL"
    if incident_type in {"DEBIT_SERVICE_MISMATCH", "CAPTURED_BUT_UNFULFILLED"}:
        return "HIGH"
    if incident_type == "RETRY_RELATED_PAYMENT_RISK":
        return "MEDIUM"
    return str(rng.choice(["MEDIUM", "HIGH"]))
