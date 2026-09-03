from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MonitoringScenario(StrEnum):
    NORMAL = "NORMAL"
    FRAUD_RISK_SPIKE = "FRAUD_RISK_SPIKE"
    PAYMENT_INCIDENT_SPIKE = "PAYMENT_INCIDENT_SPIKE"
    DEBIT_SERVICE_MISMATCH_SPIKE = "DEBIT_SERVICE_MISMATCH_SPIKE"
    COMPLAINT_SPIKE = "COMPLAINT_SPIKE"
    RETRY_SPIKE = "RETRY_SPIKE"
    MIXED_RISK_SPIKE = "MIXED_RISK_SPIKE"
    RECOVERY = "RECOVERY"


class MonitoringStatus(StrEnum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


VERY_HIGH_RISK_THRESHOLD = 0.90
DEFAULT_WINDOW_MINUTES = 15
DEFAULT_BASELINE_WINDOWS = 50
DEFAULT_EWMA_ALPHA = 0.50


MONITORED_METRICS = [
    "mean_fraud_risk",
    "review_rate",
    "very_high_risk_rate",
    "payment_incident_rate",
    "critical_high_incident_rate",
    "debit_service_mismatch_rate",
    "complaint_escalation_rate",
    "retry_risk_rate",
    "captured_unfulfilled_rate",
]


@dataclass(frozen=True)
class MonitoringRecord:
    event_id: str
    event_time: str
    transaction_id: int | None
    payment_id: str | None
    merchant_id: str
    fraud_risk_score: float | None
    fraud_decision: str | None
    payment_incident_detected: bool | None
    payment_incident_type: str | None
    payment_incident_severity: str | None
    amount: float
    payment_method: str
    scenario_type: MonitoringScenario
    expected_spike: bool
