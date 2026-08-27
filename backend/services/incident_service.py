from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.incidents.rules import evaluate_payment_incident
from src.incidents.schema import (
    IncidentType,
    PaymentEvent,
    RecommendedAction,
    Severity,
    parse_payment_event,
)


ROOT = Path(__file__).resolve().parents[2]
INCIDENT_DATA_PATH = ROOT / "data/synthetic/payment_incidents.csv"
SYNTHETIC_NOTE = (
    "Synthetic payment-event data for demonstration only; not validated on "
    "payment-provider production data."
)
DETAIL_FIELDS = [
    "payment_id",
    "merchant_id",
    "amount",
    "payment_method",
    "bank_debited",
    "gateway_status",
    "order_status",
    "service_delivered",
    "callback_received",
    "refund_status",
    "retry_count",
    "time_since_payment_minutes",
    "customer_complaint",
    "fraud_risk_score",
]


@lru_cache(maxsize=1)
def get_incident_events() -> pd.DataFrame:
    """Load packaged synthetic payment events once per process."""
    if not INCIDENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Incident demo data not found: {INCIDENT_DATA_PATH}")
    return pd.read_csv(INCIDENT_DATA_PATH)


@lru_cache(maxsize=1)
def get_evaluated_incidents() -> list[dict[str, Any]]:
    """Evaluate packaged payment events with the deterministic rule engine."""
    rows: list[dict[str, Any]] = []
    for row in get_incident_events().to_dict(orient="records"):
        rows.append(_packaged_incident(row))
    return rows


def incident_types() -> dict[str, list[str]]:
    return {
        "incident_types": [item.value for item in IncidentType],
        "severities": [item.value for item in Severity],
        "recommended_actions": [item.value for item in RecommendedAction],
    }


def incident_summary() -> dict[str, Any]:
    rows = get_evaluated_incidents()
    total = len(rows)
    active = sum(1 for row in rows if row["incident_detected"])
    severity_counts = {severity.value: 0 for severity in Severity}
    type_counts = {incident_type.value: 0 for incident_type in IncidentType}
    for row in rows:
        severity_counts[row["severity"]] += 1
        type_counts[row["incident_type"]] += 1
    return {
        "total_payments": total,
        "active_incidents": active,
        "critical": severity_counts[Severity.CRITICAL.value],
        "high": severity_counts[Severity.HIGH.value],
        "medium": severity_counts[Severity.MEDIUM.value],
        "low": severity_counts[Severity.LOW.value],
        "normal": severity_counts[Severity.NONE.value],
        "incident_rate": active / total if total else 0.0,
        "severity_distribution": [
            {"severity": key, "payments": value}
            for key, value in severity_counts.items()
        ],
        "type_distribution": [
            {"incident_type": key, "payments": value}
            for key, value in type_counts.items()
        ],
        "data_note": SYNTHETIC_NOTE,
    }


def list_incidents(
    severity: str | None = None,
    incident_type: str | None = None,
    incident_detected: bool | None = None,
    payment_method: str | None = None,
    minimum_amount: float | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    rows = get_evaluated_incidents()
    filtered = [
        row
        for row in rows
        if _matches(row, severity, incident_type, incident_detected, payment_method, minimum_amount)
    ]
    bounded_limit = min(max(limit, 1), 500)
    bounded_offset = max(offset, 0)
    page = filtered[bounded_offset : bounded_offset + bounded_limit]
    return {
        "total": len(filtered),
        "limit": bounded_limit,
        "offset": bounded_offset,
        "incidents": [_summary_row(row) for row in page],
    }


def incident_detail(payment_id: str) -> dict[str, Any]:
    for row in get_evaluated_incidents():
        if row["payment_id"] == payment_id:
            return row
    raise KeyError(f"Payment incident {payment_id} was not found.")


def evaluate_event(payload: dict[str, Any]) -> dict[str, Any]:
    event = parse_payment_event(payload)
    result = evaluate_payment_incident(event).to_dict()
    return {**_event_to_public_dict(event), **result}


def _packaged_incident(row: dict[str, Any]) -> dict[str, Any]:
    event = parse_payment_event({field: row[field] for field in DETAIL_FIELDS})
    result = evaluate_payment_incident(event).to_dict()
    return {
        **_event_to_public_dict(event),
        **result,
        "scenario": row.get("scenario"),
        "synthetic_label": bool(row.get("incident_label", False)),
        "synthetic_incident_type": row.get("incident_type"),
        "data_note": SYNTHETIC_NOTE,
    }


def _event_to_public_dict(event: PaymentEvent) -> dict[str, Any]:
    return {
        "payment_id": event.payment_id,
        "merchant_id": event.merchant_id,
        "amount": event.amount,
        "payment_method": event.payment_method.value,
        "bank_debited": event.bank_debited,
        "gateway_status": event.gateway_status.value,
        "order_status": event.order_status.value,
        "service_delivered": event.service_delivered,
        "callback_received": event.callback_received,
        "refund_status": event.refund_status.value,
        "retry_count": event.retry_count,
        "time_since_payment_minutes": event.time_since_payment_minutes,
        "customer_complaint": event.customer_complaint,
        "fraud_risk_score": event.fraud_risk_score,
    }


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "payment_id": row["payment_id"],
        "merchant_id": row["merchant_id"],
        "amount": row["amount"],
        "payment_method": row["payment_method"],
        "incident_detected": row["incident_detected"],
        "incident_type": row["incident_type"],
        "severity": row["severity"],
        "recommended_action": row["recommended_action"],
        "fraud_risk_score": row["fraud_risk_score"],
    }


def _matches(
    row: dict[str, Any],
    severity: str | None,
    incident_type: str | None,
    incident_detected: bool | None,
    payment_method: str | None,
    minimum_amount: float | None,
) -> bool:
    if severity and row["severity"] != severity:
        return False
    if incident_type and row["incident_type"] != incident_type:
        return False
    if incident_detected is not None and row["incident_detected"] is not incident_detected:
        return False
    if payment_method and row["payment_method"] != payment_method:
        return False
    if minimum_amount is not None and row["amount"] < minimum_amount:
        return False
    return True
