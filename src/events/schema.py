from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from backend.config import get_settings


EVENT_VERSION = 1
TRANSACTION_SCORED_EVENT = "transaction_scored"


@dataclass(frozen=True)
class RiskTransactionEvent:
    event_id: str
    event_version: int
    event_type: str
    event_time: str
    merchant_id: str
    transaction_id: int | None
    payment_id: str | None
    amount: float | None
    payment_method: str | None
    fraud_risk_score: float
    decision: str
    threshold: float
    priority: str
    payment_incident_detected: bool | None = None
    payment_incident_type: str | None = None
    payment_incident_severity: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_transaction_event(
    prediction: dict[str, Any],
    transaction: dict[str, Any],
    merchant_id: str | None = None,
) -> RiskTransactionEvent:
    """Build a versioned analytics event from an already-completed prediction."""
    return RiskTransactionEvent(
        event_id=str(uuid.uuid4()),
        event_version=EVENT_VERSION,
        event_type=TRANSACTION_SCORED_EVENT,
        event_time=datetime.now(tz=UTC).isoformat(),
        merchant_id=str(transaction.get("merchant_id") or merchant_id or get_settings().default_merchant_id),
        transaction_id=_optional_int(prediction.get("transaction_id") or transaction.get("TransactionID")),
        payment_id=_optional_text(transaction.get("payment_id")),
        amount=_optional_float(prediction.get("amount") or transaction.get("TransactionAmt")),
        payment_method=_optional_text(transaction.get("payment_method") or transaction.get("card4")),
        fraud_risk_score=float(prediction["risk_score"]),
        decision=str(prediction["decision"]),
        threshold=float(prediction["threshold"]),
        priority=str(prediction.get("priority") or "Unknown"),
        payment_incident_detected=_optional_bool(transaction.get("payment_incident_detected")),
        payment_incident_type=_optional_text(transaction.get("payment_incident_type")),
        payment_incident_severity=_optional_text(transaction.get("payment_incident_severity")),
    )


def validate_transaction_event(data: dict[str, Any]) -> RiskTransactionEvent:
    """Validate an incoming transaction event and return the normalized dataclass."""
    required = {
        "event_id",
        "event_version",
        "event_type",
        "event_time",
        "merchant_id",
        "fraud_risk_score",
        "decision",
        "threshold",
        "priority",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Risk event is missing required fields: {missing}")
    if int(data["event_version"]) != EVENT_VERSION:
        raise ValueError(f"Unsupported risk event version: {data['event_version']}")
    if data["event_type"] != TRANSACTION_SCORED_EVENT:
        raise ValueError(f"Unsupported risk event type: {data['event_type']}")
    if not str(data["event_id"]).strip():
        raise ValueError("event_id must be a non-empty UUID string.")
    uuid.UUID(str(data["event_id"]))
    risk_score = float(data["fraud_risk_score"])
    threshold = float(data["threshold"])
    if risk_score < 0.0 or risk_score > 1.0:
        raise ValueError("fraud_risk_score must be between 0.0 and 1.0.")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0.")
    if data["decision"] not in {"ALLOW", "REVIEW"}:
        raise ValueError("decision must be ALLOW or REVIEW.")
    datetime.fromisoformat(str(data["event_time"]).replace("Z", "+00:00"))
    return RiskTransactionEvent(
        event_id=str(data["event_id"]),
        event_version=EVENT_VERSION,
        event_type=TRANSACTION_SCORED_EVENT,
        event_time=str(data["event_time"]),
        merchant_id=str(data["merchant_id"]),
        transaction_id=_optional_int(data.get("transaction_id")),
        payment_id=_optional_text(data.get("payment_id")),
        amount=_optional_float(data.get("amount")),
        payment_method=_optional_text(data.get("payment_method")),
        fraud_risk_score=risk_score,
        decision=str(data["decision"]),
        threshold=threshold,
        priority=str(data["priority"]),
        payment_incident_detected=_optional_bool(data.get("payment_incident_detected")),
        payment_incident_type=_optional_text(data.get("payment_incident_type")),
        payment_incident_severity=_optional_text(data.get("payment_incident_severity")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
