from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PaymentMethod(StrEnum):
    CARD = "card"
    UPI = "upi"
    NETBANKING = "netbanking"
    WALLET = "wallet"


class GatewayStatus(StrEnum):
    CREATED = "created"
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"


class OrderStatus(StrEnum):
    CREATED = "created"
    CONFIRMED = "confirmed"
    FULFILLED = "fulfilled"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RefundStatus(StrEnum):
    NONE = "none"
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class IncidentType(StrEnum):
    NONE = "NORMAL_PAYMENT"
    DEBIT_SERVICE_MISMATCH = "DEBIT_SERVICE_MISMATCH"
    LATE_AUTHORIZATION_RISK = "LATE_AUTHORIZATION_RISK"
    CAPTURED_BUT_UNFULFILLED = "CAPTURED_BUT_UNFULFILLED"
    REFUND_REQUIRED = "REFUND_REQUIRED"
    RETRY_RELATED_PAYMENT_RISK = "RETRY_RELATED_PAYMENT_RISK"
    COMPLAINT_ESCALATION_RISK = "COMPLAINT_ESCALATION_RISK"


class Severity(StrEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendedAction(StrEnum):
    NO_ACTION = "NO_ACTION"
    VERIFY_PAYMENT = "VERIFY_PAYMENT"
    CHECK_ORDER = "CHECK_ORDER"
    INITIATE_REFUND = "INITIATE_REFUND"
    CONTACT_CUSTOMER = "CONTACT_CUSTOMER"
    ESCALATE_REVIEW = "ESCALATE_REVIEW"
    MONITOR = "MONITOR"


REQUIRED_PAYMENT_EVENT_FIELDS = (
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
)


@dataclass(frozen=True)
class PaymentEvent:
    payment_id: str
    merchant_id: str
    amount: float
    payment_method: PaymentMethod
    bank_debited: bool
    gateway_status: GatewayStatus
    order_status: OrderStatus
    service_delivered: bool
    callback_received: bool
    refund_status: RefundStatus
    retry_count: int
    time_since_payment_minutes: int
    customer_complaint: bool
    fraud_risk_score: float | None = None


@dataclass(frozen=True)
class PaymentIncidentResult:
    payment_id: str
    incident_detected: bool
    incident_type: IncidentType
    severity: Severity
    recommended_action: RecommendedAction
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "payment_id": self.payment_id,
            "incident_detected": self.incident_detected,
            "incident_type": self.incident_type.value,
            "severity": self.severity.value,
            "recommended_action": self.recommended_action.value,
            "reasons": list(self.reasons),
        }


def parse_payment_event(data: PaymentEvent | dict[str, object]) -> PaymentEvent:
    """Validate and normalize a payment-event mapping."""
    if isinstance(data, PaymentEvent):
        _validate_fraud_score(data.fraud_risk_score)
        return data
    missing = [field for field in REQUIRED_PAYMENT_EVENT_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Payment event is missing required fields: {missing}")
    event = PaymentEvent(
        payment_id=_required_str(data["payment_id"], "payment_id"),
        merchant_id=_required_str(data["merchant_id"], "merchant_id"),
        amount=_non_negative_float(data["amount"], "amount"),
        payment_method=PaymentMethod(str(data["payment_method"])),
        bank_debited=bool(data["bank_debited"]),
        gateway_status=GatewayStatus(str(data["gateway_status"])),
        order_status=OrderStatus(str(data["order_status"])),
        service_delivered=bool(data["service_delivered"]),
        callback_received=bool(data["callback_received"]),
        refund_status=RefundStatus(str(data["refund_status"])),
        retry_count=_non_negative_int(data["retry_count"], "retry_count"),
        time_since_payment_minutes=_non_negative_int(
            data["time_since_payment_minutes"],
            "time_since_payment_minutes",
        ),
        customer_complaint=bool(data["customer_complaint"]),
        fraud_risk_score=(
            None
            if data.get("fraud_risk_score") is None
            else float(data["fraud_risk_score"])
        ),
    )
    _validate_fraud_score(event.fraud_risk_score)
    return event


def _required_str(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return text


def _non_negative_float(value: object, field_name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number


def _non_negative_int(value: object, field_name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number


def _validate_fraud_score(value: float | None) -> None:
    if value is not None and (value < 0.0 or value > 1.0):
        raise ValueError("fraud_risk_score must be between 0.0 and 1.0 when provided.")
