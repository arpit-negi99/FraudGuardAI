from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from src.incidents.rules import evaluate_payment_incident
from src.incidents.schema import (
    GatewayStatus,
    IncidentType,
    OrderStatus,
    PaymentEvent,
    PaymentIncidentResult,
    PaymentMethod,
    RecommendedAction,
    RefundStatus,
    Severity,
)


class LifecycleEventType(StrEnum):
    PAYMENT_CREATED = "PAYMENT_CREATED"
    BANK_DEBITED = "BANK_DEBITED"
    CALLBACK_RECEIVED = "CALLBACK_RECEIVED"
    CALLBACK_MISSING = "CALLBACK_MISSING"
    PAYMENT_AUTHORIZED = "PAYMENT_AUTHORIZED"
    PAYMENT_CAPTURED = "PAYMENT_CAPTURED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    CUSTOMER_RETRY = "CUSTOMER_RETRY"
    ORDER_CONFIRMED = "ORDER_CONFIRMED"
    ORDER_FULFILLED = "ORDER_FULFILLED"
    ORDER_FAILED = "ORDER_FAILED"
    REFUND_INITIATED = "REFUND_INITIATED"
    REFUND_PROCESSED = "REFUND_PROCESSED"
    REFUND_FAILED = "REFUND_FAILED"
    CUSTOMER_COMPLAINT = "CUSTOMER_COMPLAINT"


class LifecycleStatus(StrEnum):
    NORMAL = "NORMAL"
    ACTIVE_INCIDENT = "ACTIVE_INCIDENT"
    RESOLVING = "RESOLVING"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class PaymentLifecycleEvent:
    event_id: str
    payment_id: str
    event_type: LifecycleEventType
    event_time_minutes: int
    gateway_status: GatewayStatus | None = None
    bank_debited: bool | None = None
    order_status: OrderStatus | None = None
    service_delivered: bool | None = None
    callback_received: bool | None = None
    refund_status: RefundStatus | None = None
    retry_count: int | None = None
    customer_complaint: bool | None = None


@dataclass(frozen=True)
class PaymentLifecycle:
    payment_id: str
    merchant_id: str
    amount: float
    payment_method: PaymentMethod
    fraud_risk_score: float | None
    events: list[PaymentLifecycleEvent]


def parse_payment_lifecycle(data: PaymentLifecycle | dict[str, Any]) -> PaymentLifecycle:
    """Validate and normalize a payment lifecycle mapping."""
    if isinstance(data, PaymentLifecycle):
        _validate_event_order(data.events)
        return data
    required = ["payment_id", "merchant_id", "amount", "payment_method", "events"]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Payment lifecycle is missing required fields: {missing}")
    events = [_parse_lifecycle_event(item, str(data["payment_id"])) for item in data["events"]]
    _validate_event_order(events)
    amount = float(data["amount"])
    if amount < 0:
        raise ValueError("amount must be non-negative.")
    score = data.get("fraud_risk_score")
    if score is not None:
        score = float(score)
        if score < 0.0 or score > 1.0:
            raise ValueError("fraud_risk_score must be between 0.0 and 1.0 when provided.")
    return PaymentLifecycle(
        payment_id=str(data["payment_id"]),
        merchant_id=str(data["merchant_id"]),
        amount=amount,
        payment_method=PaymentMethod(str(data["payment_method"])),
        fraud_risk_score=score,
        events=events,
    )


def reconstruct_payment_state(lifecycle: PaymentLifecycle | dict[str, Any]) -> PaymentEvent:
    """Replay all lifecycle events into the current snapshot understood by the rule engine."""
    parsed = parse_payment_lifecycle(lifecycle)
    state = _initial_state(parsed)
    for event in parsed.events:
        _apply_event(state, event)
    return _state_to_event(parsed, state)


def evaluate_payment_lifecycle(lifecycle: PaymentLifecycle | dict[str, Any]) -> dict[str, Any]:
    """Evaluate a payment timeline by replaying each event through snapshot rules."""
    parsed = parse_payment_lifecycle(lifecycle)
    state = _initial_state(parsed)
    timeline: list[dict[str, Any]] = []
    incident_history: list[dict[str, Any]] = []
    first_incident_time: int | None = None
    resolution_time: int | None = None
    highest_severity = Severity.NONE

    for event in parsed.events:
        _apply_event(state, event)
        snapshot = _state_to_event(parsed, state)
        result = _effective_incident_result(evaluate_payment_incident(snapshot), snapshot)
        if result.incident_detected and first_incident_time is None:
            first_incident_time = event.event_time_minutes
        if _severity_rank(result.severity) > _severity_rank(highest_severity):
            highest_severity = result.severity
        status = _lifecycle_status(result, first_incident_time, state)
        if status == LifecycleStatus.RESOLVED and resolution_time is None and first_incident_time is not None:
            resolution_time = event.event_time_minutes - first_incident_time
        item = {
            "time": event.event_time_minutes,
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "event_label": lifecycle_event_label(event.event_type),
            "status": status.value,
            "incident_type": result.incident_type.value,
            "severity": result.severity.value,
            "recommended_action": result.recommended_action.value,
            "reasons": list(result.reasons),
        }
        timeline.append(item)
        _append_history_if_changed(incident_history, item)

    current_state = _state_to_event(parsed, state)
    current_result = _effective_incident_result(evaluate_payment_incident(current_state), current_state)
    current_status = _lifecycle_status(current_result, first_incident_time, state)
    resolved = current_status == LifecycleStatus.RESOLVED
    return {
        "payment_id": parsed.payment_id,
        "merchant_id": parsed.merchant_id,
        "amount": parsed.amount,
        "payment_method": parsed.payment_method.value,
        "fraud_risk_score": parsed.fraud_risk_score,
        "status": current_status.value,
        "resolved": resolved,
        "current_state": current_state_to_dict(current_state),
        "current_incident": current_result.incident_type.value,
        "current_severity": current_result.severity.value,
        "highest_severity_observed": highest_severity.value,
        "first_incident_time_minutes": first_incident_time,
        "time_to_resolution_minutes": resolution_time if resolved else None,
        "recommended_action": current_result.recommended_action.value,
        "reasons": list(current_result.reasons),
        "events": [lifecycle_event_to_dict(event) for event in parsed.events],
        "timeline": timeline,
        "incident_history": incident_history,
    }


def lifecycle_event_to_dict(event: PaymentLifecycleEvent) -> dict[str, Any]:
    row = asdict(event)
    row["event_type"] = event.event_type.value
    for key in ["gateway_status", "order_status", "refund_status"]:
        if row[key] is not None:
            row[key] = row[key].value
    return row


def current_state_to_dict(event: PaymentEvent) -> dict[str, Any]:
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


def lifecycle_event_label(event_type: LifecycleEventType | str) -> str:
    return str(event_type).replace("LifecycleEventType.", "").replace("_", " ").title()


def _parse_lifecycle_event(data: dict[str, Any], payment_id: str) -> PaymentLifecycleEvent:
    required = ["event_id", "event_type", "event_time_minutes"]
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Lifecycle event is missing required fields: {missing}")
    event_payment_id = str(data.get("payment_id", payment_id))
    if event_payment_id != payment_id:
        raise ValueError("Lifecycle event payment_id must match lifecycle payment_id.")
    event_time = int(data["event_time_minutes"])
    if event_time < 0:
        raise ValueError("event_time_minutes must be non-negative.")
    return PaymentLifecycleEvent(
        event_id=str(data["event_id"]),
        payment_id=event_payment_id,
        event_type=LifecycleEventType(str(data["event_type"])),
        event_time_minutes=event_time,
        gateway_status=_optional_enum(data, "gateway_status", GatewayStatus),
        bank_debited=_optional_bool(data.get("bank_debited")),
        order_status=_optional_enum(data, "order_status", OrderStatus),
        service_delivered=_optional_bool(data.get("service_delivered")),
        callback_received=_optional_bool(data.get("callback_received")),
        refund_status=_optional_enum(data, "refund_status", RefundStatus),
        retry_count=_optional_non_negative_int(data.get("retry_count"), "retry_count"),
        customer_complaint=_optional_bool(data.get("customer_complaint")),
    )


def _validate_event_order(events: list[PaymentLifecycleEvent]) -> None:
    if not events:
        raise ValueError("Payment lifecycle must include at least one event.")
    previous = -1
    for event in events:
        if event.event_time_minutes < previous:
            raise ValueError("Lifecycle events must be ordered by non-decreasing event_time_minutes.")
        previous = event.event_time_minutes


def _initial_state(lifecycle: PaymentLifecycle) -> dict[str, Any]:
    return {
        "gateway_status": GatewayStatus.CREATED,
        "bank_debited": False,
        "order_status": OrderStatus.CREATED,
        "service_delivered": False,
        "callback_received": False,
        "refund_status": RefundStatus.NONE,
        "retry_count": 0,
        "customer_complaint": False,
        "time_since_payment_minutes": 0,
    }


def _apply_event(state: dict[str, Any], event: PaymentLifecycleEvent) -> None:
    state["time_since_payment_minutes"] = event.event_time_minutes
    if event.event_type == LifecycleEventType.PAYMENT_CREATED:
        state["gateway_status"] = GatewayStatus.CREATED
        state["order_status"] = OrderStatus.CREATED
    elif event.event_type == LifecycleEventType.BANK_DEBITED:
        state["bank_debited"] = True
    elif event.event_type == LifecycleEventType.CALLBACK_RECEIVED:
        state["callback_received"] = True
    elif event.event_type == LifecycleEventType.CALLBACK_MISSING:
        state["callback_received"] = False
    elif event.event_type == LifecycleEventType.PAYMENT_AUTHORIZED:
        state["gateway_status"] = GatewayStatus.AUTHORIZED
    elif event.event_type == LifecycleEventType.PAYMENT_CAPTURED:
        state["gateway_status"] = GatewayStatus.CAPTURED
        state["bank_debited"] = True
    elif event.event_type == LifecycleEventType.PAYMENT_FAILED:
        state["gateway_status"] = GatewayStatus.FAILED
    elif event.event_type == LifecycleEventType.CUSTOMER_RETRY:
        state["retry_count"] += 1
    elif event.event_type == LifecycleEventType.ORDER_CONFIRMED:
        state["order_status"] = OrderStatus.CONFIRMED
    elif event.event_type == LifecycleEventType.ORDER_FULFILLED:
        state["order_status"] = OrderStatus.FULFILLED
        state["service_delivered"] = True
    elif event.event_type == LifecycleEventType.ORDER_FAILED:
        state["order_status"] = OrderStatus.FAILED
        state["service_delivered"] = False
    elif event.event_type == LifecycleEventType.REFUND_INITIATED:
        state["refund_status"] = RefundStatus.PENDING
    elif event.event_type == LifecycleEventType.REFUND_PROCESSED:
        state["refund_status"] = RefundStatus.PROCESSED
        state["gateway_status"] = GatewayStatus.REFUNDED
    elif event.event_type == LifecycleEventType.REFUND_FAILED:
        state["refund_status"] = RefundStatus.FAILED
    elif event.event_type == LifecycleEventType.CUSTOMER_COMPLAINT:
        state["customer_complaint"] = True
    _apply_explicit_fields(state, event)


def _apply_explicit_fields(state: dict[str, Any], event: PaymentLifecycleEvent) -> None:
    for field in [
        "gateway_status",
        "bank_debited",
        "order_status",
        "service_delivered",
        "callback_received",
        "refund_status",
        "retry_count",
        "customer_complaint",
    ]:
        value = getattr(event, field)
        if value is not None:
            state[field] = value


def _state_to_event(lifecycle: PaymentLifecycle, state: dict[str, Any]) -> PaymentEvent:
    return PaymentEvent(
        payment_id=lifecycle.payment_id,
        merchant_id=lifecycle.merchant_id,
        amount=lifecycle.amount,
        payment_method=lifecycle.payment_method,
        bank_debited=bool(state["bank_debited"]),
        gateway_status=state["gateway_status"],
        order_status=state["order_status"],
        service_delivered=bool(state["service_delivered"]),
        callback_received=bool(state["callback_received"]),
        refund_status=state["refund_status"],
        retry_count=int(state["retry_count"]),
        time_since_payment_minutes=int(state["time_since_payment_minutes"]),
        customer_complaint=bool(state["customer_complaint"]),
        fraud_risk_score=lifecycle.fraud_risk_score,
    )


def _lifecycle_status(
    result: PaymentIncidentResult,
    first_incident_time: int | None,
    state: dict[str, Any],
) -> LifecycleStatus:
    if result.incident_detected:
        if state["refund_status"] == RefundStatus.PENDING:
            return LifecycleStatus.RESOLVING
        return LifecycleStatus.ACTIVE_INCIDENT
    if first_incident_time is not None and _still_operationally_unresolved(state):
        return LifecycleStatus.ACTIVE_INCIDENT
    if first_incident_time is not None:
        return LifecycleStatus.RESOLVED
    return LifecycleStatus.NORMAL


def _effective_incident_result(
    result: PaymentIncidentResult,
    snapshot: PaymentEvent,
) -> PaymentIncidentResult:
    if (
        result.incident_type == IncidentType.CAPTURED_BUT_UNFULFILLED
        and snapshot.order_status in {OrderStatus.CREATED, OrderStatus.CONFIRMED}
        and snapshot.time_since_payment_minutes < 30
        and not snapshot.customer_complaint
    ):
        return PaymentIncidentResult(
            payment_id=snapshot.payment_id,
            incident_detected=False,
            incident_type=IncidentType.NONE,
            severity=Severity.NONE,
            recommended_action=RecommendedAction.NO_ACTION,
            reasons=[],
        )
    return result


def _still_operationally_unresolved(state: dict[str, Any]) -> bool:
    return (
        state["refund_status"] != RefundStatus.PROCESSED
        and not bool(state["service_delivered"])
        and state["order_status"] != OrderStatus.FULFILLED
        and state["gateway_status"] in {
            GatewayStatus.AUTHORIZED,
            GatewayStatus.CAPTURED,
            GatewayStatus.PENDING,
            GatewayStatus.FAILED,
        }
    )


def _append_history_if_changed(history: list[dict[str, Any]], item: dict[str, Any]) -> None:
    current = {
        "time": item["time"],
        "incident_type": item["incident_type"],
        "severity": item["severity"],
        "status": item["status"],
        "recommended_action": item["recommended_action"],
    }
    if not history:
        history.append(current)
        return
    previous = history[-1]
    comparable = ["incident_type", "severity", "status", "recommended_action"]
    if any(previous[key] != current[key] for key in comparable):
        history.append(current)


def _severity_rank(severity: Severity) -> int:
    return {
        Severity.NONE: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }[severity]


def _optional_enum(data: dict[str, Any], key: str, enum_type):
    value = data.get(key)
    if value is None:
        return None
    return enum_type(str(value))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number
