from __future__ import annotations

from src.incidents.schema import (
    GatewayStatus,
    IncidentType,
    OrderStatus,
    PaymentEvent,
    PaymentIncidentResult,
    RecommendedAction,
    RefundStatus,
    Severity,
    parse_payment_event,
)


SIGNIFICANT_UNRESOLVED_MINUTES = 30
HIGH_RETRY_COUNT = 3


def evaluate_payment_incident(event: PaymentEvent | dict[str, object]) -> PaymentIncidentResult:
    """Evaluate deterministic payment lifecycle incident rules.

    The rules are operational consistency checks, not a payment-gateway action system.
    They do not call Module 1 fraud scoring and do not mutate payment state.
    """
    payment = parse_payment_event(event)
    checks = [
        _complaint_escalation(payment),
        _refund_required(payment),
        _captured_but_unfulfilled(payment),
        _retry_related_payment_risk(payment),
        _debit_service_mismatch(payment),
        _late_authorization_risk(payment),
    ]
    incidents = [check for check in checks if check is not None]
    if not incidents:
        return PaymentIncidentResult(
            payment_id=payment.payment_id,
            incident_detected=False,
            incident_type=IncidentType.NONE,
            severity=Severity.NONE,
            recommended_action=RecommendedAction.NO_ACTION,
            reasons=[],
        )
    incidents.sort(key=lambda item: _severity_rank(item.severity), reverse=True)
    primary = incidents[0]
    reasons: list[str] = []
    for incident in incidents:
        for reason in incident.reasons:
            if reason not in reasons:
                reasons.append(reason)
    return PaymentIncidentResult(
        payment_id=payment.payment_id,
        incident_detected=True,
        incident_type=primary.incident_type,
        severity=primary.severity,
        recommended_action=primary.recommended_action,
        reasons=reasons,
    )


def _complaint_escalation(payment: PaymentEvent) -> PaymentIncidentResult | None:
    unresolved = _payment_unresolved(payment) or not payment.service_delivered
    refund_missing = payment.refund_status in {RefundStatus.NONE, RefundStatus.FAILED}
    if payment.customer_complaint and unresolved and refund_missing:
        severity = (
            Severity.CRITICAL
            if payment.bank_debited
            and payment.gateway_status in {GatewayStatus.AUTHORIZED, GatewayStatus.CAPTURED}
            else Severity.HIGH
        )
        return _result(
            payment,
            IncidentType.COMPLAINT_ESCALATION_RISK,
            severity,
            RecommendedAction.ESCALATE_REVIEW,
            [
                "Customer complaint is present",
                "Payment or service state is unresolved",
                "Refund is not processed",
            ],
        )
    return None


def _captured_but_unfulfilled(payment: PaymentEvent) -> PaymentIncidentResult | None:
    if (
        payment.gateway_status == GatewayStatus.CAPTURED
        and not payment.service_delivered
        and payment.order_status != OrderStatus.FULFILLED
        and payment.refund_status != RefundStatus.PROCESSED
    ):
        return _result(
            payment,
            IncidentType.CAPTURED_BUT_UNFULFILLED,
            Severity.HIGH,
            RecommendedAction.CHECK_ORDER,
            [
                "Gateway status is captured",
                "Service is not delivered",
                "Order is not fulfilled",
            ],
        )
    return None


def _refund_required(payment: PaymentEvent) -> PaymentIncidentResult | None:
    if (
        payment.gateway_status in {GatewayStatus.AUTHORIZED, GatewayStatus.CAPTURED}
        and payment.order_status in {OrderStatus.FAILED, OrderStatus.CANCELLED}
        and not payment.service_delivered
        and payment.refund_status == RefundStatus.NONE
    ):
        return _result(
            payment,
            IncidentType.REFUND_REQUIRED,
            Severity.HIGH,
            RecommendedAction.INITIATE_REFUND,
            [
                "Payment is authorized or captured",
                "Order failed or was cancelled",
                "Refund has not been initiated",
            ],
        )
    return None


def _debit_service_mismatch(payment: PaymentEvent) -> PaymentIncidentResult | None:
    if (
        payment.bank_debited
        and not payment.service_delivered
        and (
            payment.gateway_status == GatewayStatus.FAILED
            or payment.order_status in {OrderStatus.FAILED, OrderStatus.CANCELLED}
        )
        and payment.refund_status != RefundStatus.PROCESSED
    ):
        severity = (
            Severity.HIGH
            if payment.time_since_payment_minutes >= SIGNIFICANT_UNRESOLVED_MINUTES
            else Severity.MEDIUM
        )
        return _result(
            payment,
            IncidentType.DEBIT_SERVICE_MISMATCH,
            severity,
            RecommendedAction.VERIFY_PAYMENT,
            [
                "Customer debit recorded",
                f"Gateway status is {payment.gateway_status.value}",
                "Service is not fulfilled",
            ],
        )
    return None


def _late_authorization_risk(payment: PaymentEvent) -> PaymentIncidentResult | None:
    if (
        payment.gateway_status in {GatewayStatus.PENDING, GatewayStatus.FAILED}
        and not payment.callback_received
        and payment.time_since_payment_minutes >= SIGNIFICANT_UNRESOLVED_MINUTES
        and payment.retry_count > 0
        and payment.refund_status != RefundStatus.PROCESSED
    ):
        return _result(
            payment,
            IncidentType.LATE_AUTHORIZATION_RISK,
            Severity.MEDIUM,
            RecommendedAction.MONITOR,
            [
                "Gateway status is pending or failed",
                "Callback is missing",
                "Payment has been unresolved for at least 30 minutes",
                "Customer retry activity is present",
            ],
        )
    return None


def _retry_related_payment_risk(payment: PaymentEvent) -> PaymentIncidentResult | None:
    if (
        payment.retry_count >= HIGH_RETRY_COUNT
        and payment.bank_debited
        and _payment_unresolved(payment)
        and payment.refund_status != RefundStatus.PROCESSED
    ):
        return _result(
            payment,
            IncidentType.RETRY_RELATED_PAYMENT_RISK,
            Severity.MEDIUM,
            RecommendedAction.VERIFY_PAYMENT,
            [
                "Retry count is unusually high",
                "Customer debit recorded",
                "Earlier payment state remains unresolved",
            ],
        )
    return None


def _payment_unresolved(payment: PaymentEvent) -> bool:
    return payment.gateway_status in {
        GatewayStatus.CREATED,
        GatewayStatus.PENDING,
        GatewayStatus.FAILED,
    } or payment.order_status in {
        OrderStatus.CREATED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
    }


def _result(
    payment: PaymentEvent,
    incident_type: IncidentType,
    severity: Severity,
    action: RecommendedAction,
    reasons: list[str],
) -> PaymentIncidentResult:
    return PaymentIncidentResult(
        payment_id=payment.payment_id,
        incident_detected=True,
        incident_type=incident_type,
        severity=severity,
        recommended_action=action,
        reasons=reasons,
    )


def _severity_rank(severity: Severity) -> int:
    return {
        Severity.NONE: 0,
        Severity.LOW: 1,
        Severity.MEDIUM: 2,
        Severity.HIGH: 3,
        Severity.CRITICAL: 4,
    }[severity]
