from __future__ import annotations

import pandas as pd
import pytest

from src.incidents.rules import evaluate_payment_incident
from src.incidents.schema import IncidentType, RecommendedAction, Severity
from src.incidents.simulator import (
    SCENARIO_TO_INCIDENT_TYPE,
    generate_payment_incident_events,
    validate_no_impossible_combinations,
)


def base_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "payment_id": "pay_demo_001",
        "merchant_id": "merchant_001",
        "amount": 120.0,
        "payment_method": "upi",
        "bank_debited": True,
        "gateway_status": "captured",
        "order_status": "fulfilled",
        "service_delivered": True,
        "callback_received": True,
        "refund_status": "none",
        "retry_count": 0,
        "time_since_payment_minutes": 5,
        "customer_complaint": False,
    }
    event.update(overrides)
    return event


def test_captured_and_fulfilled_has_no_incident() -> None:
    result = evaluate_payment_incident(base_event())

    assert result.incident_detected is False
    assert result.incident_type == IncidentType.NONE
    assert result.severity == Severity.NONE
    assert result.recommended_action == RecommendedAction.NO_ACTION


def test_failed_and_not_debited_has_no_incident() -> None:
    result = evaluate_payment_incident(
        base_event(
            bank_debited=False,
            gateway_status="failed",
            order_status="failed",
            service_delivered=False,
        )
    )

    assert result.incident_detected is False


def test_debited_failed_unfulfilled_detects_debit_service_mismatch() -> None:
    result = evaluate_payment_incident(
        base_event(
            gateway_status="failed",
            order_status="failed",
            service_delivered=False,
            callback_received=False,
            time_since_payment_minutes=45,
        )
    )

    assert result.incident_detected is True
    assert result.incident_type == IncidentType.DEBIT_SERVICE_MISMATCH
    assert result.severity == Severity.HIGH
    assert "Customer debit recorded" in result.reasons


def test_captured_unfulfilled_is_high_priority_incident() -> None:
    result = evaluate_payment_incident(
        base_event(order_status="confirmed", service_delivered=False, time_since_payment_minutes=50)
    )

    assert result.incident_type == IncidentType.CAPTURED_BUT_UNFULFILLED
    assert result.severity == Severity.HIGH
    assert result.recommended_action == RecommendedAction.CHECK_ORDER


def test_complaint_unresolved_no_refund_escalates_to_critical_when_captured() -> None:
    result = evaluate_payment_incident(
        base_event(order_status="failed", service_delivered=False, customer_complaint=True)
    )

    assert result.incident_type == IncidentType.COMPLAINT_ESCALATION_RISK
    assert result.severity == Severity.CRITICAL
    assert result.recommended_action == RecommendedAction.ESCALATE_REVIEW


def test_refund_processed_resolves_refund_related_incident() -> None:
    result = evaluate_payment_incident(
        base_event(
            gateway_status="captured",
            order_status="cancelled",
            service_delivered=False,
            refund_status="processed",
            customer_complaint=False,
        )
    )

    assert result.incident_detected is False


def test_retry_related_payment_risk_detected_without_claiming_duplicate_charge() -> None:
    result = evaluate_payment_incident(
        base_event(
            gateway_status="pending",
            order_status="created",
            service_delivered=False,
            callback_received=False,
            retry_count=4,
            time_since_payment_minutes=25,
        )
    )

    assert result.incident_type == IncidentType.RETRY_RELATED_PAYMENT_RISK
    assert result.severity == Severity.MEDIUM
    assert any("Retry count is unusually high" in reason for reason in result.reasons)


def test_late_authorization_risk_detected_for_pending_missing_callback() -> None:
    result = evaluate_payment_incident(
        base_event(
            gateway_status="pending",
            order_status="created",
            service_delivered=False,
            callback_received=False,
            retry_count=1,
            time_since_payment_minutes=45,
        )
    )

    assert result.incident_type == IncidentType.LATE_AUTHORIZATION_RISK
    assert result.recommended_action == RecommendedAction.MONITOR


def test_optional_fraud_score_does_not_break_evaluation() -> None:
    without_score = evaluate_payment_incident(base_event())
    with_score = evaluate_payment_incident(base_event(fraud_risk_score=0.72))

    assert without_score.incident_detected is False
    assert with_score.incident_detected is False


def test_fraud_score_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="fraud_risk_score"):
        evaluate_payment_incident(base_event(fraud_risk_score=1.5))


def test_simulator_produces_required_columns() -> None:
    data = generate_payment_incident_events(row_count=100, random_seed=7)

    expected = {
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
        "incident_label",
        "incident_type",
        "synthetic_data_note",
    }
    assert expected.issubset(data.columns)


def test_simulator_is_deterministic_with_fixed_seed() -> None:
    first = generate_payment_incident_events(row_count=50, random_seed=11)
    second = generate_payment_incident_events(row_count=50, random_seed=11)

    pd.testing.assert_frame_equal(first, second)


def test_simulator_avoids_impossible_combinations() -> None:
    data = generate_payment_incident_events(row_count=500, random_seed=13)

    validate_no_impossible_combinations(data)
    assert not ((data["gateway_status"] == "refunded") & (data["refund_status"] == "none")).any()
    assert not (data["service_delivered"] & (data["order_status"] == "created")).any()


def test_refund_required_has_own_incident_type() -> None:
    result = evaluate_payment_incident(
        base_event(gateway_status="captured", order_status="cancelled", service_delivered=False)
    )

    assert result.incident_type == IncidentType.REFUND_REQUIRED
    assert result.recommended_action == RecommendedAction.INITIATE_REFUND


def test_synthetic_ground_truth_is_scenario_generated_not_rule_output() -> None:
    data = generate_payment_incident_events(row_count=200, random_seed=19)

    expected_from_scenario = data["scenario"].map(SCENARIO_TO_INCIDENT_TYPE)
    assert data["incident_type"].equals(expected_from_scenario)
    assert "scenario" in data.columns
