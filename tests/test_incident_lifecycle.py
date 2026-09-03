from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from src.incidents.lifecycle import (
    LifecycleEventType,
    PaymentLifecycle,
    PaymentLifecycleEvent,
    evaluate_payment_lifecycle,
    reconstruct_payment_state,
)
from src.incidents.lifecycle_simulator import generate_payment_lifecycles
from src.incidents.schema import GatewayStatus, OrderStatus, PaymentMethod, RefundStatus


client = TestClient(app)


def test_lifecycle_event_order_accepts_non_decreasing_times() -> None:
    result = evaluate_payment_lifecycle(_lifecycle([("PAYMENT_CREATED", 0), ("BANK_DEBITED", 0)]))

    assert result["timeline"][0]["time"] == 0
    assert result["timeline"][1]["time"] == 0


def test_invalid_decreasing_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        evaluate_payment_lifecycle(_lifecycle([("PAYMENT_CREATED", 4), ("BANK_DEBITED", 2)]))


def test_state_reconstruction_replays_event_state() -> None:
    state = reconstruct_payment_state(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_CAPTURED", 4),
                ("ORDER_FULFILLED", 8),
            ]
        )
    )

    assert state.bank_debited is True
    assert state.gateway_status == GatewayStatus.CAPTURED
    assert state.order_status == OrderStatus.FULFILLED
    assert state.service_delivered is True


def test_normal_success_remains_normal() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_AUTHORIZED", 3),
                ("PAYMENT_CAPTURED", 4),
                ("ORDER_FULFILLED", 9),
            ]
        )
    )

    assert result["status"] == "NORMAL"
    assert result["current_severity"] == "NONE"


def test_safe_failure_remains_normal() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle([("PAYMENT_CREATED", 0), ("PAYMENT_FAILED", 4), ("CALLBACK_RECEIVED", 8)])
    )

    assert result["status"] == "NORMAL"


def test_debit_failure_becomes_incident() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_FAILED", 8),
                ("CALLBACK_MISSING", 35),
            ]
        )
    )

    assert result["status"] == "ACTIVE_INCIDENT"
    assert result["current_incident"] == "DEBIT_SERVICE_MISMATCH"
    assert result["current_severity"] == "HIGH"


def test_late_authorization_changes_replayed_state() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_FAILED", 8),
                ("CUSTOMER_RETRY", 18),
                ("PAYMENT_AUTHORIZED", 45),
            ]
        )
    )

    assert result["current_state"]["gateway_status"] == "authorized"
    assert result["status"] == "ACTIVE_INCIDENT"


def test_refund_processed_resolves_refund_related_incident() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_CAPTURED", 4),
                ("ORDER_FAILED", 12),
                ("REFUND_INITIATED", 52),
                ("REFUND_PROCESSED", 80),
            ]
        )
    )

    assert result["status"] == "RESOLVED"
    assert result["current_severity"] == "NONE"
    assert result["time_to_resolution_minutes"] == 68


def test_fulfilment_resolves_captured_unfulfilled_incident() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_CAPTURED", 4),
                ("CALLBACK_MISSING", 35),
                ("ORDER_FULFILLED", 50),
            ]
        )
    )

    assert result["status"] == "RESOLVED"
    assert result["current_state"]["service_delivered"] is True


def test_complaint_escalates_severity() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_CAPTURED", 4),
                ("ORDER_FAILED", 30),
                ("CUSTOMER_COMPLAINT", 75),
            ]
        )
    )

    assert result["current_incident"] == "COMPLAINT_ESCALATION_RISK"
    assert result["current_severity"] == "CRITICAL"


def test_first_detection_time_is_recorded() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle([("PAYMENT_CREATED", 0), ("BANK_DEBITED", 2), ("PAYMENT_FAILED", 8)])
    )

    assert result["first_incident_time_minutes"] == 8


def test_highest_severity_retained_after_resolution() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle(
            [
                ("PAYMENT_CREATED", 0),
                ("BANK_DEBITED", 2),
                ("PAYMENT_CAPTURED", 4),
                ("ORDER_FAILED", 12),
                ("REFUND_PROCESSED", 80),
            ]
        )
    )

    assert result["status"] == "RESOLVED"
    assert result["highest_severity_observed"] == "HIGH"
    assert result["current_severity"] == "NONE"


def test_duplicate_timeline_states_are_compressed_in_history() -> None:
    result = evaluate_payment_lifecycle(
        _lifecycle([("PAYMENT_CREATED", 0), ("CALLBACK_RECEIVED", 1), ("CALLBACK_RECEIVED", 2)])
    )

    assert len(result["timeline"]) == 3
    assert len(result["incident_history"]) == 1


def test_lifecycle_simulator_is_deterministic() -> None:
    first = generate_payment_lifecycles(count=5, random_seed=99)
    second = generate_payment_lifecycles(count=5, random_seed=99)

    assert first == second


def test_lifecycle_simulator_source_does_not_call_rule_engine() -> None:
    import src.incidents.lifecycle_simulator as simulator

    source = inspect.getsource(simulator)

    assert "evaluate_payment_incident" not in source
    assert "src.incidents.rules" not in source


def test_lifecycle_api_list_detail_and_evaluate() -> None:
    list_response = client.get("/incidents/lifecycles?limit=1")

    assert list_response.status_code == 200
    payment_id = list_response.json()["lifecycles"][0]["payment_id"]
    detail_response = client.get(f"/incidents/lifecycles/{payment_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["payment_id"] == payment_id

    payload = _lifecycle([("PAYMENT_CREATED", 0), ("BANK_DEBITED", 2), ("PAYMENT_FAILED", 8)])
    evaluate_response = client.post("/incidents/lifecycles/evaluate", json=payload)
    assert evaluate_response.status_code == 200
    assert evaluate_response.json()["status"] == "ACTIVE_INCIDENT"


def test_existing_snapshot_api_still_works() -> None:
    response = client.get("/incidents?limit=1")

    assert response.status_code == 200
    assert response.json()["incidents"]


def test_module_one_threshold_and_feature_contract_are_unchanged() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["threshold"] == 0.60
    assert Path("artifacts/preprocessors/preprocessing_metadata.json").exists()


def _lifecycle(steps: list[tuple[str, int]]) -> dict:
    payment_id = "pay_lifecycle_test"
    return {
        "payment_id": payment_id,
        "merchant_id": "merchant_001",
        "amount": 125.0,
        "payment_method": PaymentMethod.UPI.value,
        "fraud_risk_score": 0.12,
        "events": [
            {
                "event_id": f"evt_{index}",
                "payment_id": payment_id,
                "event_type": event_type,
                "event_time_minutes": time,
            }
            for index, (event_type, time) in enumerate(steps, start=1)
        ],
    }
