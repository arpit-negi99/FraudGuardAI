from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from src.incidents.evaluation import (
    INCIDENT_CLASS_ORDER,
    applicable_incident_types,
    binary_metrics,
    evaluate_dataset,
    per_class_metrics,
    precedence_artifact,
    simulator_ground_truth_is_independent,
)
from src.incidents.rules import evaluate_payment_incident
from src.incidents.simulator import (
    STRESS_SCENARIO_TO_INCIDENT_TYPE,
    generate_payment_incident_stress_events,
    validate_no_impossible_combinations,
)


def event(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "payment_id": "pay_eval_001",
        "merchant_id": "merchant_001",
        "amount": 100.0,
        "payment_method": "card",
        "bank_debited": True,
        "gateway_status": "captured",
        "order_status": "fulfilled",
        "service_delivered": True,
        "callback_received": True,
        "refund_status": "none",
        "retry_count": 0,
        "time_since_payment_minutes": 5,
        "customer_complaint": False,
        "fraud_risk_score": None,
        "scenario": "NORMAL",
        "incident_label": False,
        "incident_type": "NORMAL_PAYMENT",
    }
    row.update(overrides)
    return row


def test_binary_metric_correctness() -> None:
    metrics = binary_metrics(
        pd.Series([True, True, False, False]),
        pd.Series([True, False, True, False]),
    )

    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_per_class_evaluation_contains_support() -> None:
    rows = per_class_metrics(
        pd.Series(["NORMAL_PAYMENT", "DEBIT_SERVICE_MISMATCH"]),
        pd.Series(["NORMAL_PAYMENT", "NORMAL_PAYMENT"]),
    )

    normal = next(row for row in rows if row["incident_type"] == "NORMAL_PAYMENT")
    assert normal["support"] == 1
    assert {"precision", "recall", "f1"}.issubset(normal)


def test_confusion_matrix_dimensions() -> None:
    data = pd.DataFrame(
        [
            event(),
            event(
                payment_id="pay_eval_002",
                gateway_status="failed",
                order_status="failed",
                service_delivered=False,
                callback_received=False,
                time_since_payment_minutes=60,
                scenario="DEBIT_SERVICE_MISMATCH",
                incident_label=True,
                incident_type="DEBIT_SERVICE_MISMATCH",
            ),
        ]
    )

    result = evaluate_dataset(data, "unit")

    matrix = result["confusion_matrix"]["matrix"]
    assert len(matrix) == len(INCIDENT_CLASS_ORDER)
    assert all(len(row) == len(INCIDENT_CLASS_ORDER) for row in matrix)


def test_stress_generator_is_deterministic_with_fixed_seed() -> None:
    first = generate_payment_incident_stress_events(row_count=100, random_seed=99)
    second = generate_payment_incident_stress_events(row_count=100, random_seed=99)

    pd.testing.assert_frame_equal(first, second)


def test_stress_ground_truth_independent_of_rules() -> None:
    assert simulator_ground_truth_is_independent()


def test_simulator_module_does_not_import_rules() -> None:
    import src.incidents.simulator as simulator

    source = inspect.getsource(simulator)
    assert "from src.incidents.rules" not in source
    assert "evaluate_payment_incident(" not in source


def test_safe_failures_remain_normal() -> None:
    row = event(
        bank_debited=False,
        gateway_status="failed",
        order_status="failed",
        service_delivered=False,
        retry_count=0,
        time_since_payment_minutes=12,
    )

    result = evaluate_payment_incident(row)

    assert result.incident_detected is False


def test_refund_pending_is_not_equivalent_to_no_refund() -> None:
    pending = evaluate_payment_incident(
        event(order_status="cancelled", service_delivered=False, refund_status="pending")
    )
    none = evaluate_payment_incident(
        event(order_status="cancelled", service_delivered=False, refund_status="none")
    )

    assert pending.incident_type.value == "CAPTURED_BUT_UNFULFILLED"
    assert none.incident_type.value == "REFUND_REQUIRED"


def test_resolved_complaint_avoids_critical_classification() -> None:
    result = evaluate_payment_incident(
        event(
            gateway_status="refunded",
            order_status="cancelled",
            service_delivered=False,
            refund_status="processed",
            customer_complaint=True,
        )
    )

    assert result.incident_detected is False


def test_high_fraud_score_alone_does_not_create_lifecycle_incident() -> None:
    result = evaluate_payment_incident(event(fraud_risk_score=0.95))

    assert result.incident_detected is False


def test_low_fraud_score_clear_lifecycle_incident_is_detected() -> None:
    result = evaluate_payment_incident(
        event(
            fraud_risk_score=0.02,
            gateway_status="failed",
            order_status="failed",
            service_delivered=False,
            callback_received=False,
            time_since_payment_minutes=60,
        )
    )

    assert result.incident_detected is True
    assert result.incident_type.value == "DEBIT_SERVICE_MISMATCH"


def test_overlapping_incident_rules_handled_deterministically() -> None:
    row = event(
        order_status="failed",
        service_delivered=False,
        customer_complaint=True,
        retry_count=4,
        time_since_payment_minutes=90,
    )

    applicable = applicable_incident_types(row)
    result = evaluate_payment_incident(row)

    assert len(applicable) > 1
    assert result.incident_type.value == "COMPLAINT_ESCALATION_RISK"
    assert "Highest severity wins" in precedence_artifact()["primary_selection_rule"]


def test_module_1_frozen_artifacts_remain_present() -> None:
    assert Path("artifacts/models/xgboost_model.json").exists()
    assert Path("artifacts/preprocessors/preprocessor.joblib").exists()
    assert Path("artifacts/results/final_test_metrics.json").exists()


def test_stress_scenario_labels_are_scenario_mapped() -> None:
    data = generate_payment_incident_stress_events(row_count=300, random_seed=123)

    validate_no_impossible_combinations(data)
    expected = data["scenario"].map(STRESS_SCENARIO_TO_INCIDENT_TYPE)
    assert data["incident_type"].equals(expected)
