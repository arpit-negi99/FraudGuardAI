from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from src.incidents.schema import (
    GatewayStatus,
    IncidentType,
    OrderStatus,
    PaymentEvent,
    PaymentMethod,
    RefundStatus,
)


DEFAULT_RANDOM_SEED = 42
DEFAULT_ROW_COUNT = 10_000
STRESS_RANDOM_SEED = 137
STRESS_ROW_COUNT = 5_000
SYNTHETIC_DATA_NOTE = (
    "This dataset is simulated for hackathon demonstration and does not represent "
    "proprietary payment-provider data."
)

SCENARIO_WEIGHTS = {
    "NORMAL": 0.54,
    "FAILED_SAFE": 0.20,
    "DEBIT_SERVICE_MISMATCH": 0.08,
    "CAPTURED_UNFULFILLED": 0.05,
    "REFUND_REQUIRED": 0.04,
    "LATE_AUTHORIZATION": 0.035,
    "RETRY_RISK": 0.035,
    "COMPLAINT_ESCALATION": 0.02,
}

SCENARIO_TO_INCIDENT_TYPE = {
    "NORMAL": IncidentType.NONE.value,
    "FAILED_SAFE": IncidentType.NONE.value,
    "DEBIT_SERVICE_MISMATCH": IncidentType.DEBIT_SERVICE_MISMATCH.value,
    "CAPTURED_UNFULFILLED": IncidentType.CAPTURED_BUT_UNFULFILLED.value,
    "REFUND_REQUIRED": IncidentType.REFUND_REQUIRED.value,
    "LATE_AUTHORIZATION": IncidentType.LATE_AUTHORIZATION_RISK.value,
    "RETRY_RISK": IncidentType.RETRY_RELATED_PAYMENT_RISK.value,
    "COMPLAINT_ESCALATION": IncidentType.COMPLAINT_ESCALATION_RISK.value,
}

STRESS_SCENARIO_WEIGHTS = {
    "DELAYED_CALLBACK_SHORT": 0.12,
    "DELAYED_CALLBACK_LONG": 0.12,
    "LATE_AUTHORIZATION": 0.11,
    "REFUND_PENDING": 0.11,
    "COMPLAINT_RESOLVED": 0.10,
    "SAFE_FAILURE": 0.14,
    "HIGH_RETRIES_NO_DEBIT": 0.10,
    "SUCCESSFUL_WITH_RETRIES": 0.10,
    "HIGH_FRAUD_HEALTHY_PAYMENT": 0.06,
    "LOW_FRAUD_CLEAR_INCIDENT": 0.04,
}

STRESS_SCENARIO_TO_INCIDENT_TYPE = {
    "DELAYED_CALLBACK_SHORT": IncidentType.NONE.value,
    "DELAYED_CALLBACK_LONG": IncidentType.LATE_AUTHORIZATION_RISK.value,
    "LATE_AUTHORIZATION": IncidentType.LATE_AUTHORIZATION_RISK.value,
    "REFUND_PENDING": IncidentType.CAPTURED_BUT_UNFULFILLED.value,
    "COMPLAINT_RESOLVED": IncidentType.NONE.value,
    "SAFE_FAILURE": IncidentType.NONE.value,
    "HIGH_RETRIES_NO_DEBIT": IncidentType.NONE.value,
    "SUCCESSFUL_WITH_RETRIES": IncidentType.NONE.value,
    "HIGH_FRAUD_HEALTHY_PAYMENT": IncidentType.NONE.value,
    "LOW_FRAUD_CLEAR_INCIDENT": IncidentType.DEBIT_SERVICE_MISMATCH.value,
}


def generate_payment_incident_events(
    row_count: int = DEFAULT_ROW_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate coherent synthetic payment lifecycle events.

    Ground truth is assigned from scenario generation, not copied from rule output.
    Synthetic fraud-risk scores are representative placeholders and do not call Module 1.
    """
    if row_count <= 0:
        raise ValueError("row_count must be positive.")
    rng = np.random.default_rng(random_seed)
    scenarios = np.array(list(SCENARIO_WEIGHTS))
    probabilities = np.array(list(SCENARIO_WEIGHTS.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    sampled = rng.choice(scenarios, size=row_count, p=probabilities)
    rows = []
    for index, scenario in enumerate(sampled, start=1):
        event = _event_for_scenario(str(scenario), index, rng)
        row = _event_to_row(event)
        row["scenario"] = str(scenario)
        row["incident_label"] = SCENARIO_TO_INCIDENT_TYPE[str(scenario)] != IncidentType.NONE.value
        row["incident_type"] = SCENARIO_TO_INCIDENT_TYPE[str(scenario)]
        row["synthetic_data_note"] = SYNTHETIC_DATA_NOTE
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_incident_dataset(
    data: pd.DataFrame,
    random_seed: int,
) -> dict[str, Any]:
    """Summarize synthetic labels and deterministic-rule severity distribution."""
    if data.empty:
        raise ValueError("Cannot summarize an empty incident dataset.")
    normal_count = int((~data["incident_label"].astype(bool)).sum())
    incident_count = int(data["incident_label"].astype(bool).sum())
    return {
        "data_note": SYNTHETIC_DATA_NOTE,
        "total_rows": int(len(data)),
        "normal_count": normal_count,
        "incident_count": incident_count,
        "incident_rate": float(incident_count / len(data)),
        "distribution_by_incident_type": {
            str(key): int(value) for key, value in data["incident_type"].value_counts().to_dict().items()
        },
        "random_seed": int(random_seed),
    }


def generate_payment_incident_stress_events(
    row_count: int = STRESS_ROW_COUNT,
    random_seed: int = STRESS_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic ambiguous lifecycle edge cases with independent labels."""
    if row_count <= 0:
        raise ValueError("row_count must be positive.")
    rng = np.random.default_rng(random_seed)
    scenarios = np.array(list(STRESS_SCENARIO_WEIGHTS))
    probabilities = np.array(list(STRESS_SCENARIO_WEIGHTS.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    sampled = rng.choice(scenarios, size=row_count, p=probabilities)
    rows = []
    for index, scenario in enumerate(sampled, start=1):
        event = _stress_event_for_scenario(str(scenario), index, rng)
        row = _event_to_row(event)
        row["scenario"] = str(scenario)
        row["incident_label"] = (
            STRESS_SCENARIO_TO_INCIDENT_TYPE[str(scenario)] != IncidentType.NONE.value
        )
        row["incident_type"] = STRESS_SCENARIO_TO_INCIDENT_TYPE[str(scenario)]
        row["synthetic_data_note"] = SYNTHETIC_DATA_NOTE
        rows.append(row)
    return pd.DataFrame(rows)


def validate_no_impossible_combinations(data: pd.DataFrame) -> None:
    """Reject simulator rows that violate core lifecycle consistency constraints."""
    refunded_without_refund = (
        (data["gateway_status"] == GatewayStatus.REFUNDED.value)
        & (data["refund_status"] == RefundStatus.NONE.value)
    )
    delivered_created_order = (
        data["service_delivered"].astype(bool)
        & (data["order_status"] == OrderStatus.CREATED.value)
    )
    if refunded_without_refund.any():
        raise ValueError("Generated data contains refunded gateway rows without refund status.")
    if delivered_created_order.any():
        raise ValueError("Generated data contains delivered service with created order status.")


def _event_for_scenario(scenario: str, index: int, rng: np.random.Generator) -> PaymentEvent:
    base = {
        "payment_id": f"pay_syn_{index:06d}",
        "merchant_id": f"merchant_{int(rng.integers(1, 121)):03d}",
        "amount": round(float(rng.lognormal(mean=4.1, sigma=0.85)), 2),
        "payment_method": PaymentMethod(str(rng.choice([item.value for item in PaymentMethod]))),
        "fraud_risk_score": round(float(rng.beta(1.4, 8.5)), 6),
    }
    if scenario == "NORMAL":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.FULFILLED,
            service_delivered=True,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(2, 18)),
            customer_complaint=False,
        )
    if scenario == "FAILED_SAFE":
        return PaymentEvent(
            **base,
            bank_debited=False,
            gateway_status=GatewayStatus.FAILED,
            order_status=OrderStatus.FAILED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(2, 25)),
            customer_complaint=False,
        )
    if scenario == "DEBIT_SERVICE_MISMATCH":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.FAILED,
            order_status=OrderStatus.FAILED,
            service_delivered=False,
            callback_received=False,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 3)),
            time_since_payment_minutes=int(rng.integers(30, 180)),
            customer_complaint=False,
        )
    if scenario == "CAPTURED_UNFULFILLED":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.CONFIRMED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(12, 240)),
            customer_complaint=False,
        )
    if scenario == "REFUND_REQUIRED":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.CANCELLED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(15, 240)),
            customer_complaint=False,
        )
    if scenario == "LATE_AUTHORIZATION":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.PENDING,
            order_status=OrderStatus.CREATED,
            service_delivered=False,
            callback_received=False,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(1, 3)),
            time_since_payment_minutes=int(rng.integers(30, 160)),
            customer_complaint=False,
        )
    if scenario == "RETRY_RISK":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.PENDING,
            order_status=OrderStatus.CREATED,
            service_delivered=False,
            callback_received=False,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(3, 7)),
            time_since_payment_minutes=int(rng.integers(20, 140)),
            customer_complaint=False,
        )
    if scenario == "COMPLAINT_ESCALATION":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.FAILED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(1, 5)),
            time_since_payment_minutes=int(rng.integers(45, 360)),
            customer_complaint=True,
        )
    raise ValueError(f"Unsupported incident scenario: {scenario}")


def _stress_event_for_scenario(scenario: str, index: int, rng: np.random.Generator) -> PaymentEvent:
    base = {
        "payment_id": f"pay_stress_{index:06d}",
        "merchant_id": f"merchant_{int(rng.integers(1, 121)):03d}",
        "amount": round(float(rng.lognormal(mean=4.1, sigma=0.85)), 2),
        "payment_method": PaymentMethod(str(rng.choice([item.value for item in PaymentMethod]))),
        "fraud_risk_score": round(float(rng.beta(1.4, 8.5)), 6),
    }
    if scenario == "DELAYED_CALLBACK_SHORT":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.PENDING,
            order_status=OrderStatus.CREATED,
            service_delivered=False,
            callback_received=False,
            refund_status=RefundStatus.NONE,
            retry_count=0,
            time_since_payment_minutes=int(rng.integers(5, 12)),
            customer_complaint=False,
        )
    if scenario in {"DELAYED_CALLBACK_LONG", "LATE_AUTHORIZATION"}:
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.PENDING,
            order_status=OrderStatus.CREATED,
            service_delivered=False,
            callback_received=False,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(1, 3)),
            time_since_payment_minutes=int(rng.integers(40, 95)),
            customer_complaint=False,
        )
    if scenario == "REFUND_PENDING":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.CANCELLED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.PENDING,
            retry_count=0,
            time_since_payment_minutes=int(rng.integers(20, 180)),
            customer_complaint=False,
        )
    if scenario == "COMPLAINT_RESOLVED":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.REFUNDED,
            order_status=OrderStatus.CANCELLED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.PROCESSED,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(60, 360)),
            customer_complaint=True,
        )
    if scenario == "SAFE_FAILURE":
        return PaymentEvent(
            **base,
            bank_debited=False,
            gateway_status=GatewayStatus.FAILED,
            order_status=OrderStatus.FAILED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(0, 2)),
            time_since_payment_minutes=int(rng.integers(3, 20)),
            customer_complaint=False,
        )
    if scenario == "HIGH_RETRIES_NO_DEBIT":
        return PaymentEvent(
            **base,
            bank_debited=False,
            gateway_status=GatewayStatus.FAILED,
            order_status=OrderStatus.FAILED,
            service_delivered=False,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(4, 8)),
            time_since_payment_minutes=int(rng.integers(8, 80)),
            customer_complaint=False,
        )
    if scenario == "SUCCESSFUL_WITH_RETRIES":
        return PaymentEvent(
            **base,
            bank_debited=True,
            gateway_status=GatewayStatus.CAPTURED,
            order_status=OrderStatus.FULFILLED,
            service_delivered=True,
            callback_received=True,
            refund_status=RefundStatus.NONE,
            retry_count=int(rng.integers(3, 6)),
            time_since_payment_minutes=int(rng.integers(5, 45)),
            customer_complaint=False,
        )
    if scenario == "HIGH_FRAUD_HEALTHY_PAYMENT":
        event = _stress_event_for_scenario("SUCCESSFUL_WITH_RETRIES", index, rng)
        return PaymentEvent(**{**asdict(event), "fraud_risk_score": round(float(rng.uniform(0.9, 0.99)), 6)})
    if scenario == "LOW_FRAUD_CLEAR_INCIDENT":
        event = _event_for_scenario("DEBIT_SERVICE_MISMATCH", index, rng)
        return PaymentEvent(
            **{
                **asdict(event),
                "payment_id": f"pay_stress_{index:06d}",
                "fraud_risk_score": round(float(rng.uniform(0.0, 0.05)), 6),
            }
        )
    raise ValueError(f"Unsupported stress incident scenario: {scenario}")


def _event_to_row(event: PaymentEvent) -> dict[str, object]:
    row = asdict(event)
    row["payment_method"] = event.payment_method.value
    row["gateway_status"] = event.gateway_status.value
    row["order_status"] = event.order_status.value
    row["refund_status"] = event.refund_status.value
    return row
