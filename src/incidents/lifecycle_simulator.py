from __future__ import annotations

from typing import Any

import numpy as np

from src.incidents.lifecycle import (
    LifecycleEventType,
    PaymentLifecycle,
    PaymentLifecycleEvent,
    lifecycle_event_to_dict,
)
from src.incidents.schema import PaymentMethod
from src.incidents.simulator import SYNTHETIC_DATA_NOTE


DEFAULT_LIFECYCLE_RANDOM_SEED = 211
DEFAULT_LIFECYCLE_COUNT = 3_000
LIFECYCLE_SCENARIO_WEIGHTS = {
    "NORMAL_SUCCESS": 0.34,
    "SAFE_FAILURE": 0.18,
    "DEBIT_GATEWAY_FAILURE": 0.12,
    "LATE_AUTHORIZATION": 0.10,
    "CAPTURED_UNFULFILLED": 0.10,
    "REFUND_RESOLUTION": 0.08,
    "COMPLAINT_ESCALATION": 0.04,
    "RETRY_RELATED_RISK": 0.04,
}
SCENARIO_EXPECTATIONS = {
    "NORMAL_SUCCESS": {"expected_resolution": False, "expected_final_status": "NORMAL"},
    "SAFE_FAILURE": {"expected_resolution": False, "expected_final_status": "NORMAL"},
    "DEBIT_GATEWAY_FAILURE": {"expected_resolution": False, "expected_final_status": "ACTIVE_INCIDENT"},
    "LATE_AUTHORIZATION": {"expected_resolution": False, "expected_final_status": "ACTIVE_INCIDENT"},
    "CAPTURED_UNFULFILLED": {"expected_resolution": False, "expected_final_status": "ACTIVE_INCIDENT"},
    "REFUND_RESOLUTION": {"expected_resolution": True, "expected_final_status": "RESOLVED"},
    "COMPLAINT_ESCALATION": {"expected_resolution": False, "expected_final_status": "ACTIVE_INCIDENT"},
    "RETRY_RELATED_RISK": {"expected_resolution": False, "expected_final_status": "ACTIVE_INCIDENT"},
}


def generate_payment_lifecycles(
    count: int = DEFAULT_LIFECYCLE_COUNT,
    random_seed: int = DEFAULT_LIFECYCLE_RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Generate scenario-first synthetic payment lifecycle timelines."""
    if count <= 0:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(random_seed)
    scenarios = np.array(list(LIFECYCLE_SCENARIO_WEIGHTS))
    probabilities = np.array(list(LIFECYCLE_SCENARIO_WEIGHTS.values()), dtype=float)
    probabilities = probabilities / probabilities.sum()
    sampled = rng.choice(scenarios, size=count, p=probabilities)
    return [
        lifecycle_to_dict(_lifecycle_for_scenario(str(scenario), index, rng), str(scenario))
        for index, scenario in enumerate(sampled, start=1)
    ]


def summarize_lifecycles(lifecycles: list[dict[str, Any]], random_seed: int) -> dict[str, Any]:
    if not lifecycles:
        raise ValueError("Cannot summarize an empty lifecycle collection.")
    scenario_distribution: dict[str, int] = {}
    status_distribution: dict[str, int] = {}
    event_counts = []
    resolution_times = []
    for lifecycle in lifecycles:
        scenario = str(lifecycle["scenario_type"])
        scenario_distribution[scenario] = scenario_distribution.get(scenario, 0) + 1
        status = str(lifecycle["expected_final_status"])
        status_distribution[status] = status_distribution.get(status, 0) + 1
        event_counts.append(len(lifecycle["events"]))
        resolution_time = _expected_resolution_time(lifecycle)
        if resolution_time is not None:
            resolution_times.append(resolution_time)
    return {
        "data_note": SYNTHETIC_DATA_NOTE,
        "random_seed": random_seed,
        "lifecycle_count": len(lifecycles),
        "scenario_distribution": scenario_distribution,
        "status_distribution": status_distribution,
        "active": status_distribution.get("ACTIVE_INCIDENT", 0),
        "resolving": status_distribution.get("RESOLVING", 0),
        "resolved": status_distribution.get("RESOLVED", 0),
        "normal": status_distribution.get("NORMAL", 0),
        "average_event_count": float(np.mean(event_counts)),
        "median_resolution_time_minutes": (
            float(np.median(resolution_times)) if resolution_times else None
        ),
    }


def _expected_resolution_time(lifecycle: dict[str, Any]) -> int | None:
    if not lifecycle.get("expected_resolution"):
        return None
    detection_event = next(
        (
            event
            for event in lifecycle["events"]
            if event["event_type"] in {"ORDER_FAILED", "PAYMENT_FAILED", "CALLBACK_MISSING"}
        ),
        None,
    )
    resolution_event = next(
        (
            event
            for event in lifecycle["events"]
            if event["event_type"] in {"REFUND_PROCESSED", "ORDER_FULFILLED"}
        ),
        None,
    )
    if not detection_event or not resolution_event:
        return None
    return int(resolution_event["event_time_minutes"]) - int(detection_event["event_time_minutes"])


def lifecycle_to_dict(lifecycle: PaymentLifecycle, scenario_type: str) -> dict[str, Any]:
    return {
        "payment_id": lifecycle.payment_id,
        "merchant_id": lifecycle.merchant_id,
        "amount": lifecycle.amount,
        "payment_method": lifecycle.payment_method.value,
        "fraud_risk_score": lifecycle.fraud_risk_score,
        "events": [lifecycle_event_to_dict(event) for event in lifecycle.events],
        "scenario_type": scenario_type,
        **SCENARIO_EXPECTATIONS[scenario_type],
        "synthetic_data_note": SYNTHETIC_DATA_NOTE,
    }


def _lifecycle_for_scenario(
    scenario: str,
    index: int,
    rng: np.random.Generator,
) -> PaymentLifecycle:
    base = _base_lifecycle(index, rng)
    events = _events_for_scenario(scenario, base["payment_id"], rng)
    return PaymentLifecycle(events=events, **base)


def _base_lifecycle(index: int, rng: np.random.Generator) -> dict[str, Any]:
    return {
        "payment_id": f"pay_life_{index:06d}",
        "merchant_id": f"merchant_{int(rng.integers(1, 121)):03d}",
        "amount": round(float(rng.lognormal(mean=4.1, sigma=0.85)), 2),
        "payment_method": PaymentMethod(str(rng.choice([item.value for item in PaymentMethod]))),
        "fraud_risk_score": round(float(rng.beta(1.4, 8.5)), 6),
    }


def _events_for_scenario(
    scenario: str,
    payment_id: str,
    rng: np.random.Generator,
) -> list[PaymentLifecycleEvent]:
    if scenario == "NORMAL_SUCCESS":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (3, LifecycleEventType.PAYMENT_AUTHORIZED),
                (4, LifecycleEventType.PAYMENT_CAPTURED),
                (9, LifecycleEventType.ORDER_FULFILLED),
            ],
        )
    if scenario == "SAFE_FAILURE":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (int(rng.integers(2, 8)), LifecycleEventType.PAYMENT_FAILED),
                (int(rng.integers(8, 18)), LifecycleEventType.CALLBACK_RECEIVED),
            ],
        )
    if scenario == "DEBIT_GATEWAY_FAILURE":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (5, LifecycleEventType.CALLBACK_MISSING),
                (10, LifecycleEventType.PAYMENT_FAILED),
                (35, LifecycleEventType.CALLBACK_MISSING),
            ],
        )
    if scenario == "LATE_AUTHORIZATION":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (8, LifecycleEventType.PAYMENT_FAILED),
                (18, LifecycleEventType.CUSTOMER_RETRY),
                (45, LifecycleEventType.PAYMENT_AUTHORIZED),
            ],
        )
    if scenario == "CAPTURED_UNFULFILLED":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (4, LifecycleEventType.PAYMENT_CAPTURED),
                (16, LifecycleEventType.ORDER_FAILED),
                (38, LifecycleEventType.CALLBACK_MISSING),
            ],
        )
    if scenario == "REFUND_RESOLUTION":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (4, LifecycleEventType.PAYMENT_CAPTURED),
                (12, LifecycleEventType.ORDER_FAILED),
                (52, LifecycleEventType.REFUND_INITIATED),
                (80, LifecycleEventType.REFUND_PROCESSED),
            ],
        )
    if scenario == "COMPLAINT_ESCALATION":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (8, LifecycleEventType.PAYMENT_CAPTURED),
                (42, LifecycleEventType.ORDER_FAILED),
                (75, LifecycleEventType.CUSTOMER_COMPLAINT),
            ],
        )
    if scenario == "RETRY_RELATED_RISK":
        return _events(
            payment_id,
            [
                (0, LifecycleEventType.PAYMENT_CREATED),
                (2, LifecycleEventType.BANK_DEBITED),
                (15, LifecycleEventType.CUSTOMER_RETRY),
                (26, LifecycleEventType.CUSTOMER_RETRY),
                (37, LifecycleEventType.CUSTOMER_RETRY),
            ],
        )
    raise ValueError(f"Unsupported lifecycle scenario: {scenario}")


def _events(
    payment_id: str,
    steps: list[tuple[int, LifecycleEventType]],
) -> list[PaymentLifecycleEvent]:
    return [
        PaymentLifecycleEvent(
            event_id=f"{payment_id}_evt_{index:02d}",
            payment_id=payment_id,
            event_time_minutes=time,
            event_type=event_type,
        )
        for index, (time, event_type) in enumerate(steps, start=1)
    ]
