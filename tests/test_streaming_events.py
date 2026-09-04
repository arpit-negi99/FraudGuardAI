from __future__ import annotations

import pytest

from src.events.schema import EVENT_VERSION, build_transaction_event, validate_transaction_event


def test_build_transaction_event_uses_uuid_and_default_merchant() -> None:
    event = build_transaction_event(
        {"risk_score": 0.7, "decision": "REVIEW", "threshold": 0.6, "priority": "High"},
        {"TransactionID": 123, "TransactionAmt": 45.5, "card4": "visa"},
    )

    assert event.event_version == EVENT_VERSION
    assert event.event_type == "transaction_scored"
    assert event.merchant_id == "merchant_demo_001"
    assert event.transaction_id == 123
    assert event.payment_method == "visa"


def test_validate_transaction_event_rejects_invalid_decision() -> None:
    event = build_transaction_event(
        {"risk_score": 0.2, "decision": "ALLOW", "threshold": 0.6, "priority": "Low"},
        {"merchant_id": "merchant_001"},
    ).to_dict()
    event["decision"] = "BLOCK"

    with pytest.raises(ValueError, match="ALLOW or REVIEW"):
        validate_transaction_event(event)


def test_validate_transaction_event_rejects_invalid_risk_score() -> None:
    event = build_transaction_event(
        {"risk_score": 0.2, "decision": "ALLOW", "threshold": 0.6, "priority": "Low"},
        {"merchant_id": "merchant_001"},
    ).to_dict()
    event["fraud_risk_score"] = 1.5

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        validate_transaction_event(event)
