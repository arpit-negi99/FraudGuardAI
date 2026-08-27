from __future__ import annotations

from fastapi.testclient import TestClient
import pandas as pd

from backend.api import app
from backend.services import fraud_service, incident_service
from src.inference.predict import ArtifactLoadError
from src.incidents.rules import evaluate_payment_incident


client = TestClient(app)


def test_health_endpoint_reports_frozen_threshold() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["preprocessor_loaded"] is True
    assert body["threshold"] == 0.60


def test_demo_transactions_endpoint_returns_packaged_rows() -> None:
    response = client.get("/demo/transactions")

    assert response.status_code == 200
    rows = response.json()["transactions"]
    assert rows
    assert {"transaction_id", "risk_score", "decision", "priority"}.issubset(rows[0])


def test_single_demo_transaction_includes_shap_contributors() -> None:
    transaction_id = client.get("/demo/transactions").json()["transactions"][0]["transaction_id"]

    response = client.get(f"/demo/transactions/{transaction_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == transaction_id
    assert "contributors" in body
    assert isinstance(body["contributors"], list)


def test_predict_endpoint_uses_existing_inference_pipeline() -> None:
    transaction = _json_ready(fraud_service.get_demo_transactions().iloc[0].to_dict())

    response = client.post(
        "/predict",
        json={"transaction": transaction, "include_explanation": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["threshold"] == 0.60
    assert body["decision"] in {"ALLOW", "REVIEW"}


def test_batch_predict_endpoint_summarizes_results() -> None:
    transactions = [
        _json_ready(row)
        for row in fraud_service.get_demo_transactions().head(3).to_dict(orient="records")
    ]

    response = client.post("/predict/batch", json={"transactions": transactions})

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_count"] == 3
    assert body["review_count"] + body["allow_count"] == 3


def test_policy_presets_endpoint_returns_expected_strategies() -> None:
    response = client.get("/policy/presets")

    assert response.status_code == 200
    names = [preset["name"] for preset in response.json()["presets"]]
    assert names == ["Fraud First", "Balanced", "Low Friction"]


def test_policy_simulation_endpoint_uses_non_negative_inputs() -> None:
    response = client.post(
        "/policy/simulate",
        json={"review_cost": 5.0, "fraud_loss_multiplier": 1.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_modeled_cost"] >= 0
    assert "not actual merchant savings" in body["note"]


def test_malformed_batch_request_is_rejected() -> None:
    response = client.post("/predict/batch", json={"transactions": []})

    assert response.status_code == 422


def test_health_handles_missing_artifacts(monkeypatch) -> None:
    def broken_predictor():
        raise ArtifactLoadError("Missing model for test")

    monkeypatch.setattr(fraud_service, "get_predictor", broken_predictor)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_spike_monitor_endpoint_documents_thresholds() -> None:
    response = client.get("/risk/spike")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"Normal", "Elevated", "High"}
    assert "Elevated" in body["thresholds"]


def test_incident_summary_endpoint_returns_evaluated_counts() -> None:
    response = client.get("/incidents/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_payments"] == 10000
    assert body["active_incidents"] + body["normal"] == body["total_payments"]
    assert body["incident_rate"] > 0
    assert "severity_distribution" in body


def test_incident_list_endpoint_returns_packaged_rows() -> None:
    response = client.get("/incidents?limit=5")

    assert response.status_code == 200
    body = response.json()
    assert body["incidents"]
    assert len(body["incidents"]) <= 5
    assert {
        "payment_id",
        "merchant_id",
        "amount",
        "payment_method",
        "incident_detected",
        "incident_type",
        "severity",
        "recommended_action",
        "fraud_risk_score",
    }.issubset(body["incidents"][0])


def test_incident_detail_endpoint_returns_lifecycle_fields() -> None:
    payment_id = client.get("/incidents?incident_detected=true&limit=1").json()["incidents"][0]["payment_id"]

    response = client.get(f"/incidents/{payment_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["payment_id"] == payment_id
    assert "bank_debited" in body
    assert "gateway_status" in body
    assert isinstance(body["reasons"], list)


def test_unknown_incident_payment_id_returns_404() -> None:
    response = client.get("/incidents/pay_missing_for_test")

    assert response.status_code == 404


def test_incident_evaluate_endpoint_uses_rule_engine() -> None:
    payload = {
        "payment_id": "pay_api_test",
        "merchant_id": "merchant_001",
        "amount": 1250.0,
        "payment_method": "upi",
        "bank_debited": True,
        "gateway_status": "failed",
        "order_status": "failed",
        "service_delivered": False,
        "callback_received": False,
        "refund_status": "none",
        "retry_count": 1,
        "time_since_payment_minutes": 47,
        "customer_complaint": False,
        "fraud_risk_score": 0.12,
    }

    response = client.post("/incidents/evaluate", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["incident_detected"] is True
    assert body["incident_type"] == "DEBIT_SERVICE_MISMATCH"
    assert body["recommended_action"] == "VERIFY_PAYMENT"


def test_malformed_incident_payload_is_rejected() -> None:
    response = client.post(
        "/incidents/evaluate",
        json={
            "payment_id": "pay_bad",
            "merchant_id": "merchant_001",
            "amount": -1,
        },
    )

    assert response.status_code == 422


def test_incident_severity_filter() -> None:
    response = client.get("/incidents?severity=HIGH&limit=25")

    assert response.status_code == 200
    rows = response.json()["incidents"]
    assert rows
    assert all(row["severity"] == "HIGH" for row in rows)


def test_incident_type_filter() -> None:
    response = client.get("/incidents?incident_type=CAPTURED_BUT_UNFULFILLED&limit=25")

    assert response.status_code == 200
    rows = response.json()["incidents"]
    assert rows
    assert all(row["incident_type"] == "CAPTURED_BUT_UNFULFILLED" for row in rows)


def test_incident_detail_reasons_originate_from_rule_engine() -> None:
    row = incident_service.incident_detail("pay_syn_000001")
    expected = evaluate_payment_incident(row).to_dict()

    assert row["reasons"] == expected["reasons"]
    assert row["incident_type"] == expected["incident_type"]


def test_module_one_frozen_threshold_still_available() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["threshold"] == 0.60


def _json_ready(row: dict) -> dict:
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}
