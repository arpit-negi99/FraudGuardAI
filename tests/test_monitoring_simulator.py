from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from backend.api import app
from src.monitoring.simulator import generate_monitoring_stream


client = TestClient(app)


def test_monitoring_simulator_is_deterministic() -> None:
    first = generate_monitoring_stream(events_per_window=5, random_seed=123)
    second = generate_monitoring_stream(events_per_window=5, random_seed=123)

    assert first.equals(second)


def test_monitoring_simulator_independent_from_detectors() -> None:
    import src.monitoring.simulator as simulator

    source = inspect.getsource(simulator)

    assert "evaluate_payment_incident" not in source
    assert "FraudPredictor" not in source
    assert "src.incidents.rules" not in source


def test_monitoring_summary_endpoint() -> None:
    response = client.get("/monitoring/summary")

    assert response.status_code == 200
    assert response.json()["stream_rows"] >= 20000


def test_monitoring_current_endpoint() -> None:
    response = client.get("/monitoring/current?scenario_type=DEBIT_SERVICE_MISMATCH_SPIKE")

    assert response.status_code == 200
    assert response.json()["status"] in {"ELEVATED", "HIGH", "CRITICAL"}


def test_monitoring_windows_endpoint() -> None:
    response = client.get("/monitoring/windows?scenario_type=FRAUD_RISK_SPIKE&limit=3")

    assert response.status_code == 200
    assert len(response.json()["windows"]) == 3


def test_monitoring_alerts_endpoint() -> None:
    response = client.get("/monitoring/alerts?scenario_type=COMPLAINT_SPIKE&limit=5")

    assert response.status_code == 200
    assert response.json()["alerts"]


def test_monitoring_scenarios_endpoint() -> None:
    response = client.get("/monitoring/scenarios")

    assert response.status_code == 200
    assert "MIXED_RISK_SPIKE" in response.json()["scenarios"]


def test_malformed_monitoring_scenario_rejected() -> None:
    response = client.get("/monitoring/current?scenario_type=NOT_REAL")

    assert response.status_code == 422


def test_snapshot_and_lifecycle_modules_still_respond() -> None:
    assert client.get("/incidents?limit=1").status_code == 200
    assert client.get("/incidents/lifecycles?limit=1").status_code == 200


def test_module_one_threshold_still_frozen() -> None:
    assert client.get("/health").json()["threshold"] == 0.60
