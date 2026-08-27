from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    BatchPredictionRequest,
    PaymentIncidentEvaluationRequest,
    PolicySimulationRequest,
    TransactionRequest,
)
from backend.services import fraud_service, incident_service
from src.inference.predict import ArtifactLoadError, InferenceError, InferenceInputError


app = FastAPI(title="FraudGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict:
    try:
        return fraud_service.health()
    except ArtifactLoadError as exc:
        return {
            "status": "error",
            "model_loaded": False,
            "preprocessor_loaded": False,
            "threshold": 0.60,
            "message": str(exc),
        }


@app.get("/demo/transactions")
def demo_transactions() -> dict:
    return {"transactions": fraud_service.demo_predictions()}


@app.get("/demo/transactions/{transaction_id}")
def demo_transaction(transaction_id: int) -> dict:
    try:
        return fraud_service.demo_transaction(transaction_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: TransactionRequest) -> dict:
    try:
        return fraud_service.score_transaction(
            request.transaction,
            include_explanation=request.include_explanation,
        )
    except InferenceInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest) -> dict:
    try:
        return fraud_service.score_batch(
            request.transactions,
            include_explanations=request.include_explanations,
        )
    except InferenceInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/policy/presets")
def policy_presets() -> dict:
    return {"presets": fraud_service.policy_presets()}


@app.post("/policy/simulate")
def policy_simulate(request: PolicySimulationRequest) -> dict:
    try:
        return fraud_service.policy_simulation(
            review_cost=request.review_cost,
            fraud_loss_multiplier=request.fraud_loss_multiplier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/risk/summary")
def risk_summary() -> dict:
    return fraud_service.risk_summary()


@app.get("/risk/review-queue")
def risk_review_queue() -> dict:
    return {"transactions": fraud_service.review_queue_rows()}


@app.get("/risk/spike")
def risk_spike() -> dict:
    return fraud_service.spike_monitor()


@app.get("/evaluation/final")
def final_evaluation() -> dict:
    return fraud_service.final_evaluation()


@app.get("/incidents")
def incidents(
    severity: str | None = None,
    incident_type: str | None = None,
    incident_detected: bool | None = None,
    payment_method: str | None = None,
    minimum_amount: float | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    try:
        return incident_service.list_incidents(
            severity=severity,
            incident_type=incident_type,
            incident_detected=incident_detected,
            payment_method=payment_method,
            minimum_amount=minimum_amount,
            limit=limit,
            offset=offset,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/incidents/summary")
def incidents_summary() -> dict:
    try:
        return incident_service.incident_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/incidents/types")
def incidents_types() -> dict:
    return incident_service.incident_types()


@app.get("/incidents/{payment_id}")
def incident(payment_id: str) -> dict:
    try:
        return incident_service.incident_detail(payment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/incidents/evaluate")
def evaluate_incident(request: PaymentIncidentEvaluationRequest) -> dict:
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return incident_service.evaluate_event(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
