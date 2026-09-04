from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.config import get_cors_origins, get_settings
from backend.schemas import (
    BatchPredictionRequest,
    MonitoringEvaluationRequest,
    PaymentIncidentEvaluationRequest,
    PaymentLifecycleEvaluationRequest,
    PolicySimulationRequest,
    TransactionRequest,
)
from backend.services import fraud_service, incident_service, monitoring_service, streaming_monitor_service
from backend.services.event_producer import producer as risk_event_producer
from src.events.schema import build_transaction_event
from src.inference.predict import ArtifactLoadError, InferenceError, InferenceInputError


app = FastAPI(title="FraudGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
async def startup() -> None:
    await risk_event_producer.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    await risk_event_producer.stop()


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    stream_health = {
        "stream_mode": settings.stream_mode,
        "streaming_enabled": settings.streaming_enabled,
        "redpanda": risk_event_producer.status(),
        "redis": await streaming_monitor_service.redis_ping(settings),
        "analytics_worker": "external",
    }
    try:
        return {
            **fraud_service.health(),
            "incident_module_available": True,
            "monitoring_module_available": True,
            **stream_health,
        }
    except ArtifactLoadError as exc:
        return {
            "status": "error",
            "model_loaded": False,
            "preprocessor_loaded": False,
            "incident_module_available": True,
            "monitoring_module_available": True,
            "threshold": 0.60,
            **stream_health,
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
async def predict(request: TransactionRequest) -> dict:
    try:
        result = fraud_service.score_transaction(
            request.transaction,
            include_explanation=request.include_explanation,
        )
        event = build_transaction_event(result, request.transaction, merchant_id=request.transaction.get("merchant_id"))
        await risk_event_producer.enqueue(event)
        return result
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


@app.get("/monitoring/summary")
def monitoring_summary() -> dict:
    return monitoring_service.summary()


@app.get("/monitoring/windows")
def monitoring_windows(
    scenario_type: str | None = None,
    limit: int = Query(default=120, ge=1, le=500),
) -> dict:
    try:
        return monitoring_service.windows(scenario_type=scenario_type, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/monitoring/current")
async def monitoring_current(
    scenario_type: str | None = None,
    merchant_id: str | None = None,
) -> dict:
    if get_settings().streaming_enabled:
        return await streaming_monitor_service.current_state(merchant_id)
    try:
        return monitoring_service.current(scenario_type=scenario_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/monitoring/alerts")
async def monitoring_alerts(
    scenario_type: str | None = None,
    merchant_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    if get_settings().streaming_enabled:
        return await streaming_monitor_service.recent_alerts(merchant_id, limit)
    try:
        return monitoring_service.alerts(scenario_type=scenario_type, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/monitoring/scenarios")
def monitoring_scenarios() -> dict:
    return monitoring_service.scenarios()


@app.get("/monitoring/stream")
async def monitoring_stream(merchant_id: str | None = None) -> StreamingResponse:
    return StreamingResponse(
        streaming_monitor_service.sse_events(merchant_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/monitoring/evaluate")
def monitoring_evaluate(request: MonitoringEvaluationRequest) -> dict:
    try:
        return monitoring_service.evaluate_custom(request.records, request.window_minutes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


@app.get("/incidents/lifecycles")
def incident_lifecycles(
    status: str | None = None,
    scenario_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict:
    try:
        return incident_service.list_lifecycles(
            status=status,
            scenario_type=scenario_type,
            limit=limit,
            offset=offset,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/incidents/lifecycles/summary")
def incident_lifecycle_summary() -> dict:
    try:
        return incident_service.lifecycle_summary()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/incidents/lifecycles/{payment_id}")
def incident_lifecycle(payment_id: str) -> dict:
    try:
        return incident_service.lifecycle_detail(payment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/incidents/lifecycles/evaluate")
def evaluate_incident_lifecycle(request: PaymentLifecycleEvaluationRequest) -> dict:
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return incident_service.evaluate_lifecycle(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
