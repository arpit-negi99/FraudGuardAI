from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction: dict[str, Any]
    include_explanation: bool = True


class BatchPredictionRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1)
    include_explanations: bool = False


class PolicySimulationRequest(BaseModel):
    review_cost: float = Field(default=5.0, ge=0)
    fraud_loss_multiplier: float = Field(default=1.0, ge=0)


class PaymentIncidentEvaluationRequest(BaseModel):
    payment_id: str = Field(min_length=1)
    merchant_id: str = Field(min_length=1)
    amount: float = Field(ge=0)
    payment_method: str
    bank_debited: bool
    gateway_status: str
    order_status: str
    service_delivered: bool
    callback_received: bool
    refund_status: str
    retry_count: int = Field(ge=0)
    time_since_payment_minutes: int = Field(ge=0)
    customer_complaint: bool
    fraud_risk_score: float | None = Field(default=None, ge=0, le=1)
