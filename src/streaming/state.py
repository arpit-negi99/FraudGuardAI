from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from backend.config import StreamingSettings
from src.events.schema import RiskTransactionEvent
from src.monitoring.schema import VERY_HIGH_RISK_THRESHOLD
from src.monitoring.spike import recommended_action, relative_change


STREAM_METRICS = [
    "review_rate",
    "very_high_risk_rate",
    "payment_incident_rate",
    "critical_high_incident_rate",
    "debit_service_mismatch_rate",
    "complaint_escalation_rate",
    "retry_risk_rate",
]


@dataclass
class MerchantBucket:
    bucket_start: int
    transaction_count: int = 0
    review_count: int = 0
    very_high_risk_count: int = 0
    payment_incident_count: int = 0
    critical_high_incident_count: int = 0
    debit_service_mismatch_count: int = 0
    complaint_escalation_count: int = 0
    retry_risk_count: int = 0

    def add(self, event: RiskTransactionEvent) -> None:
        self.transaction_count += 1
        if event.decision == "REVIEW":
            self.review_count += 1
        if event.fraud_risk_score >= VERY_HIGH_RISK_THRESHOLD:
            self.very_high_risk_count += 1
        if event.payment_incident_detected:
            self.payment_incident_count += 1
        if event.payment_incident_severity in {"CRITICAL", "HIGH"}:
            self.critical_high_incident_count += 1
        if event.payment_incident_type == "DEBIT_SERVICE_MISMATCH":
            self.debit_service_mismatch_count += 1
        if event.payment_incident_type == "COMPLAINT_ESCALATION_RISK":
            self.complaint_escalation_count += 1
        if event.payment_incident_type == "RETRY_RELATED_PAYMENT_RISK":
            self.retry_risk_count += 1

    def rates(self) -> dict[str, float]:
        denominator = max(self.transaction_count, 1)
        return {
            "review_rate": self.review_count / denominator,
            "very_high_risk_rate": self.very_high_risk_count / denominator,
            "payment_incident_rate": self.payment_incident_count / denominator,
            "critical_high_incident_rate": self.critical_high_incident_count / denominator,
            "debit_service_mismatch_rate": self.debit_service_mismatch_count / denominator,
            "complaint_escalation_rate": self.complaint_escalation_count / denominator,
            "retry_risk_rate": self.retry_risk_count / denominator,
        }


@dataclass
class MerchantRuntimeState:
    merchant_id: str
    buckets: list[MerchantBucket] = field(default_factory=list)
    ewma: dict[str, float] = field(default_factory=dict)
    status: str = "NORMAL"
    primary_driver: str = "NONE"
    alerts: list[dict[str, Any]] = field(default_factory=list)
    last_alert_at: dict[str, str] = field(default_factory=dict)


def bucket_start(timestamp: datetime, window_seconds: int) -> int:
    epoch = int(timestamp.timestamp())
    return epoch - (epoch % window_seconds)


def evaluate_stream_event(
    state: MerchantRuntimeState,
    event: RiskTransactionEvent,
    settings: StreamingSettings,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Update merchant state using dynamic baseline and return current status plus events."""
    event_time = datetime.fromisoformat(event.event_time.replace("Z", "+00:00"))
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=UTC)
    bucket_id = bucket_start(event_time, settings.window_seconds)
    bucket = _get_or_create_bucket(state, bucket_id)
    bucket.add(event)
    cutoff = bucket_id - settings.window_seconds * settings.baseline_buckets
    state.buckets = [item for item in state.buckets if item.bucket_start >= cutoff]

    current_rates = bucket.rates()
    baseline_buckets = [item for item in state.buckets if item.bucket_start < bucket_id]
    baseline = _baseline_stats(baseline_buckets)
    metric_states: dict[str, dict[str, float | str]] = {}
    highest_status = "NORMAL"
    primary_driver = "NONE"
    primary_score = -1.0
    for metric in STREAM_METRICS:
        current_value = current_rates[metric]
        stats = baseline[metric]
        std = _safe_std(stats["std"], stats["mean"])
        z_score = max(0.0, (current_value - stats["mean"]) / std)
        previous_ewma = state.ewma.get(metric, stats["mean"])
        ewma_value = settings.ewma_alpha * current_value + (1.0 - settings.ewma_alpha) * previous_ewma
        state.ewma[metric] = ewma_value
        ewma_z = max(0.0, (ewma_value - stats["mean"]) / std)
        status = _status_from_score(max(z_score, ewma_z), settings)
        metric_states[metric] = {
            "current": current_value,
            "baseline": stats["mean"],
            "baseline_std": stats["std"],
            "z_score": z_score,
            "ewma": ewma_value,
            "ewma_z_score": ewma_z,
            "relative_change": relative_change(current_value, stats["mean"]),
            "status": status,
        }
        score = max(z_score, ewma_z)
        if _status_rank(status) > _status_rank(highest_status) or (
            status == highest_status and score > primary_score
        ):
            highest_status = status
            primary_driver = metric.upper()
            primary_score = score

    previous_status = state.status
    state.status = highest_status
    state.primary_driver = primary_driver
    current_state = {
        "mode": "stream",
        "merchant_id": state.merchant_id,
        "status": highest_status,
        "primary_driver": primary_driver,
        "secondary_drivers": [
            metric.upper()
            for metric, values in metric_states.items()
            if values["status"] != "NORMAL" and metric.upper() != primary_driver
        ],
        "current_metrics": current_rates,
        "baseline_metrics": {metric: baseline[metric] for metric in STREAM_METRICS},
        "metrics": metric_states,
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "window_start": datetime.fromtimestamp(bucket_id, tz=UTC).isoformat(),
        "recommended_action": recommended_action(primary_driver),
    }
    events = _alert_events(state, current_state, previous_status, settings)
    return current_state, events


def redis_keys(merchant_id: str) -> dict[str, str]:
    return {
        "current": f"risk:merchant:{merchant_id}:current",
        "events": f"risk:merchant:{merchant_id}:events",
        "alerts": f"risk:merchant:{merchant_id}:alerts",
        "baseline": f"risk:merchant:{merchant_id}:baseline",
    }


async def persist_event_window(redis_client: Any, event: RiskTransactionEvent, settings: StreamingSettings) -> int:
    """Maintain a Redis sorted-set window without timestamp-only member collisions."""
    event_time = datetime.fromisoformat(event.event_time.replace("Z", "+00:00"))
    score = event_time.timestamp()
    cutoff = score - settings.window_seconds
    keys = redis_keys(event.merchant_id)
    member = f"{int(score * 1000)}:{event.event_id}"
    payload = json.dumps(event.to_dict())
    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(keys["events"], 0, cutoff)
    pipe.zadd(keys["events"], {member: score})
    pipe.hset(f"{keys['events']}:payloads", member, payload)
    pipe.expire(keys["events"], settings.window_seconds * 2)
    pipe.expire(f"{keys['events']}:payloads", settings.window_seconds * 2)
    pipe.zcard(keys["events"])
    result = await pipe.execute()
    return int(result[-1])


async def persist_current_state(
    redis_client: Any,
    merchant_id: str,
    current_state: dict[str, Any],
    alerts: list[dict[str, Any]],
    settings: StreamingSettings,
) -> None:
    keys = redis_keys(merchant_id)
    pipe = redis_client.pipeline()
    pipe.set(keys["current"], json.dumps(current_state), ex=settings.window_seconds * 4)
    pipe.set(keys["baseline"], json.dumps(current_state.get("baseline_metrics", {})), ex=settings.window_seconds * 4)
    for alert in alerts:
        pipe.lpush(keys["alerts"], json.dumps(alert))
    pipe.ltrim(keys["alerts"], 0, 99)
    pipe.expire(keys["alerts"], settings.window_seconds * 12)
    await pipe.execute()


async def load_current_state(redis_client: Any, merchant_id: str) -> dict[str, Any] | None:
    value = await redis_client.get(redis_keys(merchant_id)["current"])
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


async def load_alert_history(redis_client: Any, merchant_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = await redis_client.lrange(redis_keys(merchant_id)["alerts"], 0, max(limit - 1, 0))
    result = []
    for row in rows:
        if isinstance(row, bytes):
            row = row.decode("utf-8")
        result.append(json.loads(row))
    return result


def _get_or_create_bucket(state: MerchantRuntimeState, bucket_id: int) -> MerchantBucket:
    for bucket in state.buckets:
        if bucket.bucket_start == bucket_id:
            return bucket
    bucket = MerchantBucket(bucket_start=bucket_id)
    state.buckets.append(bucket)
    state.buckets.sort(key=lambda item: item.bucket_start)
    return bucket


def _baseline_stats(buckets: list[MerchantBucket]) -> dict[str, dict[str, float]]:
    rows = [bucket.rates() for bucket in buckets]
    result = {}
    for metric in STREAM_METRICS:
        values = [row[metric] for row in rows]
        if not values:
            result[metric] = {"mean": 0.0, "std": 0.0}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result[metric] = {"mean": mean, "std": math.sqrt(variance)}
    return result


def _safe_std(std: float, mean: float) -> float:
    if std > 1e-9:
        return std
    return max(abs(mean) * 0.25, 0.01)


def _status_from_score(score: float, settings: StreamingSettings) -> str:
    if score >= settings.z_critical:
        return "CRITICAL"
    if score >= settings.z_high:
        return "HIGH"
    if score >= settings.z_elevated:
        return "ELEVATED"
    return "NORMAL"


def _status_rank(status: str) -> int:
    return {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}.get(status, 0)


def _alert_events(
    state: MerchantRuntimeState,
    current_state: dict[str, Any],
    previous_status: str,
    settings: StreamingSettings,
) -> list[dict[str, Any]]:
    now = datetime.now(tz=UTC)
    status = current_state["status"]
    driver = current_state["primary_driver"]
    events: list[dict[str, Any]] = []
    if status == "NORMAL" and previous_status != "NORMAL":
        events.append(
            {
                "event_type": "recovery",
                "merchant_id": state.merchant_id,
                "previous_status": previous_status,
                "status": status,
                "event_time": now.isoformat(),
                "message": f"Merchant risk recovered from {previous_status} to NORMAL.",
            }
        )
        return events
    if status == "NORMAL":
        return events

    dedupe_key = f"{driver}:{status}"
    last_seen = state.last_alert_at.get(dedupe_key)
    if last_seen:
        last_time = datetime.fromisoformat(last_seen)
        if now - last_time < timedelta(seconds=settings.alert_cooldown_seconds):
            return events
    state.last_alert_at[dedupe_key] = now.isoformat()
    metric = current_state["metrics"][driver.lower()]
    alert = {
        "event_type": "alert",
        "merchant_id": state.merchant_id,
        "severity": status,
        "primary_driver": driver,
        "current_value": metric["current"],
        "baseline_value": metric["baseline"],
        "relative_change": metric["relative_change"],
        "recommended_action": current_state["recommended_action"],
        "window_start": current_state["window_start"],
        "event_time": now.isoformat(),
    }
    state.alerts.append(alert)
    events.append(alert)
    return events
