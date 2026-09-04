from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from backend.config import StreamingSettings, get_settings
from src.streaming.state import load_alert_history, load_current_state

try:  # Optional for local demo mode.
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover - exercised when dependency is absent.
    redis_async = None


def redis_available() -> bool:
    return redis_async is not None


async def current_state(merchant_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    merchant = merchant_id or settings.default_merchant_id
    if redis_async is None:
        return _empty_state(merchant, "Redis dependency is not installed.")
    client = redis_async.from_url(settings.redis_url, decode_responses=True)
    try:
        state = await load_current_state(client, merchant)
        return state or _empty_state(merchant, "No streaming state has been written yet.")
    except Exception as exc:
        return _empty_state(merchant, f"Redis unavailable: {exc}")
    finally:
        await client.aclose()


async def recent_alerts(merchant_id: str | None = None, limit: int = 20) -> dict[str, Any]:
    settings = get_settings()
    merchant = merchant_id or settings.default_merchant_id
    if redis_async is None:
        return {"mode": "stream", "merchant_id": merchant, "alerts": [], "message": "Redis dependency is not installed."}
    client = redis_async.from_url(settings.redis_url, decode_responses=True)
    try:
        return {"mode": "stream", "merchant_id": merchant, "alerts": await load_alert_history(client, merchant, limit)}
    except Exception as exc:
        return {"mode": "stream", "merchant_id": merchant, "alerts": [], "message": f"Redis unavailable: {exc}"}
    finally:
        await client.aclose()


async def sse_events(merchant_id: str | None = None) -> AsyncIterator[str]:
    settings = get_settings()
    merchant = merchant_id or settings.default_merchant_id
    last_updated_at: str | None = None
    while True:
        try:
            state = await current_state(merchant)
            event_name = "risk_state" if state.get("updated_at") and state.get("updated_at") != last_updated_at else "heartbeat"
            if event_name == "risk_state":
                last_updated_at = state.get("updated_at")
            yield _format_sse(event_name, state)
        except Exception as exc:  # Keep the SSE connection alive in degraded mode.
            yield _format_sse("heartbeat", _empty_state(merchant, f"SSE degraded: {exc}"))
        await asyncio.sleep(settings.sse_heartbeat_seconds)


async def redis_ping(settings: StreamingSettings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    if not settings.streaming_enabled:
        return {"enabled": False, "status": "disabled"}
    if redis_async is None:
        return {"enabled": True, "status": "missing_dependency"}
    client = redis_async.from_url(settings.redis_url, decode_responses=True)
    try:
        await client.ping()
        return {"enabled": True, "status": "ok"}
    except Exception as exc:
        return {"enabled": True, "status": "unavailable", "message": str(exc)}
    finally:
        await client.aclose()


def _empty_state(merchant_id: str, message: str) -> dict[str, Any]:
    return {
        "mode": "stream",
        "merchant_id": merchant_id,
        "status": "UNKNOWN",
        "primary_driver": "NONE",
        "secondary_drivers": [],
        "current_metrics": {},
        "baseline_metrics": {},
        "metrics": {},
        "updated_at": None,
        "message": message,
    }


def _format_sse(event_name: str, data: dict[str, Any]) -> str:
    payload = {**data, "server_time": datetime.now(tz=UTC).isoformat()}
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"
