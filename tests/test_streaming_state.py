from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from backend.config import StreamingSettings
from src.events.schema import build_transaction_event
from src.streaming.state import MerchantRuntimeState, evaluate_stream_event, persist_event_window


class FakePipeline:
    def __init__(self) -> None:
        self.commands = []

    def zremrangebyscore(self, key, minimum, maximum):
        self.commands.append(("zremrangebyscore", key, minimum, maximum))
        return self

    def zadd(self, key, mapping):
        self.commands.append(("zadd", key, mapping))
        return self

    def hset(self, key, field, value):
        self.commands.append(("hset", key, field, value))
        return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl))
        return self

    def zcard(self, key):
        self.commands.append(("zcard", key))
        return self

    async def execute(self):
        return [None for _ in self.commands[:-1]] + [1]


class FakeRedis:
    def __init__(self) -> None:
        self.pipeline_instance = FakePipeline()

    def pipeline(self):
        return self.pipeline_instance


def test_persist_event_window_uses_timestamp_and_uuid_member_with_ttl() -> None:
    async def run() -> None:
        redis = FakeRedis()
        event = build_transaction_event(
            {"risk_score": 0.8, "decision": "REVIEW", "threshold": 0.6, "priority": "High"},
            {"merchant_id": "merchant_001", "TransactionID": 1},
        )

        count = await persist_event_window(redis, event, StreamingSettings(window_seconds=60))

        zadd = [command for command in redis.pipeline_instance.commands if command[0] == "zadd"][0]
        member = next(iter(zadd[2].keys()))
        assert count == 1
        assert event.event_id in member
        assert ":" in member
        assert any(command[0] == "expire" and command[2] == 120 for command in redis.pipeline_instance.commands)

    asyncio.run(run())


def test_dynamic_baseline_excludes_current_bucket_and_alerts() -> None:
    settings = StreamingSettings(window_seconds=60, baseline_buckets=5, z_elevated=1.5, z_high=2.0, z_critical=3.0)
    state = MerchantRuntimeState(merchant_id="merchant_001")
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(5):
        event = _event(base_time + timedelta(minutes=index), decision="ALLOW", score=0.1)
        evaluate_stream_event(state, event, settings)

    spike = _event(base_time + timedelta(minutes=5), decision="REVIEW", score=0.95)
    current, alerts = evaluate_stream_event(state, spike, settings)

    assert current["status"] in {"ELEVATED", "HIGH", "CRITICAL"}
    assert current["baseline_metrics"]["review_rate"]["mean"] == 0.0
    assert alerts


def test_recovery_event_is_emitted_after_alert_status_normalizes() -> None:
    settings = StreamingSettings(window_seconds=60, baseline_buckets=5, z_elevated=1.5, z_high=2.0, z_critical=3.0)
    state = MerchantRuntimeState(merchant_id="merchant_001", status="HIGH")
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(5):
        evaluate_stream_event(state, _event(base_time + timedelta(minutes=index), decision="REVIEW", score=0.95), settings)
    current, events = evaluate_stream_event(state, _event(base_time + timedelta(minutes=5), decision="REVIEW", score=0.95), settings)

    assert current["status"] == "NORMAL"
    assert events[0]["event_type"] == "recovery"


def _event(event_time: datetime, decision: str, score: float):
    event = build_transaction_event(
        {"risk_score": score, "decision": decision, "threshold": 0.6, "priority": "High"},
        {"merchant_id": "merchant_001", "TransactionID": int(event_time.timestamp())},
    )
    return type(event)(
        **{
            **event.to_dict(),
            "event_time": event_time.isoformat(),
        }
    )
