from __future__ import annotations

import asyncio

from backend.config import StreamingSettings
from backend.services.event_producer import AsyncRiskEventProducer


def test_producer_is_disabled_in_local_mode() -> None:
    async def run() -> None:
        producer = AsyncRiskEventProducer(StreamingSettings(stream_mode="local"))

        queued = await producer.enqueue({"event_id": "evt"})

        assert queued is False
        assert producer.metrics.events_enqueued == 0

    asyncio.run(run())


def test_producer_drops_when_queue_is_full() -> None:
    async def run() -> None:
        producer = AsyncRiskEventProducer(StreamingSettings(stream_mode="stream", publisher_queue_size=1))

        first = await producer.enqueue({"event_id": "evt-1"})
        second = await producer.enqueue({"event_id": "evt-2"})

        assert first is True
        assert second is False
        assert producer.metrics.events_enqueued == 1
        assert producer.metrics.events_dropped == 1

    asyncio.run(run())
