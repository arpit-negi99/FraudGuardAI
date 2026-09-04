from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from backend.config import StreamingSettings, get_settings
from src.events.schema import RiskTransactionEvent

try:  # pragma: no cover - import path depends on optional streaming install
    from aiokafka import AIOKafkaProducer
except Exception:  # pragma: no cover
    AIOKafkaProducer = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class EventProducerMetrics:
    events_enqueued: int = 0
    events_published: int = 0
    events_dropped: int = 0
    publish_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class AsyncRiskEventProducer:
    """Bounded fail-open analytics event publisher for the prediction hot path."""

    def __init__(self, settings: StreamingSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.metrics = EventProducerMetrics()
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self.settings.publisher_queue_size,
        )
        self._producer: Any | None = None
        self._task: asyncio.Task | None = None
        self._running = False

    @property
    def enabled(self) -> bool:
        return self.settings.streaming_enabled

    async def start(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        if AIOKafkaProducer is None:
            logger.warning("analytics producer disabled: aiokafka is not installed")
        else:
            try:
                self._producer = AIOKafkaProducer(
                    bootstrap_servers=self.settings.redpanda_bootstrap_servers,
                    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
                )
                await self._producer.start()
            except Exception as exc:
                self._producer = None
                self.metrics.publish_failures += 1
                logger.warning("broker unavailable; analytics events will be dropped: %s", exc)
        self._task = asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None

    async def enqueue(self, event: RiskTransactionEvent | dict[str, Any]) -> bool:
        """Queue an event without blocking prediction; return False when dropped."""
        if not self.enabled:
            return False
        payload = event.to_dict() if isinstance(event, RiskTransactionEvent) else dict(event)
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.metrics.events_dropped += 1
            logger.warning("analytics event dropped because publisher queue is full")
            return False
        self.metrics.events_enqueued += 1
        logger.info("analytics event queued", extra={"analytics_event_queued": True})
        return True

    async def _publish_loop(self) -> None:
        while self._running:
            payload = await self.queue.get()
            try:
                if self._producer is None:
                    self.metrics.publish_failures += 1
                    continue
                await asyncio.wait_for(
                    self._producer.send_and_wait(self.settings.transaction_topic, payload),
                    timeout=self.settings.publish_timeout_seconds,
                )
                self.metrics.events_published += 1
                logger.info("analytics event published", extra={"analytics_event_published": True})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.metrics.publish_failures += 1
                logger.warning("analytics event publish failed: %s", exc)
            finally:
                self.queue.task_done()

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "stream_mode": "local",
                "redpanda": "disabled",
                "analytics_worker": "disabled",
                "event_metrics": self.metrics.to_dict(),
            }
        return {
            "stream_mode": "stream",
            "redpanda": "connected" if self._producer is not None else "unavailable",
            "analytics_worker": "external",
            "queue_size": self.queue.qsize(),
            "queue_capacity": self.settings.publisher_queue_size,
            "event_metrics": self.metrics.to_dict(),
        }


producer = AsyncRiskEventProducer()
