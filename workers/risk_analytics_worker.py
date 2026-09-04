from __future__ import annotations

import asyncio
import json
import logging
import signal
from collections import defaultdict
from typing import Any

from backend.config import get_settings
from src.events.schema import validate_transaction_event
from src.streaming.state import (
    MerchantRuntimeState,
    evaluate_stream_event,
    persist_current_state,
    persist_event_window,
)

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
except ImportError:  # pragma: no cover
    AIOKafkaConsumer = None  # type: ignore[assignment]
    AIOKafkaProducer = None  # type: ignore[assignment]

try:
    import redis.asyncio as redis_async
except ImportError:  # pragma: no cover
    redis_async = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("risk_analytics_worker")


async def run_worker() -> None:
    settings = get_settings()
    if not settings.streaming_enabled:
        logger.info("RISK_STREAM_MODE is local; worker is idle.")
        return
    if AIOKafkaConsumer is None or redis_async is None:
        raise RuntimeError("Streaming mode requires aiokafka and redis dependencies.")

    redis_client = redis_async.from_url(settings.redis_url, decode_responses=True)
    consumer = AIOKafkaConsumer(
        settings.transaction_topic,
        bootstrap_servers=settings.redpanda_bootstrap_servers,
        group_id="fraudguard-risk-analytics",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        enable_auto_commit=True,
    )
    alert_producer = None
    if AIOKafkaProducer is not None:
        alert_producer = AIOKafkaProducer(
            bootstrap_servers=settings.redpanda_bootstrap_servers,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )

    states: dict[str, MerchantRuntimeState] = defaultdict(lambda: MerchantRuntimeState(merchant_id=""))
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    await consumer.start()
    if alert_producer is not None:
        await alert_producer.start()
    logger.info("Risk analytics worker consuming topic %s", settings.transaction_topic)
    try:
        while not stop_event.is_set():
            result = await consumer.getmany(timeout_ms=1000, max_records=250)
            for messages in result.values():
                for message in messages:
                    await _handle_message(message.value, states, redis_client, settings, alert_producer)
    finally:
        await consumer.stop()
        if alert_producer is not None:
            await alert_producer.stop()
        await redis_client.aclose()
        logger.info("Risk analytics worker stopped.")


async def _handle_message(
    payload: dict[str, Any],
    states: dict[str, MerchantRuntimeState],
    redis_client: Any,
    settings: Any,
    alert_producer: Any | None,
) -> None:
    try:
        event = validate_transaction_event(payload)
    except ValueError as exc:
        logger.warning("Skipping invalid risk event: %s", exc)
        return

    state = states[event.merchant_id]
    if not state.merchant_id:
        state.merchant_id = event.merchant_id
    await persist_event_window(redis_client, event, settings)
    current_state, alerts = evaluate_stream_event(state, event, settings)
    await persist_current_state(redis_client, event.merchant_id, current_state, alerts, settings)
    for alert in alerts:
        logger.info("Risk %s for merchant %s: %s", alert["event_type"], event.merchant_id, alert)
        if alert_producer is not None:
            await alert_producer.send_and_wait(settings.alert_topic, alert)


if __name__ == "__main__":
    asyncio.run(run_worker())
