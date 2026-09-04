from __future__ import annotations

import os
from dataclasses import dataclass


VALID_STREAM_MODES = {"local", "stream"}


@dataclass(frozen=True)
class StreamingSettings:
    stream_mode: str = "local"
    redpanda_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "redis://localhost:6379/0"
    transaction_topic: str = "payment-transactions"
    alert_topic: str = "risk-alerts"
    window_seconds: int = 300
    baseline_buckets: int = 12
    z_elevated: float = 2.0
    z_high: float = 3.0
    z_critical: float = 4.0
    ewma_alpha: float = 0.3
    publisher_queue_size: int = 1000
    publish_timeout_seconds: float = 1.0
    alert_cooldown_seconds: int = 300
    sse_heartbeat_seconds: int = 10
    default_merchant_id: str = "merchant_demo_001"

    @property
    def streaming_enabled(self) -> bool:
        return self.stream_mode == "stream"


def get_settings() -> StreamingSettings:
    """Read and validate FraudGuard runtime settings from environment variables."""
    mode = os.getenv("RISK_STREAM_MODE", "local").strip().lower()
    if mode not in VALID_STREAM_MODES:
        raise ValueError(f"RISK_STREAM_MODE must be one of {sorted(VALID_STREAM_MODES)}.")

    return StreamingSettings(
        stream_mode=mode,
        redpanda_bootstrap_servers=_env_text("REDPANDA_BOOTSTRAP_SERVERS", "localhost:9092"),
        redis_url=_env_text("REDIS_URL", "redis://localhost:6379/0"),
        transaction_topic=_env_text("RISK_TRANSACTION_TOPIC", "payment-transactions"),
        alert_topic=_env_text("RISK_ALERT_TOPIC", "risk-alerts"),
        window_seconds=_positive_int("RISK_WINDOW_SECONDS", 300),
        baseline_buckets=_positive_int("RISK_BASELINE_BUCKETS", 12),
        z_elevated=_positive_float("RISK_Z_ELEVATED", 2.0),
        z_high=_positive_float("RISK_Z_HIGH", 3.0),
        z_critical=_positive_float("RISK_Z_CRITICAL", 4.0),
        ewma_alpha=_bounded_float("RISK_EWMA_ALPHA", 0.3, 0.0, 1.0),
        publisher_queue_size=_positive_int("RISK_PUBLISHER_QUEUE_SIZE", 1000),
        publish_timeout_seconds=_positive_float("RISK_PUBLISH_TIMEOUT_SECONDS", 1.0),
        alert_cooldown_seconds=_positive_int("RISK_ALERT_COOLDOWN_SECONDS", 300),
        sse_heartbeat_seconds=_positive_int("RISK_SSE_HEARTBEAT_SECONDS", 10),
        default_merchant_id=_env_text("RISK_DEFAULT_MERCHANT_ID", "merchant_demo_001"),
    )


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} must not be empty.")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _bounded_float(name: str, default: float, lower: float, upper: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= lower or value > upper:
        raise ValueError(f"{name} must be greater than {lower} and at most {upper}.")
    return value
