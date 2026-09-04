# Streaming Architecture

FraudGuard AI keeps the frozen scoring path as the source of truth. Streaming mode adds an optional analytics path after prediction; it does not retrain the model, change the `0.60` threshold, alter Module 2 rules, or replace the existing synthetic monitoring demo.

## Modes

`RISK_STREAM_MODE=local` is the default. The app runs exactly as a lightweight React + FastAPI demo and uses the existing local synthetic monitoring endpoints.

`RISK_STREAM_MODE=stream` enables a production-like local analytics path:

```text
POST /predict
  -> frozen XGBoost score and ALLOW/REVIEW decision
  -> bounded async event queue
  -> Redpanda topic payment-transactions
  -> risk analytics worker
  -> Redis merchant windows/current state/alerts
  -> /monitoring/current and /monitoring/stream
  -> React Risk Monitor
```

## Event Contract

Events are schema version `1` and include a UUID `event_id`, event timestamp, merchant identifier, transaction/payment identifiers, fraud risk score, locked threshold, decision, priority, and optional payment incident fields.

The producer never blocks the request path on broker acknowledgement. Queue-full or broker-down cases drop analytics events and keep fraud scoring available.

## Redis State

Merchant state is keyed only by `merchant_id`:

```text
risk:merchant:{merchant_id}:current
risk:merchant:{merchant_id}:events
risk:merchant:{merchant_id}:alerts
risk:merchant:{merchant_id}:baseline
```

Sliding transaction windows use Redis sorted sets. Members include both timestamp and UUID to avoid overwriting two events with the same timestamp.

## Monitoring

The worker computes merchant-level review-rate and payment-incident-rate signals using fixed windows, a dynamic baseline that excludes the current bucket, z-scores, and EWMA trend smoothing. Alerts are deduplicated with cooldowns. Recovery events are emitted when a merchant returns to normal.

This is a local production-like demo architecture, not a real payment-gateway integration or production-scale load test.
