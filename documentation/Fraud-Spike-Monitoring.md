# Fraud Spike & Operational Risk Monitoring

Module 3 detects unusual increases across merchant risk streams.

It is separate from:

- Module 1 transaction fraud detection
- Module 2 payment incident detection

Module 3 consumes fraud-risk and payment-incident outputs where available, then evaluates population-level windows with explainable statistics. It does not train a new classifier.

## Synthetic Monitoring Stream

The monitoring stream is synthetic demo data with synthetic timestamps. It is not live merchant traffic and is not real payment-provider data.

Stream records include:

- `event_id`
- `event_time`
- `transaction_id`
- `payment_id`
- `merchant_id`
- `fraud_risk_score`
- `fraud_decision`
- `payment_incident_detected`
- `payment_incident_type`
- `payment_incident_severity`
- `amount`
- `payment_method`
- `scenario_type`
- `expected_spike`

## Metrics

Window-level metrics include:

- `transaction_count`
- `mean_fraud_risk`
- `review_count`
- `review_rate`
- `very_high_risk_count`
- `very_high_risk_rate`
- `payment_count`
- `payment_incident_count`
- `payment_incident_rate`
- `critical_incident_count`
- `high_incident_count`
- `critical_high_incident_rate`
- `debit_service_mismatch_count`
- `complaint_escalation_count`
- `retry_risk_count`
- `captured_unfulfilled_count`

The very-high-risk monitoring threshold is `fraud_risk_score >= 0.90`. This is monitoring-only and does not change Module 1 threshold `0.60`.

## Windowing

Default window size: 15 minutes.

The implementation is offline and simulated. It does not use Kafka, background workers, live webhooks, or production merchant streams.

## Baseline

The baseline is built from the initial normal period only. Later windows are evaluated against that historical baseline and are not used to rewrite earlier baseline statistics.

Baseline statistics include:

- mean
- standard deviation
- median

## Detection

The detector calculates z-scores:

```text
z = (current_metric - baseline_mean) / baseline_std
```

Severity thresholds:

- `NORMAL`: z < 2.0
- `ELEVATED`: 2.0 <= z < 3.0
- `HIGH`: 3.0 <= z < 4.0
- `CRITICAL`: z >= 4.0

EWMA is also calculated with alpha `0.50` to capture sustained changes.

Overall operational risk is the highest meaningful monitored anomaly severity. The system returns a primary driver and secondary drivers instead of a fake probability.

## Actions

Recommended actions are deterministic and defensive:

- review-rate or very-high-risk spike: inspect high-risk transactions and the review queue
- payment-incident-rate spike: inspect lifecycle incidents and reconciliation states
- debit-service-mismatch spike: check callback and reconciliation flow
- complaint spike: review unresolved complaints and refund status
- retry-risk spike: inspect retry patterns and unresolved earlier payment attempts

No automatic blocking, refund, chargeback, customer contact, or gateway action is performed.

## Scenarios

The synthetic stream includes:

- `NORMAL`
- `FRAUD_RISK_SPIKE`
- `PAYMENT_INCIDENT_SPIKE`
- `DEBIT_SERVICE_MISMATCH_SPIKE`
- `COMPLAINT_SPIKE`
- `RETRY_SPIKE`
- `MIXED_RISK_SPIKE`
- `RECOVERY`

Scenario identity and expected spike labels are generated first. The simulator does not call the spike detector to create labels.

## Artifacts

- `data/synthetic/monitoring_stream.csv`
- `artifacts/results/monitoring_stream_summary.json`
- `artifacts/results/spike_monitor_metrics.json`
- `artifacts/results/spike_monitor_windows.csv`
- `artifacts/results/spike_monitor_scenario_metrics.csv`
- `artifacts/results/spike_monitor_detection_delay.json`
- `artifacts/results/review_rate_over_time.png`
- `artifacts/results/payment_incident_rate_over_time.png`
- `artifacts/results/operational_risk_over_time.png`

## Current Evaluation

Generated stream rows: 24800.

Window size: 15 minutes.

Evaluation results:

- Precision: 0.9178743961352657
- Recall: 1.0
- F1: 0.9571788413098236
- False alert rate: 0.14166666666666666
- Mean detection delay: 0.0 minutes
- Median detection delay: 0.0 minutes
- Max detection delay: 0 minutes

Scenario performance:

- `NORMAL`: some false-alert windows observed, 5 of 80 windows alerted.
- `FRAUD_RISK_SPIKE`: detected.
- `PAYMENT_INCIDENT_SPIKE`: detected.
- `DEBIT_SERVICE_MISMATCH_SPIKE`: detected.
- `COMPLAINT_SPIKE`: detected.
- `RETRY_SPIKE`: detected.
- `MIXED_RISK_SPIKE`: detected.
- `RECOVERY`: returns to normal in the latest window, with 12 early recovery alert windows from elevated recent history.

## Limitations

- Synthetic monitoring stream only.
- Statistical detection only.
- No production merchant feed.
- No real Razorpay or payment-provider validation.
- No automatic blocking, refund, or operational action.
- Defense-only monitoring.
