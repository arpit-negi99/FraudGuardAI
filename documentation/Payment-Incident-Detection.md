# Payment Incident Detection

Module 2 detects payment lifecycle inconsistencies that may create merchant loss, refund burden, disputes, support tickets, or poor customer experience.

This module is separate from Module 1.

## Module 1

Module 1 is transaction fraud scoring using the IEEE-CIS Fraud Detection dataset, a frozen XGBoost model, frozen preprocessing, SHAP explanations, and an `ALLOW` / `REVIEW` threshold policy.

## Module 2

Module 2 is payment lifecycle incident detection using explicitly synthetic payment-event data.

The generated dataset is synthetic / simulated payment-event data. It is simulated for hackathon demonstration and does not represent proprietary payment-provider data. It does not claim to reproduce Razorpay production systems.

## Input Schema

Payment events contain:

- `payment_id`
- `merchant_id`
- `amount`
- `payment_method`
- `bank_debited`
- `gateway_status`
- `order_status`
- `service_delivered`
- `callback_received`
- `refund_status`
- `retry_count`
- `time_since_payment_minutes`
- `customer_complaint`
- `fraud_risk_score`

`fraud_risk_score` is optional and represents a possible Module 1 output. Module 2 still works when it is unavailable. Synthetic generation may include representative simulated fraud scores, but it does not call the XGBoost model.

## Incident Types

- `DEBIT_SERVICE_MISMATCH`
- `LATE_AUTHORIZATION_RISK`
- `CAPTURED_BUT_UNFULFILLED`
- `REFUND_REQUIRED`
- `RETRY_RELATED_PAYMENT_RISK`
- `COMPLAINT_ESCALATION_RISK`
- `NORMAL_PAYMENT`

## Severity

Severity is deterministic:

- `CRITICAL`: complaint plus customer debit/payment capture or authorization, service/refund unresolved.
- `HIGH`: captured-but-unfulfilled, refund required, or debited failed/unresolved payment older than 30 minutes.
- `MEDIUM`: unresolved late authorization or retry-related payment risk.
- `LOW`: reserved for minor future consistency checks.
- `NONE`: normal, safely failed, or already resolved lifecycle state.

Rule thresholds are intentionally simple:

- significant unresolved duration: at least 30 minutes
- high retry count: at least 3 retries

## Actions

Recommended actions are defensive and operational only:

- `NO_ACTION`
- `VERIFY_PAYMENT`
- `CHECK_ORDER`
- `INITIATE_REFUND`
- `CONTACT_CUSTOMER`
- `ESCALATE_REVIEW`
- `MONITOR`

The module does not manipulate payment gateways, refunds, orders, or customer accounts.

## Synthetic Data

The generator creates a scenario first, then generates fields consistent with that scenario. Synthetic labels come from scenario generation, not by copying the rule-engine output.

Output dataset:

`data/synthetic/payment_incidents.csv`

Summary artifact:

`artifacts/results/payment_incident_data_summary.json`

## Current Limitation

No machine-learning model exists yet for Module 2. The current baseline is deterministic operational rule logic.

## Step 2 Evaluation

The deterministic detector has been evaluated against independently scenario-generated synthetic labels.

Artifacts:

- `artifacts/results/payment_incident_rule_metrics.json`
- `artifacts/results/payment_incident_rule_per_class.csv`
- `artifacts/results/payment_incident_rule_confusion_matrix.png`
- `artifacts/results/payment_incident_stress_metrics.json`
- `artifacts/results/payment_incident_stress_per_class.csv`
- `artifacts/results/payment_incident_stress_confusion_matrix.png`
- `artifacts/results/payment_incident_error_examples.json`
- `artifacts/results/payment_incident_rule_precedence.json`

Rule precedence:

1. Highest severity wins.
2. If severities tie, deterministic rule-check order wins.

The current evaluation found no binary or incident-type errors on the standard or stress synthetic datasets. This does not prove production performance; it means the current deterministic rules match the current synthetic scenario design. The next useful work is product integration and broader scenario coverage, not adding ML just for novelty.

## Product Integration

Module 2 is integrated into the React + FastAPI product as a separate operational workflow from fraud review.

FastAPI endpoints:

- `GET /incidents`
- `GET /incidents/summary`
- `GET /incidents/{payment_id}`
- `POST /incidents/evaluate`
- `GET /incidents/types`

React UI:

- Adds `Payment Incidents` as a primary navigation item.
- Shows active incident, critical, high-priority, and incident-rate summary cards.
- Provides severity, incident-type, and payment-ID filters.
- Shows payment lifecycle fields, actual rule-engine reasons, recommended operational action, and fraud-vs-incident comparison.
- Keeps Module 1 fraud risk and Module 2 payment incident risk visually separate.

Demo payment IDs:

- `pay_syn_000007`: low fraud score with high debit-service mismatch.
- `pay_syn_000003`: captured but unfulfilled payment.
- `pay_syn_000012`: late authorization risk.
- `pay_stress_000038`: high synthetic fraud score with no payment lifecycle incident in the stress dataset.

Limitations:

- Module 2 still uses synthetic payment-event data.
- Module 2 is deterministic rule logic, not ML.
- No Razorpay production validation is claimed.
- No real gateway action, refund, chargeback, webhook, or customer action is performed.

## Multi-Step Lifecycle Reasoning

The snapshot detector remains intact through `evaluate_payment_incident(event)`.

The lifecycle engine replays ordered lifecycle events into a snapshot-compatible `PaymentEvent` after every event, then calls the existing snapshot evaluator. This produces:

- current reconstructed state
- current incident and severity
- timeline evaluations
- first incident detection time
- highest severity observed
- resolution status and resolution time where applicable
- event-by-event rule reasons

Supported lifecycle event types:

- `PAYMENT_CREATED`
- `BANK_DEBITED`
- `CALLBACK_RECEIVED`
- `CALLBACK_MISSING`
- `PAYMENT_AUTHORIZED`
- `PAYMENT_CAPTURED`
- `PAYMENT_FAILED`
- `CUSTOMER_RETRY`
- `ORDER_CONFIRMED`
- `ORDER_FULFILLED`
- `ORDER_FAILED`
- `REFUND_INITIATED`
- `REFUND_PROCESSED`
- `REFUND_FAILED`
- `CUSTOMER_COMPLAINT`

Lifecycle statuses:

- `NORMAL`
- `ACTIVE_INCIDENT`
- `RESOLVING`
- `RESOLVED`

The lifecycle simulator is scenario-first and does not call the detector to create labels. It generates synthetic timelines only.

Generated artifacts:

- `data/synthetic/payment_lifecycles.json`
- `artifacts/results/payment_lifecycle_summary.json`

Generated lifecycle scenarios:

- `NORMAL_SUCCESS`
- `SAFE_FAILURE`
- `DEBIT_GATEWAY_FAILURE`
- `LATE_AUTHORIZATION`
- `CAPTURED_UNFULFILLED`
- `REFUND_RESOLUTION`
- `COMPLAINT_ESCALATION`
- `RETRY_RELATED_RISK`

Current generated lifecycle summary:

- Lifecycle count: 3000
- Average events: 4.729333333333333
- Active: 1237
- Resolved: 244
- Normal: 1519
- Median resolution time: 68.0 minutes

Evaluated lifecycle API summary:

- Critical histories: 120
- Highest severity distribution: CRITICAL=120, HIGH=929, MEDIUM=432, NONE=1519

Lifecycle API endpoints:

- `GET /incidents/lifecycles`
- `GET /incidents/lifecycles/summary`
- `GET /incidents/lifecycles/{payment_id}`
- `POST /incidents/lifecycles/evaluate`

Lifecycle demo payment IDs:

- `pay_life_000004`: late authorization.
- `pay_life_000017`: captured/unfulfilled path that currently recommends refund initiation.
- `pay_life_000033`: refund resolution, resolved after 68 minutes.
- `pay_life_000007`: complaint escalation to critical severity.

This lifecycle upgrade still does not use real payment-provider production data and does not perform real payment actions.
