# Demo Flow

Target navigation time: under 5 minutes.

## Demo 1 - Dashboard Overview

Open the React app.

Show:

- fraud review workload
- operational risk status
- payment incident summary
- lifecycle incident summary

Message: FraudGuard separates transaction fraud, payment lifecycle incidents, and population-level operational spikes.

## Demo 2 - Fraud Transaction Review

Open `Review Queue`.

Show:

- `REVIEW` transactions ranked by risk
- risk score
- decision
- top signal

Open a transaction detail.

Show:

- risk score
- `ALLOW` / `REVIEW`
- SHAP contributors
- session-only analyst actions

Message: Module 1 is frozen XGBoost decision support, not automatic blocking.

## Demo 3 - Payment Lifecycle Incident

Open `Payment Incidents`.

Recommended IDs:

- `pay_life_000004`: late authorization
- `pay_life_000017`: captured/unfulfilled current refund-required path
- `pay_life_000033`: refund resolution
- `pay_life_000007`: complaint escalation to critical

Show:

- lifecycle summary cards
- payment lifecycle fields
- timeline events
- first incident time
- highest severity observed
- recommended action

Message: Fraud risk and payment incident risk are different operational signals.

## Demo 4 - Operational Spike

Open `Risk Monitor`.

Use `Synthetic demo scenario`.

Show:

- Normal activity: operational risk normal
- Debit-service mismatch spike: reconciliation/callback driver
- Mixed spike: fraud and payment operations move together
- Recovery: latest window returns to normal

Message: Module 3 detects population-level changes using statistical monitoring, not another classifier.

## Demo 5 - Evaluation Credibility

Open `About`.

Show:

- Module 1 held-out metrics
- synthetic scenario boundaries for Modules 2 and 3
- limitations and defense-only framing

Message: Real held-out evaluation is reported only for Module 1. Synthetic validation is clearly labeled for Modules 2 and 3.
