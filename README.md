# FraudGuard AI

Cost-aware merchant transaction fraud-risk detection using the IEEE-CIS Fraud Detection dataset.

FraudGuard AI is a defense-only decision-support system. It scores transactions, recommends `ALLOW` or `REVIEW`, explains model signals with SHAP, and helps reason about review workload and modeled fraud cost. It does not automatically block payments.

## Problem

Merchants lose money when fraudulent transactions are approved, but overly aggressive fraud controls create manual-review cost and friction for legitimate customers.

## Solution

FraudGuard combines a frozen XGBoost fraud-risk model with a validation-selected operating threshold. The client UI focuses on the merchant workflow:

- transaction risk scoring
- `ALLOW` / `REVIEW` policy
- SHAP explanations
- review queue
- batch scoring through the API
- cost-aware policy simulator
- risk monitoring with lightweight rolling review-rate spike status

Module 2 adds payment lifecycle incident detection using deterministic rules and explicitly synthetic / simulated payment-event data. It is separate from IEEE-CIS fraud scoring and does not claim to use Razorpay production data.

## Two Risk Modules

### Transaction Fraud Detection

Uses XGBoost and the IEEE-CIS Fraud Detection dataset to identify transactions that resemble historically fraudulent transactions. The frozen threshold remains `0.60`; the UI presents this as an `ALLOW` / `REVIEW` decision-support signal.

### Payment Incident Detection

Uses deterministic payment-lifecycle safeguards and synthetic payment-event data to identify unresolved payment-state inconsistencies that may lead to complaints, refunds, or disputes. This module is separate from transaction fraud scoring and performs no real gateway, refund, or chargeback action.

## Architecture

```text
                       FraudGuard AI
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Fraud Risk Engine          Payment Incident Engine
         XGBoost                     Deterministic Rules
              |                           |
              +-------------+-------------+
                            |
                            v
                     Risk Operations
                            |
             +--------------+--------------+
             |                             |
             v                             v
       Fraud Review                  Incident Response
```

The React frontend is the primary submission UI. FastAPI reuses the existing Python inference pipeline and never duplicates fraud scoring logic in JavaScript. Streamlit remains as a fallback/debug UI in `app.py`.

Module 2 API endpoints:

- `GET /incidents`
- `GET /incidents/summary`
- `GET /incidents/{payment_id}`
- `POST /incidents/evaluate`
- `GET /incidents/types`

## Final Held-Out Results

Frozen threshold: `0.60`

| Metric | Held-out test |
| --- | ---: |
| Precision | 36.4% |
| Recall | 56.3% |
| F1 | 44.2% |
| PR-AUC | 0.515 |
| ROC-AUC | 0.891 |
| Review rate | 5.38% |

These values come from the chronological held-out test artifact. The threshold was selected before held-out evaluation and must not be tuned against these results.

## Screenshots

Screenshots will be added manually after visual review.

## Local Setup

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Running Locally

Backend:

```bash
python -m uvicorn backend.api:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm run dev
```

Default frontend URL: `http://localhost:5173`

The frontend reads `VITE_API_BASE_URL`; see `frontend/.env.example`.

Primary UI sections include Dashboard, Transactions, Review Queue, Payment Incidents, Risk Monitor, Policy, and About.

## Testing

Python:

```bash
python -m pip check
python -m pytest -q
```

Frontend:

```bash
cd frontend
npm run build
npm test
```

## Limitations

- Dataset features are partly anonymized.
- Risk score is not guaranteed calibrated probability.
- Cost values are modeled assumptions, not actual merchant savings.
- FraudGuard does not automatically block transactions.
- Held-out recall is 56.3%.
- The historical IEEE-CIS dataset may not reflect current production fraud patterns.
- Payment incident data is synthetic and does not represent proprietary payment-provider systems.

## Defense-Only Design

FraudGuard is only for defensive fraud detection and review prioritization. It does not provide fraud-generation workflows, evasion guidance, offensive simulation, credential collection, or automatic payment-decline behavior.
