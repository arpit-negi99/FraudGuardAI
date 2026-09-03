# FraudGuard AI

FraudGuard AI is a defense-only merchant risk operations demo. It combines transaction fraud scoring, payment lifecycle incident detection, and operational spike monitoring in a React + FastAPI application.

The product is frozen for submission preparation. The frozen fraud threshold is `0.60`.

## What It Does

- Scores merchant transactions with a frozen XGBoost fraud-risk model.
- Recommends `ALLOW` or `REVIEW`; it never automatically blocks payments.
- Shows SHAP-based top contributors for transaction review.
- Detects simulated payment lifecycle incidents with deterministic rules.
- Monitors synthetic population-level risk spikes with rolling statistical windows.
- Clearly separates real held-out fraud evaluation from synthetic Module 2 and Module 3 demonstrations.

## Modules

| Module | Purpose | Data | Method |
| --- | --- | --- | --- |
| Module 1 | Transaction fraud risk | IEEE-CIS labeled training data | XGBoost + SHAP |
| Module 2 | Payment lifecycle incidents | Synthetic payment-event data | Deterministic lifecycle rules |
| Module 3 | Operational spike monitoring | Synthetic monitoring stream | Z-score + EWMA windows |

## Final Held-Out Module 1 Results

The held-out test split was opened only after validation-based model and threshold selection.

| Metric | Value |
| --- | ---: |
| Precision | 0.364018 |
| Recall | 0.563088 |
| F1 | 0.442180 |
| PR-AUC | 0.514931 |
| ROC-AUC | 0.891247 |
| Review rate | 0.053838 |

## Runtime Architecture

```text
React + Vite UI
    |
FastAPI backend
    |
    +-- Module 1: frozen XGBoost + preprocessor + SHAP
    +-- Module 2: payment lifecycle rules
    +-- Module 3: statistical monitoring
```

Streamlit remains available in `app.py` as a fallback/debug UI. The React frontend is the primary submission UI.

## Local Setup

```bash
python -m pip install -r requirements.txt
cd frontend
npm install
```

## Run Locally

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

The frontend can read `VITE_API_BASE_URL`; see `frontend/.env.example`.

## Key API Endpoints

- `GET /health`
- `GET /demo/transactions`
- `GET /demo/transactions/{transaction_id}`
- `POST /predict`
- `POST /predict/batch`
- `GET /risk/review-queue`
- `GET /policy/presets`
- `POST /policy/simulate`
- `GET /evaluation/final`
- `GET /incidents/lifecycles`
- `GET /incidents/lifecycles/{payment_id}`
- `GET /monitoring/current`
- `GET /monitoring/scenarios`

## Demo Guide

Use [documentation/DEMO-FLOW.md](documentation/DEMO-FLOW.md) for the 5-minute walkthrough.

Screenshots should be captured after final visual review and stored in `documentation/screenshots/`.

## Tests

Python:

```bash
python -m pip check
python -m pytest -q
```

Frontend:

```bash
cd frontend
npm test
npm run build
```

## Dataset Safety

Raw IEEE-CIS CSV files live under `data/raw/` and are intentionally ignored by Git. They are not required for normal demo runtime. Do not commit the raw Kaggle dataset.

## Limitations

- Fraud scores are model risk scores, not guaranteed calibrated probabilities.
- Held-out recall is 0.563088 at threshold `0.60`.
- IEEE-CIS features are anonymized and historical.
- Cost simulations use assumptions, not measured merchant savings.
- Module 2 and Module 3 use synthetic data for demonstration.
- No real payment-provider credentials, webhooks, refunds, chargebacks, or payment actions are included.

## Frozen System Notes

See [documentation/FROZEN-SYSTEM.md](documentation/FROZEN-SYSTEM.md) for the artifact audit, freeze rules, and module boundaries.
