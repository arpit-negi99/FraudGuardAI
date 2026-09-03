# Frozen System

FraudGuard AI is frozen for submission preparation. No additional core modules should be added unless a blocking issue is discovered.

## Runtime Architecture

```text
React + Vite
  |
FastAPI
  |
+----------------+--------------------+----------------------+
|                |                    |                      |
Module 1         Module 2             Module 3
Fraud Model      Incident Engine      Monitoring Engine
XGBoost + SHAP   Lifecycle Rules      Z-score + EWMA
```

Primary UI: React.

Backend: FastAPI.

Legacy/prototyping UI: Streamlit in `app.py`.

## Module 1

Model: XGBoost

Features: 422 transformed features

Operating threshold: 0.60

Decision: `ALLOW` / `REVIEW`

Final chronological held-out metrics:

| Metric | Value |
| --- | ---: |
| Precision | 0.364018 |
| Recall | 0.563088 |
| F1 | 0.442180 |
| PR-AUC | 0.514931 |
| ROC-AUC | 0.891247 |
| Review rate | 0.053838 |

The held-out test was opened only after model/policy selection. These results are not used for further Module 1 tuning.

## Module 2

Payment incident detection uses deterministic lifecycle rules and synthetic payment-event data.

Capabilities:

- snapshot incident evaluation
- multi-step payment lifecycle replay
- timeline reasoning
- severity/action tracking
- resolution status

Synthetic rule evaluation showed full consistency with the currently generated synthetic scenarios. This is not evidence of real-world production generalization.

Module 2 does not use real payment-provider data, does not claim Razorpay validation, and performs no automatic payment action.

## Module 3

Operational risk monitoring uses synthetic monitoring streams, 15-minute windows, historical baselines, z-scores, and EWMA trend monitoring.

Synthetic monitoring evaluation:

| Metric | Value |
| --- | ---: |
| Precision | 0.917874 |
| Recall | 1.000000 |
| F1 | 0.957179 |
| False alert rate | 0.141667 |
| Window size | 15 minutes |
| Mean detection delay | 0.0 minutes |
| Median detection delay | 0.0 minutes |
| Max detection delay | 0 minutes |

All simulated spike scenarios were detected in the first evaluated spike window. This is window-level synthetic detection, not a claim of 0-minute real-time detection.

## Runtime Artifact Audit

| Artifact | Purpose | Required at runtime |
| --- | --- | --- |
| `artifacts/models/xgboost_model.json` | Frozen fraud scoring model | yes |
| `artifacts/preprocessors/preprocessor.joblib` | Frozen feature transformation | yes |
| `artifacts/preprocessors/preprocessing_metadata.json` | Feature/schema metadata | yes |
| `artifacts/demo/demo_transactions.csv` | Packaged transaction demo rows | yes |
| `artifacts/demo/demo_labels.csv` | Demo labels for historical context | yes |
| `artifacts/results/final_test_metrics.json` | About/evaluation display | yes |
| `data/synthetic/payment_incidents.csv` | Module 2 snapshot demo data | yes |
| `data/synthetic/payment_lifecycles.json` | Module 2 lifecycle demo data | yes |
| `data/synthetic/monitoring_stream.csv` | Module 3 monitoring demo data | yes |
| `artifacts/results/*.png` | Methodology/evaluation evidence | no |
| `artifacts/results/*threshold*` | Threshold analysis evidence | no |
| `artifacts/results/*shap*` | Explainability evidence | no |
| `artifacts/results/spike_monitor_*` | Monitoring evaluation evidence | no |

## Dataset Policy

Raw IEEE-CIS CSV files are not required for the demo runtime and must not be committed to GitHub.

The repository may include small packaged demo artifacts and synthetic datasets needed for reproducible demo behavior.

## Freeze Rules

- Do not retrain Module 1.
- Do not change threshold `0.60`.
- Do not tune using held-out test results.
- Do not add another ML model.
- Do not add LLM, RAG, agents, payment gateway integrations, webhooks, databases, or automatic refunds/blocks.
- Recommended actions are advisory only.
