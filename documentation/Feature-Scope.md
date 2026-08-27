# Feature Scope — FraudGuard AI

## Product Overview

### Product name

**FraudGuard AI**

### Problem

Online merchants lose money when fraudulent transactions are approved, but overly aggressive fraud controls also create losses by sending too many legitimate transactions to review or rejection. The MVP detects high-risk transactions while explicitly measuring both fraud misses and false-positive cost.

### Target users

- merchant fraud/risk analysts,
- payment-operations teams,
- online-merchant finance/operations teams.

### Value proposition

A lightweight, explainable fraud-risk scorer that:

1. ranks online transactions by fraud risk,
2. flags only higher-risk transactions for review,
3. reports honest precision/recall on a locked held-out test set,
4. exposes the business tradeoff between missed fraud and false positives.

---

# AI/ML Capabilities

## 1. Transaction fraud scoring — P0

- **Input:** one transaction row matching the approved feature schema.
- **Processing:** schema validation → train-fitted preprocessing → XGBoost inference.
- **Model:** XGBoost binary classifier.
- **Output:** risk score in `[0,1]`, model version, review threshold.
- **Expected behavior:** higher score means higher model-estimated fraud risk. The score is not described as a calibrated probability unless calibration is explicitly implemented and validated later.

## 2. Review recommendation — P0

- **Input:** risk score and locked threshold.
- **Processing:** deterministic threshold comparison.
- **Model/algorithm:** validation-selected threshold.
- **Output:** `ALLOW` or `REVIEW`.
- **Expected behavior:** no transaction is automatically declined by the MVP.

## 3. Cost-aware threshold selection — P0

- **Input:** validation labels, validation scores, transaction amounts, declared cost assumptions.
- **Processing:** evaluate candidate thresholds and compute precision, recall, F1, false positives, missed-fraud value, and total scenario cost.
- **Algorithm:** grid/search across model-score thresholds.
- **Output:** locked default review threshold plus threshold tradeoff table.
- **Expected behavior:** threshold is chosen without viewing test results.

## 4. Held-out model evaluation — P0

- **Input:** untouched chronological test split.
- **Processing:** frozen preprocessing + frozen model + locked threshold.
- **Output:** precision, recall, F1, PR-AUC, ROC-AUC, confusion matrix, FPR, and scenario cost.
- **Expected behavior:** evaluation command produces a machine-readable results artifact and human-readable report. No metric is hardcoded.

## 5. Single-transaction explanation — P0

- **Input:** one scored transaction.
- **Processing:** SHAP tree attribution.
- **Model/algorithm:** SHAP TreeExplainer over the trained XGBoost model.
- **Output:** top 3–5 risk-raising and risk-lowering feature contributions.
- **Expected behavior:** explanation is presented as “signals influencing the model,” never as causal proof.

## 6. Batch CSV scoring — P0

- **Input:** CSV containing the approved inference feature schema.
- **Processing:** schema validation → preprocessing → vectorized inference → decision rule.
- **Output:** downloadable table containing original row identifier, risk score, and `ALLOW/REVIEW` decision.
- **Expected behavior:** invalid rows are reported clearly; batch processing does not silently drop rows.

## 7. Evaluation dashboard — P1

- **Input:** stored evaluation artifacts only.
- **Processing:** visualization.
- **Output:** precision-recall curve, threshold/cost curve, confusion matrix, class balance, and cost summary.
- **Expected behavior:** the dashboard never recomputes or retunes against the held-out test set.

## 8. Cost sensitivity controls — P1

- **Input:** alternative review/friction/loss assumptions.
- **Processing:** recompute cost for already-generated validation/test scores without retraining.
- **Output:** cost comparison across scenarios.
- **Expected behavior:** the locked default threshold remains visibly identified; scenario exploration is clearly labeled analysis.

## 9. Analyst feedback capture — P2

- **Input:** analyst marks a prediction as useful/incorrect.
- **Processing:** append feedback locally.
- **Output:** feedback log.
- **Expected behavior:** no automatic online retraining.

## 10. Fraud-rate spike indicator — P2

- **Input:** batch scores over time if timestamped batch data is available.
- **Processing:** simple rolling risk-rate comparison.
- **Output:** informational spike warning.
- **Expected behavior:** this is not a second ML detector; it is a descriptive monitoring layer only.

---

# MVP Priorities

## P0 — Critical

- Reproducible IEEE-CIS data preparation.
- Chronological train/validation/test split.
- Leakage-safe preprocessing.
- Logistic regression baseline.
- XGBoost fraud classifier.
- Validation-only threshold selection.
- Cost model with explicit false-positive cost.
- Locked held-out evaluation.
- Single-transaction scoring.
- SHAP feature-attribution explanation.
- Batch CSV scoring.
- React + Vite merchant risk console backed by FastAPI.
- Reproducible model/evaluation artifact saving.

## P1 — Important

- Precision-recall and threshold-cost visualizations.
- Cost-sensitivity controls.
- Model metadata/version panel.
- Improved feature selection after error analysis.
- One carefully justified iteration of leakage-safe feature engineering.

## P2 — Nice to Have

- Analyst feedback log.
- Descriptive fraud-risk spike indicator.
- Optional probability calibration, only if clearly beneficial and evaluated.

## OUT OF SCOPE

Do **not** build these during the 10-day MVP:

- chargeback evidence responder,
- return-risk scoring,
- abuse-ring/graph detection,
- multi-loss “risk operating system,”
- real-time payment-gateway integration,
- automatic transaction blocking,
- LLM-generated fraud decisions,
- RAG,
- vector database,
- LLM fine-tuning,
- deep neural network training,
- adversarial fraud generation or evasion testing,
- authentication/role management,
- merchant billing,
- microservices,
- Kafka,
- Kubernetes,
- heavyweight model-serving infrastructure,
- duplicated JavaScript fraud-scoring logic.

---

# AI/ML Scope

## Models

- **Primary:** XGBoost binary classifier.
- **Baseline:** Logistic Regression; majority-class baseline for sanity only.

## What will be trained

- logistic regression baseline,
- XGBoost classifier,
- preprocessing objects fit on training data only.

## What will NOT be trained

- no LLM,
- no transformer,
- no deep neural network,
- no graph neural network,
- no embedding model.

## Fine-tuning

**Not required.** There is no foundation model whose fine-tuning is justified for the selected tabular fraud task.

## RAG

**Not required.** The core task is supervised tabular classification, not document-grounded question answering.

## Embeddings

**Not required.**

## External APIs

**Not required for inference.**

A Kaggle credential/API may be used only to download the dataset during development. The deployed demo must not depend on Kaggle.

---

# Scope Protection

Any proposed feature must answer all three questions before implementation:

1. Does it directly improve fraud detection, evaluation credibility, or demo clarity?
2. Can it be completed and tested without threatening the 10-day deadline?
3. Is it defense-only and consistent with the track rules?

If any answer is “no,” defer it.

Significant scope changes require updating this file and the architecture/evaluation documentation first.
