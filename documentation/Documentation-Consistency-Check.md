# Documentation Consistency Check — FraudGuard AI

## Status

**PASS — documentation is internally consistent for implementation approval.**

This document records the final cross-check across:

- `Project-Decision.md`
- `Feature-Scope.md`
- `AI-ML-Architecture.md`
- `User-Flow-Experience.md`
- `ML-Evaluation.md`
- `Data-Strategy.md`
- `Production-Definition.md`
- `AGENTS.md`
- `Development-Plan.md`
- `README.md`

---

# 1. Core Problem Consistency

All documents define the same loss class:

**online transaction fraud**.

No document proposes building returns, chargebacks, or abuse-ring detection in P0/P1.

Result: **consistent**.

---

# 2. Model Consistency

All documents use:

- primary model: XGBoost,
- baseline: Logistic Regression,
- majority-class baseline for sanity.

No document requires a deep network, transformer, LLM, or graph model.

Result: **consistent**.

---

# 3. LLM / RAG Consistency

All documents explicitly state:

- no LLM in core,
- no RAG,
- no embeddings,
- no vector database,
- no fine-tuning,
- no model API dependency for inference.

Result: **consistent**.

---

# 4. Dataset Consistency

All documents use the IEEE-CIS labeled training data and join transaction + identity tables using `TransactionID`.

All documents state that the competition test data is not used for precision/recall claims because labels are unavailable.

All documents acknowledge:

- class imbalance,
- missing data,
- anonymized features,
- non-India-specific benchmark limitations.

Result: **consistent**.

---

# 5. Split / Leakage Consistency

All documents specify chronological splitting by `TransactionDT`:

- 70% train,
- 15% validation,
- 15% held-out test.

Training-only:

- preprocessing fit,
- feature-drop rules,
- encoders,
- class weighting,
- model fit.

Validation-only allowed for:

- early stopping/model decisions,
- threshold selection,
- cost-threshold analysis.

Test:

- final reporting only.

Result: **consistent**.

---

# 6. Product Decision Consistency

All product documents return:

- risk score,
- `ALLOW` or `REVIEW`,
- locked threshold,
- feature-attribution explanation.

No document proposes automatic blocking.

Result: **consistent**.

---

# 7. Explainability Consistency

SHAP is the only approved P0 explanation mechanism.

It is described consistently as feature attribution, not causal explanation and not hidden chain-of-thought.

Result: **consistent**.

---

# 8. Evaluation Consistency

Required metrics are aligned:

Primary:

- precision,
- recall,
- F1,
- PR-AUC.

Secondary:

- ROC-AUC,
- false-positive rate,
- confusion matrix,
- review rate.

Business:

- missed-fraud value,
- false-positive cost,
- total scenario cost.

Result: **consistent**.

---

# 9. Cost-Model Consistency

All documents treat false-positive/fraud-loss costs as **declared scenarios**, not observed merchant truth.

Threshold is chosen on validation data only.

Result: **consistent**.

---

# 10. Deployment Consistency

All implementation-facing docs support one lightweight architecture:

- Python,
- in-process inference,
- React + FastAPI demo,
- no separate backend required,
- no raw dataset in deployment.

Result: **consistent**.

---

# 11. Security / Track Compliance

All relevant docs prohibit:

- fraud generation,
- evasion tactics,
- bypass discovery,
- credential theft,
- automatic fraudulent action.

The model is positioned as a defensive review-triage tool.

Result: **consistent**.

---

# 12. Unrealistic Assumptions Removed

Removed from scope:

- bank-grade real-time streaming,
- microservices,
- Kubernetes,
- deep learning for prestige,
- LLM-generated fraud decisions,
- graph-ring detection,
- real merchant cost claims,
- India-specific performance claims,
- automatic payment decline.

Result: **10-day scope is realistic**.

---

# 13. Remaining Technical Risks

## Risk 1 — Dataset size / memory

IEEE-CIS is large. Mitigation:

- drop ultra-sparse columns using train-only rules,
- begin with selected feature groups,
- use XGBoost histogram mode,
- avoid loading the raw dataset in the demo.

## Risk 2 — Masked/anonymized feature semantics

SHAP can explain feature influence but some features are hard to translate into business language.

Mitigation:

- prioritize interpretable feature groups,
- describe masked features honestly,
- avoid causal claims.

## Risk 3 — Temporal performance degradation

Chronological holdout may score lower than random split.

Mitigation:

- accept honest metrics,
- focus on PR-AUC/cost improvement over baseline,
- perform targeted error analysis.

## Risk 4 — False-positive cost assumptions

No real merchant cost labels exist.

Mitigation:

- use transparent low/reference/high sensitivity scenarios,
- never claim the assumed values are universal.

## Risk 5 — Deployment artifact size / package compatibility

Mitigation:

- keep feature/model size controlled,
- pin dependencies,
- smoke-test deployment on Day 9,
- preserve a local backup demo.

---

# 14. Features That Should Still Be Removed if Schedule Slips

Cut in this order:

1. feedback capture,
2. fraud-spike indicator,
3. probability calibration,
4. cost-explorer polish,
5. extra feature engineering,
6. nonessential charts.

Never cut:

- held-out evaluation,
- baseline comparison,
- precision/recall,
- false-positive cost,
- leakage prevention,
- single/batch inference reliability.

---

# Final Documentation Freeze

The approved MVP specification is:

> **A CPU-friendly XGBoost online-transaction fraud detector trained on a leakage-safe chronological split of IEEE-CIS data, compared with a logistic baseline, using validation-selected cost-aware review thresholding, SHAP explanations, held-out precision/recall evaluation, and a lightweight React + FastAPI demo.**

Implementation should not start until this specification is approved.
