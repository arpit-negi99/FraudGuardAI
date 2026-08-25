# FraudGuard AI — Project Decision

## Executive Decision

Build **FraudGuard AI**, a cost-aware online transaction fraud detector for merchants.

The MVP will predict whether an incoming online payment transaction should be **flagged for manual review**. It will output:

- a fraud risk score,
- a binary `REVIEW` / `ALLOW` recommendation using a threshold chosen only on validation data,
- the top model features that pushed the score upward or downward,
- measured held-out test metrics,
- a business-cost view that explicitly includes false-positive cost.

This is intentionally a **defense-only** system. It will not generate fraud strategies, simulate evasion, identify bypasses, or provide offensive testing capabilities.

---

## Step 1 — Problem Definition

### Exact problem

Given transaction and identity attributes available at payment-risk decision time, estimate whether an online transaction is fraudulent. The merchant uses the score to decide which transactions should receive manual review.

### Is this actually an ML problem?

Yes. Fraud is a rare-event classification problem where risk is determined by combinations of amount, card, address, email, identity, device, historical-count, match, and timing signals. Simple static rules are useful as a baseline, but they struggle to model nonlinear interactions and changing combinations of weak signals.

### Non-ML baseline

A simple rules engine could flag a transaction when one or more hand-written conditions are met, for example:

- unusually large transaction amount,
- unusual/missing identity information,
- mismatched purchaser/recipient signals,
- suspicious combinations of transaction/card/device fields.

The rules baseline is easy to explain but brittle and difficult to optimize globally for precision, recall, and false-positive cost.

### Why ML is useful

A tree-based supervised model can learn nonlinear interactions across sparse and partially missing tabular features, rank transactions by risk, and support threshold tuning against business costs rather than relying on fixed rules.

---

## Input

Primary input: **tabular online transaction data**.

Expected fields are based on a selected, manageable subset of the IEEE-CIS Fraud Detection transaction and identity tables, including groups such as:

- transaction amount and product information,
- payment-card attributes,
- address and distance-related attributes,
- purchaser/recipient email-domain attributes,
- count (`C*`) features,
- time-difference (`D*`) features,
- match (`M*`) features,
- selected identity and device fields,
- missingness-derived indicators.

`TransactionDT` is used to create chronological splits. Absolute transaction time will not be used as a shortcut target proxy unless a derived feature is justified and leakage-safe.

---

## Output

For each transaction:

```json
{
  "risk_score": 0.0,
  "decision": "ALLOW | REVIEW",
  "threshold": 0.0,
  "top_risk_drivers": [
    {"feature": "...", "direction": "raises_risk | lowers_risk", "impact": 0.0}
  ],
  "model_version": "..."
}
```

For evaluation/batch views:

- precision,
- recall,
- F1,
- PR-AUC,
- ROC-AUC,
- false-positive rate,
- confusion matrix,
- missed-fraud transaction value,
- false-positive review/friction cost,
- total scenario cost.

---

## Users

Primary users:

- merchant risk analysts,
- payment operations teams,
- fraud operations teams,
- founders/finance teams at online merchants that need a lightweight risk triage layer.

This is not positioned as a replacement for a bank/payment-network fraud stack. It is a demo-ready decision-support tool.

---

## Main Use Cases

1. **Single-transaction review** — enter or select a transaction and obtain a risk score, review recommendation, and top drivers.
2. **Batch risk screening** — upload a CSV and rank transactions by fraud risk for analyst review.
3. **Threshold/cost analysis** — show how the review threshold changes precision, recall, false positives, missed fraud, and estimated cost.
4. **Held-out evaluation** — demonstrate that reported metrics come only from a locked chronological test split.

---

## Assumptions

1. Public IEEE-CIS data is acceptable for the hackathon MVP.
2. The competition training data is used because it contains labels; the competition test set is not used for final metric claims because its labels are not provided to us.
3. Data is representative of online-payment fraud behavior for an MVP, but it is **not India-specific**, so the project will not claim India-specific production validity.
4. Cost numbers used for threshold optimization are scenario assumptions, not claimed merchant ground truth.
5. A transaction flagged by the model is sent to **review**, not automatically declined.
6. The model is a risk-ranking/triage system, not a legal or compliance decision engine.
7. All preprocessing and feature selection are fit using training data only.

---

# Step 2 — AI/ML Approach Comparison

| Approach | Advantages | Disadvantages | Data | Compute | Expected performance | Dev time | 10-day suitability |
|---|---|---|---|---|---|---|---|
| Rules only | Fast, transparent, deterministic | Brittle; weak nonlinear modeling; hard to tune globally | Small labeled set optional | Very low | Low–moderate | <1 day | Good baseline, weak final model |
| Logistic Regression | Fast, reproducible, interpretable baseline | Limited nonlinear interactions; categorical preprocessing | Labeled tabular | Low | Moderate baseline | 0.5–1 day | Excellent baseline |
| **XGBoost** | Strong tabular performance; handles missing values; CPU-friendly histogram training; works with SHAP | Requires preprocessing/tuning; score may be uncalibrated | Labeled tabular | Low–moderate CPU | High for this problem class | 1–2 days | **Best primary choice** |
| Random Forest | Simple; robust baseline | Larger/slower inference; often weaker than boosting on imbalanced tabular fraud | Labeled tabular | Moderate | Moderate | ~1 day | Acceptable but not preferred |
| Deep MLP / Transformer for tabular data | Can model complex patterns | More tuning, more compute, less reliable under 10-day limit; weak justification over boosting | Large labeled tabular | Moderate–high | Uncertain vs boosting | 2–4+ days | Poor tradeoff |
| Pretrained foundation model | Little training | No natural pretrained foundation model maps directly to anonymized fraud tabular fields | Still needs task data | Varies | Uncertain | 1–3 days | Not justified |
| LLM API | Fast generated explanations | LLM is not the right fraud classifier; cost, latency, hallucination risk; harder honest precision/recall | Prompt/context data | API | Weak as core classifier | 1–2 days | Reject for core |
| RAG | Useful for document-grounded Q&A | No external knowledge retrieval is required for transaction classification | Document corpus | Low–moderate | Not relevant | 1–3 days | Reject |
| Fine-tuned LLM | Flexible text model | Wrong modality, expensive, unnecessary | Large curated text dataset | High | Not relevant | >10 days realistically | Reject |
| Graph fraud/ring model | Captures networks and coordinated abuse | Requires entity graph design, graph features, evaluation complexity | Rich entity graph | Moderate–high | Potentially strong | 4–7+ days | Too risky for MVP |

---

## Recommended Primary Approach

### Primary model

**XGBoost binary classifier** trained on a leakage-safe subset of IEEE-CIS transaction + identity features.

### Baseline

**Logistic Regression** on a compact, train-fitted preprocessing pipeline, plus a majority-class sanity baseline.

### Class imbalance

Use training-data class weighting (`scale_pos_weight` or equivalent) and threshold selection on validation data. Do not use test data to tune the model or threshold.

### Explainability

Use **SHAP TreeExplainer** to show the strongest feature contributions for an individual scored transaction. Explanations are feature-attribution summaries, not causal claims and not hidden chain-of-thought.

### Cost-aware decision rule

Select the production/demo review threshold using validation data and a declared cost function:

`total_cost = missed_fraud_loss + false_positive_review_cost + false_positive_friction_cost`

Where:

- missed fraud uses the transaction amount multiplied by a declared loss-rate assumption,
- false-positive review cost is a configurable per-review scenario cost,
- false-positive friction cost is a configurable fraction of legitimate transaction amount.

All assumptions must be shown in the UI/report.

---

## Explicitly Rejected for the 10-Day MVP

- Training any large model from scratch.
- Fine-tuning an LLM.
- RAG or vector databases.
- A second model whose only purpose is generated prose.
- Graph neural networks / abuse-ring detection.
- Real-time Kafka/streaming infrastructure.
- Microservices or Kubernetes.
- Full merchant admin/auth stack.
- Automatic payment blocking.

---

## External References Used for Planning

- IEEE-CIS Fraud Detection, Kaggle: https://www.kaggle.com/c/ieee-fraud-detection/data
- IEEE-CIS data-description discussion: https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203
- SHAP TreeExplainer documentation: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- Reserve Bank of India Annual Report 2024-25, Regulation/Supervision/Fraud section: https://www.rbi.org.in/scripts/AnnualReportPublications.aspx?Id=1436
