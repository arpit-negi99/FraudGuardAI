# User Flow & Experience — FraudGuard AI

## UX Principle

The demo should make the AI system easy to trust **without pretending it is certain**.

The user should always be able to see:

- what the model scored,
- what threshold was applied,
- what decision was produced,
- which features most influenced the score,
- what the model cannot guarantee.

The primary UI is a polished React merchant dashboard backed by FastAPI.

---

# Main User Journey

```text
Risk analyst
    ↓
Enter/select/upload transaction data
    ↓
Schema validation
    ↓
Frozen preprocessing + fraud model
    ↓
Risk score + threshold decision
    ↓
Feature-attribution explanation
    ↓
Analyst allows, reviews, exports, or inspects evaluation
```

---

# Flow 1 — Single Transaction Scoring

## User action

Open “Single Transaction” and either:

- choose a preloaded demo transaction, or
- enter the supported raw fields.

## Input

One valid transaction row.

## AI processing

1. Validate fields.
2. Apply frozen preprocessing.
3. Compute XGBoost risk score.
4. Compare score with locked review threshold.
5. Compute SHAP explanation.

## Output

- Risk score: e.g. `0.73`.
- Decision: `REVIEW` or `ALLOW`.
- Locked threshold.
- “Distance from threshold” indicator.
- Top 3–5 model signals.
- Model version.

## Loading state

“Scoring transaction…” with a small progress indicator.

Do not show fake multi-stage AI animation.

## Error state

Show the exact validation/model error in user-friendly language and log technical details.

## Empty state

Prompt the user to choose a demo sample or enter transaction data.

## Retry behavior

After correcting fields, the user can resubmit immediately.

## User feedback

P2: optional “prediction useful / incorrect” control stored locally. No automatic retraining.

---

# Flow 2 — Batch CSV Risk Screening

## User action

Upload a CSV containing the approved inference schema.

## Input

CSV with multiple transactions and an optional user-provided row identifier.

## AI processing

1. Validate columns.
2. Report missing/unexpected fields.
3. Score valid rows in batch.
4. Apply the same locked threshold.
5. Rank by risk score.

## Output

- row identifier,
- risk score,
- `ALLOW/REVIEW`,
- count and percentage flagged,
- downloadable scored CSV.

## Loading state

“Validating and scoring N transactions…”

## Error state

- malformed CSV,
- missing required columns,
- unsupported types,
- model unavailable.

Do not silently discard invalid rows.

## Empty state

Explain the required schema and provide a small safe sample CSV bundled with the demo.

## Retry behavior

User fixes the file and uploads again.

## User feedback

User can inspect the highest-risk rows individually.

---

# Flow 3 — Model Evaluation

## User action

Open “Evaluation.”

## Input

Precomputed evaluation artifact from the locked held-out test run.

## AI processing

No model tuning occurs here. The page reads stored metrics and curves.

## Output

Headline metrics:

- Precision,
- Recall,
- F1,
- PR-AUC.

Supporting metrics:

- ROC-AUC,
- false-positive rate,
- confusion matrix,
- reviewed fraction,
- missed-fraud value,
- false-positive review/friction cost,
- total scenario cost.

Also show comparison against the logistic-regression baseline.

## Loading state

Minimal; metrics should be local artifacts.

## Error state

If evaluation artifacts are missing, explicitly say “evaluation has not been run” rather than filling placeholders.

## Empty state

Before first evaluation, show a TODO state.

## Retry behavior

Evaluation is rerun from the command line after model changes; it is not recomputed interactively against the test set.

---

# Flow 4 — Cost/Threshold Explorer

## User action

Open “Cost Tradeoff.”

## Input

Stored validation/test scores and transparent scenario assumptions.

## AI processing

Recalculate threshold tradeoff tables for display.

## Output

- precision vs recall by threshold,
- false-positive count by threshold,
- missed-fraud amount by threshold,
- total modeled cost by threshold,
- locked default threshold highlighted.

## Loading state

Short calculation spinner.

## Error state

If the score artifact is unavailable, explain that the evaluation pipeline must be run first.

## Empty state

Show the default assumptions and what each cost parameter means.

## Retry behavior

User can reset scenario controls to the reference configuration.

---

# AI Transparency

## Show to users

### Confidence / score

Show the model risk score and decision threshold.

Do not call the score “certainty.”

### Sources

Not applicable to individual predictions because this is not a retrieval system.

The app may link to the dataset/model documentation in an “About” section.

### Retrieved documents

Not applicable.

### Reasoning summaries

Show concise feature-attribution summaries, e.g.:

- “This feature increased the model risk score.”
- “This feature reduced the model risk score.”

Do not expose hidden chain-of-thought.

### Model limitations

Always provide a visible limitations section:

- public benchmark data,
- partially anonymized features,
- not India-specific,
- offline MVP, not a payment-network replacement,
- risk score may not be probability-calibrated.

### Citations

Not required on every prediction. Dataset/model methodology should be documented in README/About.

### AI-generated indicator

Use “ML model output” or “AI risk score” near prediction results.

---

# Human Override

The system is advisory.

Users can:

- ignore the `REVIEW` recommendation,
- inspect the transaction manually,
- retry after correcting malformed input,
- optionally mark a result as incorrect,
- export the batch output for external review.

The MVP must never:

- auto-block a payment,
- auto-retrain from a single correction,
- claim a person committed fraud,
- expose offensive fraud tactics.
