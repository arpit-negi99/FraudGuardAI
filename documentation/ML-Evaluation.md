# ML Evaluation — FraudGuard AI

## Evaluation Objective

Prove that the detector identifies fraudulent transactions better than simple baselines **without hiding the cost of false positives**.

All headline results must come from a locked chronological held-out test split.

No metric may be copied from a paper, Kaggle notebook, README example, or previous run.

---

# 1. Dataset Used for Evaluation

Primary dataset: **IEEE-CIS Fraud Detection** labeled training data.

Public descriptions report approximately:

- 590,540 training transactions,
- about 20,663 fraud-labeled transactions,
- roughly 3.5% fraud prevalence,
- transaction and identity feature tables joined by `TransactionID`.

Final counts must be recomputed from the downloaded dataset and written to evaluation artifacts.

---

# 2. Split Methodology

Sort by `TransactionDT` and split contiguously:

- train: first 70%,
- validation: next 15%,
- test: last 15%.

Approximate row counts before any row-level validation issues:

- train: ~413,378,
- validation: ~88,581,
- test: ~88,581.

Actual split counts must be produced by code.

## Why chronological splitting

Random splitting can make fraud benchmarks overly optimistic when related transaction patterns appear across train and test. Chronological splitting better represents future inference and drift.

---

# 3. Baselines

## Baseline A — Majority class

Predict `not fraud` for all rows.

Purpose:

- demonstrate that accuracy is not a useful headline metric for this imbalanced problem,
- confirm metric implementation.

## Baseline B — Logistic Regression

Train a simple supervised logistic model using a compact train-fitted preprocessing pipeline.

Purpose:

- provide a meaningful linear baseline,
- quantify whether XGBoost’s nonlinear interactions add value.

---

# 4. Primary Metrics

## Precision

`TP / (TP + FP)`

Business meaning: among transactions flagged for review, how many are truly fraud-labeled?

## Recall

`TP / (TP + FN)`

Business meaning: what fraction of fraud-labeled transactions did we catch?

## F1

Harmonic mean of precision and recall.

Useful as a balanced classification summary, but not sufficient by itself.

## PR-AUC / Average Precision

Primary threshold-independent ranking metric because fraud is a minority class.

---

# 5. Secondary Metrics

- ROC-AUC,
- false-positive rate,
- confusion matrix,
- review/flag rate,
- fraud prevalence,
- score distribution by class.

Accuracy may be shown only as a non-headline supporting metric with an imbalance warning.

---

# 6. Business-Cost Metrics

## False-negative / missed-fraud value

For fraud rows not flagged for review:

```text
missed_fraud_loss = Σ TransactionAmt × fraud_loss_rate
```

The default `fraud_loss_rate` is a declared scenario assumption, not empirical merchant ground truth.

## False-positive cost

For legitimate rows incorrectly flagged:

```text
FP_cost = review_cost + TransactionAmt × friction_rate
```

Where:

- `review_cost` = assumed operational cost of reviewing one legitimate transaction,
- `friction_rate` = assumed margin/conversion-friction percentage associated with unnecessary review.

## Total scenario cost

```text
total_cost = missed_fraud_loss + total_false_positive_cost
```

## Cost reporting rule

Report at least three sensitivity scenarios (for example low, reference, and high false-positive cost) so the result is not dependent on one arbitrary assumption.

The exact scenario values must be declared in the evaluation report and may be adjusted before model freeze, but never chosen after looking at the test set merely to make the model look better.

---

# 7. Threshold Selection

1. Train model on train split.
2. Generate validation scores.
3. Evaluate threshold candidates.
4. Pick the **reference cost-optimal threshold** on validation only.
5. Save threshold and cost assumptions.
6. Freeze model + preprocessing + threshold.
7. Evaluate once on held-out test.

Also record the validation F1-optimal threshold for comparison.

Do not choose the final threshold using test precision/recall.

---

# 8. Success Threshold — “Good Enough for the Project”

The project is demo-ready only if all conditions are true on the held-out temporal test set:

1. XGBoost has higher PR-AUC than the logistic-regression baseline.
2. XGBoost has higher F1 than the logistic-regression baseline at each model’s locked decision threshold.
3. At the locked default XGBoost threshold:
   - precision is at least **0.35**,
   - recall is at least **0.60**,
   - false-positive rate is at most **0.05**.
4. Under the declared reference cost scenario, XGBoost total cost is lower than the logistic-regression baseline.
5. Evaluation completes with no use of test data for model, feature, or threshold tuning.

These are predeclared MVP acceptance criteria, not promised results.

If the model misses one threshold narrowly, do error analysis before changing the target. Do not lower success criteria after test inspection just to claim success.

---

# 9. Evaluation Dataset & Test Cases

## Normal test set

Locked chronological test partition.

## Edge-case slices

Report metrics or at minimum prediction behavior for slices such as:

- rows with no identity record,
- rows with high missingness,
- very small transaction amounts,
- high transaction amounts,
- unseen categorical values introduced in synthetic inference tests,
- records close to the decision threshold.

Do not invent labels for synthetic edge cases. Synthetic inputs test robustness of inference only, not classification quality.

---

# 10. Error Analysis

For every final model iteration, inspect at least:

## 50 highest-confidence false positives

Questions:

- are they concentrated in particular product/card/email/device groups?
- is missingness driving the model?
- are high amounts over-penalized?

## 50 highest-confidence false negatives

Questions:

- are there specific fraud subtypes the model cannot see?
- are fraud rows missing identity data?
- is class weighting insufficient?

## Threshold-near errors

Inspect examples just above and below the review threshold to understand sensitivity.

## SHAP aggregate review

Check whether model importance is dominated by suspicious shortcut features.

Any feature suspected of leakage must be removed and the complete training/evaluation process rerun.

---

# 11. Calibration

Probability calibration is **not P0**.

If implemented in P2/P1 after the core model is stable:

- fit calibration using validation or a dedicated calibration subset only,
- measure Brier score / calibration curve,
- update UI wording only if calibration improves.

Until then, call the output a risk score.

---

# 12. Evaluation Artifacts

The final evaluation run should save:

- `metrics.json`,
- `confusion_matrix.csv` or JSON,
- threshold table,
- cost sensitivity table,
- precision-recall curve data,
- model-vs-baseline comparison,
- test split metadata,
- model/config hash,
- timestamp/date,
- `evaluation_report.md` generated from actual results.

README result fields remain `TODO` until these artifacts exist.
