# 10-Day Development Plan — FraudGuard AI

## Planning Principle

The deadline is fixed. Scope is variable.

The goal is a **complete, measured, deployable fraud detector**, not the maximum number of features.

Do not begin this implementation until the documentation is approved.

---

# Day 1 — Problem Definition, Architecture, Environment

## Goals

- Approve documentation and scope.
- Create repository structure.
- Create Python environment and dependency file.
- Configure Kaggle dataset access.
- Download data locally outside Git.
- Implement only the minimal schema-inspection script needed to verify dataset availability.

## Deliverables

- approved docs,
- reproducible environment,
- `.gitignore` protecting raw data/secrets,
- dataset present locally,
- dataset profile with row/column/class counts.

## Exit criteria

- everyone agrees we are building only transaction fraud scoring,
- data can be loaded,
- no raw data is committed.

---

# Day 2 — Data Pipeline / Model Foundation

## Goals

- Join transaction + identity tables.
- Implement schema checks.
- Create chronological 70/15/15 split.
- Implement train-only missingness rules.
- Define P0 feature set.
- Build reusable preprocessing pipeline.
- Add pipeline tests.

## Deliverables

- reproducible prepared splits,
- preprocessing artifact fit on train only,
- split metadata,
- data validation tests.

## Exit criteria

- no leakage from validation/test into preprocessing,
- same preprocessing can transform train/validation/test.

---

# Day 3 — Baseline + First Core Model

## Goals

- Train majority-class sanity baseline.
- Train logistic-regression baseline.
- Record validation metrics.
- Train first XGBoost model.
- Use early stopping.
- Save experiment config and model artifact.

## Deliverables

- baseline metrics,
- first XGBoost validation metrics,
- first model artifacts.

## Exit criteria

- end-to-end train → validation score path works,
- no hyperparameter rabbit hole.

---

# Day 4 — Core AI/ML Completion

## Goals

- Implement cost function.
- Implement threshold analysis.
- Lock reference cost assumptions.
- Select validation-only default threshold.
- Add SHAP explanation pipeline.
- Create frozen inference function.

## Deliverables

- threshold table,
- locked threshold metadata,
- single-row inference result,
- SHAP top-driver output.

## Exit criteria

A raw transaction can go through:

`input → preprocessing → model → score → ALLOW/REVIEW → explanation`.

---

# Day 5 — MVP Integration

## Goals

Build minimal React + FastAPI interface:

1. Single Transaction.
2. Batch CSV.
3. Evaluation summary placeholder wired to stored artifacts.
4. About/limitations.

Do not train from the UI.

## Deliverables

- working local demo,
- valid sample transactions,
- batch scoring/export.

## Exit criteria

A reviewer can understand and use the core detector without running a notebook.

---

# Day 6 — Evaluation + Error Analysis

## Goals

- Freeze first serious model/threshold.
- Run held-out temporal test evaluation.
- Generate metrics artifacts.
- Compare XGBoost vs logistic baseline.
- Analyze high-confidence false positives and false negatives.
- Inspect SHAP feature importance for shortcuts/leakage.

## Deliverables

- first honest held-out test report,
- error-analysis notes,
- decision: freeze or make one improvement.

## Exit criteria

Know exactly where the model fails and whether it meets predefined success criteria.

---

# Day 7 — Improvement + Optimization

## Goals

Only address issues found on Day 6.

Allowed improvements:

- remove suspicious/leaky features,
- improve categorical handling,
- adjust regularization/class weighting,
- add one small leakage-safe feature-engineering batch,
- reduce artifact size/latency.

Do not introduce a new model family unless the current approach is fundamentally broken.

## Deliverables

- final candidate model,
- updated validation-selected threshold,
- refreshed held-out test report only after the final candidate is frozen.

## Exit criteria

Model is stable enough that further tuning has lower value than demo/reliability work.

---

# Day 8 — Demo/UI/Integration Polish

## Goals

- Add clear metric cards.
- Add precision-recall/threshold-cost visuals.
- Add strong single-transaction explanation view.
- Improve empty/error/loading states.
- Add model limitations and methodology panel.
- Ensure cost assumptions are visible.

## Deliverables

- demo-ready React UI,
- no fake metrics,
- no offensive features.

## Exit criteria

A 3–5 minute demo tells one clear story:

**merchant loss → fraud score → review decision → explanation → measured precision/recall/cost tradeoff**.

---

# Day 9 — Production Testing + Deployment

## Goals

- Run unit/data/model/integration tests.
- Benchmark latency/memory.
- Pin dependencies.
- Deploy React frontend and FastAPI backend.
- Smoke test deployed flows.
- Fix packaging/model-load issues.

## Deliverables

- live deployment,
- deployment instructions,
- reproducible requirements,
- green critical tests.

## Exit criteria

The deployed app works independently of the raw dataset and local training environment.

---

# Day 10 — Final Evaluation + Bug Fixing + Demo Preparation

## Goals

- Run final frozen evaluation one last time if and only if code/model changed after Day 9.
- Update README from real evaluation artifacts.
- Verify every claimed number.
- Prepare demo dataset/examples.
- Prepare presentation/demo script.
- Record fallback local demo.
- Freeze final commit/tag.

## Deliverables

- final evaluation report,
- completed README results,
- live demo,
- backup local demo,
- 3–5 minute presentation script,
- known limitations slide/section.

## Exit criteria

No unfinished P0 issue remains.

---

# Mandatory Cut Rules

If behind schedule:

## Cut first

- feedback capture,
- spike indicator,
- probability calibration,
- fancy visualizations,
- extra feature engineering,
- secondary deployment option.

## Never cut

- held-out evaluation,
- precision/recall,
- false-positive cost,
- baseline comparison,
- leakage prevention,
- frozen threshold,
- core demo reliability.

---

# Recommended Daily Time Allocation

- 60% core ML/data/evaluation until Day 7,
- 20% testing/reproducibility,
- 20% demo/UX.

After Day 7:

- 20% model changes maximum,
- 40% reliability/testing/deployment,
- 40% demo/story/presentation.
