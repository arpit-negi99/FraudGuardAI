# Production Definition — FraudGuard AI

## Meaning of “Production-Ready MVP”

For this 10-day project, production-ready does **not** mean bank-grade infrastructure.

It means the fraud detector is reproducible, measured honestly, fails safely, can be deployed as a stable demo, and has enough engineering discipline that another developer can run the full pipeline without hidden notebook state.

---

# 1. AI/ML Requirements

The MVP is production-ready only when:

- the XGBoost model trains reproducibly,
- logistic-regression baseline is evaluated,
- chronological split is fixed and recorded,
- no test leakage is detected,
- review threshold is chosen on validation only,
- held-out test evaluation is complete,
- precision/recall/F1/PR-AUC are reported,
- false-positive cost is reported,
- major false positives/false negatives are inspected,
- inference uses the exact preprocessing used in training,
- malformed inputs are rejected safely,
- model artifacts are versioned.

---

# 2. LLM Requirements

No LLM is part of the MVP.

Therefore the project does not need:

- prompt versioning,
- output JSON validation for LLM responses,
- hallucination mitigation for generated answers,
- token budgeting,
- LLM rate-limit handling,
- model API fallback.

This absence is intentional and reduces failure modes.

If an LLM is later introduced for narrative summaries, it must never alter the fraud score or decision and must be documented as a new scoped feature first.

---

# 3. Security

## Secrets

- No model API key is required for inference.
- Kaggle credentials, if used during dataset download, must live in environment variables or the standard Kaggle credential mechanism and never be committed.

## Input validation

- enforce schema,
- limit CSV size,
- reject unexpected binary/executable uploads,
- validate numeric ranges where semantics are known,
- handle unknown categorical values safely.

## Sensitive information

The demo must state that users should not upload:

- full card numbers,
- CVV,
- PIN,
- OTP,
- passwords,
- bank-login credentials.

## Prompt injection

Not applicable to P0 because no LLM receives untrusted text.

## Defense-only constraint

The app and documentation must not include fraud-generation, evasion, bypass, or attacker-optimization functionality.

---

# 4. Performance Targets

These are engineering targets, not claimed benchmark results.

## Single-row inference

Target: **<300 ms** on a typical CPU after model/app warm-up, including preprocessing but excluding first app startup.

## Batch inference

Target: **10,000 rows in <10 seconds** on a typical development CPU.

If the final model misses this target, reduce feature/model size before adding infrastructure.

## UI response

Non-training interactions should feel interactive; training is never triggered from the demo UI.

## Memory

Inference/deployed app target: **<1.5 GB RAM**.

The raw 1+ GB dataset is not loaded in the deployed app.

## Model artifact size

Target: **<150 MB** combined model + preprocessing artifacts.

---

# 5. Reliability

## Retry behavior

- user input errors: correct and retry immediately,
- batch row errors: report them; do not silently skip,
- SHAP failure: return score/decision without explanation,
- model artifact load failure: stop the app and show a clear error.

## Timeouts

No external API calls are in P0. Batch processing should enforce a practical row limit to avoid hanging the demo.

## Fallback behavior

There is no fallback model. If the trained model is unavailable, do not generate a prediction.

## Error handling

All exceptions shown to end users should be concise; detailed stack traces belong in logs/development mode.

---

# 6. Testing

## Unit tests

- cost function,
- threshold selection,
- decision rule,
- derived features,
- schema validation.

## Data pipeline tests

- join row-count invariant,
- split ordering invariant,
- no overlap between splits,
- preprocessing fitted only on train,
- expected target values,
- unknown category handling.

## Model tests

- model artifact loads,
- prediction shape is correct,
- scores are finite and in expected range,
- same input gives same output for frozen artifact.

## Evaluation tests

- metric calculations match a tiny hand-verified example,
- threshold uses validation only,
- test evaluation does not mutate model/threshold.

## Integration tests

- raw row → preprocessing → model → decision,
- batch CSV → scored output,
- model + SHAP explanation path.

## Critical user-flow tests

- valid single transaction,
- invalid/missing field,
- unknown category,
- batch upload success,
- malformed CSV,
- evaluation artifact present/missing.

---

# 7. Deployment

## Primary environment

React + Vite frontend with a lightweight FastAPI backend.

Streamlit may remain available as a fallback/debug interface, but it is no longer the primary client-facing deployment target after the React migration.

## Dependencies

Pin tested versions before final deployment for at least:

- Python,
- pandas,
- NumPy,
- scikit-learn,
- XGBoost,
- SHAP,
- Streamlit,
- joblib or equivalent serializer.

## Model storage

Store model/preprocessing artifacts in the repository only if their size/license is acceptable. Otherwise fetch a versioned artifact during deployment from a simple release/object-storage location.

Do not bundle raw training data.

## Environment variables

Expected P0 deployment variables: none, unless artifact hosting requires one.

Development-only Kaggle variables may be used for data download.

## Deployment steps

1. Run final data preparation.
2. Train baseline and primary model.
3. Lock threshold.
4. Run final held-out evaluation.
5. Freeze model artifacts.
6. Run automated tests.
7. Launch app locally with frozen artifacts.
8. Deploy FastAPI backend and React frontend.
9. Run API/client smoke tests against deployed app.
10. Tag/freeze final demo commit.

---

# 8. Definition of Done

- [ ] Feature scope is frozen.
- [ ] Raw data is excluded from Git.
- [ ] Data schema checks pass.
- [ ] Chronological splits are generated reproducibly.
- [ ] Logistic-regression baseline is trained and evaluated.
- [ ] XGBoost model is trained reproducibly.
- [ ] Validation-only threshold is stored with assumptions.
- [ ] Held-out test metrics are generated from code.
- [ ] Precision, recall, F1, and PR-AUC are reported.
- [ ] False-positive count and modeled false-positive cost are reported.
- [ ] At least 50 high-confidence FPs and 50 high-confidence FNs are reviewed.
- [ ] No suspicious leakage feature remains unresolved.
- [ ] Single transaction scoring works.
- [ ] Batch CSV scoring works.
- [ ] SHAP explanation works or fails gracefully.
- [ ] Demo never auto-blocks a transaction.
- [ ] No offensive fraud functionality exists.
- [ ] Unit/data/model/integration tests pass.
- [ ] App deploys successfully.
- [ ] README contains only actual measured results.
- [ ] Final demo script uses real outputs from the frozen build.
