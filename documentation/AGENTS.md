# AGENTS.md — FraudGuard AI

## Project Context

FraudGuard AI is a **defense-only, cost-aware online transaction fraud detector** for merchants.

The core task is binary classification of labeled online transactions using the IEEE-CIS Fraud Detection dataset.

Primary model: **XGBoost**.

Baseline: **Logistic Regression**.

The product outputs a fraud risk score and an `ALLOW` or `REVIEW` recommendation. It does not automatically decline transactions.

The MVP explicitly measures precision, recall, F1, PR-AUC, false positives, missed-fraud value, and false-positive cost on a chronological held-out test set.

---

# Core Principles

1. Prefer simple solutions.
2. Avoid over-engineering.
3. Reuse existing components.
4. Keep experiments reproducible.
5. Document all model assumptions.
6. Never fabricate metrics, datasets, labels, or benchmark results.
7. Do not optimize for visual complexity at the expense of evaluation quality.
8. Preserve the defense-only nature of the project.
9. Do not add an LLM because it sounds impressive.
10. A complete, measured baseline-to-model pipeline is more valuable than an unfinished advanced architecture.

---

# AI/ML Rules

- Do not change the primary model family from XGBoost without documenting a specific deficiency and obtaining approval.
- Do not change evaluation metrics without justification.
- Do not claim model performance unless the evaluation command produced the result.
- Do not copy performance claims from Kaggle, papers, or other repositories.
- Do not fabricate evaluation artifacts.
- Do not introduce data leakage.
- Keep preprocessing identical between training and inference.
- Fit encoders, imputers, scalers, missingness/drop rules, and feature-selection logic using training data only.
- Use validation data for model selection/early stopping/threshold selection.
- Never tune using the final test set.
- Do not train on `TransactionID`.
- Treat `TransactionDT` primarily as the chronological split key; adding time-derived features requires justification.
- Do not use future transactions when constructing historical features.
- Do not add SMOTE or synthetic oversampling without a documented experiment showing benefit on the chronological validation set.
- Keep model and preprocessing artifacts versioned together.
- Use a fixed random seed where applicable; default to `42` unless the experiment explicitly requires otherwise.
- Save experiment configuration alongside metrics.

---

# Safety / Defense-Only Rules

This track is strictly defensive.

Do not implement or document:

- fraud-generation workflows,
- payment-fraud tutorials,
- evasion strategies,
- ways to bypass fraud controls,
- optimization of fraudulent transactions against the detector,
- attacker simulators designed to discover bypasses,
- credential theft,
- unauthorized payment actions.

Allowed work includes:

- fraud detection,
- defensive evaluation,
- false-positive analysis,
- robustness to malformed benign input,
- data-quality checks,
- model monitoring.

If a requested feature could materially enable fraud evasion, stop and escalate rather than implement it.

---

# LLM Rules

The approved MVP does **not** use an LLM.

Therefore:

- do not add OpenAI/Anthropic/Gemini dependencies,
- do not add RAG,
- do not add embeddings/vector databases,
- do not add LLM-generated fraud decisions,
- do not add a chat interface.

If a future approved scope adds an LLM for narrative summaries only:

- version prompts,
- validate structured outputs,
- handle malformed responses,
- handle API failures/rate limits,
- record token usage/cost,
- never expose API keys,
- never let the LLM alter the XGBoost fraud decision,
- evaluate generated summaries separately.

---

# Data Rules

- Raw IEEE-CIS competition data must not be committed to Git.
- Follow dataset license/competition rules.
- Join transaction and identity tables on `TransactionID` and validate row-count invariants.
- Never silently drop rows because identity data is missing.
- Never silently drop malformed batch rows; surface validation errors.
- Never log sensitive user-uploaded transaction payloads in production mode.
- Never ask users for full PAN/card number, CVV, PIN, OTP, passwords, or bank-login credentials.

---

# Evaluation Rules

The primary reporting set is the final chronological test partition.

Required metrics:

- precision,
- recall,
- F1,
- PR-AUC,
- ROC-AUC,
- false-positive rate,
- confusion matrix,
- review rate,
- false-positive cost,
- missed-fraud value,
- total scenario cost.

Rules:

1. Lock threshold using validation only.
2. Store the threshold and cost assumptions.
3. Evaluate the test set using frozen artifacts.
4. Do not pick a better-looking threshold after seeing test metrics.
5. README values must be read from actual evaluation artifacts or manually copied after verification; never invent placeholders as results.
6. `TODO` is preferable to a fabricated number.

---

# Experimentation

Every meaningful experiment should record:

- experiment name/id,
- date,
- dataset/split version,
- selected features,
- preprocessing version,
- model family,
- hyperparameters,
- class weighting,
- random seed,
- validation metrics,
- chosen threshold,
- cost assumptions,
- notes/decision.

Do not run a broad hyperparameter search unless the baseline pipeline is already complete and the remaining schedule permits it.

Preferred sequence:

1. baseline,
2. first XGBoost,
3. one focused improvement based on error analysis,
4. freeze.

---

# Coding Rules

- Prefer Python.
- Keep modules focused.
- Avoid unnecessary classes/abstractions.
- Use type hints for public functions where practical.
- Keep configuration separate from logic.
- Use explicit schemas/constants for feature names.
- Add error handling at file/input/artifact boundaries.
- Use structured logging for training/evaluation metadata.
- Avoid notebook-only hidden state for final workflows.
- A notebook may be used for EDA, but production training/evaluation must be callable from scripts/modules.
- Do not hardcode absolute local file paths.
- Do not hardcode metric results into the app.
- Never trigger training from the Streamlit UI.

---

# Testing Before an AI/ML Task Is Complete

1. Run relevant unit tests.
2. Run data validation.
3. Run model evaluation where applicable.
4. Run lint/type checking if configured.
5. Verify frozen inference.
6. Verify edge cases.
7. Confirm no test leakage.
8. Confirm the demo uses the same preprocessing/model artifacts as evaluation.

---

# Architecture Constraints

Approved:

- pandas / NumPy,
- scikit-learn,
- XGBoost,
- SHAP,
- Streamlit,
- simple local artifacts.

Not approved for P0/P1:

- LLM APIs,
- RAG,
- vector databases,
- deep-learning frameworks solely for this model,
- graph databases,
- Kafka,
- Kubernetes,
- microservices,
- separate frontend/backend stacks.

---

# Scope

Before significant architectural changes, read:

- `Project-Decision.md`
- `Feature-Scope.md`
- `AI-ML-Architecture.md`
- `User-Flow-Experience.md`
- `ML-Evaluation.md`
- `Data-Strategy.md`
- `Production-Definition.md`
- `Development-Plan.md`

Do not implement features outside the approved scope unless the scope documents are updated first.
