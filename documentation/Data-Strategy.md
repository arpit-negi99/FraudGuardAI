# Data Strategy — FraudGuard AI

## 1. Data Source

Primary source: **IEEE-CIS Fraud Detection** competition dataset.

Use:

- `train_transaction.csv`
- `train_identity.csv`

The transaction table includes the binary target `isFraud`; identity information is joined by `TransactionID` and is not available for every transaction.

The competition data is subject to Kaggle competition rules. Do not commit or redistribute the raw files through the project repository unless the license/rules explicitly permit it.

---

# 2. Why This Dataset

It matches the selected loss class better than generic credit-card PCA benchmarks because it represents online transaction fraud and includes transaction, card, address, email, count, time-difference, match, and identity/device-related feature groups.

It is large enough to support a real held-out test while still being feasible for a 10-day tabular-ML project using careful preprocessing and CPU histogram boosting.

---

# 3. Data Requirements

## Size

Public descriptions indicate roughly 590k labeled training transactions. Final local counts must be recomputed.

## Format

CSV tables joined by `TransactionID`.

## Label

`isFraud ∈ {0,1}`.

## Quality

Expect:

- severe class imbalance,
- many missing values,
- partially anonymized features,
- high-cardinality categorical values,
- incomplete identity coverage.

## Class balance

Fraud is roughly 3.5% of the public training data according to public descriptions. Recompute exact prevalence after download.

---

# 4. Data Processing

## Join

Left join identity fields onto the transaction table by `TransactionID`.

Validate row count before and after join.

## Cleaning

- no target-dependent row removal,
- no blanket deletion of rows with missing identity data,
- preserve numeric `NaN` where the chosen model supports it,
- encode categorical fields using a train-fitted unknown-safe encoder,
- drop columns with >95% missingness based on training data only,
- exclude raw identifier columns from the model.

## Normalization

XGBoost does not require numeric standardization.

Logistic regression baseline may use scaling for numeric inputs where helpful.

## Missing values

Missingness can itself be informative in fraud data. Do not blindly mean-impute every field.

For XGBoost:

- keep numeric missing values as `NaN`,
- encode missing categorical values with an explicit category/sentinel.

## Duplicates

Investigate duplicate `TransactionID` rather than silently dropping.

## Outliers

Do not remove high transaction amounts merely because they are outliers; legitimate and fraudulent payments can both be large.

Use `log1p(TransactionAmt)` as an additional feature rather than truncating amount without evidence.

## Tokenization

Not applicable.

## Image preprocessing

Not applicable.

---

# 5. Feature Strategy

## P0 raw feature groups

Begin with a manageable subset drawn from:

- `TransactionAmt`,
- `ProductCD`,
- `card1`–`card6`,
- `addr1`, `addr2`,
- `dist1`, `dist2`,
- `P_emaildomain`, `R_emaildomain`,
- `C1`–`C14`,
- `D1`–`D15`,
- `M1`–`M9`,
- selected `id_*` features,
- `DeviceType`,
- optionally `DeviceInfo` only if cardinality/memory remain manageable,
- selected `V*` features only if train-only selection shows they materially improve performance and memory remains acceptable.

Do not force all available columns into the first model.

## P0 derived features

- log transaction amount,
- email-domain match indicator,
- identity missing-count,
- transaction-feature missing-count.

## P1 behavioral features

Only after error analysis, and only if computed from past data:

- previous transaction count for a stable entity proxy,
- prior amount mean/median,
- amount deviation from prior behavior,
- time since prior transaction.

If future rows are used to construct a past row’s features, the feature is invalid.

---

# 6. Data Splitting

Sort by `TransactionDT` and create contiguous partitions:

- 70% train,
- 15% validation,
- 15% test.

No random shuffle before splitting.

All preprocessing fit is performed after the split.

---

# 7. Data Leakage Risks

## Risk 1 — fitting preprocessing on the full dataset

Prevention: fit encoders, missingness/drop rules, feature selection, and scalers using train only.

## Risk 2 — threshold selection on the test set

Prevention: threshold is locked using validation scores only.

## Risk 3 — future-aware behavioral features

Prevention: compute historical aggregates using prior transactions only.

## Risk 4 — target-derived feature selection

Feature selection may use train labels but never validation/test labels for repeated cherry-picking.

## Risk 5 — memorizing IDs

Do not train on `TransactionID`. Be cautious with high-cardinality identity proxies that merely memorize entities.

## Risk 6 — competition-era engineered features

Some anonymized features may encode historical aggregates. This is part of the supplied dataset, but absolute conclusions about real-time deployability must be avoided because their exact construction is masked.

---

# 8. Data Privacy

The public dataset is anonymized/obfuscated, but the product architecture should still follow data-minimization principles.

For the MVP:

- never request real card numbers,
- never request CVV, PIN, OTP, passwords, or bank credentials,
- sample demo inputs use dataset-compatible anonymized values,
- uploaded CSVs are processed in-memory or temporary storage and are not retained by default,
- logs must not contain full sensitive transaction payloads.

---

# 9. Dataset Limitations

1. **Not India-specific.** Do not claim measured performance on Indian BFSI/UPI traffic.
2. **Partially anonymized features.** Some feature semantics are masked, limiting business interpretability.
3. **Historical competition data.** It does not capture current fraud tactics or present-day deployment drift.
4. **Selection bias.** Performance reflects this dataset, not all merchants.
5. **Identity sparsity.** Many transactions lack identity rows.
6. **Class distribution mismatch.** A real merchant’s fraud prevalence may be very different.
7. **Cost labels absent.** False-positive and fraud-loss costs must be modeled through transparent scenarios rather than claimed as observed merchant costs.

---

# 10. Training Dataset Requirement

A training dataset **is required** because the core model is supervised fraud classification.

Therefore this project is not an API-only or prompt-only system.

No LLM/foundation-model fine-tuning dataset is needed.

---

# 11. Deployment Data Policy

Do not deploy the full IEEE-CIS dataset.

Deploy only:

- trained model artifacts,
- preprocessing artifacts,
- a small set of non-sensitive demo/example rows,
- stored aggregate evaluation results.

For reproducibility, document how a developer with Kaggle access can obtain the dataset and rebuild the artifacts.

---

## Reference

IEEE-CIS Fraud Detection data description: https://www.kaggle.com/c/ieee-fraud-detection/data
