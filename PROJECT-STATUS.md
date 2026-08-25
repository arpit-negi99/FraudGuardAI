# Project

FraudGuard AI

## Goal

Build a cost-aware merchant transaction fraud-risk detector using the IEEE-CIS Fraud Detection dataset.

The system should detect fraudulent transactions while measuring both:

* false-positive cost
* missed-fraud cost

## Current Stage

Demo Polished — Ready to Freeze Policy

## Completed Before This Task

### Planning and Documentation

Existing documentation:

* `documentation/AGENTS.md`
* `documentation/AI-ML-Architecture.md`
* `documentation/Data-Strategy.md`
* `documentation/Development-Plan.md`
* `documentation/Documentation-Consistency-Check.md`
* `documentation/Feature-Scope.md`
* `documentation/ML-Evaluation.md`
* `documentation/Production-Definition.md`
* `documentation/Project-Decision.md`
* `documentation/README.md`
* `documentation/User-Flow-Experience.md`

### Dataset

IEEE-CIS Fraud Detection dataset downloaded.

Available files:

* `data/raw/train_transaction.csv`
* `data/raw/train_identity.csv`
* `data/raw/test_transaction.csv`
* `data/raw/test_identity.csv`
* `data/raw/sample_submission.csv`

### Environment

Python virtual environment exists at:

`.venv/`

## Core ML Decisions

Primary model:

`XGBoost`

Baseline:

`Logistic Regression`

Target:

`isFraud`

Evaluation split:

```text
70% chronological train
15% chronological validation
15% chronological held-out test
```

Split using:

`TransactionDT`

## Critical ML Rules

* Do not randomly split the labeled dataset.
* Do not use Kaggle `test_transaction.csv` for model evaluation.
* Do not use `TransactionID` as a model feature.
* Do not use `isFraud` as a model feature.
* Do not use `TransactionDT` as an initial P0 model feature.
* Fit preprocessing only on training data.
* Do not calculate imputation statistics using validation/test data.
* Do not learn category mappings from validation/test data.
* Do not determine dropped columns using validation/test data.
* Do not optimize thresholds against the final held-out test set.
* Do not use SMOTE initially.
* Never fabricate evaluation results.

## Technologies Explicitly Not Required

Do not add without explicit architectural justification:

* LLMs
* RAG
* embeddings
* vector databases
* fine-tuning
* deep learning
* agentic frameworks
* microservices
* Kubernetes

## Planned Build Order

1. Data foundation
2. Logistic Regression baseline
3. Baseline evaluation
4. XGBoost model
5. Validation-based threshold selection
6. Cost-aware decision engine
7. Error analysis
8. SHAP explainability
9. Streamlit demo
10. Final held-out evaluation
11. Deployment
12. Presentation/demo preparation

## Results

```text
Total labeled transactions: 590540
Train rows: 413378
Validation rows: 88581
Test rows: 88581

Train fraud rate: 0.035169
Validation fraud rate: 0.034341
Test fraud rate: 0.034804

Dropped high-missing columns: 9
Final transformed feature count: 422

Baseline Precision: 0.149115
Baseline Recall: 0.678172
Baseline F1: 0.244475
Baseline PR-AUC: 0.401383
Baseline ROC-AUC: 0.847032
Baseline Accuracy: 0.856053
Baseline threshold: 0.50 (fixed conventional validation threshold; not optimized)
Validation review rate: 0.156185
Validation true positives: 2063
Validation false positives: 11772
Validation true negatives: 73767
Validation false negatives: 979

Majority baseline accuracy: 0.965659
Majority baseline precision: 0.000000
Majority baseline recall: 0.000000
Majority baseline F1: 0.000000

XGBoost Precision: 0.335067
XGBoost Recall: 0.677844
XGBoost F1: 0.448456
XGBoost PR-AUC: 0.571018
XGBoost ROC-AUC: 0.918641
XGBoost Accuracy: 0.942742
XGBoost review rate: 0.069473
XGBoost true positives: 2062
XGBoost false positives: 4092
XGBoost true negatives: 81447
XGBoost false negatives: 980
XGBoost training time: 141.22 seconds
XGBoost best iteration: 799
XGBoost estimators used: 800
XGBoost scale_pos_weight: 27.434310

Threshold analysis grid: 0.01 to 0.99, 99 thresholds
Threshold 0.50 precision: 0.335067
Threshold 0.50 recall: 0.677844
Threshold 0.50 F1: 0.448456
Threshold 0.50 review rate: 0.069473

Highest-F1 validation threshold: 0.79
Highest-F1 precision: 0.655989
Highest-F1 recall: 0.464497
Highest-F1 F1: 0.543880
Highest-F1 review rate: 0.024317

Best precision with recall >= 0.60 threshold: 0.61
Best precision with recall >= 0.60 precision: 0.441078
Best precision with recall >= 0.60 recall: 0.607824
Best precision with recall >= 0.60 F1: 0.511197
Best precision with recall >= 0.60 review rate: 0.047324

Best precision with recall >= 0.70 threshold: 0.46
Best precision with recall >= 0.70 precision: 0.300451
Best precision with recall >= 0.70 recall: 0.700855
Best precision with recall >= 0.70 F1: 0.420596
Best precision with recall >= 0.70 review rate: 0.080107

Lowest review rate with recall >= 0.60 threshold: 0.61
Lowest review rate with recall >= 0.60 precision: 0.441078
Lowest review rate with recall >= 0.60 recall: 0.607824
Lowest review rate with recall >= 0.60 F1: 0.511197
Lowest review rate with recall >= 0.60 review rate: 0.047324

Cost analysis assumptions: scenario results only, not merchant ground truth
Fraud loss multiplier: 1.0
Review cost scenarios: low=1.0, medium=5.0, high=10.0
Validation TransactionAmt invalid count: 0
Allow-all modeled missed-fraud exposure: 496907.59 cost units

Low review-cost minimum-cost threshold: 0.08
Low review-cost precision: 0.064691
Low review-cost recall: 0.959237
Low review-cost F1: 0.121207
Low review-cost review rate: 0.509218
Low review-cost false positives: 42189
Low review-cost false negatives: 124
Low review-cost fraud amount detected: 487714.39
Low review-cost fraud amount missed: 9193.20
Low review-cost false-positive cost: 42189.00
Low review-cost missed-fraud cost: 9193.20
Low review-cost total estimated cost: 51382.20

Medium review-cost minimum-cost threshold: 0.23
Medium review-cost precision: 0.141892
Medium review-cost recall: 0.863248
Medium review-cost F1: 0.243724
Medium review-cost review rate: 0.208927
Medium review-cost false positives: 15881
Medium review-cost false negatives: 416
Medium review-cost fraud amount detected: 443353.28
Medium review-cost fraud amount missed: 53554.31
Medium review-cost false-positive cost: 79405.00
Medium review-cost missed-fraud cost: 53554.31
Medium review-cost total estimated cost: 132959.31

High review-cost minimum-cost threshold: 0.38
High review-cost precision: 0.239316
High review-cost recall: 0.754767
High review-cost F1: 0.363406
High review-cost review rate: 0.108308
High review-cost false positives: 7298
High review-cost false negatives: 746
High review-cost fraud amount detected: 390486.48
High review-cost fraud amount missed: 106421.11
High review-cost false-positive cost: 72980.00
High review-cost missed-fraud cost: 106421.11
High review-cost total estimated cost: 179401.11

Review-rate <= 0.05 constrained candidate threshold: 0.60
Review-rate <= 0.05 constrained precision: 0.430783
Review-rate <= 0.05 constrained recall: 0.612755
Review-rate <= 0.05 constrained F1: 0.505903
Review-rate <= 0.05 constrained review rate: 0.048848

SHAP explainer: shap.TreeExplainer
SHAP transformed feature count: 422
SHAP global validation sample size: 3000
SHAP runtime: 26.62 seconds
SHAP top global feature 1: C13
SHAP top global feature 2: TransactionAmt
SHAP top global feature 3: C1
SHAP top global feature 4: C14
SHAP top global feature 5: card1

Example explanation high-risk true positive: TransactionID 3481071, probability 0.999935
Example explanation high-risk false positive: TransactionID 3456622, probability 0.998787
Example explanation missed fraud false negative: TransactionID 3481470, probability 0.003222
Example explanation clearly legitimate true negative: TransactionID 3458851, probability 0.000015

Inference default threshold: 0.60 (validation-derived demo/default policy; not final held-out-test proof)
Inference single validation example TransactionID: 3400378
Inference single validation example risk score: 0.163838
Inference single validation example decision: ALLOW
Inference single validation example top risk contributors: V83, M4, C5, V70, V5
Inference batch demo rows: 100
Inference batch demo reviews: 4
Inference batch demo allows: 96
Inference batch demo review rate: 0.040000
Inference latency single prediction without SHAP: 114.03 ms
Inference latency single prediction with SHAP: 496.07 ms
Inference latency batch prediction 100 rows: 146.30 ms

Streamlit demo local app: `app.py`
Streamlit pages implemented: Risk Overview, Transaction Inspector, Batch Analysis, Risk Policy Lab, Model & Methodology
Streamlit local run verified: HTTP 200 from `streamlit run app.py`
Streamlit navigation verification: all 5 sections ran with 0 Streamlit test-harness exceptions
Streamlit batch upload verification: small CSV scored with 0 Streamlit test-harness exceptions
Streamlit CSV download verification: download button rendered after batch scoring
Streamlit threshold slider verification: slider reran at threshold 0.79 with 0 exceptions
Streamlit tests: 95 passed, 0 failed, 0 skipped
Streamlit dependency installed: streamlit 1.62.0
Streamlit UI support tests added: 8 passed
Streamlit held-out test evaluation: not performed

UI polish completed: Risk Overview, Transaction Inspector, Batch Analysis, Risk Policy Lab, and Model & Methodology refined
UI polish tests: 103 passed, 0 failed, 0 skipped
UI polish local app verification: HTTP 200 from `streamlit run app.py`
UI polish navigation verification: all 5 sections ran with 0 Streamlit test-harness exceptions
UI polish sample batch flow: sample CSV generated without `isFraud`, uploaded, scored, and results download rendered with 0 exceptions
UI polish policy preset verification: Highest F1 preset and threshold slider at 0.79 ran with 0 exceptions
UI polish held-out test evaluation: not performed
Second visual polish pass completed after review feedback that the UI looked too similar to the previous version
Second visual polish local app verification: HTTP 200 from `streamlit run app.py`
Second visual polish tests: 103 passed, 0 failed, 0 skipped

Selected threshold: TODO
Review rate: TODO
False positives: TODO
Missed fraud cases: TODO

Estimated false-positive cost: TODO
Estimated missed-fraud cost: TODO
```

## Completed

Data loading complete.

Transaction and identity merge validation complete.

Chronological 70/15/15 splitting complete.

Feature/target separation complete with `TransactionID`, `TransactionDT`, and `isFraud` excluded from P0 model features.

Train-only preprocessing complete.

High-missingness columns are selected using training features only.

Numerical medians are fitted using training features only.

Categorical encoding uses train-fitted `OrdinalEncoder` with unknown categories mapped to `-1`.

Preprocessor serialization complete.

Unit tests complete: 24 passed, 0 failed, 0 skipped.

Real IEEE-CIS data preparation pipeline complete.

Logistic Regression baseline complete.

Validation evaluation complete at fixed threshold 0.50.

First XGBoost validation model complete.

XGBoost validation evaluation complete at fixed threshold 0.50.

XGBoost validation threshold analysis complete.

Cost-aware validation analysis complete.

SHAP explainability complete.

Reusable single-transaction inference complete.

Batch inference complete.

Configurable threshold support complete.

Local SHAP explanation integration complete.

Inference batch summary helper complete.

Inference unit and integration tests complete: 87 passed, 0 failed, 0 skipped.

Inference demo complete using validation rows only.

Streamlit demo implemented.

UI polish completed.

UI polish improvements completed:

* Risk Overview now emphasizes validation Precision, Recall, Review Rate, and PR-AUC, with demo/default threshold metrics where appropriate.
* Logistic Regression vs XGBoost comparison now uses cleaner percentage formatting and calculated validation improvements from saved artifacts.
* Majority-baseline message is shorter and focused on why accuracy is misleading for fraud detection.
* Global SHAP importance is sorted descending and uses a clearer `Average model impact` label.
* Sidebar threshold copy is shorter while preserving the validation-derived/demo-policy caveat.
* Transaction Inspector uses a shorter header, clearer metric cards, stronger ALLOW/REVIEW status presentation, and historical false-positive/false-negative context when applicable.
* Built-in demo extra-column warnings are moved into `Technical details`; uploaded-data warnings remain available.
* SHAP tables now use `Feature`, `Observed value`, and `Risk contribution` labels with rounded contributions.
* Batch Analysis now includes a real validation-derived sample CSV download with `isFraud` removed.
* Batch results keep prominent summary cards, secondary risk metrics, highest-risk-first sorting, and scored CSV download.
* Risk Policy Lab now formats F1 as a percentage, supports policy presets, shows a dynamic validation trade-off summary, and keeps cost figures labeled as modeled cost units.
* Model & Methodology is concise and explicitly lists validation-only metrics, untouched held-out test, anonymized features, scenario-assumption costs, and decision-support framing.

UI polish local verification completed.

UI polish support tests complete: 103 passed, 0 failed, 0 skipped.

UI polish implementation note:

* This task changed Streamlit presentation/helper code only.
* A more visible Streamlit visual layer was added with a branded overview hero, styled metric cards, decision banners, dashboard panels, and a darker navigation sidebar.
* Overview cards now look distinct from default Streamlit metrics and use the threshold 0.60 validation operating point for Precision, Recall, and Review Rate.
* Transaction Inspector now presents model output in custom cards plus a clear ALLOW/REVIEW decision banner.
* Batch Analysis and Risk Policy Lab now use styled cards for summary and policy metrics.
* Model & Methodology now uses compact summary cards instead of a plain text list.
* XGBoost model, preprocessing, feature set, Logistic Regression, validation probabilities, threshold analysis, cost analysis, SHAP calculations, and inference output semantics were not changed.
* Held-out test split remains untouched.

Streamlit pages/features completed:

* Risk Overview with validation XGBoost metrics, model comparison, majority-baseline context, and SHAP global importance.
* Transaction Inspector with small representative validation examples, frozen inference scoring, policy decision, SHAP local explanation, and offline label separation.
* Batch Analysis with CSV upload, frozen batch inference, summary metrics, sorted results table, and downloadable scored CSV.
* Risk Policy Lab with validation threshold slider, policy presets, threshold trade-off chart, and cost scenario comparison.
* Model & Methodology with concise dataset/model/split/explainability/limitation notes.

Streamlit local run verified.

Streamlit UI support tests complete: 95 passed, 0 failed, 0 skipped.

Streamlit implementation note:

* The app imports and calls the existing inference layer instead of duplicating ML logic.
* The app reads existing validation result artifacts for metrics, threshold analysis, cost scenarios, and SHAP global importance.
* No model retraining, preprocessing refit, hyperparameter tuning, or held-out test evaluation is triggered by the UI.
* Default threshold 0.60 is labeled as validation-derived demo/default policy, not a final production threshold.
* Cost values are labeled as modeled cost units, not actual merchant savings.
* Batch precision/recall is not calculated for uploaded CSVs because production-style input normally has no labels.
* Streamlit is cached with `st.cache_resource` for the predictor and `st.cache_data` for static artifacts and validation examples.
* Local app test harness verified all navigation sections with 0 exceptions.
* Local app test harness verified CSV upload scoring with 0 exceptions.
* Local app test harness verified the scored CSV download button rendered.
* Local app test harness verified threshold slider rerun at 0.79 with 0 exceptions.
* Streamlit install upgraded `protobuf` to 7.36.0 and pip reported conflicts with unrelated installed Google/MediaPipe packages in the global Python environment.
* The existing preprocessor artifact emits a scikit-learn version warning when loaded because it was serialized with scikit-learn 1.9.0 and this environment has 1.5.2.
* Held-out test split remains untouched.

Inference artifacts reused:

* `artifacts/preprocessors/preprocessor.joblib`
* `artifacts/preprocessors/preprocessing_metadata.json`
* `artifacts/models/xgboost_model.json`

Inference implementation note:

* The inference layer loads frozen artifacts and does not retrain or refit preprocessing.
* Single-transaction input supports dictionaries, pandas Series, and one-row DataFrames.
* Batch inference accepts pandas DataFrames and preserves row count.
* `TransactionID` and `TransactionDT` may be retained as metadata but are excluded from model features by the fitted preprocessor contract.
* `isFraud` is not required for inference and is ignored if provided.
* Missing expected feature columns are inserted as missing values, using the existing train-fitted preprocessing behavior.
* Extra unknown columns are ignored with a warning.
* Unseen categorical values use the existing train-fitted unknown-category path.
* SHAP explanations are optional and fail gracefully without blocking fraud-risk scoring.
* LOW/MEDIUM/HIGH risk bands are display-only categories; the decision remains threshold-based.
* Held-out test split was not evaluated.

SHAP artifacts produced:

* `artifacts/results/shap_global_importance.csv`
* `artifacts/results/shap_global_importance.png`
* `artifacts/results/shap_summary.png`
* `artifacts/results/example_explanations.json`

SHAP implementation note:

* Explanations use the existing saved XGBoost model and train-fitted preprocessor.
* Global SHAP importance uses a fixed 3,000-row validation sample for runtime control.
* Individual examples use validation rows only.
* Threshold 0.50 is used only to define example confusion-matrix categories.
* SHAP values are model attributions, not causal proof.
* Anonymized IEEE-CIS features are reported by feature name only; no semantic meanings are invented.
* Held-out test split was not evaluated.

Cost analysis artifacts produced:

* `artifacts/results/xgboost_cost_analysis.csv`
* `artifacts/results/xgboost_cost_summary.json`
* `artifacts/results/total_cost_vs_threshold.png`
* `artifacts/results/cost_components_vs_threshold.png`

Cost analysis note:

* Cost values are validation-set scenario estimates in cost units.
* `TransactionAmt` is dataset-derived.
* Review cost and fraud loss multiplier are configurable assumptions, not IEEE-CIS ground truth.
* No final production threshold was selected.
* Held-out test split was not evaluated.

Threshold analysis artifacts produced:

* `artifacts/results/xgboost_threshold_analysis.csv`
* `artifacts/results/xgboost_threshold_summary.json`
* `artifacts/results/precision_recall_vs_threshold.png`
* `artifacts/results/f1_vs_threshold.png`
* `artifacts/results/review_rate_vs_threshold.png`
* `artifacts/results/false_positives_false_negatives_vs_threshold.png`

Threshold analysis note:

* Validation probabilities were recomputed from the saved XGBoost model and train-fitted preprocessor.
* Held-out test split was not evaluated.
* No final production threshold or business-cost threshold was selected.

XGBoost artifacts produced:

* `artifacts/models/xgboost_model.json`
* `artifacts/results/xgboost_validation_metrics.json`
* `artifacts/results/model_comparison.json`

XGBoost training note:

* `scale_pos_weight` calculated from the training split only.
* Validation split used for early stopping and validation metrics only.
* Held-out test split was not evaluated.

Baseline artifacts produced:

* `artifacts/models/logistic_regression.joblib`
* `artifacts/models/baseline_scaler.joblib`
* `artifacts/results/logistic_baseline_metrics.json`

Baseline training note:

* `lbfgs` solver used because `liblinear` and `saga` did not complete promptly on the full dense chronological training matrix.
* scikit-learn emitted a convergence warning because `lbfgs` reached the configured iteration limit.

Artifacts produced:

* `artifacts/preprocessors/preprocessor.joblib`
* `artifacts/preprocessors/preprocessing_metadata.json`

## Next Action

Freeze the existing XGBoost model, preprocessing, features, and threshold 0.60, then perform final held-out evaluation once.
