# Project

FraudGuard AI

## Goal

Build a cost-aware merchant transaction fraud-risk detector using the IEEE-CIS Fraud Detection dataset.

The system should detect fraudulent transactions while measuring both:

* false-positive cost
* missed-fraud cost

## Current Stage

Module 2 Payment Incident Detection Integrated into React + FastAPI

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
Streamlit held-out test evaluation: not performed during UI implementation

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

Client-facing Streamlit UI redesign completed
Client-facing UI local app verification: HTTP 200 from `streamlit run app.py`
Client-facing UI navigation verification: Home, Review Queue, Transaction Details, Risk Monitor, Policy Settings, and About FraudGuard each ran with 0 Streamlit test-harness exceptions
Client-facing UI tests: 122 passed, 0 failed, 0 skipped
Client-facing UI held-out test evaluation: not performed during UI redesign

React primary frontend implemented
FastAPI backend implemented
Shared presentation/inference helper module implemented
Streamlit retained as fallback/debug UI in `app.py`
React pages implemented: Dashboard, Transactions, Review Queue, Risk Monitor, Policy, About, Transaction Details
FastAPI endpoints implemented: `/health`, `/demo/transactions`, `/demo/transactions/{transaction_id}`, `/predict`, `/predict/batch`, `/policy/presets`, `/policy/simulate`, `/risk/summary`, `/risk/review-queue`, `/risk/spike`, `/evaluation/final`
React frontend local verification: HTTP 200 from Vite dev server at `http://127.0.0.1:5173`
FastAPI backend local verification: `/health` returned status ok, model_loaded true, preprocessor_loaded true, threshold 0.60
FastAPI prediction verification: demo transaction detail returned decision plus 5 SHAP contributors
FastAPI batch verification: one-row batch returned transaction_count 1 and consistent allow/review count
FastAPI policy verification: presets returned Fraud First, Balanced, Low Friction
FastAPI spike verification: rolling review-rate spike endpoint returned Normal
Frontend build verification: `npm run build` succeeded
Frontend utility tests: 3 passed, 0 failed, 0 skipped
Python API/integration tests: 132 passed, 0 failed, 0 skipped
React/FastAPI held-out test tuning: not performed
Frozen ML unchanged during React/FastAPI migration

Module 2 - Payment Incident Detection: Data + deterministic baseline implemented
Module 2 data source: synthetic / simulated payment-event data only; not Razorpay production data
Module 2 supported incident types: DEBIT_SERVICE_MISMATCH, LATE_AUTHORIZATION_RISK, CAPTURED_BUT_UNFULFILLED, REFUND_REQUIRED, RETRY_RELATED_PAYMENT_RISK, COMPLAINT_ESCALATION_RISK, NORMAL_PAYMENT
Module 2 recommended actions: NO_ACTION, VERIFY_PAYMENT, CHECK_ORDER, INITIATE_REFUND, CONTACT_CUSTOMER, ESCALATE_REVIEW, MONITOR
Module 2 deterministic rule tests: included in Python suite
Module 2 Python tests: 147 passed, 0 failed, 0 skipped
Module 2 synthetic dataset generated: `data/synthetic/payment_incidents.csv`
Module 2 synthetic summary artifact generated: `artifacts/results/payment_incident_data_summary.json`
Module 2 synthetic dataset rows: 10000
Module 2 synthetic normal count: 7447
Module 2 synthetic incident count: 2553
Module 2 synthetic incident rate: 0.2553
Module 2 synthetic incident distribution: NORMAL_PAYMENT=7447, DEBIT_SERVICE_MISMATCH=791, CAPTURED_BUT_UNFULFILLED=495, REFUND_REQUIRED=380, RETRY_RELATED_PAYMENT_RISK=345, LATE_AUTHORIZATION_RISK=335, COMPLAINT_ESCALATION_RISK=207
Module 2 rule severity distribution on generated data: NONE=7447, HIGH=1666, MEDIUM=680, CRITICAL=207
Module 2 random seed: 42
Module 2 known limitations: synthetic payment-event data, not validated on real payment-provider production data, deterministic demonstration rules only, no Module 2 ML model yet
Module 1 frozen XGBoost model, preprocessor, threshold 0.60, SHAP logic, and held-out results unchanged during Module 2 implementation

Module 2 Step 2 deterministic rule evaluation complete
Module 2 standard synthetic evaluation rows: 10000
Module 2 standard binary precision: 1.000000
Module 2 standard binary recall: 1.000000
Module 2 standard binary F1: 1.000000
Module 2 standard binary confusion: TP=2553, FP=0, TN=7447, FN=0
Module 2 standard macro F1: 1.000000
Module 2 standard weighted F1: 1.000000
Module 2 stress synthetic dataset generated: `data/synthetic/payment_incidents_stress.csv`
Module 2 stress synthetic evaluation rows: 5000
Module 2 stress binary precision: 1.000000
Module 2 stress binary recall: 1.000000
Module 2 stress binary F1: 1.000000
Module 2 stress binary confusion: TP=1854, FP=0, TN=3146, FN=0
Module 2 stress macro F1: 1.000000
Module 2 stress weighted F1: 1.000000
Module 2 stress zero-support classes: REFUND_REQUIRED, RETRY_RELATED_PAYMENT_RISK, COMPLAINT_ESCALATION_RISK
Module 2 overlap behavior: highest severity wins; ties follow deterministic rule-check order
Module 2 overlap audit standard rows with multiple applicable rules: 1423
Module 2 overlap audit stress rows with multiple applicable rules: 680
Module 2 weakest cases: no false positives, false negatives, or incident-type errors found on current standard/stress synthetic datasets; remaining weakness is synthetic coverage, especially absent stress support for refund-required, retry-risk, and complaint-escalation primary labels
Module 2 severity evaluation: unavailable because synthetic ground truth does not independently define expected severity
Module 2 recommended-action evaluation: unavailable because synthetic ground truth does not independently define expected recommended actions
Module 2 anti-circularity check: synthetic ground truth generation does not call the detector
Module 2 evaluation recommendation: keep deterministic rules; ML is not justified by these synthetic evaluations
Module 2 evaluation artifacts produced: `payment_incident_rule_metrics.json`, `payment_incident_rule_per_class.csv`, `payment_incident_rule_confusion_matrix.png`, `payment_incident_stress_metrics.json`, `payment_incident_stress_per_class.csv`, `payment_incident_stress_confusion_matrix.png`, `payment_incident_error_examples.json`, `payment_incident_rule_precedence.json`
Module 2 Step 2 Python tests: 161 passed, 0 failed, 0 skipped

Module 2 - Payment Incident Detection integrated into React + FastAPI.
Module 2 API endpoints implemented: `GET /incidents`, `GET /incidents/summary`, `GET /incidents/{payment_id}`, `POST /incidents/evaluate`, `GET /incidents/types`
Module 2 frontend page implemented: Payment Incidents
Module 2 frontend integration: Dashboard payment incident section, Risk Monitor payment operations charts, About two-module architecture
Module 2 UI features implemented: summary cards, severity/type/search filters, incident table, detail panel, lifecycle fields, rule-engine reasons, recommended action mapping, fraud-vs-payment-incident comparison
Module 2 packaged demo IDs: `pay_syn_000007` low fraud score + high debit-service mismatch; `pay_syn_000003` captured but unfulfilled; `pay_syn_000012` late authorization risk
Module 2 stress-only opposite demo ID: `pay_stress_000038` high synthetic fraud score + no payment lifecycle incident
Module 2 runtime backend verification: `/incidents/summary`, `/incidents?severity=HIGH&limit=2`, `/incidents/pay_syn_000007`, and `POST /incidents/evaluate` returned HTTP 200
Module 2 runtime frontend verification: Vite dev server returned HTTP 200 at `http://127.0.0.1:5173`
Module 2 frontend build: `npm run build` succeeded with existing Vite chunk-size warning
Module 2 frontend utility tests: 7 passed, 0 failed, 0 skipped
Module 2 Python tests after integration: 171 passed, 0 failed, 0 skipped
Module 2 remains synthetic payment-event data only.
Module 2 remains deterministic rules only; no Module 2 ML model added.
Module 2 limitations: not validated on Razorpay production data; no real gateway, refund, webhook, chargeback, or customer action is performed.
Module 1 frozen XGBoost model, preprocessor, 422-feature contract, threshold 0.60, SHAP logic, and held-out results unchanged during Module 2 product integration.

Selected threshold: 0.60
Review rate: 0.053838
False positives: 3033
Missed fraud cases: 1347

Estimated false-positive cost: 15165.00 under medium review-cost scenario
Estimated missed-fraud cost: 216620.97 under fraud loss multiplier 1.0
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
* Held-out test split remained untouched during UI polish.

Client-facing UI redesign note:

* Navigation now uses merchant-facing sections: Home, Review Queue, Transaction Details, Risk Monitor, Policy Settings, and About FraudGuard.
* Home focuses on review workload, highest-risk activity, current demo risk status, and priority transactions.
* Review Queue shows REVIEW transactions first, with priority and minimum-risk filters plus CSV scoring/download support.
* Transaction Details presents risk score, recommended action, transaction amount, priority, plain-language signal tables, and demo-only analyst actions.
* Technical SHAP details and model metrics are available in expanders or About FraudGuard rather than dominating the primary workflow.
* Risk Monitor is framed as current demo/batch activity; live fraud-spike detection remains a future feature.
* Policy Settings uses business-friendly review strategies while preserving threshold 0.60 as the balanced frozen policy.
* The UI redesign changed Streamlit presentation/helper code only.
* XGBoost model, preprocessing, 422-feature contract, threshold 0.60, final held-out metrics, SHAP artifacts, and inference output semantics were not changed.

React/FastAPI migration note:

* React + Vite is now the primary client-facing submission UI.
* FastAPI wraps the existing frozen `FraudPredictor`; fraud scoring logic was not duplicated in JavaScript.
* Shared presentation helpers in `src/inference/presentation.py` define priority bands, review queue sorting, policy presets, cost simulation display, historical demo outcome labels, and lightweight rolling review-rate spike status.
* Dashboard shows merchant-facing cards, priority transactions, risk distribution, and current-session activity.
* Transactions page provides search, decision filtering, priority filtering, minimum-risk filtering, sorting, and sliced table display.
* Review Queue shows only `REVIEW` transactions sorted by risk, with actual SHAP top feature names as top signals.
* Transaction Details shows risk score, amount, priority, current policy, SHAP contributors, historical demo outcome, and session-only analyst actions.
* Risk Monitor shows current demo activity charts and a transparent rolling review-rate spike detector, not a second ML model.
* Policy page uses Fraud First, Balanced, and Low Friction strategy cards plus validation-only advanced metrics and scenario cost simulation.
* About page shows the frozen architecture, technology stack, final held-out metrics, limitations, and defense-only framing.
* Streamlit remains available as fallback/debug UI but is no longer the primary submission interface.
* XGBoost model, preprocessing artifact, 422 transformed features, threshold 0.60, final held-out metrics, and SHAP calculation were not changed.
* No held-out test result was used for tuning during this migration.

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
* Default threshold 0.60 is the frozen demo policy selected from validation analysis before final held-out evaluation.
* Cost values are labeled as modeled cost units, not actual merchant savings.
* Batch precision/recall is not calculated for uploaded CSVs because production-style input normally has no labels.
* Streamlit is cached with `st.cache_resource` for the predictor and `st.cache_data` for static artifacts and validation examples.
* Local app test harness verified all navigation sections with 0 exceptions.
* Local app test harness verified CSV upload scoring with 0 exceptions.
* Local app test harness verified the scored CSV download button rendered.
* Local app test harness verified threshold slider rerun at 0.79 with 0 exceptions.
* Streamlit install upgraded `protobuf` to 7.36.0 and pip reported conflicts with unrelated installed Google/MediaPipe packages in the global Python environment.
* The preprocessor artifact originally emitted a scikit-learn version warning before deployment hardening because it was serialized with scikit-learn 1.9.0 and the runtime had 1.5.2.
* Held-out test split remained untouched during Streamlit implementation.

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

## Final Frozen System

Model: XGBoost

Model artifact: `artifacts/models/xgboost_model.json`

Preprocessor artifact: `artifacts/preprocessors/preprocessor.joblib`

Features: 422

Threshold: 0.60

Decision: ALLOW / REVIEW

Freeze warning:

`The held-out test set has been opened. No future threshold/model/feature tuning should use these test results as optimization feedback.`

Deployment hardening note:

* Runtime dependency pins are recorded in `requirements-lock.txt`.
* scikit-learn runtime aligned to 1.9.0 to match the frozen preprocessor artifact.
* Standard app/demo inference now uses small packaged demo artifacts instead of loading raw IEEE-CIS CSVs.
* No API keys or external model services are required.

## Final Held-Out Test Metrics

```text
Test rows: 88581
Test fraud rate: 0.034804

Precision: 0.364018
Recall: 0.563088
F1: 0.442180
PR-AUC: 0.514931
ROC-AUC: 0.891247
Accuracy: 0.950554
TP: 1736
FP: 3033
TN: 82465
FN: 1347
Review rate: 0.053838
```

## Generalization

Validation vs held-out test at frozen threshold 0.60:

```text
Metric | Validation | Held-out test | Test - validation
precision: 0.430783 | 0.364018 | -0.066766
recall: 0.612755 | 0.563088 | -0.049667
f1: 0.505903 | 0.442180 | -0.063723
pr_auc: 0.571018 | 0.514931 | -0.056087
roc_auc: 0.918641 | 0.891247 | -0.027394
review_rate: 0.048848 | 0.053838 | 0.004990
```

## Final Cost Simulation

Held-out modeled cost simulation only. Threshold 0.60 was not optimized on test costs.

```text
Low review cost total: 219653.97
Medium review cost total: 231785.97
High review cost total: 246950.97
```

Final held-out evaluation artifacts produced:

* `artifacts/results/final_test_metrics.json`
* `artifacts/results/final_validation_vs_test.json`
* `artifacts/results/final_validation_vs_test.csv`
* `artifacts/results/final_test_cost_simulation.json`
* `artifacts/results/final_confusion_matrix.png`
* `artifacts/results/final_validation_vs_test.png`
* `artifacts/results/final_precision_recall_curve.png`

## Deployment Hardening Verification

```text
Python version: 3.12.0
pandas version: 3.0.5
numpy version: 2.5.2
scikit-learn version: 1.9.0
XGBoost version: 3.4.1
SHAP version: 0.52.0
Streamlit version: 1.62.0
joblib version: 1.5.3
PyYAML version: 6.0.3
matplotlib version: 3.11.1
pytest version: 9.1.1

project .venv pip check: no broken requirements found
global Python pip check: failed due to unrelated globally installed Google/MediaPipe/spaCy-side package conflicts with protobuf/numpy; FraudGuard .venv is clean.
project .venv pytest: 117 passed, 0 failed, 0 skipped
demo_inference.py: succeeded
Streamlit HTTP startup: 200
Streamlit page harness: Risk Overview, Transaction Inspector, Batch Analysis, Risk Policy Lab, and Model & Methodology each ran with 0 exceptions
Preprocessor sklearn version warning: absent after runtime alignment
Raw IEEE-CIS data required for standard app/demo startup: no
External API keys required: no
Absolute local paths required by app/demo: no
Required deployment artifacts present: yes
```

## Next Action

Upgrade payment incident simulation from single snapshots to multi-step payment lifecycle timelines.
