# FraudGuard AI

Cost-aware merchant transaction fraud-risk scoring with a frozen XGBoost model.

## Project

FraudGuard AI scores IEEE-CIS-style transactions, returns a fraud risk score, and recommends `ALLOW` or `REVIEW`. It is decision support only; it never automatically blocks a payment.

## Problem

Merchant transaction fraud risk management requires balancing missed fraud against unnecessary reviews of legitimate customers.

## Differentiator

FraudGuard is not just fraud prediction:

- cost-aware policy analysis
- review-capacity thresholding
- SHAP explainability
- human-in-the-loop `ALLOW` / `REVIEW` decisioning

## Final Held-Out Metrics

Frozen system:

- Model: `XGBoost`
- Features: `422`
- Threshold: `0.60`
- Model artifact: `artifacts/models/xgboost_model.json`
- Preprocessor artifact: `artifacts/preprocessors/preprocessor.joblib`

Threshold `0.60` was selected using validation analysis before opening the held-out test set.

Held-out chronological test metrics:

- Precision: `0.364018`
- Recall: `0.563088`
- F1: `0.442180`
- PR-AUC: `0.514931`
- ROC-AUC: `0.891247`
- Review rate: `0.053838`

## Setup

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Run

```bash
python -m streamlit run app.py
```

No API keys or external model services are required.

## Testing

```bash
python -m pytest -v
python scripts/demo_inference.py
```

## Demo

- Transaction Inspector: inspect packaged historical demo transactions and SHAP contributors.
- Batch Analysis: score a small CSV-style batch and download scored results.
- Risk Policy Lab: explore validation-derived threshold tradeoffs without changing the frozen final policy.

The deployed demo uses frozen artifacts and small demo CSVs under `artifacts/demo/`; raw IEEE-CIS training CSVs are not required for normal inference startup.

## Limitations

- Many IEEE-CIS features are anonymized.
- Costs are simulated assumptions, not observed merchant economics.
- Current model recall on the held-out test is 56.3%.
- The model is decision support, not an automatic blocker.
- The risk score is not calibrated as a guaranteed probability.
- The dataset may not reflect current production fraud patterns.

See `documentation/README.md` and `PROJECT-STATUS.md` for the full project log and methodology.
