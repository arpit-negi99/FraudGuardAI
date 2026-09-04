from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DEPLOYMENT_ARTIFACTS = (
    Path("artifacts/models/xgboost_model.json"),
    Path("artifacts/preprocessors/preprocessor.joblib"),
    Path("artifacts/preprocessors/preprocessing_metadata.json"),
    Path("artifacts/results/xgboost_validation_metrics.json"),
    Path("artifacts/results/model_comparison.json"),
    Path("artifacts/results/xgboost_threshold_analysis.csv"),
    Path("artifacts/results/xgboost_threshold_summary.json"),
    Path("artifacts/results/xgboost_cost_summary.json"),
    Path("artifacts/results/shap_global_importance.csv"),
    Path("artifacts/results/final_test_metrics.json"),
    Path("artifacts/demo/demo_transactions.csv"),
    Path("artifacts/demo/demo_labels.csv"),
)


def missing_deployment_artifacts(root: Path = ROOT) -> list[Path]:
    """Return required React/FastAPI deployment artifacts missing from the local package."""
    return [path for path in REQUIRED_DEPLOYMENT_ARTIFACTS if not (root / path).exists()]
