from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """Calculate XGBoost class imbalance weight from training labels only."""
    labels = np.asarray(y_train).astype(int)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0:
        raise ValueError("Cannot calculate scale_pos_weight with zero positive samples.")
    return negative_count / positive_count


def create_xgboost_classifier(
    scale_pos_weight: float,
    early_stopping_rounds: int | None = 50,
    n_estimators: int = 800,
) -> XGBClassifier:
    """Create the first conservative XGBoost fraud classifier."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )


def fit_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """Fit XGBoost on train only while using validation only for early stopping."""
    model = create_xgboost_classifier(scale_pos_weight=scale_pos_weight)
    return model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )


def predict_fraud_probabilities(model: XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return predicted fraud probabilities for transformed features."""
    return model.predict_proba(X)[:, 1]


def save_xgboost_model(model: XGBClassifier, path: str | Path) -> None:
    """Save an XGBoost model using its native JSON format."""
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)


def load_xgboost_model(path: str | Path) -> XGBClassifier:
    """Load an XGBoost model saved with native JSON serialization."""
    model = XGBClassifier()
    model.load_model(Path(path))
    return model
