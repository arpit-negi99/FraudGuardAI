from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def create_logistic_regression_baseline() -> LogisticRegression:
    """Create the reproducible Logistic Regression fraud baseline."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        solver="lbfgs",
        random_state=42,
    )


def fit_logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Fit the Logistic Regression baseline on transformed training features."""
    model = create_logistic_regression_baseline()
    return model.fit(X_train, y_train)


def predict_fraud_probabilities(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Return predicted fraud probabilities for transformed features."""
    return model.predict_proba(X)[:, 1]
