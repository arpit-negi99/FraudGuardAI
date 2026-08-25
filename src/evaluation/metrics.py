from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_predictions(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert fraud probabilities to binary predictions at a fixed threshold."""
    probabilities = np.asarray(y_prob)
    return (probabilities >= threshold).astype(int)


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return model-independent binary fraud metrics for probabilities and labels."""
    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob, dtype=float)
    predictions = binary_predictions(probabilities, threshold)

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "pr_auc": _safe_average_precision(labels, probabilities),
        "roc_auc": _safe_roc_auc(labels, probabilities),
        "accuracy": float(accuracy_score(labels, predictions)),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
        "review_rate": float(predictions.mean()) if len(predictions) else 0.0,
    }


def evaluate_majority_legitimate_baseline(y_true: np.ndarray) -> dict[str, Any]:
    """Evaluate the trivial baseline that predicts every row as legitimate."""
    labels = np.asarray(y_true).astype(int)
    predictions = np.zeros_like(labels)
    return {
        "model": "MajorityClassLegitimate",
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def _safe_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))
