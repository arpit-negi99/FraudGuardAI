from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_binary_classifier


DEFAULT_THRESHOLDS = [round(value, 2) for value in np.arange(0.01, 1.0, 0.01)]


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate threshold-dependent classification metrics across a threshold grid."""
    rows = []
    for threshold in thresholds:
        metrics = evaluate_binary_classifier(y_true, y_prob, threshold=float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_positive": metrics["true_positive"],
                "false_positive": metrics["false_positive"],
                "true_negative": metrics["true_negative"],
                "false_negative": metrics["false_negative"],
                "review_rate": metrics["review_rate"],
            }
        )
    return pd.DataFrame(rows)


def row_with_highest_f1(threshold_table: pd.DataFrame) -> dict[str, float | int]:
    """Return the threshold row with highest F1, breaking ties by lower review rate."""
    sorted_table = threshold_table.sort_values(
        ["f1", "review_rate", "threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return _series_to_dict(sorted_table.iloc[0])


def highest_precision_with_min_recall(
    threshold_table: pd.DataFrame,
    min_recall: float,
) -> dict[str, float | int] | None:
    """Return highest-precision threshold row subject to a minimum recall constraint."""
    candidates = threshold_table[threshold_table["recall"] >= min_recall]
    if candidates.empty:
        return None
    sorted_candidates = candidates.sort_values(
        ["precision", "review_rate", "threshold"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    return _series_to_dict(sorted_candidates.iloc[0])


def lowest_review_rate_with_min_recall(
    threshold_table: pd.DataFrame,
    min_recall: float,
) -> dict[str, float | int] | None:
    """Return lowest-review-rate threshold row subject to a minimum recall constraint."""
    candidates = threshold_table[threshold_table["recall"] >= min_recall]
    if candidates.empty:
        return None
    sorted_candidates = candidates.sort_values(
        ["review_rate", "precision", "threshold"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    return _series_to_dict(sorted_candidates.iloc[0])


def metrics_at_threshold(
    threshold_table: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    """Return metrics for the grid threshold closest to the requested threshold."""
    index = (threshold_table["threshold"] - threshold).abs().idxmin()
    return _series_to_dict(threshold_table.loc[index])


def summarize_thresholds(threshold_table: pd.DataFrame) -> dict[str, object]:
    """Summarize useful validation operating-point candidates."""
    return {
        "evaluation_split": "validation",
        "threshold_grid": {
            "min": float(threshold_table["threshold"].min()),
            "max": float(threshold_table["threshold"].max()),
            "count": int(len(threshold_table)),
        },
        "metrics_at_threshold_0_50": metrics_at_threshold(threshold_table, 0.50),
        "highest_f1": row_with_highest_f1(threshold_table),
        "highest_precision_recall_at_least_0_60": highest_precision_with_min_recall(
            threshold_table, 0.60
        ),
        "highest_precision_recall_at_least_0_70": highest_precision_with_min_recall(
            threshold_table, 0.70
        ),
        "lowest_review_rate_recall_at_least_0_60": lowest_review_rate_with_min_recall(
            threshold_table, 0.60
        ),
    }


def _series_to_dict(row: pd.Series) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    integer_fields = {"true_positive", "false_positive", "true_negative", "false_negative"}
    for key, value in row.to_dict().items():
        result[key] = int(value) if key in integer_fields else float(value)
    return result
