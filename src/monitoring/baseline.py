from __future__ import annotations

from typing import Any

import pandas as pd

from src.monitoring.schema import DEFAULT_BASELINE_WINDOWS, MONITORED_METRICS


def build_historical_baseline(
    windows: pd.DataFrame,
    baseline_windows: int = DEFAULT_BASELINE_WINDOWS,
) -> dict[str, dict[str, float]]:
    """Build baseline statistics from the initial period only."""
    if windows.empty:
        raise ValueError("Cannot build monitoring baseline from empty windows.")
    if baseline_windows <= 1:
        raise ValueError("baseline_windows must be greater than 1.")
    baseline_frame = windows.sort_values("window_start").head(baseline_windows)
    return {
        metric: {
            "mean": float(baseline_frame[metric].mean()),
            "std": float(baseline_frame[metric].std(ddof=0)),
            "median": float(baseline_frame[metric].median()),
        }
        for metric in MONITORED_METRICS
    }


def baseline_to_rows(baseline: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [{"metric": metric, **values} for metric, values in baseline.items()]
