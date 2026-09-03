from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.schema import DEFAULT_BASELINE_WINDOWS, DEFAULT_WINDOW_MINUTES
from src.monitoring.simulator import generate_monitoring_stream
from src.monitoring.spike import (
    alert_rows,
    detection_delay,
    evaluate_binary_spike_detection,
    evaluate_monitoring_windows,
    scenario_performance,
)
from src.monitoring.windows import aggregate_windows


STREAM_PATH = ROOT / "data/synthetic/monitoring_stream.csv"
METRICS_PATH = ROOT / "artifacts/results/spike_monitor_metrics.json"
WINDOWS_PATH = ROOT / "artifacts/results/spike_monitor_windows.csv"
SCENARIO_PATH = ROOT / "artifacts/results/spike_monitor_scenario_metrics.csv"
DELAY_PATH = ROOT / "artifacts/results/spike_monitor_detection_delay.json"
REVIEW_PLOT_PATH = ROOT / "artifacts/results/review_rate_over_time.png"
INCIDENT_PLOT_PATH = ROOT / "artifacts/results/payment_incident_rate_over_time.png"
STATUS_PLOT_PATH = ROOT / "artifacts/results/operational_risk_over_time.png"


def main() -> None:
    if STREAM_PATH.exists():
        records = pd.read_csv(STREAM_PATH)
    else:
        records = generate_monitoring_stream()
        STREAM_PATH.parent.mkdir(parents=True, exist_ok=True)
        records.to_csv(STREAM_PATH, index=False)
    windows = aggregate_windows(records, window_minutes=DEFAULT_WINDOW_MINUTES)
    evaluated, baseline = evaluate_monitoring_windows(
        windows,
        baseline_windows=DEFAULT_BASELINE_WINDOWS,
    )
    metrics = evaluate_binary_spike_detection(evaluated)
    delays = detection_delay(evaluated)
    scenarios = scenario_performance(evaluated)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(
            {
                **metrics,
                "window_size_minutes": DEFAULT_WINDOW_MINUTES,
                "baseline_windows": DEFAULT_BASELINE_WINDOWS,
                "baseline": baseline,
                "latest": _compact_latest(evaluated.iloc[-1].to_dict()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    DELAY_PATH.write_text(json.dumps(delays, indent=2), encoding="utf-8")
    pd.DataFrame(scenarios).to_csv(SCENARIO_PATH, index=False)
    _windows_for_csv(evaluated).to_csv(WINDOWS_PATH, index=False)
    _plot_rate(evaluated, "review_rate", REVIEW_PLOT_PATH, "Review rate over synthetic time")
    _plot_rate(evaluated, "payment_incident_rate", INCIDENT_PLOT_PATH, "Payment incident rate over synthetic time")
    _plot_status(evaluated, STATUS_PLOT_PATH)
    print("FraudGuard AI - Spike Monitor Evaluation")
    print(f"Windows: {len(evaluated)}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")
    print(f"False alert rate: {metrics['false_alert_rate']:.6f}")
    print(f"Mean detection delay: {delays['mean_detection_delay_minutes']}")
    print(f"Median detection delay: {delays['median_detection_delay_minutes']}")
    print(f"Max detection delay: {delays['max_detection_delay_minutes']}")


def _windows_for_csv(evaluated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "window_start",
        "window_end",
        "scenario_type",
        "expected_spike",
        "status",
        "primary_driver",
        "review_rate",
        "payment_incident_rate",
        "critical_high_incident_rate",
        "debit_service_mismatch_rate",
        "complaint_escalation_rate",
        "retry_risk_rate",
        "recommended_action",
    ]
    return evaluated[columns].copy()


def _compact_latest(row: dict) -> dict:
    return {
        "window_start": row["window_start"],
        "window_end": row["window_end"],
        "scenario_type": row["scenario_type"],
        "status": row["status"],
        "primary_driver": row["primary_driver"],
        "recommended_action": row["recommended_action"],
    }


def _plot_rate(evaluated: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(evaluated)), evaluated[metric], color="#0f766e")
    plt.title(title)
    plt.xlabel("15-minute synthetic window")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_status(evaluated: pd.DataFrame, path: Path) -> None:
    rank = {"NORMAL": 0, "ELEVATED": 1, "HIGH": 2, "CRITICAL": 3}
    plt.figure(figsize=(10, 3.5))
    plt.step(range(len(evaluated)), [rank[item] for item in evaluated["status"]], where="post", color="#dc2626")
    plt.yticks(list(rank.values()), list(rank))
    plt.title("Operational risk over synthetic time")
    plt.xlabel("15-minute synthetic window")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
