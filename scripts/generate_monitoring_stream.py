from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.monitoring.simulator import DEFAULT_EVENTS_PER_WINDOW, DEFAULT_MONITORING_SEED, generate_monitoring_stream


OUTPUT_PATH = ROOT / "data/synthetic/monitoring_stream.csv"
SUMMARY_PATH = ROOT / "artifacts/results/monitoring_stream_summary.json"


def main() -> None:
    data = generate_monitoring_stream()
    summary = {
        "rows": int(len(data)),
        "events_per_window": DEFAULT_EVENTS_PER_WINDOW,
        "random_seed": DEFAULT_MONITORING_SEED,
        "scenario_distribution": {
            str(key): int(value) for key, value in data["scenario_type"].value_counts().to_dict().items()
        },
        "synthetic_monitoring_note": data["synthetic_monitoring_note"].iloc[0],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FraudGuard AI - Synthetic Monitoring Stream")
    print(f"Rows: {summary['rows']}")
    print(f"Events per window: {summary['events_per_window']}")
    print("Scenario distribution:")
    for scenario, count in summary["scenario_distribution"].items():
        print(f"  {scenario}: {count}")
    print(f"Saved stream: {OUTPUT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
