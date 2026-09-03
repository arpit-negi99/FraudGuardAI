from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.incidents.lifecycle_simulator import (
    DEFAULT_LIFECYCLE_COUNT,
    DEFAULT_LIFECYCLE_RANDOM_SEED,
    generate_payment_lifecycles,
    summarize_lifecycles,
)


LIFECYCLE_PATH = ROOT / "data/synthetic/payment_lifecycles.json"
SUMMARY_PATH = ROOT / "artifacts/results/payment_lifecycle_summary.json"


def main() -> None:
    lifecycles = generate_payment_lifecycles()
    summary = summarize_lifecycles(lifecycles, DEFAULT_LIFECYCLE_RANDOM_SEED)
    LIFECYCLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIFECYCLE_PATH.write_text(json.dumps(lifecycles, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FraudGuard AI - Payment Lifecycle Generation")
    print(f"Lifecycles: {summary['lifecycle_count']}")
    print(f"Average events: {summary['average_event_count']:.2f}")
    print(f"Active: {summary['active']}")
    print(f"Resolved: {summary['resolved']}")
    print(f"Normal: {summary['normal']}")
    print(f"Median resolution time: {summary['median_resolution_time_minutes']}")
    print(f"Saved: {LIFECYCLE_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
