from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.incidents.simulator import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_ROW_COUNT,
    generate_payment_incident_events,
    summarize_incident_dataset,
    validate_no_impossible_combinations,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic payment incident data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROW_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", default="data/synthetic/payment_incidents.csv")
    parser.add_argument("--summary-output", default="artifacts/results/payment_incident_data_summary.json")
    args = parser.parse_args()

    data = generate_payment_incident_events(row_count=args.rows, random_seed=args.seed)
    validate_no_impossible_combinations(data)
    summary = summarize_incident_dataset(data, random_seed=args.seed)

    output_path = ROOT / args.output
    summary_path = ROOT / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("FraudGuard AI - Synthetic Payment Incident Data")
    print(summary["data_note"])
    print(f"Rows: {summary['total_rows']}")
    print(f"Normal: {summary['normal_count']}")
    print(f"Incidents: {summary['incident_count']}")
    print(f"Incident rate: {summary['incident_rate']:.4f}")
    print("Distribution by incident type:")
    for incident_type, count in summary["distribution_by_incident_type"].items():
        print(f"  {incident_type}: {count}")
    print(f"Saved dataset: {output_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
