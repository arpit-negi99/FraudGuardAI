from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEMO_TRANSACTIONS = ROOT / "artifacts/demo/demo_transactions.csv"
SCENARIOS = {"normal", "fraud_spike", "payment_incident_spike", "mixed_spike"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay packaged demo transactions into the optional stream path.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="normal")
    parser.add_argument("--merchant-id", default="merchant_demo_001")
    parser.add_argument("--events-per-second", type=float, default=5.0)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    rows = pd.read_csv(DEMO_TRANSACTIONS)
    demo_predictions = _get_json(f"{args.api_url}/demo/transactions").get("transactions", [])
    high_risk_ids = {row["transaction_id"] for row in demo_predictions if row.get("decision") == "REVIEW"}
    high_risk_rows = rows[rows["TransactionID"].isin(high_risk_ids)]
    source_rows = high_risk_rows if args.scenario in {"fraud_spike", "mixed_spike"} and not high_risk_rows.empty else rows

    print("FraudGuard AI - stream replay")
    print(f"Scenario: {args.scenario}")
    print(f"Merchant: {args.merchant_id}")
    print(f"Events: {args.count} at {args.events_per_second}/second")

    delay = 1.0 / max(args.events_per_second, 0.1)
    sent = 0
    for index in range(args.count):
        row = source_rows.sample(1, random_state=random.randint(1, 1_000_000)).iloc[0].to_dict()
        payload = _scenario_payload(row, args.scenario, args.merchant_id, index)
        _post_json(f"{args.api_url}/predict", {"transaction": payload, "include_explanation": False})
        sent += 1
        time.sleep(delay)
    print(f"Replay complete. Posted {sent} prediction requests.")


def _scenario_payload(row: dict, scenario: str, merchant_id: str, index: int) -> dict:
    payload = {key: _json_value(value) for key, value in row.items()}
    payload["merchant_id"] = merchant_id
    payload["payment_id"] = f"stream_demo_{scenario}_{index:06d}"
    if scenario in {"payment_incident_spike", "mixed_spike"}:
        incident_types = [
            "DEBIT_SERVICE_MISMATCH",
            "COMPLAINT_ESCALATION_RISK",
            "RETRY_RELATED_PAYMENT_RISK",
        ]
        payload["payment_incident_detected"] = True
        payload["payment_incident_type"] = incident_types[index % len(incident_types)]
        payload["payment_incident_severity"] = "CRITICAL" if index % 3 == 1 else "HIGH"
    else:
        payload["payment_incident_detected"] = False
        payload["payment_incident_type"] = None
        payload["payment_incident_severity"] = None
    return payload


def _get_json(url: str) -> dict:
    try:
        with urlopen(url, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to read {url}: {exc}") from exc


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to post {url}: {exc}") from exc


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
