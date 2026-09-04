from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from backend.config import get_settings
from backend.services import fraud_service


ROOT = Path(__file__).resolve().parents[1]
DEMO_TRANSACTIONS = ROOT / "artifacts/demo/demo_transactions.csv"
OUTPUT_PATH = ROOT / "artifacts/results/hot_path_latency.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FraudGuard hot-path latency on packaged demo rows.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--skip-api", action="store_true")
    args = parser.parse_args()

    rows = pd.read_csv(DEMO_TRANSACTIONS).head(max(args.iterations, 1))
    records = rows.to_dict(orient="records")
    model_only = _measure(lambda row: fraud_service.score_transaction(row, include_explanation=False), records)
    explanation = _measure(lambda row: fraud_service.score_transaction(row, include_explanation=True), records[: min(5, len(records))])

    api_local = None
    if not args.skip_api:
        api_local = _measure(lambda row: _post_predict(args.api_url, row, include_explanation=False), records)

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stream_mode": get_settings().stream_mode,
        "iterations": len(records),
        "model_only_ms": model_only,
        "predict_api_ms": api_local,
        "shap_explanation_ms": explanation,
        "stream_overhead_ms": _overhead(api_local, model_only),
        "notes": "Benchmarked on packaged demo rows; not a production-scale load test.",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _measure(fn, records: list[dict]) -> dict[str, float]:
    timings = []
    for row in records:
        start = time.perf_counter()
        fn(row)
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "median": round(statistics.median(timings), 3),
        "p95": round(_percentile(timings, 95), 3),
        "p99": round(_percentile(timings, 99), 3),
    }


def _post_predict(api_url: str, row: dict, include_explanation: bool = False) -> dict:
    payload = {
        "transaction": {key: None if pd.isna(value) else value for key, value in row.items()},
        "include_explanation": include_explanation,
    }
    request = Request(
        f"{api_url}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"API benchmark failed: {exc}") from exc


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((percentile / 100) * (len(ordered) - 1))))
    return ordered[index]


def _overhead(api_result: dict[str, float] | None, model_result: dict[str, float]) -> dict[str, float] | None:
    if api_result is None:
        return None
    return {
        key: round(api_result[key] - model_result[key], 3)
        for key in ("median", "p95", "p99")
    }


if __name__ == "__main__":
    main()
