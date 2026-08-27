from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import FraudPredictor


DEMO_ROWS = 5
BATCH_SIZE = 100
DEMO_TRANSACTIONS_PATH = ROOT / "artifacts" / "demo" / "demo_transactions.csv"
DEMO_LABELS_PATH = ROOT / "artifacts" / "demo" / "demo_labels.csv"


def main() -> None:
    print("FraudGuard AI - Inference Demo")
    print()

    transactions = pd.read_csv(DEMO_TRANSACTIONS_PATH)
    labels = pd.read_csv(DEMO_LABELS_PATH)
    demo_df = transactions.merge(labels, how="left", on="TransactionID", validate="one_to_one")
    offline_labels = demo_df["isFraud"].astype(int).tolist()
    inference_df = demo_df.drop(columns=["isFraud", "demo_case"], errors="ignore")

    predictor = FraudPredictor(config_path=ROOT / "configs" / "config.yaml")

    started = time.perf_counter()
    single_result = predictor.predict_transaction(
        inference_df.iloc[[0]],
        include_explanation=False,
    )
    single_without_shap_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    single_with_shap = predictor.predict_transaction(
        inference_df.iloc[[0]],
        include_explanation=True,
    )
    single_with_shap_ms = (time.perf_counter() - started) * 1000

    batch_df = inference_df.head(BATCH_SIZE).copy()
    started = time.perf_counter()
    batch_predictions = predictor.predict_batch(batch_df)
    batch_ms = (time.perf_counter() - started) * 1000
    batch_summary = predictor.summarize_batch(batch_predictions)

    print("Single transaction")
    print(f"TransactionID: {single_result['transaction_id']}")
    print(f"Risk score: {single_result['risk_score']:.6f}")
    print(f"Threshold: {single_result['threshold']:.2f}")
    print(f"Decision: {single_result['decision']}")
    print(f"Offline ground truth label: {offline_labels[0]}")
    print()

    print("Top model contributors")
    contributors = single_with_shap.get("explanation", {}).get("top_risk_factors", [])
    if contributors:
        for item in contributors[:5]:
            print(f"- {item['feature']}")
    else:
        print("- unavailable")
    print()

    print("Batch summary")
    print(f"Rows: {batch_summary['transactions_scored']}")
    print(f"Reviews: {batch_summary['review_count']}")
    print(f"Allows: {batch_summary['allow_count']}")
    print(f"Review rate: {batch_summary['review_rate']:.6f}")
    print()

    print("Latency")
    print(f"Single prediction without SHAP: {single_without_shap_ms:.2f} ms")
    print(f"Single prediction with SHAP: {single_with_shap_ms:.2f} ms")
    print(f"Batch prediction ({len(batch_df)} rows): {batch_ms:.2f} ms")


if __name__ == "__main__":
    main()
