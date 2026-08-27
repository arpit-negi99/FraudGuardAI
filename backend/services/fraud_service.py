from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.predict import ArtifactLoadError, FraudPredictor, InferenceError
from src.inference.presentation import (
    DEFAULT_THRESHOLD,
    build_policy_presets,
    demo_outcome_message,
    enrich_predictions,
    historical_outcome,
    nearest_threshold_metrics,
    priority_band,
    review_queue,
    risk_distribution,
    risk_status,
    simulate_policy_cost,
    spike_status,
)


ROOT = Path(__file__).resolve().parents[2]
DEMO_TRANSACTIONS_PATH = ROOT / "artifacts/demo/demo_transactions.csv"
DEMO_LABELS_PATH = ROOT / "artifacts/demo/demo_labels.csv"
THRESHOLD_TABLE_PATH = ROOT / "artifacts/results/xgboost_threshold_analysis.csv"
THRESHOLD_SUMMARY_PATH = ROOT / "artifacts/results/xgboost_threshold_summary.json"
COST_SUMMARY_PATH = ROOT / "artifacts/results/xgboost_cost_summary.json"
FINAL_METRICS_PATH = ROOT / "artifacts/results/final_test_metrics.json"


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    return FraudPredictor()


@lru_cache(maxsize=1)
def get_demo_transactions() -> pd.DataFrame:
    return pd.read_csv(DEMO_TRANSACTIONS_PATH)


@lru_cache(maxsize=1)
def get_demo_labels() -> pd.DataFrame:
    return pd.read_csv(DEMO_LABELS_PATH)


@lru_cache(maxsize=1)
def get_threshold_table() -> pd.DataFrame:
    return pd.read_csv(THRESHOLD_TABLE_PATH)


@lru_cache(maxsize=1)
def get_threshold_summary() -> dict[str, Any]:
    return _load_json(THRESHOLD_SUMMARY_PATH)


@lru_cache(maxsize=1)
def get_cost_summary() -> dict[str, Any]:
    return _load_json(COST_SUMMARY_PATH)


@lru_cache(maxsize=1)
def get_final_metrics() -> dict[str, Any]:
    return _load_json(FINAL_METRICS_PATH)


def health() -> dict[str, Any]:
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_loaded": predictor.model is not None,
        "preprocessor_loaded": predictor.preprocessor is not None,
        "threshold": predictor.threshold,
    }


def demo_predictions(include_top_signal: bool = True) -> list[dict[str, Any]]:
    transactions = get_demo_transactions()
    predictor = get_predictor()
    scored = predictor.predict_batch(transactions, include_explanations=False)
    enriched = enrich_predictions(scored, transactions)
    labels = get_demo_labels().rename(columns={"TransactionID": "transaction_id"})
    enriched = enriched.merge(labels, how="left", on="transaction_id")
    if include_top_signal:
        enriched["top_signal"] = enriched.apply(_top_signal_for_row, axis=1)
    return [_row_to_public_dict(row) for row in enriched.to_dict(orient="records")]


def demo_transaction(transaction_id: int) -> dict[str, Any]:
    transactions = get_demo_transactions()
    selected = transactions[transactions["TransactionID"] == transaction_id]
    if selected.empty:
        raise KeyError(f"Demo transaction {transaction_id} was not found.")
    result = score_transaction(selected.iloc[0].to_dict(), include_explanation=True)
    label_row = get_demo_labels()
    label_row = label_row[label_row["TransactionID"] == transaction_id]
    if not label_row.empty:
        label = int(label_row.iloc[0]["isFraud"])
        result["historical_outcome"] = historical_outcome(label)
        result["demo_outcome"] = demo_outcome_message(label, result["decision"])
    return result


def score_transaction(transaction: dict[str, Any], include_explanation: bool = True) -> dict[str, Any]:
    predictor = get_predictor()
    result = predictor.predict_transaction(
        transaction,
        threshold=DEFAULT_THRESHOLD,
        include_explanation=include_explanation,
    )
    result["priority"] = priority_band(float(result["risk_score"]))
    result["amount"] = _optional_float(transaction.get("TransactionAmt"))
    if include_explanation:
        result["contributors"] = _contributors_from_result(result)
    return result


def score_batch(
    transactions: list[dict[str, Any]],
    include_explanations: bool = False,
) -> dict[str, Any]:
    rows = pd.DataFrame(transactions)
    predictor = get_predictor()
    scored = predictor.predict_batch(
        rows,
        threshold=DEFAULT_THRESHOLD,
        include_explanations=include_explanations,
    )
    enriched = enrich_predictions(scored, rows)
    scored_rows = [_row_to_public_dict(row) for row in enriched.to_dict(orient="records")]
    review_count = sum(1 for row in scored_rows if row["decision"] == "REVIEW")
    return {
        "transaction_count": len(scored_rows),
        "review_count": review_count,
        "allow_count": len(scored_rows) - review_count,
        "review_rate": review_count / len(scored_rows),
        "transactions": scored_rows,
    }


def risk_summary() -> dict[str, Any]:
    rows = pd.DataFrame(demo_predictions(include_top_signal=False))
    queue = review_queue(rows)
    highest = float(rows["risk_score"].max())
    review_rate = float((rows["decision"] == "REVIEW").mean())
    return {
        "transactions_analyzed": int(len(rows)),
        "needs_review": int(len(queue)),
        "allow_count": int((rows["decision"] == "ALLOW").sum()),
        "critical_count": int((rows["priority"] == "Critical").sum()),
        "high_count": int((rows["priority"] == "High").sum()),
        "review_rate": review_rate,
        "average_risk": float(rows["risk_score"].mean()),
        "highest_risk": highest,
        "risk_status": risk_status(review_rate, highest),
        "risk_distribution": risk_distribution(rows).to_dict(orient="records"),
    }


def review_queue_rows() -> list[dict[str, Any]]:
    return [_row_to_public_dict(row) for row in review_queue(pd.DataFrame(demo_predictions())).to_dict(orient="records")]


def policy_presets() -> list[dict[str, Any]]:
    table = get_threshold_table()
    presets = []
    for preset in build_policy_presets(get_threshold_summary()):
        metrics = nearest_threshold_metrics(table, float(preset["threshold"]))
        presets.append({**preset, "metrics": metrics})
    return presets


def policy_simulation(review_cost: float, fraud_loss_multiplier: float) -> dict[str, Any]:
    return simulate_policy_cost(get_cost_summary(), review_cost, fraud_loss_multiplier)


def spike_monitor() -> dict[str, Any]:
    return spike_status(pd.DataFrame(demo_predictions(include_top_signal=False)))


def final_evaluation() -> dict[str, Any]:
    return get_final_metrics()


def _top_signal_for_row(row: pd.Series) -> str | None:
    try:
        detail = demo_transaction(int(row["transaction_id"]))
    except Exception:
        return None
    contributors = detail.get("contributors", [])
    if not contributors:
        return None
    return contributors[0]["feature"]


def _contributors_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    risk_factors = result.get("explanation", {}).get("top_risk_factors", [])
    return [
        {
            "feature": item["feature"],
            "value": item["value"],
            "impact": item["shap_value"],
            "direction": "increases_risk",
        }
        for item in risk_factors
    ]


def _row_to_public_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = {
        "transaction_id": int(row["transaction_id"]) if row.get("transaction_id") is not None else None,
        "amount": _optional_float(row.get("amount")),
        "risk_score": float(row["risk_score"]),
        "threshold": float(row["threshold"]),
        "decision": row["decision"],
        "priority": row.get("priority") or priority_band(float(row["risk_score"])),
        "risk_band": row.get("risk_band"),
        "top_signal": row.get("top_signal"),
    }
    if row.get("isFraud") is not None and not pd.isna(row.get("isFraud")):
        label = int(row["isFraud"])
        item["historical_outcome"] = historical_outcome(label)
        item["demo_outcome"] = demo_outcome_message(label, row["decision"])
    return item


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
