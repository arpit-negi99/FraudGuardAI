from __future__ import annotations

import pandas as pd

from src.monitoring.schema import DEFAULT_WINDOW_MINUTES, VERY_HIGH_RISK_THRESHOLD


def aggregate_windows(
    records: pd.DataFrame,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Aggregate synthetic stream records into deterministic time windows."""
    if records.empty:
        raise ValueError("Cannot aggregate an empty monitoring stream.")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive.")
    data = records.copy()
    data["event_time"] = pd.to_datetime(data["event_time"], errors="raise")
    data = data.sort_values("event_time")
    data["window_start"] = data["event_time"].dt.floor(f"{window_minutes}min")
    rows = []
    for window_start, group in data.groupby("window_start", sort=True):
        fraud_scores = group["fraud_risk_score"].fillna(0.0).astype(float)
        incidents = group["payment_incident_detected"].fillna(False).astype(bool)
        severity = group["payment_incident_severity"].fillna("NONE")
        incident_type = group["payment_incident_type"].fillna("NORMAL_PAYMENT")
        scenario = group["scenario_type"].mode().iloc[0]
        expected_spike = bool(group["expected_spike"].astype(bool).any())
        transaction_count = int(group["transaction_id"].notna().sum())
        payment_count = int(group["payment_id"].notna().sum())
        review_count = int((group["fraud_decision"] == "REVIEW").sum())
        very_high_count = int((fraud_scores >= VERY_HIGH_RISK_THRESHOLD).sum())
        incident_count = int(incidents.sum())
        critical_count = int((severity == "CRITICAL").sum())
        high_count = int((severity == "HIGH").sum())
        rows.append(
            {
                "window_start": window_start,
                "window_end": pd.to_datetime(window_start) + pd.to_timedelta(int(window_minutes), unit="m"),
                "scenario_type": scenario,
                "expected_spike": expected_spike,
                "transaction_count": transaction_count,
                "payment_count": payment_count,
                "mean_fraud_risk": float(fraud_scores.mean()),
                "review_count": review_count,
                "review_rate": review_count / transaction_count if transaction_count else 0.0,
                "very_high_risk_count": very_high_count,
                "very_high_risk_rate": very_high_count / transaction_count if transaction_count else 0.0,
                "payment_incident_count": incident_count,
                "payment_incident_rate": incident_count / payment_count if payment_count else 0.0,
                "critical_incident_count": critical_count,
                "high_incident_count": high_count,
                "critical_high_incident_rate": (critical_count + high_count) / payment_count
                if payment_count
                else 0.0,
                "debit_service_mismatch_count": int((incident_type == "DEBIT_SERVICE_MISMATCH").sum()),
                "debit_service_mismatch_rate": float((incident_type == "DEBIT_SERVICE_MISMATCH").mean()),
                "complaint_escalation_count": int((incident_type == "COMPLAINT_ESCALATION_RISK").sum()),
                "complaint_escalation_rate": float((incident_type == "COMPLAINT_ESCALATION_RISK").mean()),
                "retry_risk_count": int((incident_type == "RETRY_RELATED_PAYMENT_RISK").sum()),
                "retry_risk_rate": float((incident_type == "RETRY_RELATED_PAYMENT_RISK").mean()),
                "captured_unfulfilled_count": int((incident_type == "CAPTURED_BUT_UNFULFILLED").sum()),
                "captured_unfulfilled_rate": float((incident_type == "CAPTURED_BUT_UNFULFILLED").mean()),
            }
        )
    result = pd.DataFrame(rows)
    result["window_start"] = result["window_start"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    result["window_end"] = result["window_end"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return result
