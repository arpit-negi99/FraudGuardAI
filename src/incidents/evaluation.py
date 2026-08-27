from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.incidents import rules
from src.incidents.rules import evaluate_payment_incident
from src.incidents.schema import IncidentType, PaymentIncidentResult, parse_payment_event
from src.incidents.simulator import (
    STRESS_SCENARIO_TO_INCIDENT_TYPE,
    generate_payment_incident_events,
    generate_payment_incident_stress_events,
)


INCIDENT_CLASS_ORDER = [item.value for item in IncidentType]
RULE_PRECEDENCE = [
    "COMPLAINT_ESCALATION_RISK",
    "REFUND_REQUIRED",
    "CAPTURED_BUT_UNFULFILLED",
    "RETRY_RELATED_PAYMENT_RISK",
    "DEBIT_SERVICE_MISMATCH",
    "LATE_AUTHORIZATION_RISK",
]
RULE_PRECEDENCE_RATIONALE = (
    "The detector evaluates all conditions, then chooses the highest severity. "
    "When severities tie, Python's stable sort keeps the rule-check order above. "
    "This makes complaint escalation critical when applicable, makes refund-required "
    "more specific than captured-but-unfulfilled, and lets retry-specific unresolved "
    "cases win over broader debit-service mismatch."
)


def evaluate_dataset(data: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Evaluate deterministic incident rules against independent synthetic labels."""
    required = {"incident_label", "incident_type"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Incident evaluation data is missing required columns: {missing}")
    predictions = [evaluate_payment_incident(row).to_dict() for row in data.to_dict(orient="records")]
    predicted = pd.DataFrame(predictions)
    y_true = data["incident_label"].astype(bool)
    y_pred = predicted["incident_detected"].astype(bool)
    true_types = data["incident_type"].astype(str)
    pred_types = predicted["incident_type"].astype(str)

    binary = binary_metrics(y_true, y_pred)
    per_class = per_class_metrics(true_types, pred_types)
    matrix = confusion_matrix(true_types, pred_types, labels=INCIDENT_CLASS_ORDER)
    present_labels = sorted(set(true_types) | set(pred_types))
    overlaps = overlap_audit(data)
    return {
        "dataset": dataset_name,
        "rows": int(len(data)),
        "binary": binary,
        "macro_f1": float(
            f1_score(true_types, pred_types, labels=present_labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true_types,
                pred_types,
                labels=present_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": INCIDENT_CLASS_ORDER,
            "matrix": matrix.tolist(),
        },
        "severity_evaluation": {
            "available": False,
            "reason": "Synthetic ground truth does not independently define expected severity.",
        },
        "recommended_action_evaluation": {
            "available": False,
            "reason": "Synthetic ground truth does not independently define expected recommended actions.",
        },
        "rule_precedence": precedence_artifact(),
        "overlap_audit": overlaps,
    }


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def per_class_metrics(true_types: pd.Series, pred_types: pd.Series) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        true_types,
        pred_types,
        labels=INCIDENT_CLASS_ORDER,
        zero_division=0,
    )
    return [
        {
            "incident_type": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(INCIDENT_CLASS_ORDER)
    ]


def save_evaluation_artifacts(
    result: dict[str, Any],
    metrics_path: Path,
    per_class_path: Path,
    confusion_matrix_path: Path,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["per_class"]).to_csv(per_class_path, index=False)
    plot_confusion_matrix(result["confusion_matrix"], confusion_matrix_path)


def collect_error_examples(
    standard_data: pd.DataFrame,
    stress_data: pd.DataFrame,
    limit: int = 8,
) -> dict[str, Any]:
    return {
        "false_positive": _examples_for_error(standard_data, "standard", "false_positive", limit)
        + _examples_for_error(stress_data, "stress", "false_positive", limit),
        "false_negative": _examples_for_error(standard_data, "standard", "false_negative", limit)
        + _examples_for_error(stress_data, "stress", "false_negative", limit),
        "incorrect_incident_type": _examples_for_error(standard_data, "standard", "type_error", limit)
        + _examples_for_error(stress_data, "stress", "type_error", limit),
        "severity_mismatch": {
            "available": False,
            "reason": "No independent expected severity exists in the synthetic datasets.",
        },
        "overlapping_rules": _overlap_examples(standard_data, "standard", limit)
        + _overlap_examples(stress_data, "stress", limit),
    }


def save_error_examples(examples: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(examples, indent=2), encoding="utf-8")


def plot_confusion_matrix(confusion: dict[str, Any], output_path: Path) -> None:
    labels = confusion["labels"]
    matrix = confusion["matrix"]
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted incident type")
    ax.set_ylabel("True incident type")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def overlap_audit(data: pd.DataFrame) -> dict[str, Any]:
    applicable_counts = Counter()
    overlap_count = 0
    for row in data.to_dict(orient="records"):
        applicable = applicable_incident_types(row)
        for item in applicable:
            applicable_counts[item] += 1
        if len(applicable) > 1:
            overlap_count += 1
    return {
        "rows_with_multiple_applicable_rules": int(overlap_count),
        "applicable_rule_counts": dict(sorted(applicable_counts.items())),
    }


def applicable_incident_types(row: dict[str, Any]) -> list[str]:
    payment = parse_payment_event(row)
    checks = [
        rules._complaint_escalation(payment),
        rules._refund_required(payment),
        rules._captured_but_unfulfilled(payment),
        rules._retry_related_payment_risk(payment),
        rules._debit_service_mismatch(payment),
        rules._late_authorization_risk(payment),
    ]
    return [check.incident_type.value for check in checks if check is not None]


def precedence_artifact() -> dict[str, Any]:
    return {
        "primary_selection_rule": "Highest severity wins; ties follow deterministic rule-check order.",
        "rule_check_order": RULE_PRECEDENCE,
        "rationale": RULE_PRECEDENCE_RATIONALE,
    }


def simulator_ground_truth_is_independent() -> bool:
    standard_source = inspect.getsource(generate_payment_incident_events)
    stress_source = inspect.getsource(generate_payment_incident_stress_events)
    forbidden = ("evaluate_payment_incident", "src.incidents.rules")
    return not any(term in standard_source or term in stress_source for term in forbidden)


def _examples_for_error(
    data: pd.DataFrame,
    dataset: str,
    category: str,
    limit: int,
) -> list[dict[str, Any]]:
    examples = []
    for row in data.to_dict(orient="records"):
        prediction = evaluate_payment_incident(row).to_dict()
        true_label = bool(row["incident_label"])
        pred_label = bool(prediction["incident_detected"])
        true_type = str(row["incident_type"])
        pred_type = str(prediction["incident_type"])
        match = (
            category == "false_positive"
            and not true_label
            and pred_label
        ) or (
            category == "false_negative"
            and true_label
            and not pred_label
        ) or (
            category == "type_error"
            and true_label
            and pred_label
            and true_type != pred_type
        )
        if match:
            examples.append(_example(row, prediction, dataset))
        if len(examples) >= limit:
            break
    return examples[:limit]


def _overlap_examples(data: pd.DataFrame, dataset: str, limit: int) -> list[dict[str, Any]]:
    examples = []
    for row in data.to_dict(orient="records"):
        applicable = applicable_incident_types(row)
        if len(applicable) > 1:
            prediction = evaluate_payment_incident(row).to_dict()
            item = _example(row, prediction, dataset)
            item["applicable_incident_types"] = applicable
            examples.append(item)
        if len(examples) >= limit:
            break
    return examples


def _example(row: dict[str, Any], prediction: dict[str, Any], dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "payment_id": row["payment_id"],
        "scenario": row.get("scenario"),
        "true_incident_label": bool(row["incident_label"]),
        "true_incident_type": row["incident_type"],
        "predicted_incident_detected": prediction["incident_detected"],
        "predicted_incident_type": prediction["incident_type"],
        "predicted_severity": prediction["severity"],
        "recommended_action": prediction["recommended_action"],
        "reasons": prediction["reasons"],
        "fields": {
            "bank_debited": row["bank_debited"],
            "gateway_status": row["gateway_status"],
            "order_status": row["order_status"],
            "service_delivered": row["service_delivered"],
            "callback_received": row["callback_received"],
            "refund_status": row["refund_status"],
            "retry_count": row["retry_count"],
            "time_since_payment_minutes": row["time_since_payment_minutes"],
            "customer_complaint": row["customer_complaint"],
            "fraud_risk_score": row.get("fraud_risk_score"),
        },
    }
