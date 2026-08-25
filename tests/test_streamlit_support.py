from __future__ import annotations

import ast
import json
from io import BytesIO

import pandas as pd
import pytest

from app import (
    build_policy_presets,
    demo_outcome_message,
    format_percent,
    format_shap_table,
    load_csv_artifact,
    load_json_artifact,
    main_warnings,
    nearest_threshold_metrics,
    parse_cost_scenarios,
    predictions_to_download_csv,
    prepare_batch_display,
    sample_batch_csv,
    sorted_shap_importance,
)


def test_validation_artifact_loading(tmp_path) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text(json.dumps({"evaluation_split": "validation", "pr_auc": 0.5}), encoding="utf-8")

    loaded = load_json_artifact(artifact)

    assert loaded["evaluation_split"] == "validation"


def test_threshold_lookup_returns_nearest_row() -> None:
    table = pd.DataFrame(
        {
            "threshold": [0.50, 0.60, 0.70],
            "precision": [0.3, 0.4, 0.5],
            "recall": [0.7, 0.6, 0.5],
        }
    )

    nearest = nearest_threshold_metrics(table, 0.62)

    assert nearest["threshold"] == 0.60


def test_shap_global_importance_is_sorted_descending() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["low", "high", "middle"],
            "mean_absolute_shap_value": [0.1, 0.9, 0.4],
            "rank": [3, 1, 2],
        }
    )

    sorted_importance = sorted_shap_importance(importance)

    assert sorted_importance.iloc[0]["feature"] == "high"
    assert "Average model impact" in sorted_importance.columns


def test_metric_percentage_formatter() -> None:
    assert format_percent(0.5059) == "50.6%"


def test_policy_preset_construction_uses_summary_values() -> None:
    summary = {
        "highest_precision_recall_at_least_0_70": {"threshold": 0.46},
        "lowest_review_rate_recall_at_least_0_60": {"threshold": 0.61},
        "highest_f1": {"threshold": 0.79},
    }

    presets = build_policy_presets(summary)

    assert [preset["name"] for preset in presets] == [
        "Higher Fraud Capture",
        "Capacity-Constrained",
        "Highest F1",
    ]
    assert [preset["metrics"]["threshold"] for preset in presets] == [0.46, 0.61, 0.79]


def test_sample_csv_does_not_contain_is_fraud() -> None:
    sample = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [10],
            "TransactionAmt": [20.0],
            "isFraud": [0],
            "demo_case": ["demo"],
        }
    )

    csv_text = sample_batch_csv(sample).decode("utf-8")

    assert "isFraud" not in csv_text
    assert "TransactionAmt" in csv_text


def test_sample_csv_is_inference_compatible_shape() -> None:
    sample = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [10, 20],
            "TransactionAmt": [20.0, 30.0],
            "ProductCD": ["W", "C"],
            "isFraud": [0, 1],
        }
    )

    parsed = pd.read_csv(BytesIO(sample_batch_csv(sample)))

    assert len(parsed) == 2
    assert "isFraud" not in parsed.columns
    assert {"TransactionID", "TransactionDT", "TransactionAmt"}.issubset(parsed.columns)


def test_batch_csv_conversion_for_download() -> None:
    predictions = pd.DataFrame(
        {
            "transaction_id": [1],
            "risk_score": [0.7],
            "threshold": [0.6],
            "decision": ["REVIEW"],
            "unused": ["x"],
        }
    )

    csv_bytes = predictions_to_download_csv(predictions)
    csv_text = csv_bytes.decode("utf-8")

    assert "transaction_id,risk_score,threshold,decision" in csv_text
    assert "unused" not in csv_text


def test_batch_display_sorts_highest_risk_first() -> None:
    predictions = pd.DataFrame(
        {
            "transaction_id": [1, 2],
            "risk_score": [0.1, 0.9],
            "threshold": [0.6, 0.6],
            "decision": ["ALLOW", "REVIEW"],
        }
    )

    display = prepare_batch_display(predictions)

    assert display.iloc[0]["transaction_id"] == 2


def test_false_negative_explanation_appears_only_for_correct_demo_condition() -> None:
    assert demo_outcome_message(label=1, decision="ALLOW").startswith("Known model miss")
    assert demo_outcome_message(label=1, decision="REVIEW") is None
    assert demo_outcome_message(label=0, decision="ALLOW") is None


def test_demo_extra_column_warning_is_suppressed_from_main_display() -> None:
    warnings = [
        "Ignored 9 extra input columns not used by the model.",
        "Malformed numeric values were coerced to missing for columns: TransactionAmt",
    ]

    visible = main_warnings(warnings, suppress_extra_column_warning=True)

    assert len(visible) == 1
    assert "Malformed numeric" in visible[0]


def test_uploaded_data_warnings_remain_available() -> None:
    warnings = ["Ignored 2 extra input columns not used by the model."]

    assert main_warnings(warnings, suppress_extra_column_warning=False) == warnings


def test_shap_table_has_friendly_labels_and_rounded_contributions() -> None:
    table = format_shap_table(
        [{"feature": "C13", "value": 2.0, "shap_value": 0.12345}]
    )

    assert list(table.columns) == ["Feature", "Observed value", "Risk contribution"]
    assert table.iloc[0]["Risk contribution"] == 0.123


def test_missing_artifact_handling(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing artifact"):
        load_csv_artifact(tmp_path / "missing.csv")


def test_cost_scenario_parsing() -> None:
    cost_summary = {
        "recommended_candidate_thresholds": {
            "low": {
                "review_cost_per_transaction": 1.0,
                "fraud_loss_multiplier": 1.0,
                "minimum_cost_threshold": {"threshold": 0.08},
                "allow_all": {"total_estimated_cost": 100.0},
                "threshold_0_50": {"total_estimated_cost": 50.0},
                "simulated_cost_reduction_vs_allow_all": 75.0,
                "simulated_cost_reduction_vs_threshold_0_50": 25.0,
            }
        }
    }

    scenarios = parse_cost_scenarios(cost_summary)

    assert scenarios[0]["scenario"] == "low"
    assert scenarios[0]["candidate"]["threshold"] == 0.08


def test_ui_helpers_do_not_trigger_held_out_test_evaluation() -> None:
    tree = ast.parse(open("app.py", encoding="utf-8").read())
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert "fit" not in call_names
    assert "evaluate_binary_classifier" not in names
    assert "X_test" not in names
