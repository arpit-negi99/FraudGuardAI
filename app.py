from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import ArtifactLoadError, FraudPredictor, InferenceError


DEFAULT_THRESHOLD = 0.60
DEMO_TRANSACTIONS_PATH = Path("artifacts/demo/demo_transactions.csv")
DEMO_LABELS_PATH = Path("artifacts/demo/demo_labels.csv")
REQUIRED_DEPLOYMENT_ARTIFACTS = (
    Path("artifacts/models/xgboost_model.json"),
    Path("artifacts/preprocessors/preprocessor.joblib"),
    Path("artifacts/preprocessors/preprocessing_metadata.json"),
    Path("artifacts/results/xgboost_validation_metrics.json"),
    Path("artifacts/results/model_comparison.json"),
    Path("artifacts/results/xgboost_threshold_analysis.csv"),
    Path("artifacts/results/xgboost_threshold_summary.json"),
    Path("artifacts/results/xgboost_cost_summary.json"),
    Path("artifacts/results/shap_global_importance.csv"),
    DEMO_TRANSACTIONS_PATH,
    DEMO_LABELS_PATH,
)
DEMO_EXAMPLE_IDS = {
    "High-risk true positive": 3481071,
    "False positive": 3456622,
    "False negative": 3481470,
    "Low-risk legitimate transaction": 3458851,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e5e7eb;
        }
        .fg-hero {
            border-radius: 8px;
            padding: 2rem 2.2rem;
            margin-bottom: 1.2rem;
            color: #f8fafc;
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(19, 78, 74, 0.94)),
                linear-gradient(90deg, rgba(20, 184, 166, 0.24), rgba(245, 158, 11, 0.12));
            border: 1px solid rgba(148, 163, 184, 0.24);
        }
        .fg-kicker {
            color: #99f6e4;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .fg-hero h1 {
            margin: 0 0 0.2rem 0;
            font-size: 2.45rem;
            line-height: 1.08;
        }
        .fg-hero h2 {
            margin: 0 0 0.9rem 0;
            color: #ccfbf1;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .fg-hero p {
            color: #dbeafe;
            max-width: 780px;
            font-size: 1rem;
            margin: 0;
        }
        .fg-card {
            border: 1px solid #d7dee8;
            border-left: 5px solid #0f766e;
            border-radius: 8px;
            padding: 1rem 1rem 0.9rem;
            background: #ffffff;
            min-height: 116px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
        }
        .fg-card.secondary {
            border-left-color: #475569;
        }
        .fg-card.warn {
            border-left-color: #d97706;
            background: #fffbeb;
        }
        .fg-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .fg-value {
            color: #0f172a;
            font-size: 1.85rem;
            line-height: 1.1;
            font-weight: 750;
        }
        .fg-note {
            color: #475569;
            font-size: 0.86rem;
            margin-top: 0.45rem;
        }
        .fg-banner {
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem;
            border: 1px solid;
            font-weight: 650;
        }
        .fg-allow {
            background: #ecfdf5;
            border-color: #86efac;
            color: #14532d;
        }
        .fg-review {
            background: #fffbeb;
            border-color: #fbbf24;
            color: #78350f;
        }
        .fg-panel {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            background: #f8fafc;
            margin: 0.7rem 0 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="fg-hero">
            <div class="fg-kicker">Validation demo · decision support</div>
            <h1>FraudGuard AI</h1>
            <h2>Cost-aware merchant fraud risk intelligence</h2>
            <p>
                Score transaction risk, prioritize suspicious activity for review,
                explain model signals, and understand the trade-off between fraud
                capture and false-positive cost.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    note_html = f'<div class="fg-note">{note}</div>' if note else ""
    return (
        f'<div class="fg-card{tone_class}">'
        f'<div class="fg-label">{label}</div>'
        f'<div class="fg-value">{value}</div>'
        f"{note_html}</div>"
    )


def render_cards(cards: list[dict[str, str]]) -> None:
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.markdown(
            metric_card(
                card["label"],
                card["value"],
                card.get("note", ""),
                card.get("tone", ""),
            ),
            unsafe_allow_html=True,
        )


def render_decision_banner(decision: str) -> None:
    if decision == "REVIEW":
        message = (
            "REVIEW - transaction should be sent for analyst review. "
            "FraudGuard does not automatically block the payment."
        )
        css_class = "fg-banner fg-review"
    else:
        message = "ALLOW - model risk score is below the selected review threshold."
        css_class = "fg-banner fg-allow"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")
    with artifact_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_csv_artifact(path: str | Path) -> pd.DataFrame:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact_path}")
    return pd.read_csv(artifact_path)


def missing_deployment_artifacts(root: Path = ROOT) -> list[Path]:
    """Return required deployment artifacts that are absent from the local package."""
    return [path for path in REQUIRED_DEPLOYMENT_ARTIFACTS if not (root / path).exists()]


def render_startup_health_check() -> bool:
    missing = missing_deployment_artifacts()
    if not missing:
        return True
    st.error("FraudGuard cannot start because required frozen artifacts are missing.")
    st.code("\n".join(str(path) for path in missing), language="text")
    st.info("Restore the frozen artifacts before launching the demo. The app will not retrain automatically.")
    return False


def nearest_threshold_metrics(threshold_table: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if threshold_table.empty:
        raise ValueError("Threshold table is empty.")
    index = (threshold_table["threshold"] - threshold).abs().idxmin()
    return threshold_table.loc[index].to_dict()


def sorted_shap_importance(shap_importance: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top global SHAP features sorted from highest to lowest impact."""
    return (
        shap_importance.sort_values("mean_absolute_shap_value", ascending=False, kind="mergesort")
        .head(top_n)
        .rename(columns={"mean_absolute_shap_value": "Average model impact"})
    )


def build_policy_presets(threshold_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "Higher Fraud Capture",
            "description": "More fraud capture, higher review load.",
            "metrics": threshold_summary["highest_precision_recall_at_least_0_70"],
        },
        {
            "name": "Capacity-Constrained",
            "description": "Keeps validation review rate near 5%.",
            "metrics": threshold_summary["lowest_review_rate_recall_at_least_0_60"],
        },
        {
            "name": "Highest F1",
            "description": "Balances validation precision and recall.",
            "metrics": threshold_summary["highest_f1"],
        },
    ]


def parse_cost_scenarios(cost_summary: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = cost_summary.get("recommended_candidate_thresholds", {})
    rows: list[dict[str, Any]] = []
    for scenario_name, details in scenarios.items():
        candidate = details["minimum_cost_threshold"]
        rows.append(
            {
                "scenario": scenario_name,
                "review_cost_per_false_positive": details["review_cost_per_transaction"],
                "fraud_loss_multiplier": details["fraud_loss_multiplier"],
                "candidate": candidate,
                "allow_all": details["allow_all"],
                "threshold_0_50": details["threshold_0_50"],
                "modeled_cost_reduction_vs_allow_all": details[
                    "simulated_cost_reduction_vs_allow_all"
                ],
                "modeled_cost_reduction_vs_threshold_0_50": details[
                    "simulated_cost_reduction_vs_threshold_0_50"
                ],
            }
        )
    return rows


def predictions_to_download_csv(predictions: pd.DataFrame) -> bytes:
    columns = [
        column
        for column in ("transaction_id", "risk_score", "threshold", "decision", "risk_band")
        if column in predictions.columns
    ]
    return predictions[columns].to_csv(index=False).encode("utf-8")


def sample_batch_csv(validation_examples: pd.DataFrame) -> bytes:
    """Create a small real validation sample for batch-demo upload, excluding labels."""
    sample = validation_examples.drop(columns=["isFraud", "demo_case"], errors="ignore").copy()
    return sample.to_csv(index=False).encode("utf-8")


def prepare_batch_display(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in ("transaction_id", "risk_score", "threshold", "decision", "risk_band")
        if column in predictions.columns
    ]
    display = predictions[columns].copy()
    if "risk_score" in display.columns:
        display = display.sort_values("risk_score", ascending=False, kind="mergesort")
    return display


def format_shap_table(contributors: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(contributors).head(5)
    if table.empty:
        return pd.DataFrame(columns=["Feature", "Observed value", "Risk contribution"])
    table = table.rename(
        columns={
            "feature": "Feature",
            "value": "Observed value",
            "shap_value": "Risk contribution",
        }
    )
    if "Risk contribution" in table.columns:
        table["Risk contribution"] = pd.to_numeric(
            table["Risk contribution"], errors="coerce"
        ).round(3)
    return table[["Feature", "Observed value", "Risk contribution"]]


def main_warnings(warnings: list[str], suppress_extra_column_warning: bool) -> list[str]:
    if not suppress_extra_column_warning:
        return warnings
    return [warning for warning in warnings if "extra input columns" not in warning]


def demo_outcome_message(label: int, decision: str) -> str | None:
    if label == 1 and decision == "ALLOW":
        return (
            "Known model miss: the model assigned a low risk score, but the historical label "
            "is fraud. This demonstrates why FraudGuard is decision support rather than an "
            "autonomous blocker."
        )
    if label == 0 and decision == "REVIEW":
        return (
            "Known false positive: the model recommended review, but the historical label is "
            "legitimate. High model risk does not guarantee actual fraud."
        )
    return None


@st.cache_resource(show_spinner=False)
def get_predictor() -> FraudPredictor:
    return FraudPredictor()


@st.cache_data(show_spinner=False)
def get_json_artifact(path: str) -> dict[str, Any]:
    return load_json_artifact(ROOT / path)


@st.cache_data(show_spinner=False)
def get_csv_artifact(path: str) -> pd.DataFrame:
    return load_csv_artifact(ROOT / path)


@st.cache_data(show_spinner=False)
def get_validation_examples() -> pd.DataFrame:
    transactions = load_csv_artifact(ROOT / DEMO_TRANSACTIONS_PATH)
    labels = load_csv_artifact(ROOT / DEMO_LABELS_PATH)
    examples = transactions.merge(labels, how="inner", on="TransactionID", validate="one_to_one")
    examples = examples[examples["demo_case"].notna()].copy()
    examples = examples.sort_values("demo_case", kind="mergesort")
    return examples


@st.cache_data(show_spinner=False)
def get_batch_sample() -> pd.DataFrame:
    return load_csv_artifact(ROOT / DEMO_TRANSACTIONS_PATH).head(10).copy()


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def render_metric_row(metrics: dict[str, Any], pr_auc: float) -> None:
    render_cards(
        [
            {
                "label": "Precision",
                "value": format_percent(metrics["precision"]),
                "note": "At demo policy threshold 0.60",
            },
            {
                "label": "Recall",
                "value": format_percent(metrics["recall"]),
                "note": "Validation fraud captured",
            },
            {
                "label": "Review Rate",
                "value": format_percent(metrics["review_rate"]),
                "note": "Approximate review load",
            },
            {
                "label": "PR-AUC",
                "value": format_metric(pr_auc),
                "note": "XGBoost validation ranking",
            },
        ]
    )


def render_overview() -> None:
    render_hero()
    st.caption("Validation performance. These are not final production metrics.")

    try:
        metrics = get_json_artifact("artifacts/results/xgboost_validation_metrics.json")
        comparison = get_json_artifact("artifacts/results/model_comparison.json")
        threshold_table = get_csv_artifact("artifacts/results/xgboost_threshold_analysis.csv")
        shap_importance = get_csv_artifact("artifacts/results/shap_global_importance.csv")
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    threshold_metrics = nearest_threshold_metrics(threshold_table, DEFAULT_THRESHOLD)
    render_metric_row(threshold_metrics, metrics["pr_auc"])

    st.subheader("Model improvement")
    st.caption("Logistic Regression -> XGBoost")
    logistic = comparison["logistic_regression"]
    xgboost = comparison["xgboost"]
    rows = []
    for model_name in ("logistic_regression", "xgboost"):
        model_metrics = comparison[model_name]
        rows.append(
            {
                "Model": "Logistic Regression" if model_name == "logistic_regression" else "XGBoost",
                "Precision": format_percent(model_metrics["precision"]),
                "Recall": format_percent(model_metrics["recall"]),
                "F1": format_percent(model_metrics["f1"]),
                "PR-AUC": format_metric(model_metrics["pr_auc"]),
                "Review rate": format_percent(model_metrics["review_rate"]),
            }
        )
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    comparison_chart = pd.DataFrame(
        [
            {
                "metric": "Precision",
                "Logistic Regression": logistic["precision"],
                "XGBoost": xgboost["precision"],
            },
            {
                "metric": "Recall",
                "Logistic Regression": logistic["recall"],
                "XGBoost": xgboost["recall"],
            },
            {
                "metric": "F1",
                "Logistic Regression": logistic["f1"],
                "XGBoost": xgboost["f1"],
            },
            {
                "metric": "Review rate",
                "Logistic Regression": logistic["review_rate"],
                "XGBoost": xgboost["review_rate"],
            },
        ]
    ).set_index("metric")
    st.bar_chart(comparison_chart)

    st.info(
        "XGBoost improved validation precision by "
        f"{format_percent(xgboost['precision'] - logistic['precision'])}, "
        "approximately preserved recall, reduced review burden by "
        f"{format_percent(logistic['review_rate'] - xgboost['review_rate'])}, "
        "and improved PR-AUC by "
        f"{format_metric(xgboost['pr_auc'] - logistic['pr_auc'])}."
    )

    majority = comparison.get("majority_baseline", {})
    st.warning(
        "A majority-class model reaches "
        f"{format_percent(majority.get('accuracy', 0.0))} accuracy while detecting "
        f"{format_percent(majority.get('recall', 0.0))} of fraud, which is why FraudGuard "
        "prioritizes precision, recall and PR-AUC over accuracy."
    )

    st.subheader("Global Model Signals")
    top_features = sorted_shap_importance(shap_importance)
    st.bar_chart(
        top_features,
        x="feature",
        y="Average model impact",
        horizontal=True,
    )
    st.caption("SHAP values describe model attribution, not causal proof.")


def render_transaction_inspector() -> None:
    st.header("Transaction Inspector")
    st.caption("Inspect historical transactions and understand the model decision.")

    try:
        examples = get_validation_examples()
        predictor = get_predictor()
    except (FileNotFoundError, ArtifactLoadError, InferenceError) as exc:
        st.error(f"Unable to initialize transaction inspector: {exc}")
        return

    if examples.empty:
        st.error("No validation demo examples were found.")
        return

    options = {
        f"{row.demo_case} - TransactionID {int(row.TransactionID)}": index
        for index, row in examples.iterrows()
    }
    selection = st.selectbox("Demo transaction", list(options))
    selected = examples.loc[[options[selection]]]
    inference_input = selected.drop(columns=["isFraud", "demo_case"], errors="ignore")

    with st.spinner("Scoring transaction and calculating local SHAP explanation..."):
        try:
            result = predictor.predict_transaction(inference_input, include_explanation=True)
        except InferenceError as exc:
            st.error(f"Prediction failed: {exc}")
            return

    score = float(result["risk_score"])
    st.subheader("Model Output")
    render_cards(
        [
            {
                "label": "Model risk score",
                "value": format_percent(score),
                "note": "Not probability-calibrated",
                "tone": "warn" if result["decision"] == "REVIEW" else "",
            },
            {
                "label": "Policy threshold",
                "value": f"{result['threshold']:.2f}",
                "note": "Validation-derived demo policy",
                "tone": "secondary",
            },
            {
                "label": "Decision",
                "value": result["decision"],
                "note": "Advisory recommendation",
                "tone": "warn" if result["decision"] == "REVIEW" else "",
            },
        ]
    )
    st.progress(min(max(score, 0.0), 1.0))
    render_decision_banner(result["decision"])

    label = int(selected.iloc[0]["isFraud"])
    outcome_message = demo_outcome_message(label, result["decision"])
    if outcome_message:
        st.info(outcome_message)

    st.caption(
        "Offline evaluation label: "
        + ("Fraud" if label == 1 else "Legitimate")
        + ". This label exists only because this is historical validation data."
    )

    warnings = result.get("warnings", [])
    visible_warnings = main_warnings(warnings, suppress_extra_column_warning=True)
    if visible_warnings:
        st.warning(" ".join(visible_warnings))
    if warnings:
        with st.expander("Technical details"):
            for warning in warnings:
                st.write(warning)

    explanation = result.get("explanation", {})
    left, right = st.columns(2)
    with left:
        st.subheader("Top factors increasing model risk")
        st.dataframe(
            format_shap_table(explanation.get("top_risk_factors", [])),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.subheader("Top factors reducing model risk")
        st.dataframe(
            format_shap_table(explanation.get("top_protective_factors", [])),
            width="stretch",
            hide_index=True,
        )


def render_batch_analysis() -> None:
    st.header("Batch Analysis")
    st.caption("Upload a CSV with transaction fields. Ground-truth labels are not required.")
    st.warning("Do not upload full card numbers, CVV, PIN, OTP, passwords, or bank-login credentials.")

    try:
        sample_df = get_batch_sample()
        st.markdown(
            '<div class="fg-panel"><b>Fast judge flow:</b> download the sample CSV, '
            "upload it here, score the batch, then download the scored results.</div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download sample CSV",
            data=sample_batch_csv(sample_df),
            file_name="fraudguard_sample_transactions.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.info(f"Sample CSV is unavailable: {exc}")

    uploaded = st.file_uploader("Transaction CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to score a batch. Results will include risk score and ALLOW/REVIEW.")
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception:
        st.error("The uploaded file could not be read as a CSV.")
        return

    if batch_df.empty:
        st.error("The uploaded CSV is empty.")
        return

    try:
        predictor = get_predictor()
        with st.spinner(f"Validating and scoring {len(batch_df)} transactions..."):
            predictions = predictor.predict_batch(batch_df)
            summary = predictor.summarize_batch(predictions)
    except (ArtifactLoadError, InferenceError, ValueError) as exc:
        st.error(f"Batch scoring failed: {exc}")
        return

    render_cards(
        [
            {"label": "Transactions scored", "value": f"{summary['transactions_scored']:,}"},
            {"label": "REVIEW", "value": f"{summary['review_count']:,}", "tone": "warn"},
            {"label": "ALLOW", "value": f"{summary['allow_count']:,}"},
            {"label": "Review rate", "value": format_percent(summary["review_rate"])},
        ]
    )

    render_cards(
        [
            {
                "label": "Average risk",
                "value": format_percent(summary["average_risk_score"]),
                "tone": "secondary",
            },
            {
                "label": "Median risk",
                "value": format_percent(summary["median_risk_score"]),
                "tone": "secondary",
            },
            {
                "label": "Maximum risk",
                "value": format_percent(summary["maximum_risk_score"]),
                "tone": "secondary",
            },
        ]
    )

    display = prepare_batch_display(predictions)
    st.dataframe(display, width="stretch", hide_index=True)
    warning_values = []
    if "warnings" in predictions.columns:
        for warnings in predictions["warnings"]:
            if isinstance(warnings, list):
                warning_values.extend(warnings)
    if warning_values:
        with st.expander("Schema and preprocessing warnings"):
            for warning in sorted(set(warning_values)):
                st.write(warning)
    st.download_button(
        "Download scored CSV",
        data=predictions_to_download_csv(predictions),
        file_name="fraudguard_scored_batch.csv",
        mime="text/csv",
    )


def render_policy_lab() -> None:
    st.header("Risk Policy Lab")
    st.caption("Validation simulation. These metrics are not live production outcomes.")

    try:
        threshold_table = get_csv_artifact("artifacts/results/xgboost_threshold_analysis.csv")
        threshold_summary = get_json_artifact("artifacts/results/xgboost_threshold_summary.json")
        cost_summary = get_json_artifact("artifacts/results/xgboost_cost_summary.json")
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    presets = build_policy_presets(threshold_summary)
    preset_options = {preset["name"]: preset for preset in presets}
    selected_preset_name = st.radio(
        "Policy preset",
        list(preset_options),
        horizontal=True,
    )
    preset_threshold = float(preset_options[selected_preset_name]["metrics"]["threshold"])
    threshold = st.slider("Review threshold", 0.10, 0.90, preset_threshold, 0.01)
    nearest = nearest_threshold_metrics(threshold_table, threshold)
    render_cards(
        [
            {"label": "Threshold", "value": f"{nearest['threshold']:.2f}", "tone": "secondary"},
            {"label": "Precision", "value": format_percent(nearest["precision"])},
            {"label": "Recall", "value": format_percent(nearest["recall"])},
            {"label": "F1", "value": format_percent(nearest["f1"])},
            {"label": "Review rate", "value": format_percent(nearest["review_rate"])},
        ]
    )
    render_cards(
        [
            {
                "label": "False positives",
                "value": f"{int(nearest['false_positive']):,}",
                "note": "Legitimate transactions reviewed",
                "tone": "warn",
            },
            {
                "label": "False negatives",
                "value": f"{int(nearest['false_negative']):,}",
                "note": "Fraud cases missed",
                "tone": "warn",
            },
        ]
    )
    st.markdown(
        '<div class="fg-panel">'
        f"At this policy, about {format_percent(nearest['review_rate'])} of transactions "
        f"are sent for review while approximately {format_percent(nearest['recall'])} of "
        "fraud cases are captured on the validation set. Legitimate transactions unnecessarily "
        f"reviewed: {int(nearest['false_positive']):,}. Fraud cases missed: "
        f"{int(nearest['false_negative']):,}.</div>",
        unsafe_allow_html=True,
    )

    chart_df = threshold_table[["threshold", "precision", "recall", "review_rate"]].set_index(
        "threshold"
    )
    st.line_chart(chart_df)

    st.subheader("Validation-Derived Policy Presets")
    preset_cols = st.columns(3)
    for column, preset in zip(preset_cols, presets):
        metrics = preset["metrics"]
        with column:
            st.markdown(
                metric_card(
                    preset["name"],
                    f"{metrics['threshold']:.2f}",
                    (
                        f"{preset['description']}<br>"
                        f"Precision {format_percent(metrics['precision'])} | "
                        f"Recall {format_percent(metrics['recall'])} | "
                        f"Review {format_percent(metrics['review_rate'])}"
                    ),
                    "secondary",
                ),
                unsafe_allow_html=True,
            )

    st.subheader("Cost-Aware Scenario Comparison")
    st.info(
        "Pure cost minimization can recommend reviewing too many transactions. FraudGuard "
        "therefore combines modeled cost with operational constraints such as review capacity "
        "and minimum fraud recall."
    )
    assumptions = cost_summary["business_assumptions"]
    st.caption(
        "Fraud loss multiplier = "
        f"{assumptions['fraud_loss_multiplier']:.1f}. Review cost per false positive: "
        "low = 1, medium = 5, high = 10 modeled cost units."
    )

    cost_rows = []
    for scenario in parse_cost_scenarios(cost_summary):
        candidate = scenario["candidate"]
        allow_all = scenario["allow_all"]
        threshold_050 = scenario["threshold_0_50"]
        cost_rows.append(
            {
                "Scenario": scenario["scenario"],
                "Review cost assumption": scenario["review_cost_per_false_positive"],
                "Fraud-loss multiplier": scenario["fraud_loss_multiplier"],
                "Candidate threshold": candidate["threshold"],
                "Precision": format_percent(candidate["precision"]),
                "Recall": format_percent(candidate["recall"]),
                "Review rate": format_percent(candidate["review_rate"]),
                "False positives": f"{int(candidate['false_positive']):,}",
                "False negatives": f"{int(candidate['false_negative']):,}",
                "False-positive cost (modeled units)": f"{candidate['false_positive_cost']:,.2f}",
                "Missed-fraud cost (modeled units)": f"{candidate['missed_fraud_cost']:,.2f}",
                "Total modeled cost": f"{candidate['total_estimated_cost']:,.2f}",
                "Allow Everything cost": f"{allow_all['total_estimated_cost']:,.2f}",
                "Threshold 0.50 cost": f"{threshold_050['total_estimated_cost']:,.2f}",
                "Modeled cost reduction vs allow all": (
                    f"{scenario['modeled_cost_reduction_vs_allow_all']:,.2f}"
                ),
            }
        )
    st.dataframe(pd.DataFrame(cost_rows), width="stretch", hide_index=True)
    st.caption("All cost figures are modeled cost units, not actual merchant savings.")


def render_methodology() -> None:
    st.header("Model & Methodology")
    render_cards(
        [
            {"label": "Dataset", "value": "IEEE-CIS", "note": "Fraud Detection benchmark"},
            {"label": "Primary model", "value": "XGBoost", "note": "Frozen local artifact"},
            {"label": "Baseline", "value": "Logistic", "note": "Regression comparison"},
        ]
    )
    render_cards(
        [
            {
                "label": "Split",
                "value": "70/15/15",
                "note": "Chronological train, validation, held-out test",
                "tone": "secondary",
            },
            {
                "label": "Features",
                "value": "422",
                "note": "Transformed model inputs",
                "tone": "secondary",
            },
            {
                "label": "Decision",
                "value": "ALLOW/REVIEW",
                "note": "No automatic payment blocking",
                "tone": "secondary",
            },
        ]
    )
    st.warning(
        "Final held-out metrics are reported from the frozen 0.60 policy. Cost values are "
        "scenario assumptions, many dataset fields are anonymized, and FraudGuard is decision "
        "support rather than an automatic payment blocker."
    )


def main() -> None:
    st.set_page_config(page_title="FraudGuard AI", page_icon=None, layout="wide")
    inject_styles()
    if not render_startup_health_check():
        st.stop()
    section = st.sidebar.radio(
        "Navigation",
        [
            "Risk Overview",
            "Transaction Inspector",
            "Batch Analysis",
            "Risk Policy Lab",
            "Model & Methodology",
        ],
    )
    st.sidebar.caption("Demo policy: threshold 0.60")
    st.sidebar.caption("Validation-selected | ~5% review capacity")

    if section == "Risk Overview":
        render_overview()
    elif section == "Transaction Inspector":
        render_transaction_inspector()
    elif section == "Batch Analysis":
        render_batch_analysis()
    elif section == "Risk Policy Lab":
        render_policy_lab()
    else:
        render_methodology()


if __name__ == "__main__":
    main()
