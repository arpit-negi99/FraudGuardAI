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
from src.inference.presentation import (
    DEFAULT_THRESHOLD,
    build_policy_presets,
    demo_outcome_message,
    enrich_predictions,
    filter_review_queue,
    historical_outcome,
    nearest_threshold_metrics,
    parse_cost_scenarios,
    priority_band,
    review_queue,
    risk_distribution,
    risk_status,
)


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
    Path("artifacts/results/final_test_metrics.json"),
    DEMO_TRANSACTIONS_PATH,
    DEMO_LABELS_PATH,
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        section[data-testid="stSidebar"] {
            background: #111827;
        }
        section[data-testid="stSidebar"] * {
            color: #f3f4f6;
        }
        .fg-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.1rem;
            padding-bottom: 0.85rem;
        }
        .fg-header h1 {
            color: #111827;
            font-size: 1.75rem;
            margin: 0;
        }
        .fg-header h2 {
            color: #0f766e;
            font-size: 1rem;
            margin: 0.2rem 0 0;
            font-weight: 650;
        }
        .fg-header p {
            color: #4b5563;
            margin: 0.35rem 0 0;
            max-width: 760px;
        }
        .fg-card {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-left: 4px solid #0f766e;
            border-radius: 8px;
            min-height: 108px;
            padding: 0.95rem 1rem;
            box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
        }
        .fg-card.warn {
            border-left-color: #d97706;
            background: #fffbeb;
        }
        .fg-card.danger {
            border-left-color: #dc2626;
            background: #fef2f2;
        }
        .fg-card.neutral {
            border-left-color: #6b7280;
        }
        .fg-label {
            color: #64748b;
            font-size: 0.76rem;
            font-weight: 750;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .fg-value {
            color: #111827;
            font-size: 1.62rem;
            line-height: 1.1;
            font-weight: 780;
        }
        .fg-note {
            color: #475569;
            font-size: 0.84rem;
            margin-top: 0.4rem;
        }
        .fg-panel {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.75rem 0 1rem;
        }
        .fg-decision {
            border-radius: 8px;
            padding: 1rem;
            margin: 0.8rem 0;
            border: 1px solid;
            font-weight: 650;
        }
        .fg-review {
            background: #fffbeb;
            border-color: #f59e0b;
            color: #78350f;
        }
        .fg-allow {
            background: #ecfdf5;
            border-color: #22c55e;
            color: #14532d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_global_header() -> None:
    st.markdown(
        """
        <div class="fg-header">
            <h1>FraudGuard AI</h1>
            <h2>Merchant fraud risk & review assistant</h2>
            <p>Prioritize suspicious transactions, understand model signals, and manage review workload.</p>
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


def sorted_shap_importance(shap_importance: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top global SHAP features sorted from highest to lowest impact."""
    return (
        shap_importance.sort_values("mean_absolute_shap_value", ascending=False, kind="mergesort")
        .head(top_n)
        .rename(columns={"mean_absolute_shap_value": "Average model impact"})
    )


def predictions_to_download_csv(predictions: pd.DataFrame) -> bytes:
    columns = [
        column
        for column in (
            "transaction_id",
            "risk_score",
            "threshold",
            "decision",
            "risk_band",
            "priority",
            "transaction_amount",
        )
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


def format_signal_table(contributors: list[dict[str, Any]], impact: str) -> pd.DataFrame:
    table = pd.DataFrame(contributors).head(5)
    if table.empty:
        return pd.DataFrame(columns=["Signal", "Observed value", "Impact"])
    table = table.rename(columns={"feature": "Signal", "value": "Observed value"})
    table["Impact"] = impact
    return table[["Signal", "Observed value", "Impact"]]


def format_shap_table(contributors: list[dict[str, Any]]) -> pd.DataFrame:
    return format_signal_table(contributors, "Increases risk")


def main_warnings(warnings: list[str], suppress_extra_column_warning: bool) -> list[str]:
    if not suppress_extra_column_warning:
        return warnings
    return [warning for warning in warnings if "extra input columns" not in warning]


def display_queue_table(queue: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "transaction_id": "Transaction ID",
        "risk_score": "Risk",
        "transaction_amount": "Transaction amount",
        "amount": "Transaction amount",
        "priority": "Priority",
        "decision": "Decision",
    }
    available = [column for column in columns if column in queue.columns]
    table = queue[available].rename(columns=columns)
    if "Risk" in table.columns:
        table["Risk"] = table["Risk"].map(format_percent)
    if "Transaction amount" in table.columns:
        table["Transaction amount"] = table["Transaction amount"].map(format_amount)
    return table


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_amount(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):,.2f}"


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
def get_demo_transactions() -> pd.DataFrame:
    return load_csv_artifact(ROOT / DEMO_TRANSACTIONS_PATH)


@st.cache_data(show_spinner=False)
def get_demo_labels() -> pd.DataFrame:
    return load_csv_artifact(ROOT / DEMO_LABELS_PATH)


@st.cache_data(show_spinner=False)
def get_validation_examples() -> pd.DataFrame:
    transactions = get_demo_transactions()
    labels = get_demo_labels()
    examples = transactions.merge(labels, how="inner", on="TransactionID", validate="one_to_one")
    examples = examples[examples["demo_case"].notna()].copy()
    examples = examples.sort_values("demo_case", kind="mergesort")
    return examples


@st.cache_data(show_spinner=False)
def get_demo_transactions_with_labels() -> pd.DataFrame:
    transactions = get_demo_transactions()
    labels = get_demo_labels()
    examples = transactions.merge(labels, how="left", on="TransactionID", validate="one_to_one")
    return examples


@st.cache_data(show_spinner=False)
def get_batch_sample() -> pd.DataFrame:
    return get_demo_transactions().head(10).copy()


def get_current_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = get_demo_transactions()
    predictor = get_predictor()
    predictions = predictor.predict_batch(transactions)
    enriched = enrich_predictions(predictions, transactions)
    return transactions, enriched


def select_transaction(transaction_id: Any) -> None:
    st.session_state["selected_transaction_id"] = int(transaction_id)


def render_decision_message(decision: str) -> None:
    if decision == "REVIEW":
        message = "Recommended action: REVIEW. A human analyst should inspect this transaction before further action."
        css_class = "fg-decision fg-review"
    else:
        message = "Recommended action: ALLOW. No manual review is recommended under the current policy."
        css_class = "fg-decision fg-allow"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)
    st.caption("FraudGuard does not automatically block transactions.")


def render_home() -> None:
    st.subheader("Today at a glance")
    st.caption("Demo values from the packaged transaction sample.")
    _, predictions = get_current_predictions()
    queue = review_queue(predictions)
    highest_risk = float(predictions["risk_score"].max())
    review_rate = float((predictions["decision"] == "REVIEW").mean())
    status = risk_status(review_rate, highest_risk)

    render_cards(
        [
            {"label": "Needs Review", "value": f"{len(queue):,}", "note": "Current demo transactions", "tone": "warn"},
            {"label": "Highest Risk", "value": format_percent(highest_risk), "note": "Top scored transaction", "tone": "danger" if highest_risk >= 0.90 else "warn"},
            {"label": "Review Workload", "value": format_percent(review_rate), "note": "Share requiring review"},
            {"label": "Risk Status", "value": status, "note": "Derived from demo sample", "tone": "danger" if status == "High" else "warn" if status == "Elevated" else ""},
        ]
    )

    st.subheader("Priority transactions")
    priority = predictions.sort_values("risk_score", ascending=False, kind="mergesort").head(5)
    table = display_queue_table(priority)
    table["Action"] = ["Review" if value == "REVIEW" else "Allow" for value in priority["decision"]]
    st.dataframe(table, width="stretch", hide_index=True)

    selected_id = st.selectbox(
        "Open transaction",
        priority["transaction_id"].tolist(),
        format_func=lambda value: f"Transaction {int(value)}",
    )
    if st.button("Open in Transaction Details"):
        select_transaction(selected_id)
        st.success("Transaction selected. Open Transaction Details from the sidebar.")

    st.subheader("How FraudGuard helps")
    render_cards(
        [
            {"label": "1", "value": "Scores risk", "note": "Higher scores indicate stronger model concern."},
            {"label": "2", "value": "Prioritizes", "note": "Suspicious transactions move to the review queue."},
            {"label": "3", "value": "Explains", "note": "Analysts see the model signals behind the score."},
        ]
    )


def render_review_queue() -> None:
    st.subheader("Review Queue")
    st.caption("Transactions recommended for human review, sorted by highest risk first.")
    _, predictions = get_current_predictions()
    queue = review_queue(predictions)

    uploaded = st.file_uploader("Upload transactions", type=["csv"])
    sample = get_batch_sample()
    st.download_button(
        "Download sample transactions",
        data=sample_batch_csv(sample),
        file_name="fraudguard_sample_transactions.csv",
        mime="text/csv",
    )
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            predictor = get_predictor()
            uploaded_predictions = predictor.predict_batch(uploaded_df)
            predictions = enrich_predictions(uploaded_predictions, uploaded_df)
            queue = review_queue(predictions)
            st.success(f"Scored {len(uploaded_df):,} uploaded transactions.")
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")
            return
    elif queue.empty:
        st.info("Upload a transaction CSV or use the sample file to see FraudGuard in action.")
        return

    priority_filter = st.radio("Priority", ["All", "Critical", "High", "Review"], horizontal=True)
    minimum_risk = st.slider("Minimum risk", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05)
    filtered = filter_review_queue(queue, priority_filter, minimum_risk)

    render_cards(
        [
            {"label": "Transactions analyzed", "value": f"{len(predictions):,}"},
            {"label": "Needs review", "value": f"{len(queue):,}", "tone": "warn"},
            {"label": "Visible after filters", "value": f"{len(filtered):,}"},
        ]
    )

    if filtered.empty:
        st.info("No transactions match the selected review filters.")
        return

    st.dataframe(display_queue_table(filtered), width="stretch", hide_index=True)
    selected_id = st.selectbox(
        "Inspect transaction",
        filtered["transaction_id"].tolist(),
        format_func=lambda value: f"Transaction {int(value)}",
    )
    if st.button("Open selected transaction"):
        select_transaction(selected_id)
        st.success("Transaction selected. Open Transaction Details from the sidebar.")

    st.download_button(
        "Download scored results",
        data=predictions_to_download_csv(predictions),
        file_name="fraudguard_scored_results.csv",
        mime="text/csv",
    )


def render_transaction_details() -> None:
    st.subheader("Transaction Details")
    st.caption("Understand the risk decision before taking action.")
    examples = get_demo_transactions_with_labels()
    predictor = get_predictor()

    default_id = st.session_state.get("selected_transaction_id")
    ids = examples["TransactionID"].astype(int).tolist()
    default_index = ids.index(default_id) if default_id in ids else 0
    selected_id = st.selectbox(
        "Transaction",
        ids,
        index=default_index,
        format_func=lambda value: f"Transaction {int(value)}",
    )
    select_transaction(selected_id)

    selected = examples[examples["TransactionID"] == selected_id].iloc[[0]]
    inference_input = selected.drop(columns=["isFraud", "demo_case"], errors="ignore")
    result = predictor.predict_transaction(inference_input, include_explanation=True)
    score = float(result["risk_score"])
    amount = selected.iloc[0].get("TransactionAmt")
    priority = priority_band(score)

    render_cards(
        [
            {"label": "Risk score", "value": format_percent(score), "note": "Higher means stronger concern", "tone": "danger" if score >= 0.90 else "warn" if result["decision"] == "REVIEW" else ""},
            {"label": "Recommended action", "value": result["decision"], "note": "Human-in-the-loop policy", "tone": "warn" if result["decision"] == "REVIEW" else ""},
            {"label": "Transaction amount", "value": format_amount(amount), "note": "From provided row"},
            {"label": "Priority", "value": priority, "note": "UI band only", "tone": "danger" if priority == "Critical" else "warn" if priority in {"High", "Review"} else ""},
        ]
    )
    st.caption("Risk level")
    st.progress(min(max(score, 0.0), 1.0))
    render_decision_message(result["decision"])

    explanation = result.get("explanation", {})
    left, right = st.columns(2)
    with left:
        st.markdown("#### Why FraudGuard flagged this")
        st.dataframe(
            format_signal_table(explanation.get("top_risk_factors", []), "Increases risk"),
            width="stretch",
            hide_index=True,
        )
    with right:
        st.markdown("#### Signals reducing concern")
        st.dataframe(
            format_signal_table(explanation.get("top_protective_factors", []), "Reduces risk"),
            width="stretch",
            hide_index=True,
        )
    st.caption("These are model-attribution signals and do not prove causation.")

    with st.expander("Technical details"):
        st.write("Explanation method: SHAP TreeExplainer over the frozen XGBoost model.")
        st.write(f"Threshold: {result['threshold']:.2f}")
        st.write(f"Model score: {score:.6f}")
        warnings = result.get("warnings", [])
        if warnings:
            st.write("Warnings:")
            for warning in warnings:
                st.write(f"- {warning}")

    if not pd.isna(selected.iloc[0].get("isFraud")):
        label = int(selected.iloc[0]["isFraud"])
        st.markdown("#### Historical outcome")
        render_cards(
            [
                {
                    "label": "Historical label",
                    "value": historical_outcome(label),
                    "note": "Shown only for packaged demo examples",
                    "tone": "warn" if label == 1 else "",
                }
            ]
        )
        outcome_message = demo_outcome_message(label, result["decision"])
        if outcome_message:
            st.info(outcome_message)

    st.markdown("#### Analyst decision")
    cols = st.columns(3)
    actions = ["Mark as suspicious", "Mark as legitimate", "Escalate"]
    for column, action in zip(cols, actions):
        if column.button(action):
            st.session_state.setdefault("analyst_actions", {})[int(selected_id)] = action
            st.success(f"Demo action recorded: {action}")
    st.caption("Demo action only - feedback storage is not enabled yet.")


def render_risk_monitor() -> None:
    st.subheader("Risk Monitor")
    st.caption("Current risk activity from the demo/batch sample. No live fraud-spike detector is enabled.")
    _, predictions = get_current_predictions()
    review_count = int((predictions["decision"] == "REVIEW").sum())
    review_rate = float(review_count / len(predictions))
    highest_risk = float(predictions["risk_score"].max())
    average_risk = float(predictions["risk_score"].mean())

    render_cards(
        [
            {"label": "Transactions analyzed", "value": f"{len(predictions):,}"},
            {"label": "Requiring review", "value": f"{review_count:,}", "tone": "warn"},
            {"label": "Review rate", "value": format_percent(review_rate)},
            {"label": "Average risk", "value": format_percent(average_risk)},
        ]
    )
    render_cards(
        [
            {"label": "Highest risk", "value": format_percent(highest_risk), "tone": "danger" if highest_risk >= 0.90 else "warn"},
            {"label": "Risk status", "value": risk_status(review_rate, highest_risk), "note": "Demo sample status"},
        ]
    )

    st.markdown("#### Risk distribution")
    distribution = risk_distribution(predictions)
    st.bar_chart(distribution, x="risk_band", y="transactions")
    st.caption("Bands are presentation labels only and do not alter the ALLOW / REVIEW policy.")

    st.markdown(
        """
        <div class="fg-panel">
            <b>Fraud Spike Monitoring</b><br>
            Coming next: identify unusual increases in high-risk transaction activity.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_policy_settings() -> None:
    st.subheader("Policy Settings")
    st.caption("Choose a review strategy. Technical values are available when needed.")

    threshold_table = get_csv_artifact("artifacts/results/xgboost_threshold_analysis.csv")
    threshold_summary = get_json_artifact("artifacts/results/xgboost_threshold_summary.json")
    cost_summary = get_json_artifact("artifacts/results/xgboost_cost_summary.json")
    presets = build_policy_presets(threshold_summary)
    preset_names = [preset["name"] for preset in presets]
    selected_name = st.radio("Review strategy", preset_names, index=1, horizontal=True)
    selected = next(preset for preset in presets if preset["name"] == selected_name)
    metrics = nearest_threshold_metrics(threshold_table, selected["threshold"])

    render_cards(
        [
            {"label": "Estimated review workload", "value": format_percent(metrics["review_rate"]), "note": "Validation simulation"},
            {"label": "Strategy", "value": selected["name"], "note": selected["description"]},
            {"label": "Trade-off", "value": selected["tradeoff"], "note": "Validation-derived preset"},
            {"label": "Threshold", "value": f"{selected['threshold']:.2f}", "note": "Balanced remains frozen at 0.60"},
        ]
    )

    with st.expander("View technical metrics"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "threshold": metrics["threshold"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "false_positives": int(metrics["false_positive"]),
                        "false_negatives": int(metrics["false_negative"]),
                        "review_rate": metrics["review_rate"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Cost controls")
    cost_scenarios = parse_cost_scenarios(cost_summary)
    names = [scenario["scenario"] for scenario in cost_scenarios]
    selected_scenario_name = st.selectbox("Cost of manual review", names, format_func=str.title)
    fraud_multiplier = st.number_input("Estimated fraud-loss multiplier", min_value=0.0, value=1.0, step=0.1)
    selected_scenario = next(item for item in cost_scenarios if item["scenario"] == selected_scenario_name)
    candidate = selected_scenario["candidate"]
    modeled_cost = candidate["total_estimated_cost"] * fraud_multiplier

    render_cards(
        [
            {"label": "Estimated modeled cost", "value": f"{modeled_cost:,.0f}", "note": "Simulation only", "tone": "warn"},
            {"label": "Suggested operating policy", "value": selected_name, "note": "Not actual merchant savings"},
        ]
    )
    with st.expander("View detailed calculation"):
        st.json(selected_scenario)


def render_about() -> None:
    st.subheader("About FraudGuard")
    st.markdown("#### What FraudGuard does")
    st.write(
        "FraudGuard scores merchant transactions, sends higher-risk cases to human review, "
        "and explains which model signals influenced each decision."
    )
    st.markdown("#### How it works")
    st.code("Transaction -> Risk Model -> Risk Score -> ALLOW / REVIEW -> Explanation", language="text")

    final_metrics = get_json_artifact("artifacts/results/final_test_metrics.json")
    metrics = final_metrics["metrics"]
    render_cards(
        [
            {"label": "Model", "value": "XGBoost", "note": "Frozen final model"},
            {"label": "Explainability", "value": "SHAP", "note": "Model-attribution signals"},
            {"label": "Threshold", "value": "0.60", "note": "Selected before held-out test"},
        ]
    )

    st.markdown("#### Evaluation")
    st.dataframe(
        pd.DataFrame(
            [
                {"Metric": "Precision", "Held-out test": metrics["precision"]},
                {"Metric": "Recall", "Held-out test": metrics["recall"]},
                {"Metric": "F1", "Held-out test": metrics["f1"]},
                {"Metric": "PR-AUC", "Held-out test": metrics["pr_auc"]},
                {"Metric": "ROC-AUC", "Held-out test": metrics["roc_auc"]},
                {"Metric": "Review rate", "Held-out test": metrics["review_rate"]},
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(
        "The held-out test was chronological. Threshold 0.60 was selected using validation "
        "analysis before the held-out test set was opened. Accuracy alone is misleading for "
        "rare fraud events."
    )

    with st.expander("Technical model comparison"):
        comparison = get_json_artifact("artifacts/results/model_comparison.json")
        st.json(comparison)

    st.markdown("#### Limitations")
    st.write(
        "- Decision support only; not an automatic payment blocker.\n"
        "- Many IEEE-CIS fields are anonymized.\n"
        "- Cost values are modeled assumptions, not observed merchant savings.\n"
        "- The dataset is historical and may not match current production fraud patterns.\n"
        "- Risk score is not guaranteed probability-calibrated.\n"
        "- Held-out test recall is 56.3%."
    )


def main() -> None:
    st.set_page_config(page_title="FraudGuard AI", page_icon=None, layout="wide")
    inject_styles()
    if not render_startup_health_check():
        st.stop()
    render_global_header()

    section = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Review Queue",
            "Transaction Details",
            "Risk Monitor",
            "Policy Settings",
            "About FraudGuard",
        ],
    )
    st.sidebar.caption("Frozen policy: threshold 0.60")
    st.sidebar.caption("ALLOW / REVIEW only")

    try:
        if section == "Home":
            render_home()
        elif section == "Review Queue":
            render_review_queue()
        elif section == "Transaction Details":
            render_transaction_details()
        elif section == "Risk Monitor":
            render_risk_monitor()
        elif section == "Policy Settings":
            render_policy_settings()
        else:
            render_about()
    except (FileNotFoundError, ArtifactLoadError, InferenceError) as exc:
        st.error(f"FraudGuard could not load the required frozen assets: {exc}")


if __name__ == "__main__":
    main()
