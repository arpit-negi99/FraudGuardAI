import { useState } from "react";
import { Badge } from "../components/Badge";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { formatAmount, formatPercent, formatScore } from "../utils/format";

function impactLabel(value) {
  const magnitude = Math.abs(Number(value || 0));
  if (magnitude >= 0.4) return "Strong model impact";
  if (magnitude >= 0.15) return "Moderate model impact";
  return "Light model impact";
}

export function TransactionDetails({ detail, loading, error, action, setAction }) {
  const [localAction, setLocalAction] = useState("");
  const choose = (value) => {
    setLocalAction(value);
    setAction?.(value);
  };

  return (
    <StateBlock loading={loading} error={error} empty={!detail}>
      <section className="case-heading">
        <div>
          <p>Case review</p>
          <h2>Transaction #{detail.transaction_id}</h2>
        </div>
        <Badge type="decision">{detail.decision}</Badge>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Risk Score" value={formatPercent(detail.risk_score)} tone={detail.decision === "REVIEW" ? "warn" : "default"} />
        <MetricCard label="Amount" value={formatAmount(detail.amount)} />
        <MetricCard label="Priority" value={detail.priority} tone={detail.priority === "Critical" ? "danger" : "warn"} />
        <MetricCard label="Current Policy" value="Balanced" note="Threshold 0.60" />
      </section>
      <div className="risk-bar">
        <span style={{ width: `${Math.min(100, Math.max(0, detail.risk_score * 100))}%` }} />
      </div>
      <section className="split-grid">
        <div className="panel">
          <div className="section-title">
            <h3>Why this transaction was flagged</h3>
            <p>Actual SHAP contributors from the frozen model</p>
          </div>
          <div className="signal-list">
            {(detail.contributors || []).slice(0, 5).map((item) => (
              <div className="signal-row" key={item.feature}>
                <div>
                  <strong>{item.feature}</strong>
                  <span>{impactLabel(item.impact)}</span>
                </div>
                <b>{Number(item.impact) >= 0 ? "+" : ""}{formatScore(item.impact)}</b>
              </div>
            ))}
          </div>
        </div>
        <div className="panel">
          <div className="section-title">
            <h3>Historical Demo Outcome</h3>
            <p>Shown only for packaged labeled examples</p>
          </div>
          <div className="outcome-box">
            <strong>{detail.historical_outcome || "Unavailable"}</strong>
            {detail.demo_outcome ? <span>{detail.demo_outcome}</span> : <span>Prediction aligned or label not available.</span>}
          </div>
          <div className="action-row">
            {["Mark Suspicious", "Mark Legitimate", "Escalate"].map((item) => (
              <button key={item} onClick={() => choose(item)} className={localAction === item || action === item ? "selected" : ""}>
                {item}
              </button>
            ))}
          </div>
          <p className="microcopy">Session-only demo action. No model retraining occurs automatically.</p>
        </div>
      </section>
      <details className="technical-details">
        <summary>Technical SHAP values</summary>
        <pre>{JSON.stringify(detail.explanation || {}, null, 2)}</pre>
      </details>
    </StateBlock>
  );
}
