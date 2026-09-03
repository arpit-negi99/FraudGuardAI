import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { formatPercent } from "../utils/format";

export function About({ evaluation, incidentSummary, monitoringSummary, loading, error }) {
  const metrics = evaluation?.metrics || {};
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>About FraudGuard</h2>
        <p>Client-facing overview of transaction fraud scoring, payment lifecycle incidents, and operational risk monitoring.</p>
      </section>
      <section className="panel about-grid">
        <div>
          <h3>Product</h3>
          <p>FraudGuard scores merchant transactions, recommends ALLOW or REVIEW, and explains model-attribution signals for analyst review.</p>
        </div>
        <div>
          <h3>Architecture</h3>
          <code>{"FraudGuard AI -> Module 1 XGBoost + Module 2 Lifecycle Rules + Module 3 Statistical Monitoring -> Risk Operations"}</code>
        </div>
        <div>
          <h3>Technology</h3>
          <p>React, FastAPI, XGBoost, SHAP, scikit-learn, deterministic lifecycle rules, z-score/EWMA monitoring.</p>
        </div>
      </section>
      <section className="split-grid">
        <div className="panel">
          <div className="section-title">
            <h3>Module 1 - Real Dataset Evaluation</h3>
            <p>Chronological IEEE-CIS held-out test</p>
          </div>
          <p className="microcopy">Uses XGBoost to identify transactions that resemble historically fraudulent transactions. The frozen operating threshold remains 0.60.</p>
          <div className="about-metric-grid">
            <MetricCard label="Precision" value={formatPercent(metrics.precision)} />
            <MetricCard label="Recall" value={formatPercent(metrics.recall)} />
            <MetricCard label="F1" value={formatPercent(metrics.f1)} />
            <MetricCard label="PR-AUC" value={Number(metrics.pr_auc || 0).toFixed(3)} />
            <MetricCard label="ROC-AUC" value={Number(metrics.roc_auc || 0).toFixed(3)} />
            <MetricCard label="Review Rate" value={formatPercent(metrics.review_rate, 2)} />
          </div>
        </div>
        <div className="panel">
          <div className="section-title">
            <h3>Module 2 - Synthetic Scenario Validation</h3>
            <p>Synthetic lifecycle data / deterministic rules</p>
          </div>
          <p className="microcopy">Deterministic payment-lifecycle safeguards were tested on synthetic scenario-generated payment events.</p>
          <div className="about-metric-grid compact-about-metrics">
            <MetricCard label="Active Incidents" value={incidentSummary.active_incidents || 0} note="Synthetic demo data" tone="warn" />
            <MetricCard label="Critical" value={incidentSummary.critical || 0} note="Synthetic demo data" tone="danger" />
            <MetricCard label="Incident Rate" value={formatPercent(incidentSummary.incident_rate, 2)} note="Synthetic demo data" />
          </div>
        </div>
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>Module 3 - Synthetic Monitoring Evaluation</h3>
          <p>Synthetic streams / statistical spike detection</p>
        </div>
        <p className="microcopy">Uses rolling windows, historical baseline statistics, z-scores, and EWMA trend signals to detect unusual increases across fraud-risk activity and payment operations. It is not a new fraud classifier and does not claim production monitoring.</p>
        <div className="about-metric-grid monitoring-eval-grid">
          <MetricCard label="Precision" value={formatPercent(monitoringSummary.precision)} note="Synthetic scenarios" />
          <MetricCard label="Recall" value={formatPercent(monitoringSummary.recall)} note="Synthetic scenarios" />
          <MetricCard label="F1" value={formatPercent(monitoringSummary.f1)} note="Synthetic scenarios" />
          <MetricCard label="False Alert Rate" value={formatPercent(monitoringSummary.false_alert_rate)} note="Synthetic scenarios" tone="warn" />
        </div>
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>Limitations</h3>
          <p>Important boundaries for honest use</p>
        </div>
        <ul className="limit-list">
          <li>Dataset features are partly anonymized.</li>
          <li>Risk score is not guaranteed calibrated probability.</li>
          <li>Costs are modeled assumptions.</li>
          <li>No automatic blocking.</li>
          <li>Held-out recall is 56.3%.</li>
          <li>Dataset may not reflect current production fraud patterns.</li>
          <li>Module 2 uses synthetic payment lifecycle data.</li>
          <li>Module 3 uses synthetic monitoring streams.</li>
          <li>No Razorpay production validation is claimed.</li>
        </ul>
      </section>
    </StateBlock>
  );
}
