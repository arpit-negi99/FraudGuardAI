import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { formatPercent } from "../utils/format";

export function About({ evaluation, loading, error }) {
  const metrics = evaluation?.metrics || {};
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>About FraudGuard</h2>
        <p>Client-facing overview of two separate risk operations modules.</p>
      </section>
      <section className="panel about-grid">
        <div>
          <h3>Product</h3>
          <p>FraudGuard scores merchant transactions, recommends ALLOW or REVIEW, and explains model-attribution signals for analyst review.</p>
        </div>
        <div>
          <h3>Architecture</h3>
          <code>{"FraudGuard AI -> Fraud Risk Engine + Payment Incident Engine -> Risk Operations -> Fraud Review + Incident Response"}</code>
        </div>
        <div>
          <h3>Technology</h3>
          <p>React, FastAPI, XGBoost, SHAP, scikit-learn, deterministic lifecycle rules.</p>
        </div>
      </section>
      <section className="split-grid">
        <div className="panel">
          <div className="section-title">
            <h3>Module 1 - Transaction Fraud Detection</h3>
            <p>IEEE-CIS / XGBoost</p>
          </div>
          <p className="microcopy">Uses XGBoost to identify transactions that resemble historically fraudulent transactions. The frozen operating threshold remains 0.60.</p>
        </div>
        <div className="panel">
          <div className="section-title">
            <h3>Module 2 - Payment Incident Detection</h3>
            <p>Synthetic lifecycle data / deterministic rules</p>
          </div>
          <p className="microcopy">Uses deterministic lifecycle safeguards to detect unresolved payment-state inconsistencies that may lead to complaints, refunds or disputes.</p>
        </div>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Precision" value={formatPercent(metrics.precision)} />
        <MetricCard label="Recall" value={formatPercent(metrics.recall)} />
        <MetricCard label="F1" value={formatPercent(metrics.f1)} />
        <MetricCard label="PR-AUC" value={Number(metrics.pr_auc || 0).toFixed(3)} />
        <MetricCard label="ROC-AUC" value={Number(metrics.roc_auc || 0).toFixed(3)} />
        <MetricCard label="Review Rate" value={formatPercent(metrics.review_rate, 2)} />
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
          <li>No Razorpay production validation is claimed.</li>
        </ul>
      </section>
    </StateBlock>
  );
}
