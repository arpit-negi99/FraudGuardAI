import { MetricCard } from "../components/MetricCard";
import { IncidentSeverityChart, IncidentTypeChart, RiskActivityChart, RiskDistributionChart } from "../components/Charts";
import { StateBlock } from "../components/StateBlock";
import { formatPercent } from "../utils/format";

export function RiskMonitor({ summary, transactions, incidentSummary, spike, loading, error }) {
  const typeData = (incidentSummary.type_distribution || [])
    .filter((row) => row.incident_type !== "NORMAL_PAYMENT")
    .map((row) => ({ ...row, label: row.incident_type.replaceAll("_", " ") }));
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Risk Monitor</h2>
        <p>Monitor current demo/batch activity without claiming live production telemetry.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Transactions Analyzed" value={summary.transactions_analyzed} />
        <MetricCard label="Review Rate" value={formatPercent(summary.review_rate)} />
        <MetricCard label="Average Risk" value={formatPercent(summary.average_risk)} />
        <MetricCard label="Highest Risk" value={formatPercent(summary.highest_risk)} tone="warn" />
      </section>
      <section className="split-grid">
        <RiskDistributionChart data={summary.risk_distribution || []} />
        <RiskActivityChart rows={transactions} />
      </section>
      <section className="split-grid monitor-secondary">
        <IncidentSeverityChart data={incidentSummary.severity_distribution || []} />
        <IncidentTypeChart data={typeData} />
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>Fraud Spike Monitor</h3>
          <p>{spike?.method}</p>
        </div>
        <div className="spike-grid">
          <MetricCard label="Spike Status" value={spike?.status || "Normal"} tone={spike?.status === "High" ? "danger" : "warn"} />
          <MetricCard label="Baseline Review Rate" value={formatPercent(spike?.baseline_review_rate)} />
          <MetricCard label="Latest Window" value={formatPercent(spike?.latest_window_review_rate)} />
          <MetricCard label="Z-Score" value={Number(spike?.z_score || 0).toFixed(2)} />
        </div>
        <p className="microcopy">{spike?.thresholds}</p>
      </section>
    </StateBlock>
  );
}
