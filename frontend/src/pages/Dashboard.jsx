import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { RiskDistributionChart } from "../components/Charts";
import { formatAction, formatIncidentType, formatPercent } from "../utils/format";

export function Dashboard({ summary, transactions, incidentSummary, incidentRows, loading, error, onOpen, onOpenIncident, events }) {
  const priority = [...transactions].sort((a, b) => b.risk_score - a.risk_score).slice(0, 6);
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Fraud Risk Overview</h2>
        <p>Monitor suspicious activity and prioritize transactions that need attention.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Needs Review" value={summary.needs_review} note="Current demo sample" tone="warn" />
        <MetricCard label="Critical Risk" value={summary.critical_count} note="Highest-priority cases" tone="danger" />
        <MetricCard label="Review Workload" value={formatPercent(summary.review_rate)} note="Share requiring review" />
        <MetricCard label="Risk Status" value={summary.risk_status} note="Computed from scored rows" tone={summary.risk_status === "High" ? "danger" : "default"} />
      </section>
      <section className="split-grid">
        <div className="panel">
          <div className="section-title">
            <h3>Priority Transactions</h3>
            <p>Highest risk first</p>
          </div>
          <DataTable rows={priority} onOpen={onOpen} />
        </div>
        <div className="side-stack">
          <RiskDistributionChart data={summary.risk_distribution || []} />
          <div className="panel">
            <div className="section-title">
              <h3>Payment Incidents</h3>
              <p>Separate lifecycle workflow</p>
            </div>
            <div className="compact-stats">
              <div>
                <span>Active</span>
                <strong>{incidentSummary.active_incidents || 0}</strong>
              </div>
              <div>
                <span>Critical / High</span>
                <strong>{incidentSummary.critical || 0} / {incidentSummary.high || 0}</strong>
              </div>
            </div>
            <div className="mini-list">
              {incidentRows.slice(0, 3).map((row) => (
                <button key={row.payment_id} onClick={() => onOpenIncident(row.payment_id)}>
                  <span>{row.payment_id}</span>
                  <strong>{formatIncidentType(row.incident_type)}</strong>
                  <small>{formatAction(row.recommended_action)}</small>
                </button>
              ))}
            </div>
            <button className="table-action wide-action" onClick={() => onOpenIncident(incidentRows[0]?.payment_id)}>
              View all incidents
            </button>
          </div>
          <div className="panel">
            <div className="section-title">
              <h3>Recent Activity</h3>
              <p>Current-session events only</p>
            </div>
            <div className="activity-feed">
              {events.length ? events.map((event) => <span key={event}>{event}</span>) : <span>Policy loaded as Balanced.</span>}
            </div>
          </div>
        </div>
      </section>
    </StateBlock>
  );
}
