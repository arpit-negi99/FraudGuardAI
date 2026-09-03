import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { RiskDistributionChart } from "../components/Charts";
import {
  formatAction,
  formatIncidentType,
  formatMonitoringDriver,
  formatPercent,
  monitoringSignalLabel,
  selectDefaultPaymentIncident,
  sortPaymentLifecycles,
} from "../utils/format";

export function Dashboard({ summary, transactions, incidentSummary, lifecycleSummary, monitoringSummary, incidentRows, loading, error, onOpen, onOpenIncident, events }) {
  const priority = [...transactions].sort((a, b) => b.risk_score - a.risk_score).slice(0, 6);
  const actionableIncidents = sortPaymentLifecycles(incidentRows).slice(0, 3);
  const topIncident = selectDefaultPaymentIncident(incidentRows);
  const monitoringStatus = monitoringSummary.latest?.status || "NORMAL";
  const monitoringDriverLabel = monitoringSignalLabel(monitoringStatus);
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Fraud Risk Overview</h2>
        <p>Monitor suspicious activity and prioritize transactions that need attention.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Needs Review" value={summary.needs_review} note="Current demo sample" tone="warn" />
        <MetricCard label="Critical Risk" value={summary.critical_count} note="Highest-priority cases" tone="danger" />
        <MetricCard label="Demo Review Workload" value={formatPercent(summary.review_rate)} note="Current demo transaction sample" />
        <MetricCard
          label="Operational Risk"
          value={monitoringStatus}
          note="Synthetic monitoring scenario"
          tone={monitoringStatus === "CRITICAL" || monitoringStatus === "HIGH" ? "danger" : "default"}
        />
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
              <h3>Operational Risk</h3>
              <p>Synthetic stream monitor</p>
            </div>
            <div className="compact-stats">
              <div>
                <span>Status</span>
                <strong>{monitoringStatus}</strong>
              </div>
              <div>
                <span>{monitoringDriverLabel}</span>
                <strong>{formatMonitoringDriver(monitoringSummary.latest?.primary_driver)}</strong>
              </div>
            </div>
            <p className="microcopy">Synthetic monitoring scenario. Driver: {formatMonitoringDriver(monitoringSummary.latest?.primary_driver)}</p>
          </div>
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
              {actionableIncidents.map((row) => (
                <button key={row.payment_id} onClick={() => onOpenIncident(row.payment_id)}>
                  <span>{row.payment_id}</span>
                  <strong>{formatIncidentType(row.current_incident)}</strong>
                  <small>{formatAction(row.recommended_action)}</small>
                </button>
              ))}
            </div>
            <button className="table-action wide-action" onClick={() => topIncident && onOpenIncident(topIncident.payment_id)} disabled={!topIncident}>
              View all incidents
            </button>
          </div>
          <div className="panel">
            <div className="section-title">
              <h3>Payment Lifecycle Incidents</h3>
              <p>Synthetic timeline reasoning</p>
            </div>
            <div className="compact-stats lifecycle-stats">
              <div>
                <span>Active</span>
                <strong>{lifecycleSummary.active || 0}</strong>
              </div>
              <div>
                <span>Resolving</span>
                <strong>{lifecycleSummary.resolving || 0}</strong>
              </div>
              <div>
                <span>Resolved</span>
                <strong>{lifecycleSummary.resolved || 0}</strong>
              </div>
              <div>
                <span>Critical</span>
                <strong>{lifecycleSummary.critical || 0}</strong>
              </div>
            </div>
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
