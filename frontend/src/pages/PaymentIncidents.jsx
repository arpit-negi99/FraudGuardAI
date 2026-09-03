import { useEffect, useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { api } from "../services/api";
import {
  actionDescription,
  formatAction,
  formatAmount,
  formatIncidentType,
  formatPercent,
  fraudIncidentMessage,
  fraudRiskBand,
  selectDefaultPaymentIncident,
  sortPaymentLifecycles,
  timelineTransitionLabel,
} from "../utils/format";

const incidentTypes = [
  ["All", "All incident types"],
  ["DEBIT_SERVICE_MISMATCH", "Debit-service mismatch"],
  ["LATE_AUTHORIZATION_RISK", "Late authorization risk"],
  ["CAPTURED_BUT_UNFULFILLED", "Captured but unfulfilled"],
  ["REFUND_REQUIRED", "Refund required"],
  ["RETRY_RELATED_PAYMENT_RISK", "Retry-related payment risk"],
  ["COMPLAINT_ESCALATION_RISK", "Complaint escalation"],
];

export function PaymentIncidents({
  summary,
  lifecycleSummary,
  summaryLoading,
  summaryError,
  selectedPaymentId,
  setSelectedPaymentId,
}) {
  const [severity, setSeverity] = useState("All");
  const [statusFilter, setStatusFilter] = useState("ACTIVE_INCIDENT");
  const [incidentType, setIncidentType] = useState("All");
  const [query, setQuery] = useState("");
  const [list, setList] = useState(null);
  const [listError, setListError] = useState("");
  const [listLoading, setListLoading] = useState(true);
  const [detail, setDetail] = useState(null);
  const [detailError, setDetailError] = useState("");
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    let active = true;
    setListLoading(true);
    setListError("");
    api
      .getLifecycles({ status: statusFilter, limit: 500 })
      .then((result) => {
        if (active) setList(result);
      })
      .catch((err) => {
        if (active) setListError(err.message ? `Unable to load payment incidents. ${err.message}` : "Unable to load payment incidents.");
      })
      .finally(() => {
        if (active) setListLoading(false);
      });
    return () => {
      active = false;
    };
  }, [statusFilter]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    let loaded = list?.lifecycles || [];
    if (severity !== "All") loaded = loaded.filter((row) => row.current_severity === severity);
    if (statusFilter !== "All") loaded = loaded.filter((row) => row.status === statusFilter);
    if (incidentType !== "All") loaded = loaded.filter((row) => row.current_incident === incidentType);
    if (needle) loaded = loaded.filter((row) => row.payment_id.toLowerCase().includes(needle));
    return sortPaymentLifecycles(loaded);
  }, [list, query, severity, statusFilter, incidentType]);

  useEffect(() => {
    if (rows.length && (!selectedPaymentId || !rows.some((row) => row.payment_id === selectedPaymentId))) {
      const firstIncident = selectDefaultPaymentIncident(rows);
      if (firstIncident) setSelectedPaymentId(firstIncident.payment_id);
    }
  }, [rows, selectedPaymentId, setSelectedPaymentId]);

  useEffect(() => {
    if (!selectedPaymentId) return;
    let active = true;
    setDetailLoading(true);
    setDetailError("");
    api
      .getLifecycle(selectedPaymentId)
      .then((result) => {
        if (active) setDetail(result);
      })
      .catch((err) => {
        if (active) {
          setDetail(null);
          setDetailError(err.message || "Unable to load payment incident.");
        }
      })
      .finally(() => {
        if (active) setDetailLoading(false);
      });
    return () => {
      active = false;
    };
  }, [selectedPaymentId]);

  return (
    <StateBlock loading={summaryLoading} error={summaryError}>
      <section className="page-heading">
        <h2>Payment Incidents</h2>
        <p>Identify unresolved payment issues before they become complaints, refunds or disputes. Synthetic demo data.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Active Incidents" value={summary.active_incidents} note="Lifecycle issues" tone="warn" />
        <MetricCard label="Critical" value={summary.critical} note="Immediate escalation" tone="danger" />
        <MetricCard label="Resolved Timelines" value={lifecycleSummary.resolved || 0} note="Issue resolved later" />
        <MetricCard label="Incident Rate" value={formatPercent(summary.incident_rate, 2)} note="Synthetic demo data" />
      </section>
      <section className="incident-education">
        <strong>Fraud risk is separate from payment incident risk.</strong>
        <span>A low fraud signal can still require urgent payment-operations action, and a high fraud signal can have no lifecycle incident.</span>
      </section>
      <section className="filters">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search Payment ID" />
        <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
          <option value="ACTIVE_INCIDENT">Active incidents</option>
          <option value="All">All</option>
          <option value="RESOLVING">Resolving</option>
          <option value="RESOLVED">Resolved</option>
          <option value="NORMAL">Normal</option>
        </select>
        <select value={severity} onChange={(event) => setSeverity(event.target.value)}>
          <option>All</option>
          <option>CRITICAL</option>
          <option>HIGH</option>
          <option>MEDIUM</option>
          <option>LOW</option>
          <option>NONE</option>
        </select>
        <select value={incidentType} onChange={(event) => setIncidentType(event.target.value)}>
          {incidentTypes.map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </section>
      <section className="split-grid incident-layout">
        <StateBlock loading={listLoading} error={listError} empty={!rows.length} emptyMessage="No payment incidents match the selected filters.">
          <div className="panel">
            <div className="section-title">
              <h3>Lifecycle Queue</h3>
              <p>{rows.length} matching timelines</p>
            </div>
            <IncidentTable rows={rows} onOpen={setSelectedPaymentId} />
          </div>
        </StateBlock>
        <StateBlock loading={detailLoading} error={detailError} empty={!detail} emptyMessage="Select a payment incident to inspect.">
          <IncidentDetail detail={detail} />
        </StateBlock>
      </section>
    </StateBlock>
  );
}

function IncidentTable({ rows, onOpen }) {
  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th>Severity</th>
            <th>Payment ID</th>
            <th>Amount</th>
            <th>Payment Method</th>
            <th>Incident</th>
            <th>Fraud Risk</th>
            <th>Status</th>
            <th>Recommended Action</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.payment_id}>
              <td><Badge type="severity">{row.current_severity}</Badge></td>
              <td>{row.payment_id}</td>
              <td>{formatAmount(row.amount)}</td>
              <td>{String(row.payment_method).toUpperCase()}</td>
              <td>{formatIncidentType(row.current_incident)}</td>
              <td>{formatPercent(row.fraud_risk_score)}</td>
              <td>{formatIncidentType(row.status)}</td>
              <td>{formatAction(row.recommended_action)}</td>
              <td>
                <button className="table-action" onClick={() => onOpen(row.payment_id)}>
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function IncidentDetail({ detail }) {
  const state = detail.current_state || detail;
  const statusLabel = formatIncidentType(detail.status);
  const incidentLabel = formatIncidentType(detail.current_incident);
  const attentionText = detail.status === "RESOLVED"
    ? "Incident resolved after analyst workflow"
    : detail.current_severity === "CRITICAL"
      ? "Requires immediate analyst attention"
      : detail.incident_detected
        ? "Requires payment-operations review"
        : "No payment lifecycle incident is currently detected";
  const lifecycle = [
    ["Customer debited", state.bank_debited ? "Yes" : "No"],
    ["Gateway status", formatIncidentType(state.gateway_status)],
    ["Order status", formatIncidentType(state.order_status)],
    ["Service delivered", state.service_delivered ? "Yes" : "No"],
    ["Callback received", state.callback_received ? "Yes" : "No"],
    ["Refund status", formatIncidentType(state.refund_status)],
    ["Retry count", state.retry_count],
    ["Time unresolved", `${state.time_since_payment_minutes} min`],
    ["Customer complaint", state.customer_complaint ? "Yes" : "No"],
  ];
  return (
    <div className="side-stack">
      <section className="case-heading">
        <div>
          <p>Payment Incident</p>
          <h2>{detail.payment_id}</h2>
          <strong>{incidentLabel}</strong>
          <span>{attentionText}</span>
        </div>
        <Badge type="severity">{detail.current_severity}</Badge>
      </section>
      <section className="panel resolved-summary">
        <div>
          <span>Current state</span>
          <strong>{statusLabel}</strong>
        </div>
        <div>
          <span>Highest severity observed</span>
          <strong>{detail.highest_severity_observed}</strong>
        </div>
        {detail.time_to_resolution_minutes !== null ? (
          <div>
            <span>Resolution time</span>
            <strong>{detail.time_to_resolution_minutes} min</strong>
          </div>
        ) : null}
      </section>
      <section className="metrics-grid incident-detail-metrics">
        <MetricCard label="Amount" value={formatAmount(detail.amount)} />
        <MetricCard label="Current State" value={formatIncidentType(detail.status)} tone={detail.status === "ACTIVE_INCIDENT" ? "warn" : "default"} />
        <MetricCard label="Current Severity" value={detail.current_severity} tone={detail.current_severity === "CRITICAL" ? "danger" : "warn"} />
        <MetricCard label="Highest Severity" value={detail.highest_severity_observed} tone={detail.highest_severity_observed === "CRITICAL" ? "danger" : "warn"} />
        <MetricCard label="First Detected" value={detail.first_incident_time_minutes === null ? "n/a" : `${detail.first_incident_time_minutes} min`} />
        {detail.time_to_resolution_minutes !== null ? (
          <MetricCard label="Resolution Time" value={`${detail.time_to_resolution_minutes} min`} />
        ) : null}
        <MetricCard label="Transaction Fraud Signal" value={formatPercent(detail.fraud_risk_score)} />
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>Payment Timeline</h3>
          <p>Event replay with rule evaluation after each step</p>
        </div>
        <div className="timeline">
          {(detail.timeline || []).map((item, index, timeline) => (
            <div
              className={`timeline-item timeline-${String(item.status || "NORMAL").toLowerCase()} severity-${String(item.severity || "NONE").toLowerCase()}`}
              key={`${item.event_id}-${item.time}`}
            >
              <div className="timeline-dot" />
              <div>
                <span>{item.time} min</span>
                <b className="timeline-transition">{timelineTransitionLabel(item, timeline[index - 1])}</b>
                <strong>{item.event_label}</strong>
                <small>{formatIncidentType(item.status)}</small>
                {item.incident_type !== "NORMAL_PAYMENT" || item.status === "RESOLVED" ? (
                  <div className="timeline-meta">
                    <Badge type="severity">{item.severity}</Badge>
                    <em>{formatIncidentType(item.incident_type)}</em>
                    <b>{formatAction(item.recommended_action)}</b>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>Payment Lifecycle</h3>
          <p>Current synthetic event fields</p>
        </div>
        <div className="lifecycle-grid">
          {lifecycle.map(([label, value]) => (
            <div className="lifecycle-row" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <div className="section-title">
          <h3>What went wrong</h3>
          <p>Actual rule-engine reasons</p>
        </div>
        {detail.reasons?.length ? (
          <ul className="reason-list">
            {detail.reasons.map((reason) => <li key={reason}>{reason}</li>)}
          </ul>
        ) : (
          <p className="microcopy">No payment lifecycle incident is currently detected.</p>
        )}
      </section>
      <section className="panel recommended-action">
        <div>
          <span>Recommended Action</span>
          <strong>{formatAction(detail.recommended_action)}</strong>
        </div>
        <p>{actionDescription(detail.recommended_action)}</p>
      </section>
      <section className="panel fraud-incident-card">
        <div className="comparison-grid">
          <div>
            <span>Fraud Risk</span>
            <strong>{formatPercent(detail.fraud_risk_score)}</strong>
            <Badge>{fraudRiskBand(detail.fraud_risk_score)}</Badge>
          </div>
          <div>
            <span>Payment Incident</span>
            <strong>{detail.current_severity}</strong>
            <small>{formatIncidentType(detail.current_incident)}</small>
          </div>
        </div>
        <p>{fraudIncidentMessage(detail.fraud_risk_score, detail.current_severity, detail.status !== "NORMAL" && detail.status !== "RESOLVED")}</p>
      </section>
    </div>
  );
}
