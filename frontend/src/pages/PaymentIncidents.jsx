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
  summaryLoading,
  summaryError,
  selectedPaymentId,
  setSelectedPaymentId,
}) {
  const [severity, setSeverity] = useState("All");
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
      .getIncidents({
        severity: severity === "All" ? "" : severity,
        incident_type: incidentType === "All" ? "" : incidentType,
        limit: 200,
      })
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
  }, [severity, incidentType]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const loaded = list?.incidents || [];
    if (!needle) return loaded;
    return loaded.filter((row) => row.payment_id.toLowerCase().includes(needle));
  }, [list, query]);

  useEffect(() => {
    if (!selectedPaymentId && rows.length) {
      const firstIncident = rows.find((row) => row.incident_detected) || rows[0];
      setSelectedPaymentId(firstIncident.payment_id);
    }
  }, [rows, selectedPaymentId, setSelectedPaymentId]);

  useEffect(() => {
    if (!selectedPaymentId) return;
    let active = true;
    setDetailLoading(true);
    setDetailError("");
    api
      .getIncident(selectedPaymentId)
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
        <p>Identify unresolved payment issues before they become complaints, refunds or disputes.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Active Incidents" value={summary.active_incidents} note="Lifecycle issues" tone="warn" />
        <MetricCard label="Critical" value={summary.critical} note="Immediate escalation" tone="danger" />
        <MetricCard label="High Priority" value={summary.high} note="Payment ops review" tone="warn" />
        <MetricCard label="Incident Rate" value={formatPercent(summary.incident_rate, 2)} note="Synthetic demo data" />
      </section>
      <section className="incident-education">
        <strong>Fraud risk is separate from payment incident risk.</strong>
        <span>A low fraud signal can still require urgent payment-operations action, and a high fraud signal can have no lifecycle incident.</span>
      </section>
      <section className="filters">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search Payment ID" />
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
        <StateBlock loading={listLoading} error={listError} empty={!rows.length} emptyMessage="No matching payment incidents.">
          <div className="panel">
            <div className="section-title">
              <h3>Incident Queue</h3>
              <p>{list?.total || 0} matching payments</p>
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
            <th>Recommended Action</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.payment_id}>
              <td><Badge type="severity">{row.severity}</Badge></td>
              <td>{row.payment_id}</td>
              <td>{formatAmount(row.amount)}</td>
              <td>{String(row.payment_method).toUpperCase()}</td>
              <td>{formatIncidentType(row.incident_type)}</td>
              <td>{formatPercent(row.fraud_risk_score)}</td>
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
  const lifecycle = [
    ["Customer debited", detail.bank_debited ? "Yes" : "No"],
    ["Gateway status", formatIncidentType(detail.gateway_status)],
    ["Order status", formatIncidentType(detail.order_status)],
    ["Service delivered", detail.service_delivered ? "Yes" : "No"],
    ["Callback received", detail.callback_received ? "Yes" : "No"],
    ["Refund status", formatIncidentType(detail.refund_status)],
    ["Retry count", detail.retry_count],
    ["Time unresolved", `${detail.time_since_payment_minutes} min`],
    ["Customer complaint", detail.customer_complaint ? "Yes" : "No"],
  ];
  return (
    <div className="side-stack">
      <section className="case-heading">
        <div>
          <p>Payment Incident</p>
          <h2>{detail.payment_id}</h2>
        </div>
        <Badge type="severity">{detail.severity}</Badge>
      </section>
      <section className="metrics-grid incident-detail-metrics">
        <MetricCard label="Amount" value={formatAmount(detail.amount)} />
        <MetricCard label="Incident Severity" value={detail.severity} tone={detail.severity === "CRITICAL" ? "danger" : "warn"} />
        <MetricCard label="Time Unresolved" value={`${detail.time_since_payment_minutes} min`} />
        <MetricCard label="Transaction Fraud Signal" value={formatPercent(detail.fraud_risk_score)} />
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
            <strong>{detail.severity}</strong>
            <small>{formatIncidentType(detail.incident_type)}</small>
          </div>
        </div>
        <p>{fraudIncidentMessage(detail.fraud_risk_score, detail.severity, detail.incident_detected)}</p>
      </section>
    </div>
  );
}
