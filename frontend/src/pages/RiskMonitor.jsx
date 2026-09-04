import { useEffect, useMemo, useState } from "react";
import { MetricCard } from "../components/MetricCard";
import { MonitoringRateChart, OperationalRiskStrip } from "../components/Charts";
import { StateBlock } from "../components/StateBlock";
import { useRiskStream } from "../hooks/useRiskStream";
import { api } from "../services/api";
import {
  formatMonitoringDriver,
  formatPercent,
  monitoringSignalLabel,
  splitMonitoringAlerts,
} from "../utils/format";
import { isStreamMode, normalizeStreamCurrent, streamStatusLabel } from "../utils/stream";

const scenarioLabels = {
  NORMAL: "Normal activity",
  FRAUD_RISK_SPIKE: "Fraud spike",
  PAYMENT_INCIDENT_SPIKE: "Payment incident spike",
  DEBIT_SERVICE_MISMATCH_SPIKE: "Debit-service mismatch spike",
  COMPLAINT_SPIKE: "Complaint spike",
  RETRY_SPIKE: "Retry spike",
  MIXED_RISK_SPIKE: "Mixed spike",
  RECOVERY: "Recovery",
};

export function RiskMonitor({ monitoringSummary, loading, error, health }) {
  const [scenario, setScenario] = useState("NORMAL");
  const [current, setCurrent] = useState(null);
  const [windows, setWindows] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [monitorError, setMonitorError] = useState("");
  const [monitorLoading, setMonitorLoading] = useState(true);
  const streamEnabled = isStreamMode(health);
  const { state: streamState, connectionState } = useRiskStream({ enabled: streamEnabled });

  useEffect(() => {
    let active = true;
    setMonitorLoading(true);
    setMonitorError("");
    Promise.all([
      api.monitoringCurrent(streamEnabled ? "" : scenario),
      api.monitoringWindows(scenario, 80),
      api.monitoringAlerts(scenario, 20),
      api.monitoringScenarios(),
    ])
      .then(([currentResult, windowResult, alertResult, scenarioResult]) => {
        if (!active) return;
        setCurrent(currentResult);
        setWindows(windowResult.windows || []);
        setAlerts(alertResult.alerts || []);
        setScenarios(scenarioResult.scenarios || []);
      })
      .catch((err) => {
        if (active) setMonitorError(err.message || "Unable to load risk data. Check that the FraudGuard API is running.");
      })
      .finally(() => {
        if (active) setMonitorLoading(false);
      });
    return () => {
      active = false;
    };
  }, [scenario, streamEnabled]);

  useEffect(() => {
    const normalized = normalizeStreamCurrent(streamState);
    if (normalized) setCurrent(normalized);
  }, [streamState]);

  const { currentAlerts, historyAlerts } = useMemo(
    () => splitMonitoringAlerts(current, alerts),
    [current, alerts],
  );
  const status = current?.status || "NORMAL";
  const signalLabel = monitoringSignalLabel(status);

  return (
    <StateBlock loading={loading || monitorLoading} error={error || monitorError}>
      <section className="page-heading">
        <h2>Risk Monitor</h2>
        <p>Detect unusual changes across fraud activity and payment operations using the active demo stream.</p>
      </section>
      <section className="stream-status-row">
        <span className={`stream-dot ${streamEnabled && connectionState === "live" ? "is-live" : ""}`} />
        <strong>{streamStatusLabel(connectionState, streamEnabled)}</strong>
        <span>{streamEnabled ? "Redpanda/Redis analytics mode" : "Synthetic local monitoring mode"}</span>
      </section>
      <section className="filters monitoring-filters">
        <label>
          Synthetic Demo Scenario
          <select value={scenario} onChange={(event) => setScenario(event.target.value)} disabled={streamEnabled}>
            {scenarios.map((item) => (
              <option key={item} value={item}>{scenarioLabels[item] || item}</option>
            ))}
          </select>
          <span>{streamEnabled ? "Scenario selection is disabled while reading live merchant state." : "Demonstration using generated payment-stream data."}</span>
        </label>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Current Status" value={status} note="Selected synthetic window" tone={["HIGH", "CRITICAL"].includes(status) ? "danger" : "default"} />
        <MetricCard label={signalLabel} value={formatMonitoringDriver(current?.primary_driver)} note="Selected demo stream" />
        <MetricCard label="Review Rate" value={formatPercent(current?.current_review_rate)} note="Current synthetic window" />
        <MetricCard label="Incident Rate" value={formatPercent(current?.current_incident_rate)} note="Current synthetic window" />
      </section>
      <section className="panel status-panel">
        <div className="section-title">
          <h3>Operational Risk Timeline</h3>
          <p>Synthetic monitoring stream; each block is one 15-minute window</p>
        </div>
        <OperationalRiskStrip rows={windows} />
      </section>
      <section>
        <MonitoringRateChart rows={windows} />
      </section>
      <section className="split-grid monitor-secondary">
        <div className="panel">
          <div className="section-title">
            <h3>Current Alerts</h3>
            <p>Current selected window only</p>
          </div>
          <div className="alert-list">
            {currentAlerts.length ? currentAlerts.map((alert) => (
              <div className="alert-card" key={alert.alert_id}>
                <strong>{alert.severity}</strong>
                <span>{formatMonitoringDriver(alert.primary_driver)}</span>
                <div className="alert-measures">
                  <small>Current <b>{formatPercent(alert.current_value)}</b></small>
                  <small>Baseline <b>{formatPercent(alert.baseline_value)}</b></small>
                  <small>Change <b>+{formatPercent(alert.relative_change, 0)}</b></small>
                </div>
                <p>{alert.recommended_action}</p>
              </div>
            )) : <p className="microcopy">No active operational alerts.</p>}
          </div>
        </div>
        <div className="panel">
          <div className="section-title">
            <h3>Recent Alert History</h3>
            <p>Previous synthetic windows</p>
          </div>
          <div className="history-list">
            {historyAlerts.slice(-10).reverse().map((alert) => (
              <div key={alert.alert_id}>
                <span>{alert.window_start}</span>
                <strong>{alert.severity}</strong>
                <small>{formatMonitoringDriver(alert.primary_driver)} - {formatPercent(alert.relative_change, 0)}</small>
              </div>
            ))}
            {!historyAlerts.length ? <p className="microcopy">No recent historical alerts for this scenario.</p> : null}
          </div>
        </div>
      </section>
      <details className="technical-details">
        <summary>How was this detected?</summary>
        <div className="section-title technical-title">
          <h3>Synthetic Monitoring Evaluation</h3>
          <p>These metrics are measured on synthetic monitoring scenarios, not live merchant traffic.</p>
        </div>
        <div className="spike-grid">
          <MetricCard label="Precision" value={formatPercent(monitoringSummary.precision)} />
          <MetricCard label="Recall" value={formatPercent(monitoringSummary.recall)} />
          <MetricCard label="F1" value={formatPercent(monitoringSummary.f1)} />
          <MetricCard label="False Alert Rate" value={formatPercent(monitoringSummary.false_alert_rate)} tone="warn" />
        </div>
        <p className="microcopy technical-note">All simulated spike scenarios were detected in their first evaluated spike window.</p>
        <pre>{JSON.stringify(current?.metrics || {}, null, 2)}</pre>
      </details>
    </StateBlock>
  );
}
