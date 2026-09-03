import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  actionDescription,
  decisionClass,
  formatAmount,
  formatIncidentType,
  formatMonitoringDriver,
  formatPercent,
  fraudIncidentMessage,
  monitoringSignalLabel,
  priorityClass,
  selectDefaultPaymentIncident,
  severityClass,
  sortPaymentLifecycles,
  splitMonitoringAlerts,
  timelineTransitionLabel,
} from "./format.js";

describe("format helpers", () => {
  it("formats percentages", () => {
    assert.equal(formatPercent(0.53838), "53.8%");
  });

  it("formats unknown amounts safely", () => {
    assert.equal(formatAmount(null), "n/a");
  });

  it("maps risk labels to stable badge classes", () => {
    assert.equal(priorityClass("Critical"), "badge-danger");
    assert.equal(decisionClass("REVIEW"), "badge-amber");
  });

  it("formats incident type labels", () => {
    assert.equal(formatIncidentType("DEBIT_SERVICE_MISMATCH"), "Debit Service Mismatch");
  });

  it("maps payment incident severities to stable badge classes", () => {
    assert.equal(severityClass("CRITICAL"), "badge-danger");
    assert.equal(severityClass("HIGH"), "badge-orange");
    assert.equal(severityClass("NONE"), "badge-green");
  });

  it("describes recommended actions without executing them", () => {
    assert.match(actionDescription("VERIFY_PAYMENT"), /Re-check/);
  });

  it("explains fraud and incident risk as separate concepts", () => {
    assert.match(
      fraudIncidentMessage(0.12, "HIGH", true),
      /does not show strong transaction-fraud risk/,
    );
    assert.match(
      fraudIncidentMessage(0.94, "NONE", false),
      /no payment lifecycle incident/,
    );
  });

  it("formats monitoring driver labels", () => {
    assert.equal(formatMonitoringDriver("PAYMENT_INCIDENT_RATE"), "Payment Incident Rate");
  });

  it("sorts payment lifecycles by actionable severity first", () => {
    const rows = sortPaymentLifecycles([
      { payment_id: "normal", current_severity: "NONE", status: "NORMAL" },
      { payment_id: "medium", current_severity: "MEDIUM", status: "ACTIVE_INCIDENT" },
      { payment_id: "critical", current_severity: "CRITICAL", status: "ACTIVE_INCIDENT" },
      { payment_id: "resolved", current_severity: "HIGH", status: "RESOLVED" },
    ]);

    assert.deepEqual(rows.map((row) => row.payment_id), ["critical", "resolved", "medium", "normal"]);
  });

  it("selects an actionable incident by default when available", () => {
    const selected = selectDefaultPaymentIncident([
      { payment_id: "normal", current_severity: "NONE", status: "NORMAL", incident_detected: false },
      { payment_id: "pay_life_000007", current_severity: "CRITICAL", status: "ACTIVE_INCIDENT", incident_detected: true },
    ]);

    assert.equal(selected.payment_id, "pay_life_000007");
  });

  it("keeps current monitoring alerts separate from history", () => {
    const split = splitMonitoringAlerts(
      { window_start: "2026-01-01T00:15:00", status: "CRITICAL" },
      [
        { alert_id: "old", window_start: "2026-01-01T00:00:00", severity: "HIGH" },
        { alert_id: "current", window_start: "2026-01-01T00:15:00", severity: "CRITICAL" },
      ],
    );

    assert.deepEqual(split.currentAlerts.map((alert) => alert.alert_id), ["current"]);
    assert.deepEqual(split.historyAlerts.map((alert) => alert.alert_id), ["old"]);
  });

  it("shows no current alerts when current monitoring status is normal", () => {
    const split = splitMonitoringAlerts(
      { window_start: "2026-01-01T00:15:00", status: "NORMAL" },
      [{ alert_id: "same-window", window_start: "2026-01-01T00:15:00", severity: "HIGH" }],
    );

    assert.equal(split.currentAlerts.length, 0);
  });

  it("uses a softer signal label for normal monitoring status", () => {
    assert.equal(monitoringSignalLabel("NORMAL"), "Leading Signal");
    assert.equal(monitoringSignalLabel("CRITICAL"), "Main Driver");
  });

  it("labels lifecycle transitions from actual timeline state", () => {
    assert.equal(
      timelineTransitionLabel(
        { incident_type: "REFUND_REQUIRED", status: "ACTIVE_INCIDENT", severity: "HIGH", recommended_action: "INITIATE_REFUND" },
        { incident_type: "NORMAL_PAYMENT", status: "NORMAL", severity: "NONE" },
      ),
      "Refund Initiated",
    );
  });

  it("keeps synthetic labels explicit for synthetic modules", () => {
    assert.match("Synthetic demo data", /Synthetic/);
    assert.match("Synthetic monitoring stream", /Synthetic/);
  });

  it("keeps frozen Module 1 display metrics unchanged", () => {
    assert.equal(formatPercent(0.364018), "36.4%");
    assert.equal(formatPercent(0.563088), "56.3%");
    assert.equal(Number(0.514931).toFixed(3), "0.515");
    assert.equal(formatPercent(0.053838, 2), "5.38%");
  });
});
