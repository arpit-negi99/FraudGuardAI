export function formatPercent(value, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

export function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

export function formatAmount(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

export function priorityClass(priority) {
  const key = String(priority || "").toLowerCase();
  if (key === "critical") return "badge-danger";
  if (key === "high") return "badge-amber";
  if (key === "review") return "badge-blue";
  if (key === "allow" || key === "low") return "badge-green";
  return "badge-slate";
}

export function decisionClass(decision) {
  return decision === "REVIEW" ? "badge-amber" : "badge-green";
}

export function severityClass(severity) {
  const key = String(severity || "").toUpperCase();
  if (key === "CRITICAL") return "badge-danger";
  if (key === "HIGH") return "badge-orange";
  if (key === "MEDIUM") return "badge-amber";
  if (key === "LOW") return "badge-blue";
  if (key === "NONE") return "badge-green";
  return "badge-slate";
}

export function formatIncidentType(value) {
  return String(value || "NORMAL_PAYMENT")
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatAction(value) {
  return formatIncidentType(value || "NO_ACTION");
}

export function formatMonitoringDriver(value) {
  if (!value || value === "NONE") return "None";
  return String(value)
    .replace("_RATE", " rate")
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function severityRank(severity) {
  return {
    CRITICAL: 5,
    HIGH: 4,
    MEDIUM: 3,
    LOW: 2,
    ELEVATED: 2,
    NONE: 1,
    NORMAL: 0,
  }[String(severity || "NONE").toUpperCase()] ?? 0;
}

export function statusRank(status) {
  return {
    ACTIVE_INCIDENT: 3,
    RESOLVING: 2,
    RESOLVED: 1,
    NORMAL: 0,
  }[String(status || "NORMAL").toUpperCase()] ?? 0;
}

export function sortPaymentLifecycles(rows = []) {
  return [...rows].sort((a, b) => {
    const severityDelta = severityRank(b.current_severity) - severityRank(a.current_severity);
    if (severityDelta) return severityDelta;
    const statusDelta = statusRank(b.status) - statusRank(a.status);
    if (statusDelta) return statusDelta;
    return String(a.payment_id || "").localeCompare(String(b.payment_id || ""));
  });
}

export function selectDefaultPaymentIncident(rows = []) {
  const sorted = sortPaymentLifecycles(rows);
  const preferred = ["pay_life_000007", "pay_life_000017", "pay_life_000033", "pay_life_000004"];
  return (
    sorted.find((row) => preferred.includes(row.payment_id) && row.status !== "NORMAL") ||
    sorted.find((row) => row.incident_detected || row.status === "ACTIVE_INCIDENT" || row.status === "RESOLVING") ||
    sorted.find((row) => row.status === "RESOLVED" && severityRank(row.highest_severity_observed) >= severityRank("HIGH")) ||
    sorted[0] ||
    null
  );
}

export function splitMonitoringAlerts(current, alerts = []) {
  const currentWindow = current?.window_start;
  const currentStatus = String(current?.status || "NORMAL").toUpperCase();
  const nonNormal = alerts.filter((alert) => String(alert.severity || "NORMAL").toUpperCase() !== "NORMAL");
  const currentAlerts =
    currentStatus === "NORMAL" || !currentWindow
      ? []
      : nonNormal.filter((alert) => alert.window_start === currentWindow);
  const historyAlerts = nonNormal.filter((alert) => alert.window_start !== currentWindow);
  return {
    currentAlerts,
    historyAlerts,
  };
}

export function monitoringSignalLabel(status) {
  return String(status || "NORMAL").toUpperCase() === "NORMAL" ? "Leading Signal" : "Main Driver";
}

export function timelineTransitionLabel(item, previous) {
  const incident = item?.incident_type || "NORMAL_PAYMENT";
  const status = item?.status || "NORMAL";
  const severity = item?.severity || "NONE";
  const action = item?.recommended_action || "NO_ACTION";
  const previousSeverity = previous?.severity || "NONE";
  const previousIncident = previous?.incident_type || "NORMAL_PAYMENT";

  if (status === "RESOLVED") return "Resolved";
  if (status === "RESOLVING") return "Incident Resolving";
  if (action === "INITIATE_REFUND") return "Refund Initiated";
  if (severityRank(severity) > severityRank(previousSeverity) && previousIncident !== "NORMAL_PAYMENT") {
    return `Severity Escalated To ${formatIncidentType(severity)}`;
  }
  if (incident !== "NORMAL_PAYMENT" && previousIncident === "NORMAL_PAYMENT") return "Incident Detected";
  if (incident !== "NORMAL_PAYMENT") return "Incident Still Active";
  return "Normal Event";
}

export const actionDescriptions = {
  VERIFY_PAYMENT: "Re-check the payment state before taking further action.",
  CHECK_ORDER: "Verify whether the associated order or service was completed.",
  INITIATE_REFUND: "Review the payment and begin the refund process if fulfilment cannot be completed.",
  CONTACT_CUSTOMER: "Contact the customer with the current payment resolution status.",
  ESCALATE_REVIEW: "Escalate this incident to a risk or payment operations analyst.",
  MONITOR: "Continue monitoring while the payment state resolves.",
  NO_ACTION: "No operational intervention is currently recommended.",
};

export function actionDescription(action) {
  return actionDescriptions[action] || actionDescriptions.NO_ACTION;
}

export function fraudRiskBand(score) {
  const value = Number(score || 0);
  if (value >= 0.75) return "HIGH";
  if (value >= 0.4) return "MEDIUM";
  return "LOW";
}

export function fraudIncidentMessage(score, severity, incidentDetected) {
  const fraudBand = fraudRiskBand(score);
  if (incidentDetected && fraudBand === "LOW" && ["HIGH", "CRITICAL"].includes(severity)) {
    return "This payment does not show strong transaction-fraud risk, but its payment lifecycle requires attention.";
  }
  if (!incidentDetected && fraudBand === "HIGH") {
    return "This payment has a high transaction-fraud signal, but no payment lifecycle incident is currently detected.";
  }
  if (incidentDetected) {
    return "Payment lifecycle status requires operational review. Treat fraud risk as a separate signal.";
  }
  return "No payment lifecycle incident is currently detected. Fraud risk remains a separate transaction signal.";
}
