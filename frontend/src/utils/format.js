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
  return String(value || "NO_ACTION").replaceAll("_", " ");
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
