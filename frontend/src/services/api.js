const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = "FraudGuard API request failed.";
    try {
      const body = await response.json();
      message = body.detail || body.message || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  demoTransactions: () => request("/demo/transactions"),
  demoTransaction: (transactionId) => request(`/demo/transactions/${transactionId}`),
  getIncidents: (filters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "" && value !== "All") {
        params.set(key, value);
      }
    });
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return request(`/incidents${suffix}`);
  },
  getIncidentSummary: () => request("/incidents/summary"),
  getIncident: (paymentId) => request(`/incidents/${paymentId}`),
  evaluateIncident: (payload) =>
    request("/incidents/evaluate", { method: "POST", body: JSON.stringify(payload) }),
  getLifecycles: (filters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "" && value !== "All") {
        params.set(key, value);
      }
    });
    const suffix = params.toString() ? `?${params.toString()}` : "";
    return request(`/incidents/lifecycles${suffix}`);
  },
  getLifecycleSummary: () => request("/incidents/lifecycles/summary"),
  getLifecycle: (paymentId) => request(`/incidents/lifecycles/${paymentId}`),
  evaluateLifecycle: (payload) =>
    request("/incidents/lifecycles/evaluate", { method: "POST", body: JSON.stringify(payload) }),
  riskSummary: () => request("/risk/summary"),
  reviewQueue: () => request("/risk/review-queue"),
  riskSpike: () => request("/risk/spike"),
  monitoringSummary: () => request("/monitoring/summary"),
  monitoringCurrent: (scenarioType = "") =>
    request(`/monitoring/current${scenarioType ? `?scenario_type=${encodeURIComponent(scenarioType)}` : ""}`),
  monitoringWindows: (scenarioType = "", limit = 120) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (scenarioType) params.set("scenario_type", scenarioType);
    return request(`/monitoring/windows?${params.toString()}`);
  },
  monitoringAlerts: (scenarioType = "", limit = 20) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (scenarioType) params.set("scenario_type", scenarioType);
    return request(`/monitoring/alerts?${params.toString()}`);
  },
  monitoringScenarios: () => request("/monitoring/scenarios"),
  policyPresets: () => request("/policy/presets"),
  policySimulate: (payload) =>
    request("/policy/simulate", { method: "POST", body: JSON.stringify(payload) }),
  finalEvaluation: () => request("/evaluation/final"),
  predict: (transaction, includeExplanation = true) =>
    request("/predict", {
      method: "POST",
      body: JSON.stringify({ transaction, include_explanation: includeExplanation }),
    }),
  predictBatch: (transactions) =>
    request("/predict/batch", {
      method: "POST",
      body: JSON.stringify({ transactions, include_explanations: false }),
    }),
};
