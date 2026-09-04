export function isStreamMode(health) {
  return health?.stream_mode === "stream" || health?.streaming_enabled === true;
}

export function streamStatusLabel(connectionState, streamEnabled) {
  if (!streamEnabled) return "Local polling";
  if (connectionState === "live") return "Live stream";
  if (connectionState === "connecting") return "Connecting";
  return "Polling fallback";
}

export function normalizeStreamCurrent(state) {
  if (!state) return null;
  const metrics = state.current_metrics || {};
  return {
    ...state,
    current_review_rate: metrics.review_rate ?? state.current_review_rate ?? 0,
    current_incident_rate: metrics.payment_incident_rate ?? state.current_incident_rate ?? 0,
  };
}
