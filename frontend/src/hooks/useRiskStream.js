import { useEffect, useState } from "react";
import { monitoringStreamUrl } from "../services/api";

export function useRiskStream({ enabled, merchantId = "merchant_demo_001" }) {
  const [state, setState] = useState(null);
  const [connectionState, setConnectionState] = useState(enabled ? "connecting" : "disabled");

  useEffect(() => {
    if (!enabled || typeof EventSource === "undefined") {
      setConnectionState(enabled ? "fallback" : "disabled");
      return undefined;
    }
    const source = new EventSource(monitoringStreamUrl(merchantId));
    setConnectionState("connecting");
    source.addEventListener("open", () => setConnectionState("live"));
    source.addEventListener("risk_state", (event) => {
      setState(JSON.parse(event.data));
      setConnectionState("live");
    });
    source.addEventListener("heartbeat", () => setConnectionState("live"));
    source.addEventListener("error", () => setConnectionState("fallback"));
    return () => {
      source.close();
    };
  }, [enabled, merchantId]);

  return { state, connectionState };
}
