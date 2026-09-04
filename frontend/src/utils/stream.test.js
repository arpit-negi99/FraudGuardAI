import assert from "node:assert/strict";
import test from "node:test";

import { isStreamMode, normalizeStreamCurrent, streamStatusLabel } from "./stream.js";

test("isStreamMode accepts explicit backend flags", () => {
  assert.equal(isStreamMode({ stream_mode: "stream" }), true);
  assert.equal(isStreamMode({ streaming_enabled: true }), true);
  assert.equal(isStreamMode({ stream_mode: "local" }), false);
});

test("streamStatusLabel reports live and fallback states", () => {
  assert.equal(streamStatusLabel("live", true), "Live stream");
  assert.equal(streamStatusLabel("fallback", true), "Polling fallback");
  assert.equal(streamStatusLabel("live", false), "Local polling");
});

test("normalizeStreamCurrent maps stream metrics to existing monitor fields", () => {
  const current = normalizeStreamCurrent({
    status: "HIGH",
    current_metrics: { review_rate: 0.2, payment_incident_rate: 0.1 },
  });
  assert.equal(current.current_review_rate, 0.2);
  assert.equal(current.current_incident_rate, 0.1);
});
