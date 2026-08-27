import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  actionDescription,
  decisionClass,
  formatAmount,
  formatIncidentType,
  formatPercent,
  fraudIncidentMessage,
  priorityClass,
  severityClass,
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
});
