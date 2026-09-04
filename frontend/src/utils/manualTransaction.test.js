import assert from "node:assert/strict";
import test from "node:test";

import { buildManualTransactionPayload, numberOrNull } from "./manualTransaction.js";

test("numberOrNull preserves numeric values and maps blanks to null", () => {
  assert.equal(numberOrNull("12.5"), 12.5);
  assert.equal(numberOrNull(""), null);
  assert.equal(numberOrNull("not-number"), null);
});

test("buildManualTransactionPayload creates model input fields", () => {
  const payload = buildManualTransactionPayload({
    transactionId: "9001",
    transactionDt: "12345",
    amount: "4999",
    productCode: "W",
    cardNetwork: "visa",
    cardType: "debit",
    emailDomain: "gmail.com",
    billingRegion: "204",
    distance: "",
    c1: "2",
    c13: "8",
  });

  assert.equal(payload.TransactionID, 9001);
  assert.equal(payload.TransactionAmt, 4999);
  assert.equal(payload.card4, "visa");
  assert.equal(payload.dist1, null);
  assert.equal(payload.C13, 8);
  assert.equal(payload.merchant_id, "merchant_demo_001");
});
