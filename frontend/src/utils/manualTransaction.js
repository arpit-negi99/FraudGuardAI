export function buildManualTransactionPayload(form) {
  const transactionId = Number(form.transactionId);
  const amount = Number(form.amount);
  const transactionDt = Number(form.transactionDt || 1);

  return {
    TransactionID: Number.isFinite(transactionId) ? transactionId : Date.now(),
    TransactionDT: Number.isFinite(transactionDt) ? transactionDt : 1,
    TransactionAmt: Number.isFinite(amount) ? amount : 0,
    ProductCD: form.productCode || "W",
    card4: form.cardNetwork || null,
    card6: form.cardType || null,
    P_emaildomain: form.emailDomain || null,
    addr1: numberOrNull(form.billingRegion),
    dist1: numberOrNull(form.distance),
    C1: numberOrNull(form.c1),
    C13: numberOrNull(form.c13),
    merchant_id: "merchant_demo_001",
    payment_id: `manual_${Date.now()}`,
  };
}

export function numberOrNull(value) {
  if (value === "" || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}
