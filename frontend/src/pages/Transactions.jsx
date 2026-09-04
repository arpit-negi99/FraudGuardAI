import { useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import { DataTable } from "../components/DataTable";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { api } from "../services/api";
import { formatAmount, formatPercent } from "../utils/format";
import { buildManualTransactionPayload } from "../utils/manualTransaction";

const initialForm = {
  transactionId: "9000001",
  transactionDt: "86400",
  amount: "2499",
  productCode: "W",
  cardNetwork: "visa",
  cardType: "debit",
  emailDomain: "gmail.com",
  billingRegion: "204",
  distance: "12",
  c1: "1",
  c13: "2",
};

export function Transactions({ transactions, loading, error, onOpen }) {
  const [query, setQuery] = useState("");
  const [decision, setDecision] = useState("All");
  const [priority, setPriority] = useState("All");
  const [minRisk, setMinRisk] = useState(0);
  const [sort, setSort] = useState("risk_desc");
  const [form, setForm] = useState(initialForm);
  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState("");
  const [scoreResult, setScoreResult] = useState(null);

  const rows = useMemo(() => {
    let filtered = transactions.filter((row) =>
      String(row.transaction_id).includes(query.trim()),
    );
    if (decision !== "All") filtered = filtered.filter((row) => row.decision === decision);
    if (priority !== "All") filtered = filtered.filter((row) => row.priority === priority);
    filtered = filtered.filter((row) => row.risk_score >= minRisk);
    return filtered.sort((a, b) =>
      sort === "risk_desc" ? b.risk_score - a.risk_score : a.risk_score - b.risk_score,
    );
  }, [transactions, query, decision, priority, minRisk, sort]);

  const updateForm = (field) => (event) => {
    setForm((current) => ({ ...current, [field]: event.target.value }));
  };

  const scoreManualTransaction = async (event) => {
    event.preventDefault();
    setScoring(true);
    setScoreError("");
    setScoreResult(null);
    try {
      const result = await api.predict(buildManualTransactionPayload(form), true);
      setScoreResult(result);
    } catch (err) {
      setScoreError(err.message || "Unable to score this transaction.");
    } finally {
      setScoring(false);
    }
  };

  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Transactions</h2>
        <p>Score a new transaction with the backend model, then compare it with packaged demo examples.</p>
      </section>
      <section className="panel manual-score-panel">
        <div className="section-title">
          <h3>Score New Transaction</h3>
          <p>Runs through the FastAPI backend and frozen XGBoost pipeline.</p>
        </div>
        <form className="manual-transaction-form" onSubmit={scoreManualTransaction}>
          <label>
            Transaction ID
            <input value={form.transactionId} onChange={updateForm("transactionId")} inputMode="numeric" />
          </label>
          <label>
            Amount
            <input value={form.amount} onChange={updateForm("amount")} inputMode="decimal" />
          </label>
          <label>
            Product
            <select value={form.productCode} onChange={updateForm("productCode")}>
              <option value="W">W</option>
              <option value="C">C</option>
              <option value="R">R</option>
              <option value="H">H</option>
              <option value="S">S</option>
            </select>
          </label>
          <label>
            Card Network
            <select value={form.cardNetwork} onChange={updateForm("cardNetwork")}>
              <option value="visa">Visa</option>
              <option value="mastercard">Mastercard</option>
              <option value="discover">Discover</option>
              <option value="american express">American Express</option>
            </select>
          </label>
          <label>
            Card Type
            <select value={form.cardType} onChange={updateForm("cardType")}>
              <option value="debit">Debit</option>
              <option value="credit">Credit</option>
            </select>
          </label>
          <label>
            Email Domain
            <input value={form.emailDomain} onChange={updateForm("emailDomain")} />
          </label>
          <label>
            Billing Region
            <input value={form.billingRegion} onChange={updateForm("billingRegion")} inputMode="numeric" />
          </label>
          <label>
            Distance
            <input value={form.distance} onChange={updateForm("distance")} inputMode="decimal" />
          </label>
          <label>
            C1 Signal
            <input value={form.c1} onChange={updateForm("c1")} inputMode="decimal" />
          </label>
          <label>
            C13 Signal
            <input value={form.c13} onChange={updateForm("c13")} inputMode="decimal" />
          </label>
          <div className="manual-form-actions">
            <button type="submit" disabled={scoring}>{scoring ? "Scoring..." : "Score Transaction"}</button>
          </div>
        </form>
        {scoreError ? <p className="form-error">{scoreError}</p> : null}
        {scoreResult ? (
          <div className="manual-result">
            <div className="case-heading compact-case-heading">
              <div>
                <p>Model Decision</p>
                <h2>{scoreResult.decision}</h2>
              </div>
              <Badge type="decision">{scoreResult.decision}</Badge>
            </div>
            <div className="metrics-grid manual-result-grid">
              <MetricCard label="Risk Score" value={formatPercent(scoreResult.risk_score)} tone={scoreResult.decision === "REVIEW" ? "warn" : "default"} />
              <MetricCard label="Threshold" value={formatPercent(scoreResult.threshold)} note="Frozen policy" />
              <MetricCard label="Amount" value={formatAmount(form.amount)} />
              <MetricCard label="Action" value={scoreResult.decision === "REVIEW" ? "Manual Review" : "Allow"} tone={scoreResult.decision === "REVIEW" ? "warn" : "default"} />
            </div>
            <div className="risk-bar">
              <span style={{ width: `${Math.min(100, Math.max(0, scoreResult.risk_score * 100))}%` }} />
            </div>
            <div className="signal-list compact-signal-list">
              {(scoreResult.explanation?.top_risk_factors || []).slice(0, 3).map((item) => (
                <div className="signal-row" key={item.feature}>
                  <div>
                    <strong>{item.feature}</strong>
                    <span>Risk contributor</span>
                  </div>
                  <b>{Number(item.shap_value) >= 0 ? "+" : ""}{Number(item.shap_value || 0).toFixed(3)}</b>
                </div>
              ))}
            </div>
            {scoreResult.warnings?.length ? (
              <details className="technical-details inline-technical-details">
                <summary>Input warnings</summary>
                <pre>{JSON.stringify(scoreResult.warnings, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        ) : null}
      </section>
      <section className="filters">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search TransactionID" />
        <select value={decision} onChange={(event) => setDecision(event.target.value)}>
          <option>All</option>
          <option>REVIEW</option>
          <option>ALLOW</option>
        </select>
        <select value={priority} onChange={(event) => setPriority(event.target.value)}>
          <option>All</option>
          <option>Critical</option>
          <option>High</option>
          <option>Review</option>
          <option>Medium</option>
          <option>Low</option>
        </select>
        <select value={sort} onChange={(event) => setSort(event.target.value)}>
          <option value="risk_desc">Risk high to low</option>
          <option value="risk_asc">Risk low to high</option>
        </select>
        <label className="range-label">
          Minimum risk {Math.round(minRisk * 100)}%
          <input type="range" min="0" max="1" step="0.05" value={minRisk} onChange={(event) => setMinRisk(Number(event.target.value))} />
        </label>
      </section>
      <StateBlock empty={!rows.length}>
        <div className="panel">
          <DataTable rows={rows.slice(0, 25)} onOpen={onOpen} />
        </div>
      </StateBlock>
    </StateBlock>
  );
}
