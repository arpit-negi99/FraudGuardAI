import { useMemo, useState } from "react";
import { DataTable } from "../components/DataTable";
import { StateBlock } from "../components/StateBlock";

export function Transactions({ transactions, loading, error, onOpen }) {
  const [query, setQuery] = useState("");
  const [decision, setDecision] = useState("All");
  const [priority, setPriority] = useState("All");
  const [minRisk, setMinRisk] = useState(0);
  const [sort, setSort] = useState("risk_desc");

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

  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Transactions</h2>
        <p>Search and filter the packaged demo transaction sample.</p>
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
