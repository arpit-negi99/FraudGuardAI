import { MetricCard } from "../components/MetricCard";
import { DataTable } from "../components/DataTable";
import { StateBlock } from "../components/StateBlock";
import { formatPercent } from "../utils/format";

export function ReviewQueue({ rows, loading, error, onOpen }) {
  const critical = rows.filter((row) => row.priority === "Critical").length;
  const high = rows.filter((row) => row.priority === "High").length;
  const averageRisk = rows.length
    ? rows.reduce((sum, row) => sum + row.risk_score, 0) / rows.length
    : 0;
  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Review Queue</h2>
        <p>Transactions recommended for analyst review, ranked by risk.</p>
      </section>
      <section className="metrics-grid">
        <MetricCard label="Total Waiting" value={rows.length} tone="warn" />
        <MetricCard label="Critical" value={critical} tone="danger" />
        <MetricCard label="High" value={high} tone="warn" />
        <MetricCard label="Average Risk" value={formatPercent(averageRisk)} />
      </section>
      <div className="panel">
        <DataTable rows={rows} onOpen={onOpen} showSignal />
      </div>
    </StateBlock>
  );
}
