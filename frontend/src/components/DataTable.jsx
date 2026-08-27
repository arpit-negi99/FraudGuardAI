import { Badge } from "./Badge";
import { formatAmount, formatPercent } from "../utils/format";

export function DataTable({ rows, onOpen, showSignal = false }) {
  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            {showSignal ? <th>Priority</th> : null}
            <th>Transaction ID</th>
            <th>Amount</th>
            <th>Risk</th>
            <th>Decision</th>
            {!showSignal ? <th>Priority</th> : null}
            {showSignal ? <th>Top Signal</th> : null}
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.transaction_id}>
              {showSignal ? (
                <td>
                  <Badge>{row.priority}</Badge>
                </td>
              ) : null}
              <td>#{row.transaction_id}</td>
              <td>{formatAmount(row.amount)}</td>
              <td>{formatPercent(row.risk_score)}</td>
              <td>
                <Badge type="decision">{row.decision}</Badge>
              </td>
              {!showSignal ? (
                <td>
                  <Badge>{row.priority}</Badge>
                </td>
              ) : null}
              {showSignal ? <td>{row.top_signal || "n/a"}</td> : null}
              <td>
                <button className="table-action" onClick={() => onOpen(row.transaction_id)}>
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
