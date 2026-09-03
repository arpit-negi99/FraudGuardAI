import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

export function RiskDistributionChart({ data }) {
  return (
    <div className="chart-card">
      <div className="section-title">
        <h3>Risk Distribution</h3>
        <p>Current demo transaction sample</p>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="risk_band" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="transactions" fill="#0f766e" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RiskActivityChart({ rows }) {
  const data = rows.map((row, index) => ({
    index: index + 1,
    risk: Number(row.risk_score),
  }));
  return (
    <div className="chart-card">
      <div className="section-title">
        <h3>Risk Activity</h3>
        <p>Sequence order from the current scored sample</p>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id="riskFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#d97706" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#d97706" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="index" />
          <YAxis domain={[0, 1]} />
          <Tooltip />
          <Area type="monotone" dataKey="risk" stroke="#d97706" fill="url(#riskFill)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IncidentSeverityChart({ data }) {
  return (
    <div className="chart-card">
      <div className="section-title">
        <h3>Payment Operations Risk</h3>
        <p>Incidents by evaluated severity</p>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="severity" />
          <YAxis allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="payments" fill="#ea580c" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function IncidentTypeChart({ data }) {
  return (
    <div className="chart-card">
      <div className="section-title">
        <h3>Incidents By Type</h3>
        <p>Deterministic lifecycle rule output</p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" margin={{ left: 48 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" allowDecimals={false} />
          <YAxis type="category" dataKey="label" width={150} />
          <Tooltip />
          <Bar dataKey="payments" fill="#0f766e" radius={[0, 6, 6, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MonitoringRateChart({ rows }) {
  const data = rows.map((row, index) => ({
    ...row,
    index: index + 1,
    review_rate_percent: Number(row.review_rate || 0) * 100,
    incident_rate_percent: Number(row.payment_incident_rate || 0) * 100,
  }));
  return (
    <div className="chart-card">
      <div className="section-title">
        <h3>Risk Trend</h3>
        <p>Review rate and payment incident rate by 15-minute window</p>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="index" />
          <YAxis unit="%" />
          <Tooltip />
          <ReferenceLine y={5} stroke="#94a3b8" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="review_rate_percent" name="Review rate" stroke="#dc2626" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="incident_rate_percent" name="Incident rate" stroke="#0f766e" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function OperationalRiskStrip({ rows }) {
  return (
    <div className="risk-strip" aria-label="Operational risk timeline">
      {rows.map((row) => (
        <span key={row.window_start} className={`risk-strip-cell status-${String(row.status).toLowerCase()}`} title={`${row.window_start}: ${row.status}`} />
      ))}
    </div>
  );
}
