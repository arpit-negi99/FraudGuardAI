import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
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
