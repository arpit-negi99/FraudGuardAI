export function MetricCard({ label, value, note, tone = "default" }) {
  return (
    <div className={`metric-card tone-${tone}`}>
      <p>{label}</p>
      <strong>{value}</strong>
      {note ? <span>{note}</span> : null}
    </div>
  );
}
