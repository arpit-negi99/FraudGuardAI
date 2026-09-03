export function StateBlock({ loading, error, empty, children, emptyMessage = "No matching transactions." }) {
  if (loading) return <div className="state-block">Loading FraudGuard data...</div>;
  if (error) return <div className="state-block error">Unable to load risk data. Check that the FraudGuard API is running.</div>;
  if (empty) return <div className="state-block">{emptyMessage}</div>;
  return children;
}
