import { useEffect, useState } from "react";
import { MetricCard } from "../components/MetricCard";
import { StateBlock } from "../components/StateBlock";
import { api } from "../services/api";
import { formatPercent } from "../utils/format";

export function Policy({ presets, loading, error }) {
  const [selected, setSelected] = useState("balanced");
  const [reviewCost, setReviewCost] = useState(5);
  const [fraudMultiplier, setFraudMultiplier] = useState(1);
  const [simulation, setSimulation] = useState(null);
  const [simError, setSimError] = useState("");

  useEffect(() => {
    api
      .policySimulate({ review_cost: reviewCost, fraud_loss_multiplier: fraudMultiplier })
      .then(setSimulation)
      .catch((err) => setSimError(err.message));
  }, [reviewCost, fraudMultiplier]);

  const current = presets.find((preset) => preset.key === selected) || presets[1];

  return (
    <StateBlock loading={loading} error={error}>
      <section className="page-heading">
        <h2>Policy</h2>
        <p>Choose a review strategy. The default operating policy remains Balanced at threshold 0.60.</p>
      </section>
      <section className="policy-cards">
        {presets.map((preset) => (
          <button key={preset.key} className={selected === preset.key ? "selected" : ""} onClick={() => setSelected(preset.key)}>
            <strong>{preset.name}</strong>
            <span>{preset.description}</span>
            <small>{preset.tradeoff}</small>
            <b>{Number(preset.threshold).toFixed(2)}</b>
          </button>
        ))}
      </section>
      {current ? (
        <details className="technical-details">
          <summary>Advanced Metrics</summary>
          <div className="metrics-grid">
            <MetricCard label="Threshold" value={Number(current.metrics.threshold).toFixed(2)} />
            <MetricCard label="Validation Precision" value={formatPercent(current.metrics.precision)} />
            <MetricCard label="Validation Recall" value={formatPercent(current.metrics.recall)} />
            <MetricCard label="Validation F1" value={formatPercent(current.metrics.f1)} />
            <MetricCard label="Validation Review Rate" value={formatPercent(current.metrics.review_rate)} />
            <MetricCard label="FP / FN" value={`${current.metrics.false_positive} / ${current.metrics.false_negative}`} />
          </div>
        </details>
      ) : null}
      <section className="panel">
        <div className="section-title">
          <h3>Cost Simulation</h3>
          <p>Scenario simulation only - not actual merchant savings.</p>
        </div>
        <div className="filters">
          <label>
            Review cost
            <input type="number" min="0" value={reviewCost} onChange={(event) => setReviewCost(Number(event.target.value))} />
          </label>
          <label>
            Fraud loss multiplier
            <input type="number" min="0" step="0.1" value={fraudMultiplier} onChange={(event) => setFraudMultiplier(Number(event.target.value))} />
          </label>
        </div>
        {simError ? <div className="state-block error">{simError}</div> : null}
        {simulation ? (
          <div className="metrics-grid">
            <MetricCard label="Modeled Review Cost" value={Number(simulation.modeled_review_cost).toFixed(0)} />
            <MetricCard label="Modeled Missed Fraud Cost" value={Number(simulation.modeled_missed_fraud_cost).toFixed(0)} />
            <MetricCard label="Total Modeled Cost" value={Number(simulation.total_modeled_cost).toFixed(0)} tone="warn" />
            <MetricCard label="Scenario Source" value={simulation.scenario} />
          </div>
        ) : null}
      </section>
    </StateBlock>
  );
}
