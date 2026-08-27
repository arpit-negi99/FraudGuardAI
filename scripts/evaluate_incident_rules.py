from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.incidents.evaluation import (
    collect_error_examples,
    evaluate_dataset,
    precedence_artifact,
    save_error_examples,
    save_evaluation_artifacts,
    simulator_ground_truth_is_independent,
)
from src.incidents.simulator import (
    STRESS_RANDOM_SEED,
    STRESS_ROW_COUNT,
    generate_payment_incident_events,
    generate_payment_incident_stress_events,
    validate_no_impossible_combinations,
)


STANDARD_DATA_PATH = ROOT / "data/synthetic/payment_incidents.csv"
STRESS_DATA_PATH = ROOT / "data/synthetic/payment_incidents_stress.csv"
RESULTS_DIR = ROOT / "artifacts/results"


def main() -> None:
    if not simulator_ground_truth_is_independent():
        raise RuntimeError("Synthetic ground truth generator appears to call the detector.")

    standard = _load_or_generate_standard()
    stress = generate_payment_incident_stress_events(
        row_count=STRESS_ROW_COUNT,
        random_seed=STRESS_RANDOM_SEED,
    )
    validate_no_impossible_combinations(standard)
    validate_no_impossible_combinations(stress)
    STRESS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    stress.to_csv(STRESS_DATA_PATH, index=False)

    standard_result = evaluate_dataset(standard, "standard_synthetic")
    stress_result = evaluate_dataset(stress, "stress_synthetic")

    save_evaluation_artifacts(
        standard_result,
        RESULTS_DIR / "payment_incident_rule_metrics.json",
        RESULTS_DIR / "payment_incident_rule_per_class.csv",
        RESULTS_DIR / "payment_incident_rule_confusion_matrix.png",
    )
    save_evaluation_artifacts(
        stress_result,
        RESULTS_DIR / "payment_incident_stress_metrics.json",
        RESULTS_DIR / "payment_incident_stress_per_class.csv",
        RESULTS_DIR / "payment_incident_stress_confusion_matrix.png",
    )
    save_error_examples(
        collect_error_examples(standard, stress),
        RESULTS_DIR / "payment_incident_error_examples.json",
    )
    (RESULTS_DIR / "payment_incident_rule_precedence.json").write_text(
        json.dumps(precedence_artifact(), indent=2),
        encoding="utf-8",
    )

    print("FraudGuard AI - Payment Incident Rule Evaluation")
    print("Ground truth source: independently scenario-generated synthetic payment-event data.")
    print()
    _print_summary("Standard synthetic evaluation", standard_result)
    print()
    _print_summary("Stress synthetic evaluation", stress_result)
    print()
    print("Anti-circularity: synthetic ground truth generation does not call the detector.")
    print("Recommendation:", _recommendation(standard_result, stress_result))


def _load_or_generate_standard() -> pd.DataFrame:
    if STANDARD_DATA_PATH.exists():
        return pd.read_csv(STANDARD_DATA_PATH)
    data = generate_payment_incident_events()
    STANDARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(STANDARD_DATA_PATH, index=False)
    return data


def _print_summary(title: str, result: dict) -> None:
    binary = result["binary"]
    print(title)
    print(f"Rows: {result['rows']}")
    print(f"Precision: {binary['precision']:.6f}")
    print(f"Recall: {binary['recall']:.6f}")
    print(f"F1: {binary['f1']:.6f}")
    print(f"TP: {binary['true_positive']}")
    print(f"FP: {binary['false_positive']}")
    print(f"TN: {binary['true_negative']}")
    print(f"FN: {binary['false_negative']}")
    print(f"Macro F1: {result['macro_f1']:.6f}")
    print(f"Weighted F1: {result['weighted_f1']:.6f}")
    print("Per-class F1:")
    for row in result["per_class"]:
        print(f"  {row['incident_type']}: {row['f1']:.6f} (support {row['support']})")


def _recommendation(standard_result: dict, stress_result: dict) -> str:
    standard_f1 = standard_result["binary"]["f1"]
    stress_f1 = stress_result["binary"]["f1"]
    if standard_f1 >= 0.95 and stress_f1 >= 0.90:
        return "Keep deterministic rules; ML is not justified by these synthetic evaluations."
    if standard_f1 >= 0.90 and stress_f1 < 0.90:
        return "Add hybrid ML risk scorer only after retaining rules for hard operational constraints."
    return "Improve data/schema first before adding ML."


if __name__ == "__main__":
    main()
