"""Verify the SEC evaluation harness and generated reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluator import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_EVAL_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_SUMMARY_PATH,
    REQUIRED_METRICS,
    load_evaluation_questions,
    run_evaluation,
)


def main() -> None:
    failures: list[str] = []
    dataset_path = DEFAULT_EVAL_PATH
    results_path = DEFAULT_RESULTS_PATH
    summary_path = DEFAULT_SUMMARY_PATH
    comparison_path = DEFAULT_COMPARISON_PATH

    if not dataset_path.exists():
        failures.append(f"Evaluation dataset is missing: {dataset_path}")
        print(json.dumps({"failures": failures}, indent=2))
        raise SystemExit("Evaluation verification failed.")

    questions = load_evaluation_questions(dataset_path)
    if len(questions) < 15:
        failures.append(f"Expected at least 15 evaluation questions, found {len(questions)}")

    evaluation = run_evaluation(
        dataset_path=dataset_path,
        results_path=results_path,
        summary_path=summary_path,
        top_k=5,
        provider_name="extractive",
    )

    if not results_path.exists():
        failures.append(f"Evaluation results file missing: {results_path}")
    if not summary_path.exists():
        failures.append(f"Evaluation summary file missing: {summary_path}")
    if not comparison_path.exists():
        failures.append(f"Evaluation comparison file missing: {comparison_path}")

    metrics = evaluation.get("summary", {}).get("metrics", {})
    missing_metrics = [metric for metric in [
        "retrieval_result_count_avg",
        "top_k_ticker_match",
        "expected_section_match",
        "expected_form_type_match",
        "keyword_hit_rate",
        "citation_coverage",
        "source_url_coverage",
        "answer_non_empty",
        "answer_has_citations",
        "weak_evidence_rate",
        "no_answer_handling",
        "latency_ms_avg",
    ] if metric not in metrics]
    if missing_metrics:
        failures.append(f"Missing aggregate metrics: {missing_metrics}")

    if evaluation.get("required_metrics") != REQUIRED_METRICS:
        failures.append("Required metrics list in evaluation output does not match evaluator definition")

    no_answer_cases = [case for case in evaluation.get("cases", []) if not case.get("answerable", True)]
    if not no_answer_cases:
        failures.append("Expected at least one no-answer case in the evaluation dataset")
    elif any(case["metrics"].get("no_answer_handling") != 1.0 for case in no_answer_cases):
        failures.append("At least one no-answer case was not handled honestly")

    if "source_url_coverage" not in metrics:
        failures.append("Source URL coverage metric was not reported")

    output = {
        "dataset_exists": dataset_path.exists(),
        "question_count": len(questions),
        "results_exists": results_path.exists(),
        "summary_exists": summary_path.exists(),
        "comparison_exists": comparison_path.exists(),
        "reported_metrics": sorted(metrics.keys()),
        "no_answer_case_count": len(no_answer_cases),
        "source_url_coverage": metrics.get("source_url_coverage"),
        "failures": failures,
    }
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit("Evaluation verification failed.")


if __name__ == "__main__":
    main()
