"""Run SEC retrieval and grounded-answer evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluator import (
    DEFAULT_EVAL_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_SUMMARY_PATH,
    run_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SEC evaluation harness.")
    parser.add_argument("--dataset", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_PATH))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", default="extractive")
    args = parser.parse_args()

    evaluation = run_evaluation(
        dataset_path=args.dataset,
        results_path=args.results,
        summary_path=args.summary,
        top_k=args.top_k,
        provider_name=args.provider,
    )
    metrics = evaluation["summary"]["metrics"]
    compact = {
        "questions": evaluation["summary"]["question_count"],
        "answerable": evaluation["summary"]["answerable_count"],
        "no_answer": evaluation["summary"]["no_answer_count"],
        "keyword_hit_rate": round(metrics["keyword_hit_rate"], 4),
        "citation_coverage": round(metrics["citation_coverage"], 4),
        "source_url_coverage": round(metrics["source_url_coverage"], 4),
        "weak_evidence_rate": round(metrics["weak_evidence_rate"], 4),
        "no_answer_handling": round(metrics["no_answer_handling"], 4),
        "latency_ms_avg": round(metrics["latency_ms_avg"], 2),
        "results_path": str(Path(args.results)),
        "summary_path": str(Path(args.summary)),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
