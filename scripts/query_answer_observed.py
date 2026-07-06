"""Run the grounded answer pipeline, evaluate it, and log an observability record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_eval_kit import evaluate_rag_response, log_rag_observation
from src.answering.grounded_answer import get_grounded_answerer


def _filters_from_args(args: argparse.Namespace) -> dict:
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.form_type:
        filters["form_type"] = args.form_type
    if args.section:
        filters["section"] = args.section
    return filters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ticker")
    parser.add_argument("--form-type")
    parser.add_argument("--section")
    parser.add_argument("--provider", default="extractive")
    parser.add_argument("--expected-keyword", action="append", dest="expected_keywords")
    args = parser.parse_args()

    answerer = get_grounded_answerer()
    try:
        result = answerer.answer_question(
            args.query,
            top_k=args.top_k,
            filters=_filters_from_args(args),
            provider_name=args.provider,
        )
    except RuntimeError as exc:
        raise SystemExit(
            "Indexes are not available for observed querying. Build the SEC indexes locally with "
            "`python scripts/build_indexes.py` and verify them with `python scripts/verify_indexes.py`.\n"
            f"Details: {exc}"
        ) from exc

    eval_result = evaluate_rag_response(
        question=result["question"],
        answer=result["answer"],
        contexts=result.get("retrieval_results", []),
        expected_keywords=args.expected_keywords,
    )
    log_rag_observation(
        question=result["question"],
        answer=result["answer"],
        contexts=result.get("retrieval_results", []),
        citations=result.get("citations", []),
        eval_result=eval_result,
        latency_ms=result.get("latency_ms"),
    )

    print(f"question: {result['question']}")
    print(f"answer: {result['answer']}")
    print(f"grounding_status: {result.get('grounding_status')}")
    print(f"used_provider: {result.get('used_provider')}")
    print("citations:")
    for citation in result.get("citations", []):
        print(f"  [Source {citation['source_num']}] {citation.get('source_url', '')}")
    print("metrics:")
    print(json.dumps(eval_result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
