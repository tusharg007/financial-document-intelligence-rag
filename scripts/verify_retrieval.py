"""Smoke-test the production retrieval pipeline against real SEC indexes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.pipeline import REQUIRED_METADATA_FIELDS, get_retrieval_pipeline


SMOKE_TESTS = [
    {
        "name": "Apple risk factors",
        "query": "What are Apple's main risk factors?",
        "filters": {"ticker": "AAPL", "section": "Risk Factors"},
        "top_k": 5,
        "expected_ticker": "AAPL",
    },
    {
        "name": "Microsoft revenue",
        "query": "What does Microsoft say about revenue?",
        "filters": {"ticker": "MSFT", "form_type": "10-K"},
        "top_k": 5,
        "expected_ticker": "MSFT",
    },
    {
        "name": "Tesla risk factors",
        "query": "Tesla risk factors",
        "filters": {"ticker": "TSLA", "section": "Risk Factors"},
        "top_k": 5,
        "expected_ticker": "TSLA",
    },
    {
        "name": "Nvidia business",
        "query": "Nvidia business",
        "filters": {"ticker": "NVDA"},
        "top_k": 5,
        "expected_ticker": "NVDA",
    },
    {
        "name": "JPM financial statements",
        "query": "JPM financial statements",
        "filters": {"ticker": "JPM"},
        "top_k": 5,
        "expected_ticker": "JPM",
    },
]


def main() -> None:
    pipeline = get_retrieval_pipeline()
    failures = []
    reports = []

    for case in SMOKE_TESTS:
        results = pipeline.retrieve(case["query"], top_k=case["top_k"], filters=case["filters"])
        doc_ids = [result.get("doc_id") for result in results]
        case_report = {
            "name": case["name"],
            "query": case["query"],
            "filters": case["filters"],
            "result_count": len(results),
            "doc_ids": doc_ids,
            "top_tickers": [result.get("ticker", "") for result in results[:3]],
        }

        if not results:
            failures.append(f"{case['name']}: no results returned")
        if len(set(doc_ids)) != len(doc_ids):
            failures.append(f"{case['name']}: duplicate doc_id in top results")
        if case.get("expected_ticker") and any(
            result.get("ticker") != case["expected_ticker"] for result in results
        ):
            failures.append(f"{case['name']}: filtered results did not preserve expected ticker {case['expected_ticker']}")
        for result in results:
            if not result.get("source_url"):
                failures.append(f"{case['name']}: source_url missing for doc_id {result.get('doc_id')}")
                break
            missing = [field for field in ["ticker", "company", "section", "source_url"] if not result.get(field)]
            if missing:
                failures.append(
                    f"{case['name']}: missing metadata {missing} for doc_id {result.get('doc_id')}"
                )
                break
        reports.append(case_report)

    output = {
        "smoke tests": reports,
        "required metadata fields": REQUIRED_METADATA_FIELDS,
        "failures": failures,
    }
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit("Retrieval verification failed.")


if __name__ == "__main__":
    main()
