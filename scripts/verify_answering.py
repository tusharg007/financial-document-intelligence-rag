"""Smoke-test grounded answer generation against real SEC indexes."""
from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answering.grounded_answer import GroundedAnswerer, get_grounded_answerer


SMOKE_TESTS = [
    {
        "name": "Apple risk factors",
        "query": "What are Apple's main risk factors?",
        "filters": {"ticker": "AAPL", "section": "Risk Factors"},
    },
    {
        "name": "Microsoft revenue",
        "query": "What does Microsoft say about revenue?",
        "filters": {"ticker": "MSFT", "form_type": "10-K"},
    },
    {
        "name": "Tesla risk factors",
        "query": "What does Tesla say about risk factors?",
        "filters": {"ticker": "TSLA", "section": "Risk Factors"},
    },
]


@contextmanager
def _without_external_keys():
    saved = {
        "GROQ_API_KEY": os.environ.pop("GROQ_API_KEY", None),
        "HF_TOKEN": os.environ.pop("HF_TOKEN", None),
        "HUGGINGFACEHUB_API_TOKEN": os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None),
    }
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _validate_result(case: dict, result: dict, failures: list[str]) -> None:
    if not result.get("answer", "").strip():
        failures.append(f"{case['name']}: answer is empty")
    citations = result.get("citations", [])
    if not citations:
        failures.append(f"{case['name']}: no citations returned")
        return
    for citation in citations:
        if not citation.get("source_url"):
            failures.append(f"{case['name']}: citation missing source_url")
            break
        missing = [field for field in ["ticker", "company", "form_type", "filing_date", "section"] if not citation.get(field)]
        if missing:
            failures.append(f"{case['name']}: citation missing metadata {missing}")
            break
    answer = result.get("answer", "")
    if citations and "[Source" not in answer:
        failures.append(f"{case['name']}: answer is missing in-text source references")
    if len(answer.strip()) < 80:
        failures.append(f"{case['name']}: answer is too short to be meaningfully grounded")
    lowered = answer.lower()
    boilerplate_hits = [
        phrase for phrase in [
            "forward-looking statements",
            "no obligation to revise or update",
            "item 1a. risk factors",
            "item 2. unregistered sales",
            "part i",
            "table of contents",
            "mine safety disclosures",
        ]
        if phrase in lowered
    ]
    if boilerplate_hits:
        failures.append(f"{case['name']}: answer still contains boilerplate/TOC text {boilerplate_hits}")
    if answer.count("[Source") < 1:
        failures.append(f"{case['name']}: answer does not contain usable citations")


def main() -> None:
    answerer = get_grounded_answerer()
    reports = []
    failures: list[str] = []

    for case in SMOKE_TESTS:
        result = answerer.answer_question(
            case["query"],
            top_k=5,
            filters=case["filters"],
            provider_name="extractive",
        )
        _validate_result(case, result, failures)
        reports.append({
            "name": case["name"],
            "used_provider": result.get("used_provider"),
            "grounding_status": result.get("grounding_status"),
            "citation_count": len(result.get("citations", [])),
            "warnings": result.get("warnings", []),
        })

    with _without_external_keys():
        fallback_answerer = GroundedAnswerer(retriever=answerer.retriever)
        fallback = fallback_answerer.answer_question(
            "What are Apple's main risk factors?",
            top_k=3,
            filters={"ticker": "AAPL", "section": "Risk Factors"},
            provider_name="auto",
        )
        _validate_result({"name": "Extractive fallback"}, fallback, failures)
        if fallback.get("used_provider") != "extractive":
            failures.append("Extractive fallback: expected used_provider=extractive when external keys are absent")
        reports.append({
            "name": "Extractive fallback",
            "used_provider": fallback.get("used_provider"),
            "grounding_status": fallback.get("grounding_status"),
            "citation_count": len(fallback.get("citations", [])),
            "warnings": fallback.get("warnings", []),
        })

    output = {"smoke_tests": reports, "failures": failures}
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit("Answering verification failed.")


if __name__ == "__main__":
    main()
