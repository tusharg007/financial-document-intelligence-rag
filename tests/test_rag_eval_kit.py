from __future__ import annotations

import json


def test_evaluate_rag_response_returns_expected_fields():
    from rag_eval_kit import evaluate_rag_response

    result = evaluate_rag_response(
        question="What revenue risks are discussed?",
        answer="Revenue may be affected by competition and demand. [Source 1]",
        contexts=["Revenue may be affected by competition and customer demand."],
        expected_keywords=["revenue", "competition", "demand"],
    )

    payload = result.to_dict()
    for field in [
        "faithfulness",
        "context_relevance",
        "citation_coverage",
        "answer_completeness",
        "hallucination_flag",
        "guardrail_status",
        "warnings",
    ]:
        assert field in payload


def test_context_relevance_increases_for_relevant_context():
    from rag_eval_kit import evaluate_rag_response

    relevant = evaluate_rag_response(
        question="What does the filing say about cloud demand?",
        answer="Cloud demand is discussed. [Source 1]",
        contexts=["Cloud demand and customer spending are discussed in the filing."],
    )
    irrelevant = evaluate_rag_response(
        question="What does the filing say about cloud demand?",
        answer="Cloud demand is discussed. [Source 1]",
        contexts=["The filing discusses unrelated board governance matters."],
    )
    assert relevant.context_relevance > irrelevant.context_relevance


def test_hallucination_flag_triggers_for_unsupported_answer():
    from rag_eval_kit import evaluate_rag_response

    result = evaluate_rag_response(
        question="What does the filing say about dividend policy?",
        answer="The company expects major dividend increases next year.",
        contexts=["The filing discusses market competition and customer demand."],
    )
    assert result.hallucination_flag is True
    assert result.guardrail_status == "fail"


def test_citation_coverage_detects_source_markers():
    from rag_eval_kit import evaluate_rag_response

    with_citation = evaluate_rag_response(
        question="What are the risks?",
        answer="Competition is a risk. [Source 1]",
        contexts=["Competition is discussed as a risk."],
    )
    without_citation = evaluate_rag_response(
        question="What are the risks?",
        answer="Competition is a risk.",
        contexts=["Competition is discussed as a risk."],
    )
    assert with_citation.citation_coverage > without_citation.citation_coverage


def test_cli_writes_json(tmp_path):
    from rag_eval_kit.cli import main
    import sys

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps(
            {
                "question": "What are the risks?",
                "answer": "Competition is a risk. [Source 1]",
                "contexts": ["Competition is discussed as a risk."],
                "expected_keywords": ["competition"],
            }
        ),
        encoding="utf-8",
    )
    argv = sys.argv
    sys.argv = ["rag_eval_kit.cli", "--input", str(input_path), "--output", str(output_path)]
    try:
        main()
    finally:
        sys.argv = argv
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "faithfulness" in payload
    assert payload["guardrail_status"] in {"pass", "warn", "fail"}


def test_query_answer_observed_imports_safely():
    import scripts.query_answer_observed as module

    assert hasattr(module, "main")
