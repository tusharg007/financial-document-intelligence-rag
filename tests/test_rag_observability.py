from __future__ import annotations

import json


def test_observability_logger_writes_jsonl(tmp_path):
    from rag_eval_kit.observability import log_rag_observation
    from rag_eval_kit.schemas import RAGEvalResult

    log_path = tmp_path / "rag_observability.jsonl"
    result = RAGEvalResult(
        question="What are the risks?",
        answer="Competition is a risk. [Source 1]",
        faithfulness=0.8,
        context_relevance=0.7,
        citation_coverage=1.0,
        answer_completeness=1.0,
        hallucination_flag=False,
        guardrail_status="pass",
        warnings=[],
    )
    record = log_rag_observation(
        question=result.question,
        answer=result.answer,
        contexts=["Competition is a risk."],
        citations=[{"source_num": 1}],
        eval_result=result,
        latency_ms=12.3,
        user_feedback="helpful",
        log_path=log_path,
    )
    assert record["guardrail_status"] == "pass"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["citation_count"] == 1
    assert payload["latency_ms"] == 12.3
