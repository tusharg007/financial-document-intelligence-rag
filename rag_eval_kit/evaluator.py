"""High-level RAG response evaluation entrypoint."""
from __future__ import annotations

from typing import Any, Iterable, List, Sequence

from rag_eval_kit.guardrails import assess_guardrails
from rag_eval_kit.metrics import (
    answer_completeness,
    citation_coverage,
    context_relevance,
    faithfulness,
)
from rag_eval_kit.schemas import RAGEvalResult


def _normalize_contexts(contexts: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for context in contexts or []:
        if isinstance(context, dict):
            normalized.append(
                str(
                    context.get("content")
                    or context.get("content_preview")
                    or context.get("text")
                    or ""
                )
            )
        else:
            normalized.append(str(context))
    return [item for item in normalized if item.strip()]


def evaluate_rag_response(
    *,
    question: str,
    answer: str,
    contexts: Sequence[Any],
    ground_truth: str | None = None,
    expected_keywords: Sequence[str] | None = None,
) -> RAGEvalResult:
    del ground_truth  # Deterministic heuristics only for this lightweight kit.

    context_texts = _normalize_contexts(contexts)
    relevance = context_relevance(question, context_texts)
    faithful = faithfulness(answer, context_texts)
    coverage = citation_coverage(answer, context_texts)
    completeness = answer_completeness(answer, expected_keywords)
    guardrail_status, hallucination_flag, warnings = assess_guardrails(
        faithfulness=faithful,
        context_relevance=relevance,
        citation_coverage=coverage,
        answer_completeness=completeness,
    )
    return RAGEvalResult(
        question=question,
        answer=answer,
        expected_keywords=list(expected_keywords) if expected_keywords else None,
        faithfulness=faithful,
        context_relevance=relevance,
        citation_coverage=coverage,
        answer_completeness=completeness,
        hallucination_flag=hallucination_flag,
        guardrail_status=guardrail_status,
        warnings=warnings,
    )
