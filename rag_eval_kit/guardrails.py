"""Simple deterministic guardrails for developer-facing RAG evaluation."""
from __future__ import annotations

from typing import List, Tuple


def assess_guardrails(
    *,
    faithfulness: float,
    context_relevance: float,
    citation_coverage: float,
    answer_completeness: float,
) -> Tuple[str, bool, List[str]]:
    warnings: List[str] = []
    hallucination_flag = False

    if citation_coverage == 0.0:
        warnings.append("Answer is missing citation markers.")
    if faithfulness < 0.35:
        warnings.append("Answer has low lexical support from the provided contexts.")
        hallucination_flag = True
    if context_relevance < 0.2:
        warnings.append("Retrieved contexts may be weakly related to the question.")
    if answer_completeness < 0.5:
        warnings.append("Answer may be incomplete relative to expected keywords.")

    if hallucination_flag or citation_coverage == 0.0:
        return "fail", True, warnings
    if warnings:
        return "warn", False, warnings
    return "pass", False, warnings
