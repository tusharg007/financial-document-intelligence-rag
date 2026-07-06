"""Schemas for the lightweight RAG evaluation kit."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RAGEvalResult:
    faithfulness: float
    context_relevance: float
    citation_coverage: float
    answer_completeness: float
    hallucination_flag: bool
    guardrail_status: str
    warnings: List[str] = field(default_factory=list)
    question: str = ""
    answer: str = ""
    expected_keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
