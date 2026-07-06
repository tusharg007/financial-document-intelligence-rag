"""Observability helpers for logging RAG evaluation events."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from config.settings import PROJECT_ROOT
from rag_eval_kit.schemas import RAGEvalResult

DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "rag_observability.jsonl"


def log_rag_observation(
    *,
    question: str,
    answer: str,
    contexts: Iterable[Any],
    eval_result: RAGEvalResult,
    citations: Optional[Iterable[Any]] = None,
    latency_ms: float | None = None,
    user_feedback: str | None = None,
    log_path: str | Path | None = None,
) -> Dict[str, Any]:
    path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    citation_count = len(list(citations or []))
    context_count = len(list(contexts or []))
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer_preview": (answer or "")[:240],
        "retrieved_doc_count": context_count,
        "citation_count": citation_count,
        "faithfulness": eval_result.faithfulness,
        "context_relevance": eval_result.context_relevance,
        "citation_coverage": eval_result.citation_coverage,
        "hallucination_flag": eval_result.hallucination_flag,
        "guardrail_status": eval_result.guardrail_status,
        "latency_ms": latency_ms,
        "user_feedback": user_feedback,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
