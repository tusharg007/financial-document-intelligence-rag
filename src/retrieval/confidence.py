"""Confidence scoring and abstention helpers."""
from typing import Any, Dict, List


def compute_confidence(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"score": 0.0, "label": "low", "answerable": False, "reason": "No retrieved evidence."}
    scores = []
    sources = set()
    for result in results:
        score = result.get("rerank_score", result.get("rrf_score", result.get("score", 0.0)))
        try:
            scores.append(float(score))
        except Exception:
            pass
        meta = result.get("metadata", {})
        sources.add((
            meta.get("ticker") or meta.get("company"),
            meta.get("accession_number") or meta.get("filing_date"),
        ))
    normalized = min(1.0, max(0.0, sum(scores) / max(len(scores), 1))) if scores else 0.0
    agreement_bonus = min(0.2, 0.05 * max(len(sources) - 1, 0))
    score = min(1.0, normalized + agreement_bonus)
    label = "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
    return {"score": round(score, 4), "label": label, "answerable": score >= 0.2, "source_count": len(sources)}


def refusal_message(reason: str = "Retrieved evidence is too weak.") -> str:
    return f"I do not have enough grounded evidence in the indexed filings to answer confidently. {reason}"
