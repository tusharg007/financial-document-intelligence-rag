"""Grounded answer generation over retrieved SEC filing evidence."""
from __future__ import annotations

import html
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.llm.base import BaseLLM
from src.llm.factory import ExtractiveProvider, provider_for
from src.retrieval.confidence import compute_confidence, refusal_message
from src.retrieval.pipeline import RetrievalPipeline, get_retrieval_pipeline
from src.utils.logger import get_logger

logger = get_logger("grounded_answer")

MIN_CITATIONS = 1
BOILERPLATE_PHRASES = [
    "forward-looking statements",
    "no obligation to revise or update",
    "item 1a. risk factors",
    "item 2. unregistered sales",
    "part i",
    "table of contents",
    "mine safety disclosures",
    "unresolved staff comments",
]
MEANINGFUL_HINTS = [
    "risk",
    "revenue",
    "competition",
    "competitive",
    "supply chain",
    "manufacturing",
    "regulatory",
    "legal proceedings",
    "legal",
    "cybersecurity",
    "financial condition",
    "demand",
    "operations",
    "business",
    "market",
    "products",
    "reputation",
    "material adverse",
    "market acceptance",
    "stock price",
]
NOISY_FINANCIAL_PHRASES = [
    "interest rate risk",
    "trade receivables",
    "marketable securities",
    "term debt",
    "derivative instruments",
    "foreign currency exchange",
    "foreign currency transactions",
    "current and non-current",
    "credit insurance",
    "third-party financing",
    "hedged assets",
    "hedged liabilities",
]


class GroundedAnswerer:
    """Answer questions using only retrieved SEC filing evidence."""

    def __init__(
        self,
        retriever: Optional[RetrievalPipeline] = None,
        provider: Optional[BaseLLM] = None,
        provider_name: Optional[str] = None,
    ):
        self.retriever = retriever or get_retrieval_pipeline()
        self._provider = provider
        self.provider_name = provider_name

    def _select_provider(self, provider_name: Optional[str] = None) -> BaseLLM:
        if self._provider is not None:
            return self._provider

        requested = (provider_name or self.provider_name or "auto").lower()
        if requested != "auto":
            return provider_for(requested)

        if os.getenv("GROQ_API_KEY"):
            return provider_for("groq")
        if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            return provider_for("huggingface")
        return ExtractiveProvider()

    @staticmethod
    def _build_citations(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        citations = []
        for rank, result in enumerate(results, start=1):
            citation = {
                "source_num": rank,
                "doc_id": result.get("doc_id", ""),
                "ticker": result.get("ticker", ""),
                "company": result.get("company", ""),
                "form_type": result.get("form_type", ""),
                "filing_date": result.get("filing_date", ""),
                "fiscal_year": result.get("fiscal_year", ""),
                "fiscal_period": result.get("fiscal_period", ""),
                "section": result.get("section", ""),
                "accession_number": result.get("accession_number", ""),
                "source_url": result.get("source_url", ""),
                "dense_score": result.get("dense_score"),
                "bm25_score": result.get("bm25_score"),
                "fused_score": result.get("fused_score"),
                "reranker_score": result.get("reranker_score"),
                "content_preview": result.get("content_preview", ""),
                "content": result.get("content", ""),
            }
            citations.append(citation)
        return citations

    @staticmethod
    def _build_context(question: str, citations: List[Dict[str, Any]]) -> str:
        parts = [
            "Answer the question using only the SEC filing excerpts below.",
            "If the evidence is weak, incomplete, or contradictory, say so explicitly.",
            f"Question: {question}",
            "",
        ]
        for citation in citations:
            parts.append(
                f"[Source {citation['source_num']}] "
                f"{citation['ticker']} | {citation['company']} | {citation['form_type']} | "
                f"{citation['filing_date']} | FY {citation['fiscal_year']} | "
                f"{citation['section']} | {citation['source_url']}"
            )
            parts.append(citation.get("content_preview", ""))
            parts.append("")
        return "\n".join(parts).strip()

    @staticmethod
    def _query_terms(question: str) -> List[str]:
        return [term for term in re.findall(r"\b[\w\-]+\b", question.lower()) if len(term) > 2]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(text or "")).strip()

    @classmethod
    def _is_sentence_fragment(cls, sentence: str) -> bool:
        clean = cls._normalize_text(sentence)
        if not clean:
            return True
        if re.match(r"^[a-z]{1,8}\b", clean):
            return True
        if not re.match(r"^[A-Z0-9\"'(]", clean):
            return True
        if re.search(r"\b[a-z]{1,4}\.$", clean) and not re.search(r"\b(Inc|Ltd|Corp|Co|etc)\.$", clean):
            return True
        return False

    @classmethod
    def _is_boilerplate_text(cls, text: str) -> bool:
        lowered = cls._normalize_text(text).lower()
        if not lowered:
            return True
        return any(phrase in lowered for phrase in BOILERPLATE_PHRASES)

    @classmethod
    def _sentence_quality_score(cls, sentence: str, query_terms: List[str]) -> float:
        clean = cls._normalize_text(sentence)
        lowered = clean.lower()
        if len(clean) < 40:
            return -10.0
        score = 0.0
        if cls._is_sentence_fragment(clean):
            score -= 5.0
        score += sum(2.0 for term in query_terms if term in lowered)
        score += sum(0.75 for hint in MEANINGFUL_HINTS if hint in lowered)
        score += min(1.5, len(re.findall(r"\b(and|or|because|including|related|depend|impact|risk|revenue|business)\b", lowered)) * 0.2)
        if cls._is_boilerplate_text(clean):
            score -= 8.0
        if any(phrase in lowered for phrase in NOISY_FINANCIAL_PHRASES):
            score -= 5.0
        if re.search(r"\b(item|part)\s+\d", lowered):
            score -= 4.0
        if re.search(r"\|\s*\d", clean) or clean.count("|") >= 2:
            score -= 3.0
        if re.search(r"\b\d+\s+item\b", lowered):
            score -= 2.0
        if "table of contents" in lowered:
            score -= 8.0
        if len(re.findall(r"\bitem\b", lowered)) > 2:
            score -= 3.0
        if clean.count("&#") >= 2:
            score -= 2.5
        if clean.count("$") >= 2:
            score -= 2.5
        if lowered.count(",") >= 2 and sum(1 for hint in MEANINGFUL_HINTS if hint in lowered) >= 2:
            score += 1.5
        if any(phrase in lowered for phrase in [
            "can be affected by a number of factors",
            "subject to intense competition",
            "ability to",
            "materially and adversely affected",
            "market acceptance",
        ]):
            score += 2.0
        return score

    @classmethod
    def _chunk_quality_score(cls, citation: Dict[str, Any], query_terms: List[str]) -> float:
        text = cls._normalize_text(citation.get("content", "") or citation.get("content_preview", ""))
        lowered = text.lower()
        score = 0.0
        score += float(citation.get("reranker_score") or 0.0)
        score += float(citation.get("fused_score") or 0.0) * 10
        score += sum(1.5 for term in query_terms if term in lowered)
        score += sum(0.5 for hint in MEANINGFUL_HINTS if hint in lowered)
        if citation.get("section", "").lower() == "risk factors" and "risk" in query_terms:
            score += 2.0
        if cls._is_boilerplate_text(text):
            score -= 6.0
        if any(phrase in lowered for phrase in NOISY_FINANCIAL_PHRASES):
            score -= 5.0
        if re.search(r"\bitem\s+1a\.\s*risk factors\b", lowered):
            score -= 4.0
        if re.search(r"\bpart i\b", lowered):
            score -= 3.0
        if lowered.count(",") >= 2 and sum(1 for hint in MEANINGFUL_HINTS if hint in lowered) >= 2:
            score += 1.5
        if any(phrase in lowered for phrase in [
            "the company's business",
            "results of operations",
            "financial condition",
            "stock price can be",
            "subject to intense competition",
            "poor market acceptance",
            "reduced demand for products and services",
            "legal matters",
            "regulatory",
        ]):
            score += 2.0
        return score

    @classmethod
    def _select_best_citations(cls, question: str, citations: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
        query_terms = cls._query_terms(question)
        ranked = sorted(
            citations,
            key=lambda citation: cls._chunk_quality_score(citation, query_terms),
            reverse=True,
        )
        best = [citation for citation in ranked if cls._chunk_quality_score(citation, query_terms) > -2][:limit]
        return best or ranked[:limit]

    @classmethod
    def _best_chunk_score(cls, question: str, citations: List[Dict[str, Any]]) -> float:
        if not citations:
            return float("-inf")
        query_terms = cls._query_terms(question)
        return max(cls._chunk_quality_score(citation, query_terms) for citation in citations)

    @classmethod
    def _count_substantive_chunks(cls, question: str, citations: List[Dict[str, Any]]) -> int:
        query_terms = cls._query_terms(question)
        return sum(1 for citation in citations if cls._chunk_quality_score(citation, query_terms) > 1.5)

    @classmethod
    def _select_extractive_sentences(cls, question: str, citations: List[Dict[str, Any]], max_points: int = 4) -> List[tuple[int, str]]:
        query_terms = cls._query_terms(question)
        selected_citations = cls._select_best_citations(question, citations, limit=max_points + 1)
        candidates: List[tuple[float, int, str]] = []
        seen = set()
        for citation in selected_citations:
            text = citation.get("content", "") or citation.get("content_preview", "")
            sentences = re.split(r"(?<=[.!?])\s+|(?<=;)\s+", cls._normalize_text(text))
            for sentence in sentences:
                clean = cls._normalize_text(sentence)
                key = clean.lower()
                if not clean or key in seen:
                    continue
                seen.add(key)
                if cls._is_sentence_fragment(clean):
                    continue
                score = cls._sentence_quality_score(clean, query_terms)
                if score <= 0:
                    continue
                candidates.append((score, citation["source_num"], clean))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [(source_num, sentence) for _, source_num, sentence in candidates[:max_points]]

    @classmethod
    def _compress_sentence(cls, sentence: str) -> str:
        clean = cls._normalize_text(sentence)
        if cls._is_sentence_fragment(clean):
            return ""
        clean = re.sub(r"^[^A-Za-z0-9]+", "", clean)
        clean = re.sub(r"\b(the company (?:assumes|does not|shall|may)\b.*)", "", clean, flags=re.I)
        clean = re.sub(r"\b(unless otherwise stated\b.*)", "", clean, flags=re.I)
        clean = cls._normalize_text(clean)
        return clean.rstrip(" ,;:")

    @staticmethod
    def _dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for result in results:
            doc_id = result.get("doc_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(result)
        return deduped

    @staticmethod
    def _relax_filters(filters: Optional[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], List[str]]:
        if not filters:
            return None, []
        relaxed = dict(filters)
        removed: List[str] = []
        for key in ("section", "fiscal_period", "filing_date"):
            if key in relaxed:
                removed.append(key)
                relaxed.pop(key, None)
        if not removed and "form_type" in relaxed:
            removed.append("form_type")
            relaxed.pop("form_type", None)
        return relaxed or None, removed

    @classmethod
    def _should_expand_evidence(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        top_k: int,
    ) -> bool:
        if not citations:
            return True
        if not filters:
            return False
        substantive = cls._count_substantive_chunks(question, citations)
        best_score = cls._best_chunk_score(question, citations)
        selected_sentences = cls._select_extractive_sentences(question, citations, max_points=min(top_k, 4))
        return substantive < min(2, top_k) or best_score < 2.5 or len(selected_sentences) < min(2, top_k)

    def _prepare_evidence(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, Any]]:
        candidate_k = max(top_k * 6, 18)
        retrieval_results = self.retriever.retrieve(query, top_k=candidate_k, filters=filters)
        raw_results = list(retrieval_results)
        citations = self._build_citations(self._dedupe_results(retrieval_results))
        warnings: List[str] = []

        if self._should_expand_evidence(query, citations, filters, top_k):
            relaxed_filters, removed = self._relax_filters(filters)
            if removed:
                supplemental = self.retriever.retrieve(query, top_k=candidate_k, filters=relaxed_filters)
                if supplemental:
                    raw_results.extend(supplemental)
                    combined_results = self._dedupe_results(raw_results)
                    combined_citations = self._build_citations(combined_results)
                    if self._best_chunk_score(query, combined_citations) > self._best_chunk_score(query, citations):
                        citations = combined_citations
                        retrieval_results = combined_results
                        warnings.append(
                            "Primary evidence was too boilerplate-heavy; widened evidence selection by relaxing "
                            + ", ".join(removed)
                            + " while preserving company-level grounding."
                        )

        best_citations = self._select_best_citations(query, citations, limit=max(top_k, MIN_CITATIONS))
        best_doc_ids = {citation.get("doc_id") for citation in best_citations}
        best_results = [result for result in retrieval_results if result.get("doc_id") in best_doc_ids]
        best_results.sort(
            key=lambda result: self._chunk_quality_score(
                self._build_citations([result])[0],
                self._query_terms(query),
            ),
            reverse=True,
        )
        best_citations = self._build_citations(best_results)
        confidence = compute_confidence(best_results or retrieval_results)
        return best_results, best_citations, warnings, confidence

    def _extractive_answer(self, question: str, citations: List[Dict[str, Any]]) -> str:
        selected = self._select_extractive_sentences(question, citations, max_points=4)
        if not selected:
            ranked_citations = self._select_best_citations(question, citations, limit=2)
            fallback_points = []
            for citation in ranked_citations:
                preview = self._compress_sentence(citation.get("content_preview", ""))
                if preview and not self._is_boilerplate_text(preview):
                    fallback_points.append((citation["source_num"], preview))
            selected = fallback_points[:2]

        if not selected:
            return refusal_message("No extractive evidence was available.")

        bullets = []
        for source_num, sentence in selected:
            sentence = self._compress_sentence(sentence)
            if not sentence:
                continue
            if not sentence.endswith("."):
                sentence += "."
            bullets.append(f"- {sentence} [Source {source_num}]")

        if not bullets:
            return refusal_message("No extractive evidence was available.")
        return "Based on the retrieved SEC filings:\n" + "\n".join(bullets)

    @staticmethod
    def _grounding_warnings(question: str, results: List[Dict[str, Any]], confidence: Dict[str, Any]) -> List[str]:
        warnings = []
        if not results:
            warnings.append("No retrieved evidence was available.")
            return warnings
        if confidence.get("label") == "low":
            warnings.append("Evidence strength is low; treat the answer as tentative.")
        sections = {result.get("section", "") for result in results if result.get("section")}
        if len(sections) > 2:
            warnings.append("Evidence spans multiple sections; check citations for context.")
        tickers = {result.get("ticker", "") for result in results if result.get("ticker")}
        query_upper = question.upper()
        mentioned = {ticker for ticker in tickers if ticker and ticker in query_upper}
        if mentioned and tickers - mentioned:
            warnings.append("Some retrieved evidence includes additional companies beyond the query focus.")
        return warnings

    def answer_question(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        retrieval_results, citations, preparation_warnings, confidence = self._prepare_evidence(query, top_k, filters)
        warnings = preparation_warnings + self._grounding_warnings(query, retrieval_results, confidence)

        if not retrieval_results:
            answer = refusal_message("No relevant filing excerpts were retrieved.")
            status = "no_evidence"
            used_provider = "none"
            latency_ms = round((time.time() - started) * 1000, 2)
            return {
                "question": query,
                "answer": answer,
                "citations": [],
                "retrieval_results": [],
                "confidence": confidence,
                "grounding_status": status,
                "used_provider": used_provider,
                "warnings": warnings,
                "latency_ms": latency_ms,
            }

        provider = self._select_provider(provider_name)
        answer_text = ""
        used_provider = provider.name

        try:
            if isinstance(provider, ExtractiveProvider):
                answer_text = self._extractive_answer(query, citations)
                used_provider = provider.name
            else:
                context = self._build_context(query, citations)
                response = provider.generate(
                    [
                        {"role": "system", "content": "You answer only from provided SEC filing evidence and must cite sources as [Source N]."},
                        {"role": "user", "content": context},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )
                answer_text = str(response.get("text", "")).strip()
                used_provider = str(response.get("provider", provider.name))
        except Exception as exc:
            logger.warning("Provider '%s' unavailable or failed; using extractive fallback. Error: %s", provider.name, exc)
            fallback = ExtractiveProvider()
            answer_text = self._extractive_answer(query, citations)
            used_provider = fallback.name
            warnings.append(f"Generative provider unavailable; used deterministic extractive fallback instead ({provider.name}).")

        if not answer_text:
            answer_text = self._extractive_answer(query, citations)
            used_provider = "extractive"
            warnings.append("Provider returned empty output; used deterministic extractive fallback instead.")

        if "[Source" not in answer_text and citations:
            answer_text = f"{answer_text.rstrip()} [Source 1]"

        if not confidence.get("answerable", False):
            warnings.append("Grounding confidence is below the answerable threshold.")

        grounding_status = "grounded" if confidence.get("answerable", False) else "weak_evidence"
        if warnings and grounding_status == "grounded":
            grounding_status = "grounded_with_warnings"

        latency_ms = round((time.time() - started) * 1000, 2)
        return {
            "question": query,
            "answer": answer_text,
            "citations": citations[: max(MIN_CITATIONS, min(top_k, len(citations)))],
            "retrieval_results": retrieval_results,
            "confidence": confidence,
            "grounding_status": grounding_status,
            "used_provider": used_provider,
            "warnings": warnings,
            "latency_ms": latency_ms,
        }


_answerer: Optional[GroundedAnswerer] = None


def get_grounded_answerer() -> GroundedAnswerer:
    global _answerer
    if _answerer is None:
        _answerer = GroundedAnswerer()
    return _answerer


def answer_question(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return get_grounded_answerer().answer_question(query=query, top_k=top_k, filters=filters)
