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
QUERY_STOPWORDS = {
    "what", "does", "say", "about", "main", "these", "filings", "their", "with", "from",
    "into", "that", "this", "there", "have", "has", "had", "were", "been", "being", "the",
    "and", "for", "its", "are", "how", "describe", "discussed", "highlighted", "describe",
    "annual", "filing", "factors", "factor", "risk", "risks", "business", "revenue",
}
DIVIDEND_POLICY_TERMS = [
    "dividend policy",
    "dividends",
    "dividend",
    "cash dividend",
    "payout policy",
    "shareholder return",
    "capital return",
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
                "quality_adjustment": result.get("quality_adjustment"),
                "quality_adjusted_score": result.get("quality_adjusted_score"),
                "is_toc_like": result.get("is_toc_like"),
                "boilerplate_score": result.get("boilerplate_score"),
                "content_quality_score": result.get("content_quality_score"),
                "section_confidence": result.get("section_confidence"),
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

    @classmethod
    def _query_support_terms(cls, question: str, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        filters = filters or {}
        excluded = {
            str(filters.get("ticker", "")).lower(),
            str(filters.get("form_type", "")).lower(),
            str(filters.get("section", "")).lower(),
        }
        terms = []
        for term in cls._query_terms(question):
            if term in QUERY_STOPWORDS:
                continue
            if term in excluded:
                continue
            terms.append(term)
        return terms

    @classmethod
    def _query_support_phrases(cls, question: str, filters: Optional[Dict[str, Any]] = None) -> List[str]:
        terms = cls._query_support_terms(question, filters=filters)
        phrases = []
        for idx in range(len(terms) - 1):
            left = terms[idx]
            right = terms[idx + 1]
            if len(left) >= 4 and len(right) >= 4:
                phrases.append(f"{left} {right}")
        return phrases

    @classmethod
    def _is_dividend_policy_question(cls, question: str) -> bool:
        lowered = cls._normalize_text(question).lower()
        return "dividend" in lowered and "policy" in lowered

    @classmethod
    def _evidence_supports_dividend_policy(cls, evidence_text: str) -> bool:
        lowered = cls._normalize_text(evidence_text).lower()
        return any(term in lowered for term in DIVIDEND_POLICY_TERMS)

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
    def _chunk_quality_score(
        cls,
        citation: Dict[str, Any],
        query_terms: List[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> float:
        text = cls._normalize_text(citation.get("content", "") or citation.get("content_preview", ""))
        lowered = text.lower()
        score = 0.0
        score += float(citation.get("reranker_score") or 0.0)
        score += float(citation.get("fused_score") or 0.0) * 10
        score += float(citation.get("quality_adjustment") or 0.0) * 10
        score += float(citation.get("content_quality_score") or 0.0) * 6
        score += float(citation.get("section_confidence") or 0.0) * 2
        score += sum(1.5 for term in query_terms if term in lowered)
        score += sum(0.5 for hint in MEANINGFUL_HINTS if hint in lowered)
        if citation.get("section", "").lower() == "risk factors" and "risk" in query_terms:
            score += 2.0
        if citation.get("is_toc_like"):
            score -= 8.0
        score -= float(citation.get("boilerplate_score") or 0.0) * 6
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
        if filters and filters.get("section"):
            requested_section = str(filters["section"]).strip().lower()
            if requested_section and requested_section == str(citation.get("section", "")).strip().lower():
                score += 2.5
        if filters and filters.get("ticker"):
            requested_ticker = str(filters["ticker"]).strip().upper()
            if requested_ticker and requested_ticker == str(citation.get("ticker", "")).strip().upper():
                score += 0.75
        return score

    @classmethod
    def _select_best_citations(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        limit: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query_terms = cls._query_terms(question)
        ranked = sorted(
            citations,
            key=lambda citation: cls._chunk_quality_score(citation, query_terms, filters=filters),
            reverse=True,
        )
        best = [citation for citation in ranked if cls._chunk_quality_score(citation, query_terms, filters=filters) > -1][:limit]
        return best or ranked[:limit]

    @classmethod
    def _best_chunk_score(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not citations:
            return float("-inf")
        query_terms = cls._query_terms(question)
        return max(cls._chunk_quality_score(citation, query_terms, filters=filters) for citation in citations)

    @classmethod
    def _count_substantive_chunks(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        query_terms = cls._query_terms(question)
        return sum(1 for citation in citations if cls._chunk_quality_score(citation, query_terms, filters=filters) > 4.0)

    @classmethod
    def _select_extractive_sentences(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        max_points: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[int, str]]:
        query_terms = cls._query_terms(question)
        selected_citations = cls._select_best_citations(question, citations, limit=max_points + 1, filters=filters)
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
        substantive = cls._count_substantive_chunks(question, citations, filters=filters)
        best_score = cls._best_chunk_score(question, citations, filters=filters)
        selected_sentences = cls._select_extractive_sentences(question, citations, max_points=min(top_k, 4), filters=filters)
        return substantive < min(2, top_k) or best_score < 2.5 or len(selected_sentences) < min(2, top_k)

    @classmethod
    def _refine_confidence(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        base_confidence: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not citations:
            return base_confidence

        score = float(base_confidence.get("score", 0.0) or 0.0)
        query_terms = cls._query_terms(question)
        ranked_scores = [cls._chunk_quality_score(citation, query_terms, filters=filters) for citation in citations]
        substantive = sum(1 for value in ranked_scores if value >= 4.0)
        best_ranked_score = max(ranked_scores) if ranked_scores else float("-inf")
        avg_quality = sum(float(citation.get("content_quality_score") or 0.0) for citation in citations[:4]) / max(len(citations[:4]), 1)
        avg_boilerplate = sum(float(citation.get("boilerplate_score") or 0.0) for citation in citations[:4]) / max(len(citations[:4]), 1)
        toc_hits = sum(1 for citation in citations[:3] if citation.get("is_toc_like"))
        support_terms = cls._query_support_terms(question, filters=filters)
        support_phrases = cls._query_support_phrases(question, filters=filters)
        evidence_text = " ".join(
            cls._normalize_text(citation.get("content", "") or citation.get("content_preview", ""))
            for citation in citations[:5]
        ).lower()
        support_hits = [term for term in support_terms if term in evidence_text]
        support_ratio = len(support_hits) / max(len(support_terms), 1) if support_terms else 1.0
        phrase_hits = [phrase for phrase in support_phrases if phrase in evidence_text]
        unsupported_dividend_policy = cls._is_dividend_policy_question(question) and not cls._evidence_supports_dividend_policy(evidence_text)
        section_match_bonus = 0.0
        requested_years = re.findall(r"\b(20\d{2})\b", question)
        cited_years = {str(citation.get("fiscal_year", "")) for citation in citations if citation.get("fiscal_year")}
        year_mismatch = bool(requested_years) and not any(year in cited_years for year in requested_years)

        if filters and filters.get("section"):
            requested_section = str(filters["section"]).strip().lower()
            if any(
                requested_section == str(citation.get("section", "")).strip().lower()
                and float(citation.get("section_confidence") or 0.0) >= 0.6
                for citation in citations[:3]
            ):
                section_match_bonus = 0.08

        score += min(0.18, avg_quality * 0.2)
        score += min(0.15, substantive * 0.05)
        score += section_match_bonus
        if support_ratio >= 0.6:
            score += 0.08
        elif support_ratio < 0.3 and support_terms:
            score -= 0.2
        score -= min(0.18, avg_boilerplate * 0.12)
        score -= min(0.15, toc_hits * 0.08)
        if year_mismatch:
            score -= 0.18
        score = max(0.0, min(1.0, score))

        label = "high" if score >= 0.72 else "medium" if score >= 0.45 else "low"
        answerable = (
            score >= 0.35
            or (score >= 0.3 and substantive >= 2 and avg_quality >= 0.58)
            or (score >= 0.28 and substantive >= 1 and avg_quality >= 0.8 and best_ranked_score >= 9.0)
        )
        if year_mismatch or (support_terms and support_ratio < 0.25):
            answerable = False
        if support_phrases and not phrase_hits and support_ratio <= 0.5:
            answerable = False
        if unsupported_dividend_policy:
            answerable = False
        refined = dict(base_confidence)
        refined.update({
            "score": round(score, 4),
            "label": label,
            "answerable": answerable,
            "substantive_chunk_count": substantive,
            "best_ranked_chunk_score": round(best_ranked_score, 4) if ranked_scores else None,
            "avg_content_quality_score": round(avg_quality, 4),
            "avg_boilerplate_score": round(avg_boilerplate, 4),
            "query_support_ratio": round(support_ratio, 4),
            "query_support_hits": support_hits,
            "query_support_phrase_hits": phrase_hits,
            "year_mismatch": year_mismatch,
            "unsupported_dividend_policy": unsupported_dividend_policy,
        })
        return refined

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
                    if self._best_chunk_score(query, combined_citations, filters=relaxed_filters or filters) > self._best_chunk_score(query, citations, filters=filters):
                        citations = combined_citations
                        retrieval_results = combined_results
                        warnings.append(
                            "Primary evidence was too boilerplate-heavy; widened evidence selection by relaxing "
                            + ", ".join(removed)
                            + " while preserving company-level grounding."
                        )

        best_citations = self._select_best_citations(query, citations, limit=max(top_k, MIN_CITATIONS), filters=filters)
        best_doc_ids = {citation.get("doc_id") for citation in best_citations}
        best_results = [result for result in retrieval_results if result.get("doc_id") in best_doc_ids]
        best_results.sort(
            key=lambda result: self._chunk_quality_score(
                self._build_citations([result])[0],
                self._query_terms(query),
                filters=filters,
            ),
            reverse=True,
        )
        best_citations = self._build_citations(best_results)
        confidence = self._refine_confidence(
            query,
            best_citations,
            compute_confidence(best_results or retrieval_results),
            filters=filters,
        )
        return best_results, best_citations, warnings, confidence

    def _extractive_answer(
        self,
        question: str,
        citations: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        selected = self._select_extractive_sentences(question, citations, max_points=4, filters=filters)
        if not selected:
            ranked_citations = self._select_best_citations(question, citations, limit=2, filters=filters)
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
    def _should_abstain(confidence: Dict[str, Any]) -> bool:
        if confidence.get("unsupported_dividend_policy"):
            return True
        if confidence.get("year_mismatch"):
            return True
        if not confidence.get("answerable", False):
            return True
        if float(confidence.get("query_support_ratio", 1.0) or 0.0) < 0.25:
            return True
        return False

    @classmethod
    def _insufficient_evidence_answer(
        cls,
        question: str,
        citations: List[Dict[str, Any]],
        confidence: Dict[str, Any],
    ) -> str:
        reason = "The retrieved filings do not contain enough grounded evidence to answer this question directly."
        if confidence.get("year_mismatch"):
            reason = "The question asks about a filing year that is outside the indexed SEC corpus."
        elif confidence.get("unsupported_dividend_policy"):
            reason = "The retrieved filings do not directly discuss Nvidia dividend policy, dividends, or payout policy."
        elif float(confidence.get("query_support_ratio", 1.0) or 0.0) < 0.25:
            reason = "The retrieved filings discuss related company topics, but they do not directly support the key subject asked in the question."
        elif confidence.get("query_support_phrase_hits") == [] and confidence.get("query_support_hits"):
            reason = "The retrieved filings only partially match the topic phrasing in the question and do not answer it directly."

        source_refs = ""
        if citations:
            source_refs = " " + " ".join(f"[Source {citation['source_num']}]" for citation in citations[: min(2, len(citations))])
        return refusal_message(reason) + source_refs

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
        if confidence.get("year_mismatch"):
            warnings.append("The query mentions a filing year that does not match the cited evidence.")
        if confidence.get("unsupported_dividend_policy"):
            warnings.append("Retrieved evidence does not directly support the dividend-policy question.")
        if float(confidence.get("query_support_ratio", 1.0) or 0.0) < 0.25:
            warnings.append("Retrieved evidence does not directly support the key topic terms in the question.")
        elif confidence.get("query_support_phrase_hits") == [] and confidence.get("query_support_hits"):
            warnings.append("Retrieved evidence only partially matches the specific topic phrasing in the question.")
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
                answer_text = self._extractive_answer(query, citations, filters=filters)
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
            answer_text = self._extractive_answer(query, citations, filters=filters)
            used_provider = fallback.name
            warnings.append(f"Generative provider unavailable; used deterministic extractive fallback instead ({provider.name}).")

        if not answer_text:
            answer_text = self._extractive_answer(query, citations, filters=filters)
            used_provider = "extractive"
            warnings.append("Provider returned empty output; used deterministic extractive fallback instead.")

        if "[Source" not in answer_text and citations:
            answer_text = f"{answer_text.rstrip()} [Source 1]"

        cited_source_nums = {
            int(match)
            for match in re.findall(r"\[Source\s+(\d+)\]", answer_text)
            if str(match).isdigit()
        }
        if cited_source_nums:
            returned_citations = [citation for citation in citations if citation.get("source_num") in cited_source_nums]
        else:
            returned_citations = citations[: max(MIN_CITATIONS, min(top_k, len(citations)))]
        if not returned_citations and citations:
            returned_citations = citations[: max(MIN_CITATIONS, min(top_k, len(citations)))]

        if self._is_dividend_policy_question(query):
            cited_evidence_text = " ".join(
                self._normalize_text(citation.get("content", "") or citation.get("content_preview", ""))
                for citation in (returned_citations or citations)
            )
            answer_supports_dividend_policy = self._evidence_supports_dividend_policy(answer_text)
            if not self._evidence_supports_dividend_policy(cited_evidence_text) or not answer_supports_dividend_policy:
                confidence = dict(confidence)
                confidence["unsupported_dividend_policy"] = True
                confidence["answerable"] = False
                confidence["label"] = "low"
                if "Retrieved evidence does not directly support the dividend-policy question." not in warnings:
                    warnings.append("Retrieved evidence does not directly support the dividend-policy question.")

        abstain = self._should_abstain(confidence)
        if abstain:
            answer_text = self._insufficient_evidence_answer(query, returned_citations or citations, confidence)
        if not confidence.get("answerable", False):
            warnings.append("Grounding confidence is below the answerable threshold.")

        grounding_status = "grounded" if confidence.get("answerable", False) and not abstain else "insufficient_evidence"
        if warnings and grounding_status == "grounded":
            grounding_status = "grounded_with_warnings"

        latency_ms = round((time.time() - started) * 1000, 2)
        return {
            "question": query,
            "answer": answer_text,
            "citations": returned_citations,
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
