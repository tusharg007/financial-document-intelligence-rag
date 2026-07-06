from __future__ import annotations

from src.llm.base import BaseLLM
from src.llm.factory import ExtractiveProvider


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, top_k=5, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self.results)


class MappingRetriever:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def retrieve(self, query, top_k=5, filters=None):
        key = tuple(sorted((filters or {}).items()))
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self.mapping.get(key, []))


class FakeProvider(BaseLLM):
    name = "fake"

    def __init__(self, text="Grounded answer from evidence [Source 1]"):
        self.text = text

    def health_check(self):
        return {"provider": self.name, "ok": True}

    def generate(self, messages, temperature=0.2, max_tokens=512):
        return {"text": self.text, "provider": self.name, "model": "fake-model"}


class FailingProvider(BaseLLM):
    name = "failing"

    def health_check(self):
        return {"provider": self.name, "ok": False}

    def generate(self, messages, temperature=0.2, max_tokens=512):
        raise RuntimeError("provider down")


def _sample_results():
    return [
        {
            "doc_id": "doc-1",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "section": "Risk Factors",
            "accession_number": "0000320193-24-000123",
            "source_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
            "content": "Apple faces supply chain concentration and cybersecurity risks.",
            "content_preview": "Apple faces supply chain concentration and cybersecurity risks.",
            "dense_score": 0.9,
            "bm25_score": 7.2,
            "fused_score": 0.02,
            "reranker_score": 2.5,
            "is_toc_like": False,
            "boilerplate_score": 0.05,
            "content_quality_score": 0.9,
            "section_confidence": 0.95,
        }
    ]


def _boilerplate_results():
    return [
        {
            "doc_id": "toc-1",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form_type": "10-Q",
            "filing_date": "2023-02-03",
            "fiscal_year": 2022,
            "fiscal_period": "Q1",
            "section": "Risk Factors",
            "accession_number": "0000320193-23-000006",
            "source_url": "https://www.sec.gov/example1",
            "content": "Item 1A. Risk Factors Part I Item 2. Unregistered Sales Mine Safety Disclosures forward-looking statements no obligation to revise or update.",
            "content_preview": "Item 1A. Risk Factors Part I Item 2. Unregistered Sales Mine Safety Disclosures forward-looking statements no obligation to revise or update.",
            "dense_score": 0.95,
            "bm25_score": 8.0,
            "fused_score": 0.03,
            "reranker_score": 2.6,
            "is_toc_like": True,
            "boilerplate_score": 0.95,
            "content_quality_score": 0.05,
            "section_confidence": 0.6,
        },
        {
            "doc_id": "real-1",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "section": "Risk Factors",
            "accession_number": "0000320193-24-000123",
            "source_url": "https://www.sec.gov/example2",
            "content": "Apple notes that its business is subject to intense competition, supply chain disruption, product demand volatility, and regulatory or legal risks in multiple markets.",
            "content_preview": "Apple notes that its business is subject to intense competition, supply chain disruption, product demand volatility, and regulatory or legal risks in multiple markets.",
            "dense_score": 0.7,
            "bm25_score": 6.1,
            "fused_score": 0.02,
            "reranker_score": 2.2,
            "is_toc_like": False,
            "boilerplate_score": 0.1,
            "content_quality_score": 0.92,
            "section_confidence": 0.96,
        },
    ]


def _unsupported_topic_results():
    return [
        {
            "doc_id": "nvda-1",
            "ticker": "NVDA",
            "company": "NVIDIA CORP",
            "form_type": "10-Q",
            "filing_date": "2023-11-21",
            "fiscal_year": 2023,
            "fiscal_period": "Q3",
            "section": "MD&A",
            "accession_number": "0001045810-23-000227",
            "source_url": "https://www.sec.gov/example-nvda-1",
            "content": "Overview Our Company and Our Businesses Since our founding in 1993, NVIDIA has been a pioneer in accelerated computing.",
            "content_preview": "Overview Our Company and Our Businesses Since our founding in 1993, NVIDIA has been a pioneer in accelerated computing.",
            "dense_score": 0.8,
            "bm25_score": 3.5,
            "fused_score": 0.02,
            "reranker_score": 1.1,
            "is_toc_like": False,
            "boilerplate_score": 0.05,
            "content_quality_score": 0.7,
            "section_confidence": 0.8,
        },
        {
            "doc_id": "nvda-2",
            "ticker": "NVDA",
            "company": "NVIDIA CORP",
            "form_type": "10-Q",
            "filing_date": "2024-05-29",
            "fiscal_year": 2024,
            "fiscal_period": "Q1",
            "section": "Risk Factors",
            "accession_number": "0001045810-24-000124",
            "source_url": "https://www.sec.gov/example-nvda-2",
            "content": "The program does not obligate NVIDIA to acquire any particular amount of common stock and the program may be suspended at any time at our discretion.",
            "content_preview": "The program does not obligate NVIDIA to acquire any particular amount of common stock and the program may be suspended at any time at our discretion.",
            "dense_score": 0.75,
            "bm25_score": 3.1,
            "fused_score": 0.018,
            "reranker_score": 1.0,
            "is_toc_like": False,
            "boilerplate_score": 0.05,
            "content_quality_score": 0.6,
            "section_confidence": 0.75,
        },
    ]


def test_answer_result_schema():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_sample_results()), provider=FakeProvider())
    result = answerer.answer_question("What are Apple's main risk factors?", top_k=3)

    assert result["question"]
    assert result["answer"]
    assert result["citations"]
    assert result["retrieval_results"]
    assert result["confidence"]["score"] >= 0
    assert result["grounding_status"]
    assert result["used_provider"] == "fake"
    assert isinstance(result["warnings"], list)
    assert result["latency_ms"] >= 0


def test_citations_schema():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_sample_results()), provider=FakeProvider())
    citation = answerer.answer_question("What are Apple's main risk factors?", top_k=3)["citations"][0]

    for field in [
        "source_num",
        "doc_id",
        "ticker",
        "company",
        "form_type",
        "filing_date",
        "fiscal_year",
        "section",
        "accession_number",
        "source_url",
        "content_preview",
    ]:
        assert field in citation


def test_extractive_fallback_behavior():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_sample_results()), provider=FailingProvider())
    result = answerer.answer_question("What are Apple's main risk factors?", top_k=3)

    assert result["used_provider"] == "extractive"
    assert "[Source 1]" in result["answer"]
    assert result["citations"]


def test_no_results_behavior():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever([]), provider=FakeProvider())
    result = answerer.answer_question("Unknown question", top_k=3)

    assert result["grounding_status"] == "no_evidence"
    assert result["citations"] == []
    assert "grounded evidence" in result["answer"].lower()


def test_prompt_context_builder_behavior():
    from src.answering.grounded_answer import GroundedAnswerer

    citations = GroundedAnswerer._build_citations(_sample_results())
    context = GroundedAnswerer._build_context("What are Apple's main risk factors?", citations)

    assert "Question: What are Apple's main risk factors?" in context
    assert "[Source 1]" in context
    assert "https://www.sec.gov/" in context
    assert "Risk Factors" in context


def test_provider_fallback_behavior_using_fake_retriever():
    from src.answering.grounded_answer import GroundedAnswerer

    retriever = FakeRetriever(_sample_results())
    answerer = GroundedAnswerer(retriever=retriever, provider=FakeProvider("Provider answer [Source 1]"))
    result = answerer.answer_question(
        "What are Apple's main risk factors?",
        top_k=4,
        filters={"ticker": "AAPL", "section": "Risk Factors"},
    )

    assert retriever.calls[0]["filters"] == {"ticker": "AAPL", "section": "Risk Factors"}
    assert result["used_provider"] == "fake"
    assert result["answer"].endswith("[Source 1]")


def test_boilerplate_filtering_prefers_substantive_evidence():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_boilerplate_results()), provider=FailingProvider())
    result = answerer.answer_question("What are Apple's main risk factors?", top_k=3)

    answer_lower = result["answer"].lower()
    assert "forward-looking statements" not in answer_lower
    assert "no obligation to revise or update" not in answer_lower
    assert "competition" in answer_lower or "supply chain" in answer_lower


def test_extractive_sentence_selection_returns_bullets_with_citations():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_boilerplate_results()), provider=ExtractiveProvider())
    result = answerer.answer_question("What are Apple's main risk factors?", top_k=3)

    assert "Based on the retrieved SEC filings:" in result["answer"]
    assert "- " in result["answer"]
    assert "[Source 1]" in result["answer"] or "[Source 2]" in result["answer"]


def test_weak_evidence_warning_still_works():
    from src.answering.grounded_answer import GroundedAnswerer

    weak = _sample_results()
    weak[0]["reranker_score"] = 0.01
    weak[0]["fused_score"] = 0.001
    answerer = GroundedAnswerer(retriever=FakeRetriever(weak), provider=ExtractiveProvider())
    result = answerer.answer_question("What are Apple's main risk factors?", top_k=3)

    assert any("low" in warning.lower() or "threshold" in warning.lower() for warning in result["warnings"])


def test_high_quality_evidence_can_be_grounded_with_warnings_not_weak():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_boilerplate_results()), provider=ExtractiveProvider())
    result = answerer.answer_question(
        "What are Apple's main risk factors?",
        top_k=3,
        filters={"ticker": "AAPL", "section": "Risk Factors"},
    )

    assert result["grounding_status"] in {"grounded", "grounded_with_warnings"}
    assert result["confidence"]["answerable"] is True


def test_relaxed_filter_evidence_recovery_uses_better_company_level_chunks():
    from src.answering.grounded_answer import GroundedAnswerer

    strict_results = [
        {
            "doc_id": "strict-1",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form_type": "10-Q",
            "filing_date": "2023-02-03",
            "fiscal_year": 2022,
            "fiscal_period": "Q1",
            "section": "Risk Factors",
            "accession_number": "0000320193-23-000006",
            "source_url": "https://www.sec.gov/strict",
            "content": "Item 1A. Risk Factors Part I Item 2. Unregistered Sales forward-looking statements no obligation to revise or update.",
            "content_preview": "Item 1A. Risk Factors Part I Item 2. Unregistered Sales forward-looking statements no obligation to revise or update.",
            "dense_score": 0.9,
            "bm25_score": 7.9,
            "fused_score": 0.02,
            "reranker_score": 2.2,
            "is_toc_like": True,
            "boilerplate_score": 0.95,
            "content_quality_score": 0.05,
            "section_confidence": 0.3,
        }
    ]
    relaxed_results = [
        {
            "doc_id": "relaxed-1",
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "form_type": "10-K",
            "filing_date": "2024-11-01",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "section": "Financial Statements",
            "accession_number": "0000320193-24-000123",
            "source_url": "https://www.sec.gov/relaxed",
            "content": "Apple states that its main risk factors include intense competition, supply chain disruption, product demand volatility, and regulatory or legal risks across multiple markets.",
            "content_preview": "Apple states that its main risk factors include intense competition, supply chain disruption, product demand volatility, and regulatory or legal risks across multiple markets.",
            "dense_score": 0.6,
            "bm25_score": 6.8,
            "fused_score": 0.015,
            "reranker_score": 1.8,
            "is_toc_like": False,
            "boilerplate_score": 0.05,
            "content_quality_score": 0.95,
            "section_confidence": 0.92,
        }
    ]
    retriever = MappingRetriever({
        (("section", "Risk Factors"), ("ticker", "AAPL")): strict_results,
        (("ticker", "AAPL"),): relaxed_results,
    })
    answerer = GroundedAnswerer(retriever=retriever, provider=ExtractiveProvider())

    result = answerer.answer_question(
        "What are Apple's main risk factors?",
        top_k=3,
        filters={"ticker": "AAPL", "section": "Risk Factors"},
    )

    assert len(retriever.calls) == 2
    assert retriever.calls[1]["filters"] == {"ticker": "AAPL"}
    assert "supply chain disruption" in result["answer"].lower()
    assert any("widened evidence selection" in warning.lower() for warning in result["warnings"])


def test_no_answer_year_outside_corpus_returns_insufficient_evidence():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_boilerplate_results()), provider=ExtractiveProvider())
    result = answerer.answer_question(
        "What did Apple disclose about 2021 risk factors?",
        top_k=3,
        filters={"ticker": "AAPL", "section": "Risk Factors"},
    )

    assert result["grounding_status"] == "insufficient_evidence"
    assert "do not have enough grounded evidence" in result["answer"].lower()
    assert "outside the indexed sec corpus" in result["answer"].lower() or any("year" in warning.lower() for warning in result["warnings"])


def test_no_answer_unsupported_topic_is_not_confident():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_unsupported_topic_results()), provider=ExtractiveProvider())
    result = answerer.answer_question(
        "What does Nvidia say about dividend policy in these filings?",
        top_k=3,
        filters={"ticker": "NVDA"},
    )

    assert result["grounding_status"] == "insufficient_evidence"
    assert result["confidence"]["answerable"] is False
    assert (
        "do not directly support" in result["answer"].lower()
        or "do not answer it directly" in result["answer"].lower()
        or "do not directly discuss" in result["answer"].lower()
    )
    assert result["citations"]


def test_sec_eval_018_dividend_policy_requires_actual_dividend_evidence():
    from src.answering.grounded_answer import GroundedAnswerer

    answerer = GroundedAnswerer(retriever=FakeRetriever(_unsupported_topic_results()), provider=ExtractiveProvider())
    result = answerer.answer_question(
        "What does Nvidia say about dividend policy in these filings?",
        top_k=5,
        filters={"ticker": "NVDA"},
    )

    assert result["grounding_status"] == "insufficient_evidence"
    assert result["confidence"]["unsupported_dividend_policy"] is True
    assert any("dividend-policy" in warning.lower() for warning in result["warnings"])
