from __future__ import annotations

import pytest


class FakeDense:
    def __init__(self, results=None, available=True, error=None):
        self.results = results or []
        self.available = available
        self.error = error
        self.calls = []

    def collection_exists(self):
        if self.error:
            raise self.error
        return self.available

    def search(self, query, top_k=None, where=None):
        self.calls.append({"query": query, "top_k": top_k, "where": where})
        if self.error:
            raise self.error
        return self.results


class FakeSparse:
    def __init__(self, results=None, exists=True, error=None):
        self.results = results or []
        self.exists = exists
        self.error = error
        self.calls = []

    def index_exists(self):
        if self.error:
            raise self.error
        return self.exists

    def load_index(self):
        if self.error:
            raise self.error
        return self.exists

    def search(self, query, top_k=None, filters=None):
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        if self.error:
            raise self.error
        return self.results


class FakeReranker:
    def rerank(self, query, documents, top_k=None):
        results = []
        for doc in documents:
            scored = doc.copy()
            scored["rerank_score"] = 10.0 if doc["doc_id"] == "shared" else 1.0
            results.append(scored)
        results.sort(key=lambda item: item["rerank_score"], reverse=True)
        return results[:top_k]


def test_rrf_fusion_and_deduplication():
    from src.retrieval.pipeline import RetrievalPipeline

    dense = FakeDense(results=[
        {"doc_id": "shared", "content": "Apple risk factors", "metadata": {"ticker": "AAPL", "company": "Apple Inc.", "form_type": "10-K", "filing_date": "2024-11-01", "fiscal_year": 2024, "fiscal_period": "FY", "section": "Risk Factors", "accession_number": "a", "source_url": "https://sec/a"}},
        {"doc_id": "dense-only", "content": "Apple products", "metadata": {"ticker": "AAPL", "company": "Apple Inc.", "form_type": "10-K", "filing_date": "2024-11-01", "fiscal_year": 2024, "fiscal_period": "FY", "section": "Business", "accession_number": "b", "source_url": "https://sec/b"}},
    ])
    sparse = FakeSparse(results=[
        {"doc_id": "shared", "content": "Apple risk factors", "ticker": "AAPL", "company": "Apple Inc.", "form_type": "10-K", "filing_date": "2024-11-01", "fiscal_year": 2024, "fiscal_period": "FY", "section": "Risk Factors", "accession_number": "a", "source_url": "https://sec/a", "score": 4.0, "bm25_score": 4.0},
        {"doc_id": "sparse-only", "content": "Apple margin", "ticker": "AAPL", "company": "Apple Inc.", "form_type": "10-K", "filing_date": "2024-11-01", "fiscal_year": 2024, "fiscal_period": "FY", "section": "MD&A", "accession_number": "c", "source_url": "https://sec/c", "score": 3.0, "bm25_score": 3.0},
    ])
    pipeline = RetrievalPipeline(dense_embedder=dense, sparse_embedder=sparse, reranker=FakeReranker())

    results = pipeline.retrieve("Apple risk factors", top_k=5)

    assert [result["doc_id"] for result in results] == ["shared", "dense-only", "sparse-only"]
    assert results[0]["found_by"] == ["dense", "sparse"]
    assert results[0]["fused_score"] > results[1]["fused_score"]


def test_metadata_filters_are_forwarded():
    from src.retrieval.pipeline import RetrievalPipeline

    dense = FakeDense(results=[])
    sparse = FakeSparse(results=[])
    pipeline = RetrievalPipeline(dense_embedder=dense, sparse_embedder=sparse, reranker=FakeReranker())

    results = pipeline.retrieve("Microsoft revenue", top_k=3, filters={"ticker": "MSFT", "section": "Risk Factors"})

    assert results == []
    assert dense.calls[0]["where"] == {"$and": [{"ticker": "MSFT"}, {"section": "Risk Factors"}]}
    assert sparse.calls[0]["filters"] == {"ticker": "MSFT", "section": "Risk Factors"}


def test_result_schema_contains_required_fields():
    from src.retrieval.pipeline import REQUIRED_METADATA_FIELDS, RetrievalPipeline

    dense = FakeDense(results=[
        {"doc_id": "d1", "content": "Tesla revenue", "score": 0.9, "metadata": {"ticker": "TSLA", "company": "Tesla, Inc.", "form_type": "10-Q", "filing_date": "2024-10-24", "fiscal_year": 2024, "fiscal_period": "Q3", "section": "Management's Discussion and Analysis", "accession_number": "z", "source_url": "https://sec/z"}}
    ])
    sparse = FakeSparse(results=[])
    pipeline = RetrievalPipeline(dense_embedder=dense, sparse_embedder=sparse, reranker=FakeReranker())

    result = pipeline.retrieve("Tesla revenue", top_k=1)[0]

    for field in REQUIRED_METADATA_FIELDS:
        assert field in result
    assert result["content_preview"].startswith("Tesla revenue")
    assert "dense_score" in result
    assert "bm25_score" in result
    assert "fused_score" in result
    assert "reranker_score" in result


def test_pipeline_falls_back_if_one_backend_fails():
    from src.retrieval.pipeline import RetrievalPipeline

    dense = FakeDense(results=[], available=False, error=RuntimeError("dense offline"))
    sparse = FakeSparse(results=[
        {"doc_id": "s1", "content": "JPM statements", "ticker": "JPM", "company": "JPMorgan Chase & Co.", "form_type": "10-K", "filing_date": "2024-02-16", "fiscal_year": 2023, "fiscal_period": "FY", "section": "Financial Statements", "accession_number": "j", "source_url": "https://sec/j", "score": 2.0, "bm25_score": 2.0}
    ])
    pipeline = RetrievalPipeline(dense_embedder=dense, sparse_embedder=sparse, reranker=FakeReranker())

    results = pipeline.retrieve("JPM financial statements", top_k=3)

    assert [result["doc_id"] for result in results] == ["s1"]
    assert results[0]["retrieval_type"] == "sparse"


def test_pipeline_fails_clearly_if_both_backends_missing():
    from src.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        dense_embedder=FakeDense(results=[], available=False),
        sparse_embedder=FakeSparse(results=[], exists=False),
        reranker=FakeReranker(),
    )

    with pytest.raises(RuntimeError, match="No retrieval backend is available"):
        pipeline.retrieve("Any query", top_k=3)
