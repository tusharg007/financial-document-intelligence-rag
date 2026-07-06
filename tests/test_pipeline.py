"""
Unit and integration tests for the Financial Document Intelligence System.

Tests cover:
- Data processing pipeline
- Dense and sparse embedding
- Hybrid retrieval with RRF
- Cross-encoder reranking
- RAG pipeline end-to-end
- Evaluation metrics
"""
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Unit Tests: Data Layer
# ============================================================================

class TestSampleData:
    """Tests for sample data module."""

    def test_get_all_documents(self):
        from src.data.sample_data import get_all_documents
        docs = get_all_documents()
        assert len(docs) > 20, f"Expected 20+ documents, got {len(docs)}"

    def test_document_structure(self):
        from src.data.sample_data import get_all_documents
        docs = get_all_documents()
        required_fields = ["content", "company", "filing_type", "section", "filing_date"]
        for doc in docs:
            for field in required_fields:
                assert field in doc, f"Missing field: {field}"

    def test_get_documents_by_company(self):
        from src.data.sample_data import get_documents_by_company
        tesla_docs = get_documents_by_company("Tesla")
        assert len(tesla_docs) > 0, "No Tesla documents found"
        for doc in tesla_docs:
            assert "tesla" in doc["company"].lower() or "TSLA" in doc.get("ticker", "")

    def test_get_documents_by_type(self):
        from src.data.sample_data import get_documents_by_type
        k_docs = get_documents_by_type("10-K")
        assert len(k_docs) > 0
        for doc in k_docs:
            assert "10-K" in doc["filing_type"]

    def test_get_companies(self):
        from src.data.sample_data import get_companies
        companies = get_companies()
        assert "Tesla Inc." in companies
        assert "Apple Inc." in companies
        assert len(companies) >= 5

    def test_evaluation_pairs(self):
        from src.data.sample_data import get_evaluation_pairs
        pairs = get_evaluation_pairs()
        assert len(pairs) >= 10
        for pair in pairs:
            assert "question" in pair
            assert "ground_truth" in pair


class TestDocumentParser:
    """Tests for document parser."""

    def test_chunk_text(self):
        from src.utils.helpers import chunk_text
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 1200  # Some tolerance

    def test_clean_text(self):
        from src.utils.helpers import clean_text
        dirty = "  <p>Hello</p>  <br>  World  "
        clean = clean_text(dirty)
        assert "<p>" not in clean
        assert clean.strip() == clean

    def test_generate_doc_id(self):
        from src.utils.helpers import generate_doc_id
        id1 = generate_doc_id("Hello world")
        id2 = generate_doc_id("Different text")
        assert id1 != id2
        assert len(id1) == 12

    def test_process_sample_documents(self):
        from src.data.document_parser import DocumentParser
        from src.data.sample_data import get_all_documents
        
        parser = DocumentParser()
        docs = get_all_documents()
        processed = parser.process_sample_documents(docs)
        
        assert len(processed) == len(docs)
        for doc in processed:
            assert "doc_id" in doc
            assert "content" in doc


# ============================================================================
# Unit Tests: Embedding Layer
# ============================================================================

class TestSparseEmbedder:
    """Tests for BM25 sparse retrieval."""

    def test_tokenize(self):
        from src.embeddings.sparse_embedder import SparseEmbedder
        embedder = SparseEmbedder()
        tokens = embedder.tokenize("Tesla reported revenue of $96 billion")
        assert "tesla" in tokens
        assert "revenue" in tokens
        assert len(tokens) > 0

    def test_build_and_search(self, tmp_path):
        from src.embeddings.sparse_embedder import SparseEmbedder
        from src.data.sample_data import get_all_documents
        from src.data.document_parser import DocumentParser
        
        parser = DocumentParser()
        docs = parser.process_sample_documents(get_all_documents())
        
        embedder = SparseEmbedder(persist_path=str(tmp_path / "test_bm25.pkl"))
        embedder.build_index(docs)
        
        results = embedder.search("Tesla revenue 2023", top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0

    def test_stats(self):
        from src.embeddings.sparse_embedder import SparseEmbedder
        embedder = SparseEmbedder()
        stats = embedder.get_stats()
        assert "total_documents" in stats
        assert "index_built" in stats


# ============================================================================
# Unit Tests: Retrieval Layer
# ============================================================================

class TestHybridRetriever:
    """Tests for hybrid retrieval."""

    def test_rrf_fusion(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.alpha = 0.5
        retriever.rrf_k = 60
        
        dense_results = [
            {"doc_id": "d1", "content": "doc 1", "score": 0.9},
            {"doc_id": "d2", "content": "doc 2", "score": 0.8},
            {"doc_id": "d3", "content": "doc 3", "score": 0.7},
        ]
        
        sparse_results = [
            {"doc_id": "d2", "content": "doc 2", "score": 5.0},
            {"doc_id": "d4", "content": "doc 4", "score": 4.0},
            {"doc_id": "d1", "content": "doc 1", "score": 3.0},
        ]
        
        merged = retriever.reciprocal_rank_fusion(dense_results, sparse_results)
        
        assert len(merged) == 4  # d1, d2, d3, d4
        
        # d2 should rank high (appears in both)
        doc_ids = [d["doc_id"] for d in merged]
        assert "d2" in doc_ids[:2] or "d1" in doc_ids[:2]
        
        # All should have RRF scores
        for doc in merged:
            assert "rrf_score" in doc
            assert doc["retrieval_type"] == "hybrid"


class TestMultiQuery:
    """Tests for multi-query generation."""

    def test_template_generation(self):
        from src.retrieval.multi_query import MultiQueryGenerator
        gen = MultiQueryGenerator(num_queries=3)
        
        queries = gen.generate_queries_template(
            "What are Tesla's risk factors?"
        )
        
        assert len(queries) > 0
        for q in queries:
            assert len(q) > 5

    def test_generate_includes_original(self):
        from src.retrieval.multi_query import MultiQueryGenerator
        gen = MultiQueryGenerator(num_queries=3)
        
        original = "What was Apple's revenue?"
        queries = gen.generate(original)
        
        assert original in queries
        assert len(queries) >= 2


# ============================================================================
# Unit Tests: Evaluation
# ============================================================================

class TestEvaluationMetrics:
    """Tests for evaluation metric calculations."""

    def test_precision_at_k(self):
        from src.evaluation.eval_suite import EvaluationSuite
        suite = EvaluationSuite()
        
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = ["d1", "d3", "d6"]
        
        p = suite.precision_at_k(retrieved, relevant, k=5)
        assert p == 2 / 5  # d1, d3 are relevant

    def test_recall_at_k(self):
        from src.evaluation.eval_suite import EvaluationSuite
        suite = EvaluationSuite()
        
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = ["d1", "d3", "d6"]
        
        r = suite.recall_at_k(retrieved, relevant, k=5)
        assert r == 2 / 3  # 2 of 3 relevant found

    def test_mrr(self):
        from src.evaluation.eval_suite import EvaluationSuite
        suite = EvaluationSuite()
        
        # Relevant doc at position 3 (0-indexed)
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = ["d3"]
        
        mrr = suite.mean_reciprocal_rank(retrieved, relevant)
        assert mrr == 1 / 3  # d3 is at rank 3

    def test_ndcg(self):
        from src.evaluation.eval_suite import EvaluationSuite
        suite = EvaluationSuite()
        
        retrieved = ["d1", "d2", "d3"]
        relevant = ["d1", "d3"]  # Perfect would be d1, d3, ...
        
        ndcg = suite.ndcg_at_k(retrieved, relevant, k=3)
        assert 0 <= ndcg <= 1
        assert ndcg > 0  # Should have some relevance

    def test_answer_quality(self):
        from src.evaluation.eval_suite import EvaluationSuite
        suite = EvaluationSuite()
        
        metrics = suite.evaluate_answer_quality(
            question="What was Tesla's revenue?",
            answer="Tesla reported revenue of $96.77 billion in 2023 [Source 1].",
            contexts=["Tesla's total revenue for fiscal year 2023 was $96.77 billion."],
            ground_truth="Tesla's total revenue in 2023 was $96.77 billion."
        )
        
        assert metrics["has_citations"] is True
        assert metrics["num_citations"] >= 1
        assert metrics["factual_overlap"] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full RAG pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up a temporary environment for tests."""
        # Override settings for test
        os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")
        os.environ["LOG_DIR"] = str(tmp_path / "logs")

    def test_classify_query(self):
        """Test query classification logic without loading models."""
        # Build a minimal RAGPipeline without initializing heavy models
        import types
        
        # Import only the classify method logic
        from src.agents.rag_agent import RAGPipeline
        pipeline = object.__new__(RAGPipeline)
        # Manually set attributes needed for classify_query only
        pipeline.max_retries = 2
        
        state = {
            "query": "Compare Tesla and Ford's risk factors",
            "query_type": "",
        }
        state = pipeline.classify_query(state)
        assert state["query_type"] == "comparison"
        
        state["query"] = "What was Tesla's revenue?"
        state = pipeline.classify_query(state)
        assert state["query_type"] == "simple"
        
        state["query"] = "How has Tesla's revenue changed over time?"
        state = pipeline.classify_query(state)
        assert state["query_type"] == "temporal"


class TestReranker:
    """Tests for cross-encoder reranker."""

    def test_rerank_ordering(self):
        """Test that reranker changes document ordering."""
        from src.retrieval.reranker import CrossEncoderReranker
        
        def fake_score(query, document):
            query_terms = {t.lower() for t in query.split()}
            return sum(1 for term in query_terms if term in document.lower())

        reranker = CrossEncoderReranker(scoring_fn=fake_score)
        
        docs = [
            {"doc_id": "1", "content": "The weather is nice today", "rrf_score": 0.9},
            {"doc_id": "2", "content": "Tesla reported revenue of $96 billion", "rrf_score": 0.8},
            {"doc_id": "3", "content": "Financial performance was strong in Q4", "rrf_score": 0.7},
        ]
        
        reranked = reranker.rerank("Tesla revenue", docs, top_k=3)
        
        assert len(reranked) == 3
        # Tesla revenue doc should be ranked higher after reranking
        assert reranked[0]["doc_id"] == "2"


# ============================================================================
# API Tests
# ============================================================================

class TestFastAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_companies_endpoint(self):
        from fastapi.testclient import TestClient
        from api import app
        
        client = TestClient(app)
        response = client.get("/companies")
        
        assert response.status_code == 200
        data = response.json()
        assert "companies" in data
        assert "demo_mode" in data


# ============================================================================
# Run tests with: python -m pytest tests/ -v
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
