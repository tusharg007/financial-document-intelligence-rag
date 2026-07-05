from pathlib import Path


def test_bm25_persists_and_reloads(tmp_path):
    from src.embeddings.sparse_embedder import SparseEmbedder

    docs = [
        {"doc_id": "a", "content": "Tesla revenue grew", "ticker": "TSLA"},
        {"doc_id": "b", "content": "Apple supply chain risk", "ticker": "AAPL"},
    ]
    path = tmp_path / "bm25" / "bm25_index.pkl"
    first = SparseEmbedder(persist_path=str(path))
    first.build_index(docs)
    assert path.exists()
    second = SparseEmbedder(persist_path=str(path))
    assert second.load_index()
    assert second.search("Tesla revenue", top_k=1)[0]["doc_id"] == "a"


def test_sparse_metadata_filters(tmp_path):
    from src.embeddings.sparse_embedder import SparseEmbedder

    docs = [
        {"doc_id": "a", "content": "revenue growth", "ticker": "TSLA"},
        {"doc_id": "b", "content": "revenue growth", "ticker": "AAPL"},
    ]
    sparse = SparseEmbedder(persist_path=str(tmp_path / "idx.pkl"))
    sparse.build_index(docs)
    results = sparse.search("revenue", top_k=5, filters={"ticker": "AAPL"})
    assert [r["doc_id"] for r in results] == ["b"]


def test_reranker_fallback_changes_order():
    from src.retrieval.reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker._model = None
    reranker.available = False
    docs = [
        {"doc_id": "weather", "content": "weather today", "score": 0.9},
        {"doc_id": "finance", "content": "Tesla revenue and financial performance", "score": 0.1},
    ]
    ranked = reranker._lexical_rerank("Tesla revenue", docs, top_k=2)
    assert ranked[0]["doc_id"] == "finance"
