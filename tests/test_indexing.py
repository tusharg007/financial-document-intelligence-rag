def test_build_bm25_index(tmp_path, monkeypatch):
    from src.indexing.build_indexes import build_bm25

    chunks = [{"doc_id": "1", "content": "financial revenue", "ticker": "T"}]
    sparse = build_bm25(chunks, rebuild=True)
    assert sparse.get_stats()["index_built"] is True
