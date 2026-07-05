def test_chunk_sections_preserves_metadata(tmp_path):
    from src.data.chunking import chunk_sections, save_chunks

    sections = [{"text": "Revenue " * 300, "ticker": "TSLA", "company": "Tesla", "form_type": "10-K", "section": "MD&A"}]
    chunks = chunk_sections(sections, chunk_size=100, chunk_overlap=10)
    assert chunks
    assert chunks[0]["ticker"] == "TSLA"
    paths = save_chunks(chunks, tmp_path)
    assert paths["jsonl"]
