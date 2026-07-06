def test_chunk_sections_preserves_metadata(tmp_path):
    from src.data.chunking import chunk_sections, save_chunks

    sections = [{
        "text": "Revenue growth remained strong and customer demand improved. " * 40,
        "ticker": "TSLA",
        "company": "Tesla",
        "form_type": "10-K",
        "section": "MD&A",
        "source_url": "https://www.sec.gov/example",
        "filing_date": "2024-01-01",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "accession_number": "abc",
        "section_confidence": 0.8,
    }]
    chunks = chunk_sections(sections, chunk_size=300, chunk_overlap=30)
    assert chunks
    assert chunks[0]["ticker"] == "TSLA"
    assert chunks[0]["source_url"] == "https://www.sec.gov/example"
    assert "content_quality_score" in chunks[0]
    assert "is_toc_like" in chunks[0]
    paths = save_chunks(chunks, tmp_path)
    assert paths["jsonl"]
