def test_sec_ingestor_requires_user_agent(monkeypatch, tmp_path):
    from src.data.sec_edgar_ingestion import SecEdgarIngestor

    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "")
    try:
        SecEdgarIngestor(user_agent="", raw_dir=tmp_path)
    except ValueError as exc:
        assert "SEC_EDGAR_USER_AGENT" in str(exc)
