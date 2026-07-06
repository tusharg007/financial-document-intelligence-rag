from __future__ import annotations


def test_toc_like_detection():
    from src.data.chunking import is_toc_like

    toc_text = (
        "Item 1A. Risk Factors 19 Item 2. Unregistered Sales of Equity Securities and Use of Proceeds 20 "
        "Item 3. Defaults Upon Senior Securities 20 Item 4. Mine Safety Disclosures 20 Item 5. Other Information 20"
    )
    assert is_toc_like(toc_text) is True


def test_boilerplate_detection():
    from src.data.chunking import boilerplate_score

    text = "Forward-looking statements. The Company assumes no obligation to revise or update any forward-looking statements except as required by law."
    assert boilerplate_score(text) >= 0.6


def test_content_quality_scoring_prefers_substantive_risk_text():
    from src.data.chunking import content_quality_score

    good = (
        "The company's business and results of operations may be adversely affected by competition, supply chain disruptions, "
        "regulatory changes, and lower customer demand in key markets."
    )
    bad = "Item 1A. Risk Factors 19 Item 2. Unregistered Sales 20 Item 3. Defaults 21 Item 4. Mine Safety Disclosures 21"
    assert content_quality_score(good, section="Risk Factors") > content_quality_score(bad, section="Risk Factors")


def test_section_extraction_skips_toc_risk_factor_match():
    from src.data.sec_parser import extract_sections

    text = """
    Table of Contents
    Item 1A. Risk Factors 5
    Item 1B. Unresolved Staff Comments 17
    Item 2. Properties 18

    Item 1. Business
    We design and sell products worldwide.

    Item 1A. Risk Factors
    Our business is subject to intense competition, supply chain constraints, and regulatory uncertainty.
    These risks can materially affect our business, results of operations, and financial condition.

    Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
    Revenue increased during the period.

    Item 8. Financial Statements and Supplementary Data
    Consolidated Statements of Operations follow.
    """
    sections = extract_sections(text, {"form_type": "10-K"})
    risk = next(section for section in sections if section["section"] == "Risk Factors")

    assert "intense competition" in risk["text"]
    assert "Unresolved Staff Comments" not in risk["text"]
    assert risk["section_confidence"] > 0.5


def test_chunk_metadata_preservation_and_source_url():
    from src.data.chunking import chunk_sections

    sections = [{
        "text": (
            "The company faces cybersecurity, competition, and demand risk. "
            "These factors may materially affect results of operations and financial condition. "
        ) * 20,
        "ticker": "AAPL",
        "company": "Apple Inc.",
        "form_type": "10-K",
        "filing_date": "2024-11-01",
        "fiscal_year": 2024,
        "fiscal_period": "FY",
        "section": "Risk Factors",
        "accession_number": "0000320193-24-000123",
        "source_url": "https://www.sec.gov/example-aapl",
        "section_confidence": 0.92,
    }]

    chunks = chunk_sections(sections, chunk_size=300, chunk_overlap=30)

    assert chunks
    assert all(chunk["source_url"] == "https://www.sec.gov/example-aapl" for chunk in chunks)
    assert all("content_quality_score" in chunk for chunk in chunks)
    assert all("section_confidence" in chunk for chunk in chunks)
