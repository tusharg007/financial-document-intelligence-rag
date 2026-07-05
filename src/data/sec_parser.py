"""SEC filing parser with section and table extraction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


SECTION_MARKERS = [
    ("Business", r"item\s+1[\.\s-]+business"),
    ("Risk Factors", r"item\s+1a[\.\s-]+risk\s+factors"),
    ("MD&A", r"item\s+7[\.\s-]+management.?s\s+discussion"),
    ("Quantitative and Qualitative Disclosures", r"item\s+7a[\.\s-]+quantitative\s+and\s+qualitative"),
    ("Financial Statements", r"item\s+8[\.\s-]+financial\s+statements"),
    ("Notes", r"notes\s+to\s+(?:consolidated\s+)?financial\s+statements"),
]


def clean_boilerplate(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"Table of Contents", " ", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def html_to_text(raw: str) -> str:
    if BeautifulSoup is None:
        return clean_boilerplate(re.sub(r"<[^>]+>", " ", raw or ""))
    soup = BeautifulSoup(raw or "", "lxml")
    for tag in soup(["script", "style", "ix:header", "meta"]):
        tag.decompose()
    return clean_boilerplate(soup.get_text(" "))


def extract_tables(raw: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        parsed = pd.read_html(raw)
    except Exception:
        return tables
    for i, df in enumerate(parsed):
        if df.empty:
            continue
        tables.append({
            **metadata,
            "section": "Table",
            "table_id": f"table_{i}",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "text": df.to_csv(index=False),
        })
    return tables


def extract_sections(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = clean_boilerplate(text)
    matches = []
    for name, pattern in SECTION_MARKERS:
        match = re.search(pattern, text, flags=re.I)
        if match:
            matches.append((match.start(), name))
    matches.sort()
    if not matches:
        return [{**metadata, "section": "Full Document", "text": text[:100000]}]

    sections: List[Dict[str, Any]] = []
    for idx, (start, name) in enumerate(matches):
        end = matches[idx + 1][0] if idx + 1 < len(matches) else min(len(text), start + 120000)
        section_text = clean_boilerplate(text[start:end])
        if section_text:
            sections.append({**metadata, "section": name, "text": section_text})
    return sections


def parse_filing_file(path: str | Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    is_html = path.suffix.lower() in {".html", ".htm"} or "<html" in raw[:1000].lower()
    text = html_to_text(raw) if is_html else clean_boilerplate(raw)
    return extract_sections(text, metadata) + extract_tables(raw, metadata)
