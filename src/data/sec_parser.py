"""SEC filing parser with section extraction tuned for real SEC filings."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


SECTION_PRIORITY = {
    "10-K": [
        "Business",
        "Risk Factors",
        "MD&A",
        "Quantitative and Qualitative Disclosures",
        "Financial Statements",
        "Notes",
    ],
    "10-Q": [
        "Financial Statements",
        "MD&A",
        "Quantitative and Qualitative Disclosures",
        "Risk Factors",
        "Notes",
    ],
}

SECTION_PATTERNS = {
    "10-K": {
        "Business": [r"(?im)^\s*item\s+1[\.\-:\s]+business\b"],
        "Risk Factors": [r"(?im)^\s*item\s+1a[\.\-:\s]+risk\s+factors\b"],
        "MD&A": [r"(?im)^\s*item\s+7[\.\-:\s]+management(?:['’]s)?\s+discussion"],
        "Quantitative and Qualitative Disclosures": [
            r"(?im)^\s*item\s+7a[\.\-:\s]+quantitative\s+and\s+qualitative\s+disclosures"
        ],
        "Financial Statements": [r"(?im)^\s*item\s+8[\.\-:\s]+financial\s+statements"],
        "Notes": [
            r"(?im)^\s*notes\s+to\s+(?:consolidated\s+)?financial\s+statements\b",
            r"(?im)^\s*note\s+1[\.\-:\s]+",
        ],
    },
    "10-Q": {
        "Financial Statements": [r"(?im)^\s*item\s+1[\.\-:\s]+financial\s+statements\b"],
        "MD&A": [r"(?im)^\s*item\s+2[\.\-:\s]+management(?:['’]s)?\s+discussion"],
        "Quantitative and Qualitative Disclosures": [
            r"(?im)^\s*item\s+3[\.\-:\s]+quantitative\s+and\s+qualitative\s+disclosures"
        ],
        "Risk Factors": [
            r"(?im)^\s*item\s+1a[\.\-:\s]+risk\s+factors\b",
            r"(?im)^\s*part\s+ii[\.\-:\s]*other\s+information\s*$[\s\S]{0,400}^\s*item\s+1a[\.\-:\s]+risk\s+factors\b",
        ],
        "Notes": [
            r"(?im)^\s*notes\s+to\s+(?:condensed\s+consolidated|consolidated)\s+financial\s+statements\b",
            r"(?im)^\s*note\s+1[\.\-:\s]+",
        ],
    },
}

TOC_PATTERNS = [
    r"table\s+of\s+contents",
    r"item\s+1a[\.\-:\s]+risk\s+factors\s+\d+",
    r"item\s+1b[\.\-:\s]+unresolved\s+staff\s+comments",
    r"item\s+2[\.\-:\s]+unregistered\s+sales",
    r"item\s+3[\.\-:\s]+defaults",
    r"item\s+4[\.\-:\s]+mine\s+safety\s+disclosures",
    r"item\s+5[\.\-:\s]+other\s+information",
    r"item\s+6[\.\-:\s]+exhibits",
]
FORWARD_LOOKING_PATTERNS = [
    r"forward-looking\s+statements",
    r"no\s+obligation\s+to\s+revise\s+or\s+update",
    r"except\s+as\s+required\s+by\s+law",
]
RISK_HINTS = [
    "competition",
    "demand",
    "supply chain",
    "manufacturing",
    "regulatory",
    "legal",
    "cybersecurity",
    "financial condition",
    "results of operations",
    "business",
]


def _clean_inline(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def clean_boilerplate(text: str) -> str:
    """Normalize text while preserving line breaks for section detection."""
    text = html.unescape(text or "")
    text = text.replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def normalize_for_chunking(text: str) -> str:
    return re.sub(r"\s+", " ", clean_boilerplate(text)).strip()


def _is_toc_like(text: str) -> bool:
    lowered = text.lower()
    toc_hits = sum(1 for pattern in TOC_PATTERNS if re.search(pattern, lowered, flags=re.I))
    item_hits = len(re.findall(r"\bitem\s+\d+[a-z]?\b", lowered))
    sentence_hits = len(re.findall(r"[.!?]", text))
    return "table of contents" in lowered or toc_hits >= 2 or (item_hits >= 4 and sentence_hits <= 2)


def _boilerplate_score(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for pattern in FORWARD_LOOKING_PATTERNS if re.search(pattern, lowered, flags=re.I))
    return min(1.0, hits / max(len(FORWARD_LOOKING_PATTERNS), 1))


def _section_score(name: str, candidate_text: str, match_start: int) -> float:
    lowered = candidate_text.lower()
    score = 0.0
    if len(candidate_text) >= 800:
        score += 0.35
    if len(candidate_text) >= 2000:
        score += 0.2
    if name == "Risk Factors":
        score += min(0.25, sum(0.05 for hint in RISK_HINTS if hint in lowered))
    if re.search(r"\bmay\b|\bcould\b|\brisk\b|\badverse\b", lowered):
        score += 0.15
    if _is_toc_like(candidate_text[:1200]):
        score -= 0.65
    score -= _boilerplate_score(candidate_text[:1500]) * 0.3
    if match_start < max(800, len(candidate_text) // 20):
        score -= 0.1
    return score


def _build_section_candidates(text: str, form_type: str) -> Dict[str, List[Dict[str, Any]]]:
    patterns = SECTION_PATTERNS.get(form_type.upper(), SECTION_PATTERNS.get("10-K", {}))
    candidates: Dict[str, List[Dict[str, Any]]] = {name: [] for name in patterns}

    all_headers = []
    for section_name, regexes in patterns.items():
        for pattern in regexes:
            for match in re.finditer(pattern, text):
                all_headers.append((match.start(), match.end(), section_name))

    all_headers.sort()
    if not all_headers:
        return candidates

    for section_name, regexes in patterns.items():
        for pattern in regexes:
            for match in re.finditer(pattern, text):
                start = match.start()
                following_headers = [header_start for header_start, _, _ in all_headers if header_start > start]
                end = following_headers[0] if following_headers else len(text)
                candidate_text = clean_boilerplate(text[start:end])
                if not candidate_text:
                    continue
                score = _section_score(section_name, candidate_text, start)
                candidates[section_name].append({
                    "start": start,
                    "end": end,
                    "text": candidate_text,
                    "score": score,
                })
    return candidates


def _choose_sections(text: str, form_type: str) -> List[Dict[str, Any]]:
    candidates = _build_section_candidates(text, form_type)
    selected: List[Dict[str, Any]] = []
    for section_name in SECTION_PRIORITY.get(form_type.upper(), SECTION_PRIORITY["10-K"]):
        options = sorted(candidates.get(section_name, []), key=lambda item: item["score"], reverse=True)
        if not options:
            continue
        best = options[0]
        section_confidence = max(0.0, min(1.0, 0.5 + best["score"]))
        selected.append({
            "section": section_name,
            "start": best["start"],
            "text": best["text"],
            "section_confidence": round(section_confidence, 3),
        })

    if not selected:
        return []

    selected.sort(key=lambda item: item["start"])
    carved: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected):
        start = item["start"]
        end = selected[idx + 1]["start"] if idx + 1 < len(selected) else len(text)
        section_text = clean_boilerplate(text[start:end])
        if not section_text:
            continue
        if item["section"] == "Risk Factors" and _is_toc_like(section_text[:1600]):
            continue
        carved.append({
            "section": item["section"],
            "text": section_text,
            "section_confidence": item["section_confidence"],
        })
    return carved


def html_to_text(raw: str) -> str:
    if BeautifulSoup is None:
        text = re.sub(r"(?i)<br\s*/?>", "\n", raw or "")
        text = re.sub(r"(?i)</?(div|p|tr|li|table|h\d|section|article)[^>]*>", "\n", text)
        return clean_boilerplate(re.sub(r"<[^>]+>", " ", text))

    soup = BeautifulSoup(raw or "", "lxml")
    for tag in soup(["script", "style", "ix:header", "ix:hidden", "meta", "link", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=True):
        attrs = getattr(tag, "attrs", None) or {}
        style = str(attrs.get("style", "")).lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()
    text = soup.get_text("\n")
    return clean_boilerplate(text)


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
            "section_confidence": 1.0,
            "table_id": f"table_{i}",
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "text": df.to_csv(index=False),
        })
    return tables


def extract_sections(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    cleaned = clean_boilerplate(text)
    form_type = str(metadata.get("form_type", "")).upper()
    sections = _choose_sections(cleaned, form_type)
    if not sections:
        return [{
            **metadata,
            "section": "Full Document",
            "section_confidence": 0.2,
            "text": cleaned,
        }]

    records: List[Dict[str, Any]] = []
    for section in sections:
        records.append({
            **metadata,
            "section": section["section"],
            "section_confidence": section["section_confidence"],
            "text": section["text"],
        })
    return records


def parse_filing_file(path: str | Path, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    is_html = path.suffix.lower() in {".html", ".htm"} or "<html" in raw[:2000].lower()
    text = html_to_text(raw) if is_html else clean_boilerplate(raw)
    return extract_sections(text, metadata) + extract_tables(raw, metadata)
