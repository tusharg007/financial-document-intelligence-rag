"""Section-aware chunking utilities for processed financial documents."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config.settings import PROJECT_ROOT, settings
from src.utils.helpers import generate_doc_id


REQUIRED_METADATA = [
    "ticker",
    "company",
    "form_type",
    "filing_date",
    "fiscal_year",
    "fiscal_period",
    "section",
    "accession_number",
    "source_url",
]

TOC_PATTERNS = [
    r"table\s+of\s+contents",
    r"item\s+1a[\.\-:\s]+risk\s+factors",
    r"item\s+1b[\.\-:\s]+unresolved\s+staff\s+comments",
    r"item\s+2[\.\-:\s]+unregistered\s+sales",
    r"item\s+3[\.\-:\s]+defaults",
    r"item\s+4[\.\-:\s]+mine\s+safety\s+disclosures",
    r"item\s+5[\.\-:\s]+other\s+information",
    r"item\s+6[\.\-:\s]+exhibits",
]
BOILERPLATE_PATTERNS = [
    r"forward-looking\s+statements",
    r"no\s+obligation\s+to\s+revise\s+or\s+update",
    r"except\s+as\s+required\s+by\s+law",
    r"unless\s+otherwise\s+stated",
]
SUBSTANTIVE_HINTS = [
    "risk",
    "competition",
    "demand",
    "manufacturing",
    "supply chain",
    "regulatory",
    "financial condition",
    "results of operations",
    "business",
    "revenue",
    "customers",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[str]:
    """Split text without crossing section boundaries supplied by the caller."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    text = _normalize_text(text)
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start + int(chunk_size * 0.6):
                end = split_at
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def is_toc_like(text: str) -> bool:
    normalized = _normalize_text(text).lower()
    toc_hits = sum(1 for pattern in TOC_PATTERNS if re.search(pattern, normalized, flags=re.I))
    item_hits = len(re.findall(r"\bitem\s+\d+[a-z]?\b", normalized))
    punctuation_hits = len(re.findall(r"[.!?]", normalized))
    return "table of contents" in normalized or toc_hits >= 2 or (item_hits >= 4 and punctuation_hits <= 2)


def boilerplate_score(text: str) -> float:
    normalized = _normalize_text(text).lower()
    hits = sum(1 for pattern in BOILERPLATE_PATTERNS if re.search(pattern, normalized, flags=re.I))
    return min(1.0, hits / max(len(BOILERPLATE_PATTERNS), 1))


def content_quality_score(text: str, section: str = "") -> float:
    normalized = _normalize_text(text)
    lowered = normalized.lower()
    score = 0.3
    if len(normalized) >= 180:
        score += 0.15
    if len(normalized) >= 500:
        score += 0.1
    score += min(0.2, sum(0.04 for hint in SUBSTANTIVE_HINTS if hint in lowered))
    sentence_count = len(re.findall(r"[.!?]", normalized))
    if sentence_count >= 2:
        score += 0.1
    if section in {"Financial Statements", "Notes", "Table"} and re.search(r"\$|\bnet income\b|\brevenue\b|\bassets\b", lowered):
        score += 0.15
    if is_toc_like(normalized):
        score -= 0.55
    score -= boilerplate_score(normalized) * 0.35
    return round(max(0.0, min(1.0, score)), 3)


def _should_exclude_chunk(section: str, text: str, quality: Dict[str, Any]) -> bool:
    normalized = _normalize_text(text)
    if len(normalized) < 100:
        return True
    if quality["is_toc_like"] and quality["content_quality_score"] < 0.3:
        return True
    if section in {"Risk Factors", "Business", "MD&A", "Quantitative and Qualitative Disclosures"}:
        if quality["boilerplate_score"] >= 0.7 and quality["content_quality_score"] < 0.4:
            return True
    return False


def assess_chunk_quality(text: str, section: str = "", section_confidence: float | None = None) -> Dict[str, Any]:
    quality = {
        "is_toc_like": is_toc_like(text),
        "boilerplate_score": boilerplate_score(text),
        "content_quality_score": content_quality_score(text, section=section),
    }
    quality["section_confidence"] = (
        float(section_confidence)
        if section_confidence not in (None, "")
        else 0.0
    )
    return quality


def chunk_sections(
    sections: Iterable[Dict[str, Any]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Dict[str, Any]]:
    """Create chunks from parser section records while preserving metadata."""
    output: List[Dict[str, Any]] = []
    for section in sections:
        base_meta = {k: section.get(k, "") for k in REQUIRED_METADATA}
        base_meta["section"] = section.get("section", base_meta.get("section", "Full Document"))
        if section.get("table_id"):
            base_meta["table_id"] = section["table_id"]
        section_confidence = float(section.get("section_confidence", 0.0) or 0.0)
        raw_text = section.get("text", "")
        for idx, text in enumerate(split_text(raw_text, chunk_size, chunk_overlap)):
            quality = assess_chunk_quality(text, section=base_meta["section"], section_confidence=section_confidence)
            if _should_exclude_chunk(base_meta["section"], text, quality):
                continue
            record = {
                "doc_id": generate_doc_id(text, {**base_meta, "chunk_index": idx}),
                "content": text,
                "chunk_index": idx,
                **base_meta,
                **quality,
            }
            output.append(record)
    return output


def save_chunks(chunks: List[Dict[str, Any]], output_dir: str | Path | None = None) -> Dict[str, str]:
    """Save chunks to JSONL and Parquet."""
    output_dir = Path(output_dir or PROJECT_ROOT / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "chunks.jsonl"
    parquet_path = output_dir / "chunks.parquet"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    try:
        pd.DataFrame(chunks).to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = output_dir / "chunks.parquet.unavailable"
        parquet_path.write_text("Parquet support unavailable; use chunks.jsonl.\n", encoding="utf-8")
    return {"jsonl": str(jsonl_path), "parquet": str(parquet_path)}


def load_chunks(path: str | Path | None = None) -> List[Dict[str, Any]]:
    """Load chunks from parquet or JSONL."""
    path = Path(path or PROJECT_ROOT / "data" / "processed" / "chunks.parquet")
    if not path.exists():
        alt = path.with_suffix(".jsonl")
        if alt.exists():
            path = alt
        else:
            return []
    if path.suffix == ".parquet":
        return pd.read_parquet(path).fillna("").to_dict(orient="records")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
