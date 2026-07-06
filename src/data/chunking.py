"""Section-aware chunking utilities for processed financial documents."""
from __future__ import annotations

import json
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


def split_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[str]:
    """Split text without crossing section boundaries supplied by the caller."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


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
        for idx, text in enumerate(split_text(section.get("text", ""), chunk_size, chunk_overlap)):
            if len(text.strip()) < 100:
                continue
            record = {
                "doc_id": generate_doc_id(text, {**base_meta, "chunk_index": idx}),
                "content": text,
                "chunk_index": idx,
                **base_meta,
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
