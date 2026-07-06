"""Verify that real SEC ingestion produced manifest and chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT


REQUIRED_CHUNK_COLUMNS = [
    "ticker",
    "company",
    "form_type",
    "filing_date",
    "fiscal_year",
    "section",
    "accession_number",
    "source_url",
]


def _read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    manifest_csv = processed / "filing_manifest.csv"
    manifest_parquet = processed / "filing_manifest.parquet"
    chunks_parquet = processed / "chunks.parquet"
    chunks_jsonl = processed / "chunks.jsonl"

    missing_files = [
        str(path)
        for path in [manifest_csv, manifest_parquet, chunks_parquet, chunks_jsonl]
        if not path.exists()
    ]
    if missing_files:
        raise SystemExit("Missing required dataset files:\n" + "\n".join(missing_files))

    manifest = pd.read_csv(manifest_csv).fillna("")
    manifest_pq = pd.read_parquet(manifest_parquet).fillna("")
    chunks = pd.read_parquet(chunks_parquet).fillna("")
    jsonl_count = _read_jsonl_count(chunks_jsonl)

    if len(manifest) != len(manifest_pq):
        raise SystemExit(f"Manifest row mismatch: csv={len(manifest)} parquet={len(manifest_pq)}")
    if len(chunks) != jsonl_count:
        raise SystemExit(f"Chunk count mismatch: parquet={len(chunks)} jsonl={jsonl_count}")

    missing_columns = [col for col in REQUIRED_CHUNK_COLUMNS if col not in chunks.columns]
    empty_metadata = [
        col for col in REQUIRED_CHUNK_COLUMNS
        if col in chunks.columns and chunks[col].astype(str).str.strip().eq("").any()
    ]
    sample_columns = []
    for col in REQUIRED_CHUNK_COLUMNS + ["content"]:
        if col in chunks.columns and col not in sample_columns:
            sample_columns.append(col)

    print(f"number of companies: {manifest['company'].nunique() if 'company' in manifest.columns else 0}")
    print(f"number of filings: {len(manifest)}")
    print(f"number of chunks: {len(chunks)}")
    print(f"chunk columns: {list(chunks.columns)}")
    print(f"missing metadata columns: {missing_columns}")
    print(f"empty required metadata columns: {empty_metadata}")
    print("3 sample chunks with metadata:")
    for row in chunks.head(3)[sample_columns].to_dict(orient="records"):
        preview = dict(row)
        if "content" in preview:
            preview["content"] = str(preview["content"])[:300]
        print(json.dumps(preview, indent=2, default=str))
    print("source URLs:")
    for url in manifest["source_url"].dropna().astype(str).head(10):
        print(url)

    if missing_columns or empty_metadata:
        raise SystemExit("Dataset verification failed: required metadata is missing or empty.")


if __name__ == "__main__":
    main()
