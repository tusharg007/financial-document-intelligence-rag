"""Generate reports/dataset_card.md from actual local files."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT
from src.data.chunking import load_chunks


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_dataset_card() -> Dict[str, Any]:
    processed = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    manifest_path = processed / "filing_manifest.csv"
    filings: List[Dict[str, Any]] = []
    if manifest_path.exists():
        filings = pd.read_csv(manifest_path).fillna("").to_dict(orient="records")
    companies = {row.get("company") for row in filings if row.get("company")} or {c.get("company") for c in chunks if c.get("company")}
    forms = Counter((row.get("form_type") or row.get("filing_type") or "") for row in filings or chunks)
    years = [int(str(row.get("fiscal_year", "0"))[:4]) for row in filings or chunks if str(row.get("fiscal_year", "")).isdigit()]
    sections = Counter(c.get("section", "") for c in chunks)
    eval_counts = {
        "financebench": _count_jsonl(processed / "eval_financebench.jsonl"),
        "tatqa": _count_jsonl(processed / "eval_tatqa.jsonl"),
        "finqa": _count_jsonl(processed / "eval_finqa.jsonl"),
        "fiqa": _count_jsonl(processed / "eval_fiqa.jsonl"),
        "demo": _count_jsonl(processed / "eval_demo.jsonl"),
    }
    stats = {
        "companies": len([c for c in companies if c]),
        "filings": len(filings),
        "chunks": len(chunks),
        "filing_types": dict(forms),
        "year_coverage": [min(years), max(years)] if years else [],
        "sections": dict(sections),
        "tables": sum(1 for c in chunks if c.get("table_id")),
        "qa_pairs": sum(eval_counts.values()),
        "source_breakdown": eval_counts,
    }
    lines = [
        "# Dataset Card",
        "",
        "All numbers below are computed from local files. Missing values mean the relevant ingestion/preparation step has not been run.",
        "",
        f"- Companies: {stats['companies']}",
        f"- Filings: {stats['filings']}",
        f"- Chunks: {stats['chunks']}",
        f"- Filing types: {json.dumps(stats['filing_types'])}",
        f"- Year coverage: {stats['year_coverage'] or 'not available'}",
        f"- Sections extracted: {json.dumps(stats['sections'])}",
        f"- Tables: {stats['tables']}",
        f"- QA pairs: {stats['qa_pairs']}",
        f"- Source breakdown: {json.dumps(stats['source_breakdown'])}",
    ]
    (reports / "dataset_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(build_dataset_card(), indent=2))
