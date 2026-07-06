"""Query the grounded answer generation layer against real SEC indexes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.answering.grounded_answer import get_grounded_answerer


def _filters_from_args(args: argparse.Namespace) -> dict:
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.form_type:
        filters["form_type"] = args.form_type
    if args.section:
        filters["section"] = args.section
    return filters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ticker")
    parser.add_argument("--form-type")
    parser.add_argument("--section")
    parser.add_argument("--provider", default="extractive")
    args = parser.parse_args()

    answerer = get_grounded_answerer()
    result = answerer.answer_question(
        args.query,
        top_k=args.top_k,
        filters=_filters_from_args(args),
        provider_name=args.provider,
    )

    print(f"question: {result['question']}")
    print(f"answer: {result['answer']}")
    print(f"grounding_status: {result['grounding_status']}")
    print(f"used_provider: {result['used_provider']}")
    print(f"warnings: {json.dumps(result.get('warnings', []))}")
    print()
    print("citations:")
    for citation in result.get("citations", []):
        print(f"  [Source {citation['source_num']}] {citation.get('ticker', '')} | {citation.get('company', '')}")
        print(f"    form_type: {citation.get('form_type', '')}")
        print(f"    filing_date: {citation.get('filing_date', '')}")
        print(f"    fiscal_year: {citation.get('fiscal_year', '')}")
        print(f"    section: {citation.get('section', '')}")
        print(f"    accession_number: {citation.get('accession_number', '')}")
        print(f"    source_url: {citation.get('source_url', '')}")
        print(f"    evidence_preview: {citation.get('content_preview', '')}")
        print()


if __name__ == "__main__":
    main()
