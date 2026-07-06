"""Query the production retrieval pipeline against the built SEC indexes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval.pipeline import get_retrieval_pipeline


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
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    pipeline = get_retrieval_pipeline()
    results = pipeline.retrieve(
        args.query,
        top_k=args.top_k,
        filters=_filters_from_args(args),
        use_reranker=not args.no_rerank,
    )

    for rank, result in enumerate(results, start=1):
        print(f"rank: {rank}")
        print(f"ticker: {result.get('ticker', '')}")
        print(f"company: {result.get('company', '')}")
        print(f"form_type: {result.get('form_type', '')}")
        print(f"filing_date: {result.get('filing_date', '')}")
        print(f"fiscal_year: {result.get('fiscal_year', '')}")
        print(f"section: {result.get('section', '')}")
        print(f"fused score: {result.get('fused_score', 0.0):.6f}")
        print(f"dense score: {result.get('dense_score')}")
        print(f"BM25 score: {result.get('bm25_score')}")
        print(f"reranker score: {result.get('reranker_score')}")
        print(f"source_url: {result.get('source_url', '')}")
        print(f"content preview: {result.get('content_preview', '')}")
        print()


if __name__ == "__main__":
    main()
