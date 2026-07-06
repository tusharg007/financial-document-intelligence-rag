"""Build dense and sparse indexes from processed chunks."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from config.settings import PROJECT_ROOT
from src.data.chunking import load_chunks
from src.embeddings.dense_embedder import DenseEmbedder
from src.embeddings.sparse_embedder import SparseEmbedder


def build_bm25(
    chunks: List[Dict[str, Any]],
    rebuild: bool = False,
    persist_path: str | Path | None = None,
) -> SparseEmbedder:
    path = Path(persist_path) if persist_path else PROJECT_ROOT / "data" / "indexes" / "bm25" / "bm25_index.pkl"
    sparse = SparseEmbedder(persist_path=str(path))
    if rebuild:
        sparse._cleanup_persisted_files()
    sparse.build_index(chunks)
    return sparse


def build_dense(chunks: List[Dict[str, Any]], rebuild: bool = False) -> DenseEmbedder:
    persist_dir = PROJECT_ROOT / "data" / "indexes" / "chroma"
    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
    try:
        dense = DenseEmbedder(persist_dir=str(persist_dir))
        dense.add_documents(chunks)
    except Exception as exc:
        raise RuntimeError(
            f"Dense index build failed for {persist_dir}. "
            f"Chroma import/init or upsert error: {exc}"
        ) from exc
    return dense


def build_indexes(chunks_path: str | Path | None = None, rebuild: bool = False, skip_dense: bool = False) -> Dict[str, Any]:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise FileNotFoundError("No chunks found. Run SEC ingestion/chunking or scripts/prepare_datasets.py first.")
    sparse = build_bm25(chunks, rebuild=rebuild)
    result = {"chunks": len(chunks), "bm25": sparse.get_stats()}
    if not skip_dense:
        dense = build_dense(chunks, rebuild=rebuild)
        result["dense"] = dense.get_collection_stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=str(PROJECT_ROOT / "data" / "processed" / "chunks.parquet"))
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of both indexes.")
    parser.add_argument("--incremental", action="store_true", help="Do not clear existing dense/BM25 indexes before building.")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()
    rebuild = True
    if args.incremental:
        rebuild = False
    if args.rebuild:
        rebuild = True
    result = build_indexes(args.chunks, rebuild=rebuild, skip_dense=args.skip_dense)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
