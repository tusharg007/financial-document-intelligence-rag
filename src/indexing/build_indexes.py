"""Build dense and sparse indexes from processed chunks."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

from config.settings import PROJECT_ROOT
from src.data.chunking import load_chunks
from src.embeddings.dense_embedder import DenseEmbedder
from src.embeddings.sparse_embedder import SparseEmbedder


def build_bm25(chunks: List[Dict[str, Any]], rebuild: bool = False) -> SparseEmbedder:
    path = PROJECT_ROOT / "data" / "indexes" / "bm25" / "bm25_index.pkl"
    if rebuild and path.exists():
        path.unlink()
    sparse = SparseEmbedder(persist_path=str(path))
    sparse.build_index(chunks)
    return sparse


def build_dense(chunks: List[Dict[str, Any]], rebuild: bool = False) -> DenseEmbedder:
    persist_dir = PROJECT_ROOT / "data" / "indexes" / "chroma"
    if rebuild and persist_dir.exists():
        shutil.rmtree(persist_dir)
    dense = DenseEmbedder(persist_dir=str(persist_dir))
    dense.add_documents(chunks)
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
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()
    print(build_indexes(args.chunks, rebuild=args.rebuild, skip_dense=args.skip_dense))


if __name__ == "__main__":
    main()
