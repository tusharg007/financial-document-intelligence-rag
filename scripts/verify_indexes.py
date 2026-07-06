"""Verify dense ChromaDB and BM25 indexes built from real SEC chunks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT, settings
from src.embeddings.sparse_embedder import SparseEmbedder


def _dense_stats(chroma_dir: Path) -> dict:
    if not chroma_dir.exists():
        return {
            "exists": False,
            "count": 0,
            "sample_metadata": {},
            "error": "Dense index directory does not exist.",
        }
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(settings.chroma_collection_name)
        sample = collection.get(limit=1, include=["metadatas", "documents"])
        sample_metadata = {}
        if sample.get("metadatas"):
            sample_metadata = sample["metadatas"][0] or {}
        if sample.get("documents"):
            sample_metadata["content_preview"] = (sample["documents"][0] or "")[:240]
        return {
            "exists": True,
            "count": collection.count(),
            "sample_metadata": sample_metadata,
            "error": "",
        }
    except Exception as exc:
        return {"exists": True, "count": 0, "sample_metadata": {}, "error": str(exc)}


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    chunks_path = processed_dir / "chunks.parquet"
    bm25_path = PROJECT_ROOT / "data" / "indexes" / "bm25" / "bm25_index.pkl"
    chroma_dir = PROJECT_ROOT / "data" / "indexes" / "chroma"

    if not chunks_path.exists():
        raise SystemExit(f"Missing chunks parquet: {chunks_path}")

    chunks = pd.read_parquet(chunks_path)
    sparse = SparseEmbedder(persist_path=str(bm25_path))
    bm25_reload_works = sparse.load_index()
    bm25_count = len(sparse.documents) if bm25_reload_works else 0
    dense = _dense_stats(chroma_dir)

    sample_bm25_metadata = {}
    if bm25_reload_works and sparse.documents:
        doc = sparse.documents[0]
        sample_bm25_metadata = {
            k: v for k, v in doc.items()
            if k not in {"content"} and isinstance(v, (str, int, float, bool))
        }
        sample_bm25_metadata["content_preview"] = str(doc.get("content", ""))[:240]

    mismatches = []
    if dense["count"] != len(chunks):
        mismatches.append(f"dense index count {dense['count']} != chunks.parquet count {len(chunks)}")
    if bm25_count != len(chunks):
        mismatches.append(f"BM25 index count {bm25_count} != chunks.parquet count {len(chunks)}")
    if dense.get("error"):
        mismatches.append(f"dense index error: {dense['error']}")
    if not bm25_reload_works:
        mismatches.append("BM25 reload failed in fresh SparseEmbedder object")

    report = {
        "number of chunks in chunks.parquet": len(chunks),
        "number of dense-indexed chunks": dense["count"],
        "number of BM25-indexed chunks": bm25_count,
        "dense index exists": dense["exists"],
        "BM25 index exists": bm25_path.exists(),
        "BM25 reload works in fresh object": bm25_reload_works,
        "dense index location": str(chroma_dir),
        "BM25 index location": str(bm25_path),
        "sample dense indexed document metadata": dense["sample_metadata"],
        "sample BM25 indexed document metadata": sample_bm25_metadata,
        "mismatch between chunk count and index count": mismatches,
    }
    print(json.dumps(report, indent=2, default=str))
    if mismatches:
        raise SystemExit("Index verification failed.")


if __name__ == "__main__":
    main()
