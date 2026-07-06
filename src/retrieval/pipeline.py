"""Production retrieval pipeline for SEC filing chunks."""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.embeddings.dense_embedder import DenseEmbedder, get_dense_embedder
from src.embeddings.sparse_embedder import SparseEmbedder, get_sparse_embedder
from src.retrieval.reranker import CrossEncoderReranker, get_reranker
from src.utils.logger import get_logger

logger = get_logger("retrieval_pipeline")

REQUIRED_METADATA_FIELDS = [
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


class RetrievalPipeline:
    """Hybrid retrieval over dense and sparse SEC filing indexes."""

    def __init__(
        self,
        dense_embedder: Optional[DenseEmbedder] = None,
        sparse_embedder: Optional[SparseEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        alpha: Optional[float] = None,
        rrf_k: int = 60,
        use_reranker: bool = True,
    ):
        self.dense = dense_embedder or get_dense_embedder()
        self.sparse = sparse_embedder or get_sparse_embedder()
        self.reranker = reranker or get_reranker()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker

    @staticmethod
    def _to_chroma_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        clean = {k: v for k, v in filters.items() if v not in ("", None, [], {})}
        if not clean:
            return None
        if len(clean) == 1:
            return clean
        return {"$and": [{k: v} for k, v in clean.items()]}

    @staticmethod
    def _normalize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {}
        if isinstance(doc.get("metadata"), dict):
            metadata.update(doc["metadata"])
        for key, value in doc.items():
            if key in {"metadata", "content", "score", "rrf_score", "rerank_score", "dense_score", "bm25_score", "lexical_overlap_score", "retrieval_type", "found_by", "original_score", "rerank_fallback"}:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        normalized = {
            "doc_id": doc.get("doc_id", ""),
            "content": doc.get("content", ""),
            "content_preview": str(doc.get("content", ""))[:280],
            "metadata": metadata,
        }
        for field in REQUIRED_METADATA_FIELDS:
            if field == "form_type":
                normalized[field] = metadata.get("form_type", metadata.get("filing_type", ""))
            else:
                normalized[field] = metadata.get(field, "")
        return normalized

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        alpha: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Fuse dense and sparse results with weighted Reciprocal Rank Fusion."""
        alpha = alpha if alpha is not None else self.alpha
        k = self.rrf_k
        fused_scores: defaultdict[str, float] = defaultdict(float)
        merged_docs: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results, start=1):
            normalized = self._normalize_document(doc)
            doc_id = normalized["doc_id"]
            merged = merged_docs.setdefault(doc_id, normalized)
            dense_score = float(doc.get("score", 0.0))
            merged["dense_score"] = dense_score
            merged["dense_rank"] = rank
            merged["bm25_score"] = merged.get("bm25_score")
            merged["reranker_score"] = merged.get("reranker_score")
            merged["found_by"] = sorted(set(merged.get("found_by", []) + ["dense"]))
            fused_scores[doc_id] += alpha * (1.0 / (k + rank))

        for rank, doc in enumerate(sparse_results, start=1):
            normalized = self._normalize_document(doc)
            doc_id = normalized["doc_id"]
            merged = merged_docs.setdefault(doc_id, normalized)
            bm25_score = float(doc.get("bm25_score", doc.get("score", 0.0)))
            merged["bm25_score"] = bm25_score
            merged["sparse_rank"] = rank
            merged["dense_score"] = merged.get("dense_score")
            merged["reranker_score"] = merged.get("reranker_score")
            merged["found_by"] = sorted(set(merged.get("found_by", []) + ["sparse"]))
            fused_scores[doc_id] += (1 - alpha) * (1.0 / (k + rank))

        fused = []
        for doc_id, doc in merged_docs.items():
            result = doc.copy()
            result["fused_score"] = fused_scores[doc_id]
            result["retrieval_type"] = "hybrid"
            fused.append(result)

        fused.sort(
            key=lambda item: (
                item.get("fused_score", 0.0),
                item.get("dense_score", float("-inf")) if item.get("dense_score") is not None else float("-inf"),
                item.get("bm25_score", float("-inf")) if item.get("bm25_score") is not None else float("-inf"),
            ),
            reverse=True,
        )
        return fused

    def _dense_available(self) -> bool:
        try:
            return self.dense.collection_exists()
        except Exception as exc:
            logger.warning("Dense retrieval unavailable: %s", exc)
            return False

    def _sparse_available(self) -> bool:
        try:
            return self.sparse.index_exists() or self.sparse.load_index()
        except Exception as exc:
            logger.warning("Sparse retrieval unavailable: %s", exc)
            return False

    def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        available: bool,
    ) -> List[Dict[str, Any]]:
        if not available:
            logger.warning("Dense index is missing; falling back away from Chroma retrieval.")
            return []
        where = self._to_chroma_where(filters)
        try:
            return self.dense.search(query, top_k=top_k, where=where)
        except Exception as exc:
            logger.warning("Dense retrieval failed for query '%s': %s", query, exc)
            return []

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        available: bool,
    ) -> List[Dict[str, Any]]:
        if not available:
            logger.warning("BM25 index is missing; falling back away from sparse retrieval.")
            return []
        try:
            return self.sparse.search(query, top_k=top_k, filters=filters)
        except Exception as exc:
            logger.warning("Sparse retrieval failed for query '%s': %s", query, exc)
            return []

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve SEC chunks with hybrid fusion and optional reranking."""
        if not query or not query.strip():
            raise ValueError("retrieve(query=...) requires a non-empty query.")

        start = time.time()
        candidate_k = max(top_k * 3, top_k)
        dense_available = self._dense_available()
        sparse_available = self._sparse_available()
        dense_results = self._dense_search(query, candidate_k, filters, dense_available)
        sparse_results = self._sparse_search(query, candidate_k, filters, sparse_available)

        if not dense_available and not sparse_available:
            raise RuntimeError(
                "No retrieval backend is available. Build the dense Chroma index and/or BM25 index first, "
                "then rerun scripts/build_indexes.py and scripts/verify_indexes.py."
            )
        if not dense_results and not sparse_results:
            logger.info("Retrieval pipeline returned 0 results for query: %s", query[:80])
            return []

        if dense_results and sparse_results:
            fused = self.reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            source_results = dense_results or sparse_results
            source_name = "dense" if dense_results else "sparse"
            fused = []
            for rank, doc in enumerate(source_results, start=1):
                normalized = self._normalize_document(doc)
                normalized["dense_score"] = float(doc.get("score", 0.0)) if source_name == "dense" else None
                normalized["bm25_score"] = float(doc.get("bm25_score", doc.get("score", 0.0))) if source_name == "sparse" else None
                normalized["fused_score"] = 1.0 / (self.rrf_k + rank)
                normalized["retrieval_type"] = source_name
                normalized["found_by"] = [source_name]
                fused.append(normalized)

        rerank_enabled = self.use_reranker if use_reranker is None else use_reranker
        if rerank_enabled and fused:
            reranked = self.reranker.rerank(query, fused, top_k=min(len(fused), max(top_k, settings.rerank_top_k)))
            reranked_map = {doc["doc_id"]: doc for doc in reranked}
            final_docs = []
            for doc in fused:
                merged = doc.copy()
                if doc["doc_id"] in reranked_map:
                    reranked_doc = reranked_map[doc["doc_id"]]
                    merged["reranker_score"] = reranked_doc.get("rerank_score")
                    if "rerank_fallback" in reranked_doc:
                        merged["rerank_fallback"] = reranked_doc["rerank_fallback"]
                final_docs.append(merged)
            final_docs.sort(
                key=lambda item: (
                    item.get("reranker_score", float("-inf")) if item.get("reranker_score") is not None else float("-inf"),
                    item.get("fused_score", 0.0),
                ),
                reverse=True,
            )
        else:
            final_docs = fused

        elapsed = time.time() - start
        logger.info(
            "Retrieval pipeline returned %s results in %.3fs for query: %s",
            min(top_k, len(final_docs)),
            elapsed,
            query[:80],
        )
        return final_docs[:top_k]


_pipeline: Optional[RetrievalPipeline] = None


def get_retrieval_pipeline() -> RetrievalPipeline:
    """Get the global retrieval pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline
