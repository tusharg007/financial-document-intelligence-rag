"""
Hybrid retrieval module combining dense and sparse retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge results from dense (semantic)
and sparse (BM25) retrieval into a single ranked list.
"""
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.embeddings.dense_embedder import DenseEmbedder, get_dense_embedder
from src.embeddings.sparse_embedder import SparseEmbedder, get_sparse_embedder
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("hybrid_retriever")


class HybridRetriever:
    """
    Combines dense (semantic) and sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF).
    
    RRF is a parameter-free rank fusion method that produces robust
    merged rankings without requiring score normalization.
    
    RRF Score = Σ (1 / (k + rank_i)) for each ranking system
    where k is a constant (default 60)
    """

    def __init__(
        self,
        dense_embedder: DenseEmbedder = None,
        sparse_embedder: SparseEmbedder = None,
        alpha: float = None,
        rrf_k: int = 60
    ):
        """
        Args:
            dense_embedder: Dense retrieval instance
            sparse_embedder: Sparse retrieval instance  
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only)
            rrf_k: RRF constant (higher = more weight to top ranks)
        """
        self.dense = dense_embedder or get_dense_embedder()
        self.sparse = sparse_embedder or get_sparse_embedder()
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        alpha: float = None
    ) -> List[Dict[str, Any]]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            alpha: Weight for dense (1-alpha for sparse)
            
        Returns:
            Merged and re-ranked results
        """
        alpha = alpha if alpha is not None else self.alpha
        k = self.rrf_k
        
        # Build RRF scores
        rrf_scores = defaultdict(float)
        doc_map = {}  # doc_id -> full document info
        
        # Dense scores (weighted by alpha)
        for rank, doc in enumerate(dense_results):
            doc_id = doc["doc_id"]
            rrf_scores[doc_id] += alpha * (1.0 / (k + rank + 1))
            doc_map[doc_id] = doc
        
        # Sparse scores (weighted by 1-alpha)
        for rank, doc in enumerate(sparse_results):
            doc_id = doc["doc_id"]
            rrf_scores[doc_id] += (1 - alpha) * (1.0 / (k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        # Sort by RRF score
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )
        
        # Build final results
        merged = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = rrf_scores[doc_id]
            doc["retrieval_type"] = "hybrid"
            
            # Track which systems found this document
            in_dense = any(d["doc_id"] == doc_id for d in dense_results)
            in_sparse = any(d["doc_id"] == doc_id for d in sparse_results)
            doc["found_by"] = []
            if in_dense:
                doc["found_by"].append("dense")
            if in_sparse:
                doc["found_by"].append("sparse")
            
            merged.append(doc)
        
        return merged

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        where: Dict = None,
        filters: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining dense and sparse search.
        
        Args:
            query: User query
            top_k: Number of results to return
            alpha: Override weight for dense vs sparse
            where: Metadata filter for dense search
            
        Returns:
            Hybrid-ranked list of documents
        """
        top_k = top_k or settings.dense_top_k
        
        start = time.time()
        
        filters = filters or where or {}
        dense_where = where or self._to_chroma_where(filters)

        # Run both retrievers
        dense_results = self.dense.search(query, top_k=top_k, where=dense_where)
        sparse_results = self.sparse.search(query, top_k=top_k, filters=filters)
        
        # Merge with RRF
        merged = self.reciprocal_rank_fusion(
            dense_results, sparse_results, alpha=alpha
        )
        
        elapsed = time.time() - start
        
        logger.info(
            f"Hybrid retrieval: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse -> {len(merged)} merged "
            f"({elapsed:.3f}s) | Query: {query[:50]}..."
        )
        
        return merged[:top_k]

    @staticmethod
    def _to_chroma_where(filters: Dict[str, Any] = None):
        """Convert simple equality filters to Chroma where syntax."""
        if not filters:
            return None
        clean = {k: v for k, v in filters.items() if v not in ("", None, [], {})}
        if not clean:
            return None
        if len(clean) == 1:
            return clean
        return {"$and": [{k: v} for k, v in clean.items()]}

    def retrieve_dense_only(
        self,
        query: str,
        top_k: int = None,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        """Dense-only retrieval (for benchmarking)."""
        return self.dense.search(query, top_k=top_k, where=where)

    def retrieve_sparse_only(
        self,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Sparse-only retrieval (for benchmarking)."""
        return self.sparse.search(query, top_k=top_k)

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics for both retrieval systems."""
        return {
            "dense": self.dense.get_collection_stats(),
            "sparse": self.sparse.get_stats(),
            "hybrid_alpha": self.alpha,
            "rrf_k": self.rrf_k,
        }


# Singleton
_retriever = None

def get_hybrid_retriever() -> HybridRetriever:
    """Get the global HybridRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
