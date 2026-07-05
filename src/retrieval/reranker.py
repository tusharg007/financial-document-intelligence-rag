"""
Cross-encoder reranking module.

Takes candidate documents from hybrid retrieval and reranks them
using a fine-grained cross-encoder model for precise relevance scoring.
"""
import time
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("reranker")


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model.
    
    Cross-encoders process query-document pairs together, producing
    more accurate relevance scores than bi-encoders but at higher
    computational cost. Used as a second-stage ranker on the top-K
    results from hybrid retrieval.
    
    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    This model is trained on the MS MARCO passage ranking dataset.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker_model_id
        self._model = None
        self.available = True

    @property
    def model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading reranker model: {self.model_name}")
            start = time.time()
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, max_length=512)
                elapsed = time.time() - start
                logger.info(f"Reranker loaded in {elapsed:.2f}s")
            except Exception as e:
                self.available = False
                logger.warning(f"Reranker unavailable ({e}); using lexical fallback.")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using the cross-encoder.
        
        Args:
            query: User query
            documents: List of candidate documents from retrieval
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with cross-encoder scores
        """
        if not documents:
            return []

        top_k = top_k or settings.rerank_top_k
        
        start = time.time()

        model = self.model
        if model is None:
            return self._lexical_rerank(query, documents, top_k)

        # Create query-document pairs
        pairs = [[query, doc["content"]] for doc in documents]

        # Get cross-encoder scores
        scores = model.predict(pairs)

        # Attach scores and sort
        scored_docs = []
        for doc, score in zip(documents, scores):
            scored_doc = doc.copy()
            scored_doc["rerank_score"] = float(score)
            scored_doc["original_score"] = doc.get("rrf_score", doc.get("score", 0))
            scored_docs.append(scored_doc)

        # Sort by rerank score (descending)
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        elapsed = time.time() - start
        logger.info(
            f"Reranked {len(documents)} docs -> top {top_k} in {elapsed:.3f}s"
        )

        return scored_docs[:top_k]

    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair."""
        model = self.model
        if model is None:
            return self._lexical_score(query, document)
        return float(model.predict([[query, document]])[0])

    def _lexical_score(self, query: str, document: str) -> float:
        q_terms = {t.lower() for t in query.split() if len(t) > 2}
        d = document.lower()
        return float(sum(1 for t in q_terms if t in d)) / max(len(q_terms), 1)

    def _lexical_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        scored_docs = []
        for doc in documents:
            scored = doc.copy()
            scored["rerank_score"] = self._lexical_score(query, doc.get("content", ""))
            scored["original_score"] = doc.get("rrf_score", doc.get("score", 0))
            scored["rerank_fallback"] = "lexical"
            scored_docs.append(scored)
        scored_docs.sort(
            key=lambda x: (x["rerank_score"], x.get("original_score", 0)),
            reverse=True
        )
        return scored_docs[:top_k]


# Singleton
_reranker = None

def get_reranker() -> CrossEncoderReranker:
    """Get the global CrossEncoderReranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
