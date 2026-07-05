"""
Sparse retrieval module using BM25 (Best Matching 25).

Provides keyword-based retrieval as a complement to dense vector search,
enabling the hybrid retrieval pipeline.
"""
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
try:
    from rank_bm25 import BM25Okapi
except Exception:
    class BM25Okapi:
        """Small fallback scorer for smoke tests when rank_bm25 is unavailable."""

        def __init__(self, corpus):
            self.corpus = corpus

        def get_scores(self, query):
            q = set(query)
            return np.array([sum(1 for token in doc if token in q) for doc in self.corpus], dtype=float)
try:
    import nltk
except Exception:
    nltk = None

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("sparse_embedder")

# Financial domain stopwords to supplement NLTK defaults
FINANCIAL_STOPWORDS = {
    "company", "fiscal", "year", "quarter", "period", "ended",
    "including", "may", "also", "could", "would", "certain",
    "approximately", "respectively", "related", "pursuant",
    "herein", "thereof", "thereto", "hereby",
}


class SparseEmbedder:
    """
    BM25-based sparse retrieval for keyword matching.
    
    Complements dense retrieval by excelling at exact keyword/term matching,
    which is particularly important for financial terminology, ticker symbols,
    and specific numerical values.
    """

    def __init__(self, persist_path: str = None):
        self.persist_path = persist_path or str(
            Path("data") / "indexes" / "bm25" / "bm25_index.pkl"
        )
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self._stopwords = None

    @property
    def stopwords(self):
        """Lazy-load NLTK stopwords + financial domain words."""
        if self._stopwords is None:
            if nltk is None:
                self._stopwords = {
                    "the", "and", "or", "of", "to", "in", "for", "a", "an",
                    "is", "was", "were", "with", "on", "by",
                } | FINANCIAL_STOPWORDS
                return self._stopwords
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words('english')) | FINANCIAL_STOPWORDS
        return self._stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Applies lowercasing, stopword removal, and basic financial
        term preservation (keeps numbers, tickers, etc.).
        """
        if not text:
            return []
        
        # Lowercase
        text = text.lower()
        
        # Split on non-alphanumeric (but keep hyphens in terms like "10-K")
        import re
        tokens = re.findall(r'\b[\w\-]+\b', text)
        
        # Remove stopwords but keep financial terms and numbers
        filtered = [
            token for token in tokens
            if token not in self.stopwords
            or token.replace("-", "").isdigit()
            or len(token) <= 2  # Keep short abbreviations
        ]
        
        return filtered

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document dicts with 'content' key
            
        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0

        start = time.time()
        
        self.documents = documents
        self.tokenized_corpus = [
            self.tokenize(doc["content"]) for doc in documents
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        elapsed = time.time() - start
        logger.info(
            f"Built BM25 index: {len(documents)} documents in {elapsed:.2f}s"
        )
        
        # Persist index
        self.save_index()
        
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25 scoring.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of matching documents with BM25 scores
        """
        if self.bm25 is None and not self.load_index():
            logger.warning("BM25 index not built. Call build_index() first.")
            return []

        top_k = top_k or settings.sparse_top_k
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        if not tokenized_query:
            return []

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][: max(top_k * 5, top_k)]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive scores
                doc = self.documents[idx].copy()
                if filters and any(
                    str(doc.get(k, "")).lower() != str(v).lower()
                    for k, v in filters.items()
                    if v not in ("", None, [], {})
                ):
                    continue
                results.append({
                    "doc_id": doc.get("doc_id", f"doc_{idx}"),
                    "content": doc["content"],
                    "metadata": {
                        k: v for k, v in doc.items()
                        if k not in ("content", "doc_id")
                        and isinstance(v, (str, int, float, bool))
                    },
                    "score": float(scores[idx]),
                    "retrieval_type": "sparse"
                })
                if len(results) >= top_k:
                    break

        logger.info(
            f"BM25 search returned {len(results)} results for: {query[:50]}..."
        )
        return results

    def save_index(self):
        """Persist BM25 index to disk."""
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
        }
        
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"BM25 index saved to {self.persist_path}")

    def load_index(self) -> bool:
        """
        Load BM25 index from disk.
        
        Returns:
            True if index was loaded successfully
        """
        path = Path(self.persist_path)
        if not path.exists():
            return False
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self.bm25 = data["bm25"]
            self.documents = data["documents"]
            self.tokenized_corpus = data["tokenized_corpus"]
            
            logger.info(
                f"BM25 index loaded: {len(self.documents)} documents"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def index_exists(self) -> bool:
        """Check if a persisted BM25 index exists."""
        return Path(self.persist_path).exists()

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_documents": len(self.documents),
            "index_built": self.bm25 is not None,
            "persist_path": self.persist_path,
            "avg_doc_length": (
                np.mean([len(t) for t in self.tokenized_corpus])
                if self.tokenized_corpus else 0
            ),
        }


# Singleton
_embedder = None

def get_sparse_embedder() -> SparseEmbedder:
    """Get the global SparseEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = SparseEmbedder()
    return _embedder
