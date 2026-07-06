"""
Sparse retrieval module using BM25 (Best Matching 25).

Provides keyword-based retrieval as a complement to dense vector search,
enabling the hybrid retrieval pipeline.
"""
import json
import os
import pickle
import time
from gzip import open as gzip_open
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
        self.persist_dir = Path(self.persist_path).parent
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []
        self._stopwords = None
        self.format_version = 2

    def _documents_path(self) -> Path:
        return self.persist_dir / "bm25_documents.jsonl.gz"

    def _tokens_path(self) -> Path:
        return self.persist_dir / "bm25_tokens.jsonl.gz"

    def _cleanup_persisted_files(self):
        for path in [Path(self.persist_path), self._documents_path(), self._tokens_path()]:
            path.unlink(missing_ok=True)

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

    def _metadata_matches(self, doc: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Return True when a document satisfies simple equality metadata filters."""
        if not filters:
            return True
        metadata = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        for key, value in filters.items():
            if value in ("", None, [], {}):
                continue
            actual = doc.get(key, metadata.get(key, ""))
            if str(actual).lower() != str(value).lower():
                return False
        return True

    @staticmethod
    def _lexical_overlap_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Positive deterministic fallback score for tiny corpora with zero BM25 IDF."""
        if not query_tokens or not doc_tokens:
            return 0.0
        query_set = set(query_tokens)
        doc_set = set(doc_tokens)
        overlap = len(query_set & doc_set)
        if overlap == 0:
            return 0.0
        return overlap / len(query_set)

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
            lexical_score = self._lexical_overlap_score(
                tokenized_query,
                self.tokenized_corpus[idx] if idx < len(self.tokenized_corpus) else []
            )
            final_score = float(scores[idx]) if scores[idx] > 0 else lexical_score
            if final_score > 0:
                doc = self.documents[idx].copy()
                if not self._metadata_matches(doc, filters):
                    continue
                metadata = {
                    k: v for k, v in doc.items()
                    if k not in ("content", "doc_id")
                    and isinstance(v, (str, int, float, bool))
                }
                if isinstance(doc.get("metadata"), dict):
                    metadata.update({
                        k: v for k, v in doc["metadata"].items()
                        if isinstance(v, (str, int, float, bool))
                    })
                results.append({
                    "doc_id": doc.get("doc_id", f"doc_{idx}"),
                    "content": doc["content"],
                    "ticker": metadata.get("ticker", ""),
                    "company": metadata.get("company", ""),
                    "form_type": metadata.get("form_type", ""),
                    "filing_date": metadata.get("filing_date", ""),
                    "fiscal_year": metadata.get("fiscal_year", ""),
                    "section": metadata.get("section", ""),
                    "accession_number": metadata.get("accession_number", ""),
                    "source_url": metadata.get("source_url", ""),
                    "metadata": metadata,
                    "score": final_score,
                    "bm25_score": float(scores[idx]),
                    "lexical_overlap_score": lexical_score,
                    "retrieval_type": "sparse"
                })
                if len(results) >= top_k:
                    break

        logger.info(
            f"BM25 search returned {len(results)} results for: {query[:50]}..."
        )
        return results

    def save_index(self):
        """Persist BM25 index to disk using compact sidecar files and atomic writes."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = Path(self.persist_path)
        docs_path = self._documents_path()
        tokens_path = self._tokens_path()

        temp_manifest = manifest_path.with_suffix(".tmp")
        temp_docs = docs_path.with_suffix(".tmp")
        temp_tokens = tokens_path.with_suffix(".tmp")

        try:
            with gzip_open(temp_docs, "wt", encoding="utf-8") as f:
                for doc in self.documents:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

            with gzip_open(temp_tokens, "wt", encoding="utf-8") as f:
                for tokens in self.tokenized_corpus:
                    f.write(json.dumps(tokens, ensure_ascii=False) + "\n")

            manifest = {
                "format_version": self.format_version,
                "document_count": len(self.documents),
                "documents_file": docs_path.name,
                "tokens_file": tokens_path.name,
            }
            with open(temp_manifest, "wb") as f:
                pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

            os.replace(temp_docs, docs_path)
            os.replace(temp_tokens, tokens_path)
            os.replace(temp_manifest, manifest_path)
            logger.info(
                "BM25 index saved to %s using compact persistence for %s documents",
                self.persist_dir,
                len(self.documents),
            )
        finally:
            for temp_path in [temp_manifest, temp_docs, temp_tokens]:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)

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

            # Backward compatibility for older all-in-one pickle files.
            if isinstance(data, dict) and {"bm25", "documents", "tokenized_corpus"} <= set(data.keys()):
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
                self.tokenized_corpus = data["tokenized_corpus"]
            else:
                docs_path = self.persist_dir / data["documents_file"]
                tokens_path = self.persist_dir / data["tokens_file"]
                if not docs_path.exists() or not tokens_path.exists():
                    raise FileNotFoundError(
                        f"BM25 sidecar files missing: {docs_path} / {tokens_path}"
                    )
                with gzip_open(docs_path, "rt", encoding="utf-8") as f:
                    self.documents = [
                        json.loads(line) for line in f if line.strip()
                    ]
                with gzip_open(tokens_path, "rt", encoding="utf-8") as f:
                    self.tokenized_corpus = [
                        json.loads(line) for line in f if line.strip()
                    ]
                self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            logger.info(
                f"BM25 index loaded: {len(self.documents)} documents"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            self.bm25 = None
            self.documents = []
            self.tokenized_corpus = []
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
            "persist_dir": str(self.persist_dir),
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
