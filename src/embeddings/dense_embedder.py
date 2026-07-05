"""
Dense embedding module using sentence-transformers and ChromaDB.

Provides vector storage and similarity search for financial documents.
"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("dense_embedder")


class DenseEmbedder:
    """
    Dense embedding and retrieval using sentence-transformers + ChromaDB.
    
    Uses all-MiniLM-L6-v2 for fast, lightweight embeddings that run on CPU.
    Stores vectors in ChromaDB for persistent similarity search.
    """

    def __init__(
        self,
        model_name: str = None,
        collection_name: str = None,
        persist_dir: str = None
    ):
        self.model_name = model_name or settings.embedding_model_id
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        
        self._model = None
        self._client = None
        self._collection = None

    @property
    def model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is False:
            return None
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            start = time.time()
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Dense embedding model unavailable ({e}); dense retrieval disabled until dependencies are installed.")
                self._model = False
                return None
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.2f}s")
        return self._model

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info(f"ChromaDB initialized at {self.persist_dir}")
        return self._client

    @property
    def collection(self):
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(
                f"Collection '{self.collection_name}' has "
                f"{self._collection.count()} documents"
            )
        return self._collection

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start = time.time()
        model = self.model
        if model is None:
            return []
        embeddings = model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
            normalize_embeddings=True
        )
        elapsed = time.time() - start
        logger.info(f"Embedded {len(texts)} texts in {elapsed:.2f}s")
        
        return embeddings.tolist()

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dicts with 'content' and metadata
            batch_size: Number of documents to process at once
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            ids = [doc.get("doc_id", f"doc_{i+j}") for j, doc in enumerate(batch)]
            texts = [doc["content"] for doc in batch]
            
            # Build metadata (ChromaDB only supports str, int, float, bool)
            metadatas = []
            for doc in batch:
                meta = {}
                for key, value in doc.items():
                    if key in ("content", "doc_id"):
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        meta[key] = value
                metadatas.append(meta)

            # Generate embeddings
            embeddings = self.embed_texts(texts)

            # Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")

        logger.info(f"Total documents in collection: {self.collection.count()}")
        return total_added

    def search(
        self,
        query: str,
        top_k: int = None,
        where: Dict = None,
        where_document: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            where: Metadata filter (ChromaDB format)
            where_document: Document content filter
            
        Returns:
            List of matching documents with scores
        """
        top_k = top_k or settings.dense_top_k
        
        embeddings = self.embed_texts([query])
        if not embeddings:
            return []
        query_embedding = embeddings[0]
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }
        
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = self.collection.query(**kwargs)

        documents = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc = {
                    "doc_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "retrieval_type": "dense"
                }
                documents.append(doc)

        logger.info(f"Dense search returned {len(documents)} results for: {query[:50]}...")
        return documents

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_model": self.model_name,
            "persist_dir": self.persist_dir,
        }

    def clear_collection(self):
        """Delete all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Cleared collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")

    def collection_exists(self) -> bool:
        """Check if the collection has documents."""
        try:
            return self.collection.count() > 0
        except Exception:
            return False


# Singleton embedder
_embedder = None

def get_dense_embedder() -> DenseEmbedder:
    """Get the global DenseEmbedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = DenseEmbedder()
    return _embedder
