"""
Multi-query generation module.

Generates multiple alternative search queries for a single user question
to improve retrieval recall, then deduplicates results.
Uses HuggingFace Inference API for query generation.
"""
import time
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("multi_query")


# Fallback query expansion templates when LLM is unavailable
QUERY_TEMPLATES = {
    "financial_metrics": "financial performance metrics {topic}",
    "risk_analysis": "risk factors challenges {topic}",
    "comparison": "comparison analysis {topic}",
    "temporal": "quarterly annual changes trends {topic}",
    "regulatory": "regulatory compliance disclosure {topic}",
}


class MultiQueryGenerator:
    """
    Generates multiple search queries from a single user question.
    
    This improves retrieval recall by capturing different aspects
    and phrasings of the original query. Uses HuggingFace Inference
    API when available, falls back to template-based expansion.
    """

    def __init__(self, num_queries: int = None):
        self.num_queries = num_queries or settings.num_generated_queries
        self._client = None

    @property
    def hf_client(self):
        """Lazy-load HuggingFace Inference client."""
        if self._client is None and settings.huggingface_api_token:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    token=settings.huggingface_api_token
                )
                logger.info("HuggingFace Inference client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize HF client: {e}")
        return self._client

    def generate_queries_llm(self, query: str) -> List[str]:
        """
        Generate alternative queries using HuggingFace LLM.
        
        Args:
            query: Original user query
            
        Returns:
            List of alternative queries
        """
        if not self.hf_client:
            return []

        prompt = f"""You are a financial document search expert. Generate {self.num_queries} alternative search queries for the following question. Each query should capture a different aspect or phrasing to improve document retrieval from SEC filings.

Original question: {query}

Generate exactly {self.num_queries} queries, one per line. Only output the queries, nothing else."""

        try:
            response = self.hf_client.text_generation(
                prompt,
                model=settings.llm_model_id,
                max_new_tokens=200,
                temperature=0.7,
            )
            
            # Parse response into individual queries
            queries = [
                q.strip().lstrip("0123456789.-) ")
                for q in response.strip().split("\n")
                if q.strip() and len(q.strip()) > 10
            ]
            
            return queries[:self.num_queries]
            
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}")
            return []

    def generate_queries_template(self, query: str) -> List[str]:
        """
        Generate alternative queries using template-based expansion.
        
        Falls back to this method when LLM is unavailable.
        """
        # Extract key terms from the query
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove common question words
        stop_words = {
            "what", "how", "why", "when", "where", "which", "who",
            "is", "are", "was", "were", "did", "do", "does",
            "the", "a", "an", "in", "of", "for", "to", "and",
            "their", "its", "this", "that", "these", "those"
        }
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]
        topic = " ".join(key_terms[:5])
        
        queries = []
        
        # Company-specific expansion
        company_terms = [
            w for w in key_terms
            if w[0].isupper() or w.upper() in [
                "tesla", "apple", "ford", "microsoft", "nvidia",
                "amazon", "google", "jpmorgan", "alphabet"
            ]
        ]
        
        # Generate template-based queries
        if "risk" in query.lower() or "challenge" in query.lower():
            queries.append(f"risk factors and challenges {topic}")
            queries.append(f"material risks disclosures {topic}")
        
        if "revenue" in query.lower() or "financial" in query.lower():
            queries.append(f"financial performance revenue earnings {topic}")
            queries.append(f"income statement results {topic}")
        
        if "supply chain" in query.lower():
            queries.append(f"supply chain disruptions logistics {topic}")
            queries.append(f"procurement manufacturing challenges {topic}")
        
        if "compare" in query.lower() or "comparison" in query.lower():
            queries.append(f"comparison analysis between {topic}")
            queries.append(f"competitive position {topic}")
        
        # Always add a semantic expansion
        queries.append(f"{topic} SEC filing disclosure analysis")
        
        # Deduplicate and limit
        seen = set()
        unique_queries = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries[:self.num_queries]

    def generate(self, query: str) -> List[str]:
        """
        Generate multiple search queries from a single question.
        
        Tries LLM-based generation first, falls back to templates.
        Always includes the original query.
        
        Args:
            query: Original user query
            
        Returns:
            List of queries (original + generated)
        """
        start = time.time()
        
        # Always start with the original query
        all_queries = [query]
        
        # Try LLM-based generation
        llm_queries = self.generate_queries_llm(query)
        if llm_queries:
            all_queries.extend(llm_queries)
            source = "LLM"
        else:
            # Fallback to template-based
            template_queries = self.generate_queries_template(query)
            all_queries.extend(template_queries)
            source = "template"
        
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in all_queries:
            normalized = q.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(q)
        
        elapsed = time.time() - start
        logger.info(
            f"Generated {len(unique)} queries ({source}) in {elapsed:.3f}s"
        )
        
        return unique

    def retrieve_with_multi_query(
        self,
        query: str,
        retriever,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval using multiple generated queries.
        
        Args:
            query: Original user query
            retriever: HybridRetriever instance
            top_k: Total number of results to return
            
        Returns:
            Deduplicated merged results from all queries
        """
        top_k = top_k or settings.dense_top_k
        
        queries = self.generate(query)
        
        # Retrieve for each query
        all_results = OrderedDict()
        
        for q in queries:
            results = retriever.retrieve(q, top_k=top_k)
            for doc in results:
                doc_id = doc["doc_id"]
                if doc_id not in all_results:
                    all_results[doc_id] = doc
                else:
                    # Keep the higher score
                    existing_score = all_results[doc_id].get("rrf_score", 0)
                    new_score = doc.get("rrf_score", 0)
                    if new_score > existing_score:
                        all_results[doc_id] = doc
        
        merged = list(all_results.values())
        merged.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        
        logger.info(
            f"Multi-query retrieval: {len(queries)} queries -> "
            f"{len(merged)} unique documents"
        )
        
        return merged[:top_k]


# Singleton
_generator = None

def get_multi_query_generator() -> MultiQueryGenerator:
    """Get the global MultiQueryGenerator instance."""
    global _generator
    if _generator is None:
        _generator = MultiQueryGenerator()
    return _generator
