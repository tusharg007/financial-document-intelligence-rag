"""
Agentic RAG pipeline using LangGraph.

Implements a stateful agentic workflow that decides when to retrieve,
when to synthesize, and self-corrects when documents are insufficiently relevant.
Uses HuggingFace Inference API for LLM operations.
"""
import time
from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field

from src.retrieval.pipeline import RetrievalPipeline, get_retrieval_pipeline
from src.retrieval.reranker import CrossEncoderReranker, get_reranker
from src.retrieval.multi_query import MultiQueryGenerator, get_multi_query_generator
from src.utils.logger import get_logger, tracer
from src.utils.helpers import format_citations
from config.settings import settings

logger = get_logger("rag_agent")


# ============================================================================
# State Definition
# ============================================================================

class RAGState(TypedDict):
    """State that flows through the RAG pipeline graph."""
    query: str
    query_type: str  # simple, complex, comparison, temporal
    generated_queries: List[str]
    retrieved_documents: List[Dict[str, Any]]
    reranked_documents: List[Dict[str, Any]]
    relevant_documents: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    processing_time: float
    error: Optional[str]


# ============================================================================
# LLM Interface (HuggingFace)
# ============================================================================

class HuggingFaceLLM:
    """Interface for HuggingFace Inference API."""
    
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
        if self._client is None and settings.huggingface_api_token:
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    token=settings.huggingface_api_token
                )
            except Exception as e:
                logger.warning(f"HF client init failed: {e}")
        return self._client
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate text using HuggingFace API."""
        if self.client:
            try:
                response = self.client.text_generation(
                    prompt,
                    model=settings.llm_model_id,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    return_full_text=False,
                )
                return response.strip()
            except Exception as e:
                logger.warning(f"HF generation failed: {e}")
        
        return self._local_generate(prompt)
    
    def _local_generate(self, prompt: str) -> str:
        """Fallback generation using local template-based responses."""
        # Smart local response generation without LLM
        return ""


# ============================================================================
# Pipeline Nodes
# ============================================================================

class RAGPipeline:
    """
    Agentic RAG pipeline with self-corrective retrieval.
    
    Graph structure:
        START -> classify_query -> retrieve -> grade -> [generate | rewrite]
                                                  ↑         ↓
                                                  └─ rewrite ┘
    
    Nodes:
    1. classify_query: Determine query type and complexity
    2. retrieve: Hybrid retrieval + reranking
    3. grade: Evaluate document relevance
    4. generate: Produce answer with citations
    5. rewrite: Reformulate query if documents are irrelevant
    """

    def __init__(self):
        self.retriever = get_retrieval_pipeline()
        self.reranker = get_reranker()
        self.multi_query = get_multi_query_generator()
        self.max_retries = 2
        
        # Select LLM backend based on configuration
        if settings.use_finetuned_model:
            from src.finetuning.lora_inference import FineTunedLLM
            self.llm = FineTunedLLM(
                base_model=settings.finetuned_base_model,
                lora_adapter=settings.finetuned_adapter_path,
                load_in_4bit=settings.finetuned_load_in_4bit,
            )
            logger.info("Using fine-tuned LoRA model for generation")
        else:
            self.llm = HuggingFaceLLM()
            logger.info("Using HuggingFace API for generation")

    def classify_query(self, state: RAGState) -> RAGState:
        """
        Classify the query type to route processing.
        
        Types:
        - simple: Direct factual question
        - complex: Multi-faceted question requiring synthesis
        - comparison: Comparing multiple companies/periods
        - temporal: Tracking changes over time
        """
        tracer.trace("classify_query", {"query": state["query"]})
        
        query = state["query"].lower()
        
        # Classification rules
        if any(word in query for word in ["compare", "versus", "vs", "difference between", "side by side"]):
            query_type = "comparison"
        elif any(word in query for word in ["over time", "trend", "changed", "evolution", "quarter", "year over year", "historically"]):
            query_type = "temporal"
        elif any(word in query for word in ["analyze", "explain", "discuss", "impact", "implications", "summarize"]):
            query_type = "complex"
        else:
            query_type = "simple"
        
        state["query_type"] = query_type
        logger.info(f"Query classified as: {query_type}")
        
        return state

    def retrieve(self, state: RAGState) -> RAGState:
        """Perform hybrid retrieval with multi-query expansion."""
        tracer.trace("retrieve", {"query_type": state["query_type"]})
        
        query = state["query"]
        
        # Use multi-query for complex queries
        if state["query_type"] in ("complex", "comparison"):
            documents = self.multi_query.retrieve_with_multi_query(
                query, self.retriever, top_k=15
            )
        else:
            documents = self.retriever.retrieve(query, top_k=15)
        
        state["retrieved_documents"] = documents
        
        # Rerank
        reranked = self.reranker.rerank(query, documents, top_k=settings.rerank_top_k)
        state["reranked_documents"] = reranked
        
        logger.info(f"Retrieved {len(documents)} -> Reranked top {len(reranked)}")
        
        return state

    def grade_documents(self, state: RAGState) -> RAGState:
        """
        Grade retrieved documents for relevance.
        
        Uses rerank scores to determine if documents are sufficiently
        relevant to answer the query.
        """
        tracer.trace("grade_documents")
        
        docs = state["reranked_documents"]
        
        if not docs:
            state["is_relevant"] = False
            state["relevant_documents"] = []
            return state
        
        # Filter by rerank score threshold
        # ms-marco cross-encoder outputs logits; positive = relevant
        relevant = [
            doc for doc in docs
            if doc.get("rerank_score", 0) > -2.0  # Lenient threshold
        ]
        
        # Check if we have enough relevant documents
        state["is_relevant"] = len(relevant) >= 1
        state["relevant_documents"] = relevant if relevant else docs[:3]
        
        logger.info(
            f"Grading: {len(relevant)}/{len(docs)} documents deemed relevant"
        )
        
        return state

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer from relevant documents with citations.
        """
        tracer.trace("generate_answer")
        
        query = state["query"]
        docs = state["relevant_documents"]
        
        if not docs:
            state["answer"] = "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a specific company's SEC filing."
            state["citations"] = []
            return state
        
        # Build context from documents
        context_parts = []
        citations = []
        
        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            company = meta.get("company", "Unknown")
            filing = meta.get("filing_type", "N/A")
            date = meta.get("filing_date", "N/A")
            section = meta.get("section", "N/A")
            
            context_parts.append(
                f"[Source {i+1}] {company} | {filing} | {date} | {section}:\n"
                f"{doc['content']}"
            )
            
            citations.append({
                "source_num": i + 1,
                "company": company,
                "filing_type": filing,
                "filing_date": date,
                "section": section,
                "content_preview": doc["content"][:200] + "...",
                "relevance_score": round(doc.get("rerank_score", 0), 4),
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Based on the following SEC filing excerpts, provide a comprehensive answer to the question. Include specific data points, numbers, and cite your sources using [Source N] format.

Question: {query}

Context from SEC Filings:
{context}

Instructions:
- Answer the question directly and comprehensively
- Include specific numbers, percentages, and data points
- Cite sources using [Source N] references
- If information is insufficient, say so clearly
- Use a professional financial analysis tone

Answer:"""
        
        # Try LLM generation
        answer = self.llm.generate(prompt, max_tokens=1024)
        
        if not answer:
            # Fallback: Generate a structured answer from the documents
            answer = self._generate_local_answer(query, docs, citations)
        
        state["answer"] = answer
        state["citations"] = citations
        
        return state

    def _generate_local_answer(
        self,
        query: str,
        docs: List[Dict],
        citations: List[Dict]
    ) -> str:
        """Generate a structured answer without LLM (fallback mode)."""
        
        query_lower = query.lower()
        
        # Build answer from document content
        answer_parts = []
        answer_parts.append(f"**Analysis based on SEC filing data:**\n")
        
        for i, doc in enumerate(docs):
            meta = doc.get("metadata", {})
            company = meta.get("company", "Unknown")
            filing = meta.get("filing_type", "")
            section = meta.get("section", "")
            content = doc["content"]
            
            # Extract key sentences from the document
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
            
            # Filter to most relevant sentences
            relevant_sentences = []
            query_terms = set(query_lower.split())
            
            for sent in sentences:
                sent_lower = sent.lower()
                relevance = sum(1 for term in query_terms if term in sent_lower)
                if relevance > 0:
                    relevant_sentences.append((sent, relevance))
            
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:3]]
            
            if not top_sentences:
                top_sentences = sentences[:2]
            
            if top_sentences:
                answer_parts.append(
                    f"\n**{company}** ({filing}, {section}) [Source {i+1}]:"
                )
                for sent in top_sentences:
                    answer_parts.append(f"- {sent.strip()}.")
        
        if len(answer_parts) <= 1:
            # Very basic fallback
            answer_parts.append("\nKey findings from SEC filings:")
            for i, doc in enumerate(docs[:3]):
                preview = doc["content"][:300]
                answer_parts.append(f"\n[Source {i+1}]: {preview}...")
        
        return "\n".join(answer_parts)

    def rewrite_query(self, state: RAGState) -> RAGState:
        """Rewrite query if initial retrieval was insufficient."""
        tracer.trace("rewrite_query", {"retry": state["retry_count"]})
        
        original = state["query"]
        
        # Simple query expansion
        expansions = {
            "revenue": "total revenue net sales financial results",
            "risk": "risk factors material risks challenges disclosures",
            "supply chain": "supply chain logistics procurement manufacturing",
            "earnings": "earnings income profit loss financial performance",
        }
        
        expanded = original
        for term, expansion in expansions.items():
            if term in original.lower():
                expanded = f"{original} {expansion}"
                break
        
        if expanded == original:
            expanded = f"SEC filing disclosure: {original}"
        
        state["query"] = expanded
        state["retry_count"] = state.get("retry_count", 0) + 1
        
        logger.info(f"Query rewritten (retry {state['retry_count']}): {expanded[:80]}...")
        
        return state

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline.
        
        This is the main entry point that orchestrates the agentic workflow.
        
        Args:
            query: User question
            
        Returns:
            Dict with answer, citations, and pipeline metadata
        """
        start = time.time()
        tracer.clear()
        
        # Initialize state
        state: RAGState = {
            "query": query,
            "query_type": "simple",
            "generated_queries": [],
            "retrieved_documents": [],
            "reranked_documents": [],
            "relevant_documents": [],
            "answer": "",
            "citations": [],
            "is_relevant": False,
            "retry_count": 0,
            "processing_time": 0,
            "error": None,
        }
        
        try:
            # Step 1: Classify query
            state = self.classify_query(state)
            
            # Step 2: Retrieve (with retry loop)
            while state["retry_count"] <= self.max_retries:
                state = self.retrieve(state)
                state = self.grade_documents(state)
                
                if state["is_relevant"] or state["retry_count"] >= self.max_retries:
                    break
                
                state = self.rewrite_query(state)
            
            # Step 3: Generate answer
            state = self.generate_answer(state)
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            state["error"] = str(e)
            state["answer"] = f"An error occurred while processing your query: {str(e)}"
        
        state["processing_time"] = time.time() - start
        
        logger.info(
            f"Pipeline completed in {state['processing_time']:.2f}s | "
            f"Type: {state['query_type']} | "
            f"Docs: {len(state['relevant_documents'])} | "
            f"Retries: {state['retry_count']}"
        )
        
        return {
            "answer": state["answer"],
            "citations": state["citations"],
            "query_type": state["query_type"],
            "num_documents_retrieved": len(state["retrieved_documents"]),
            "num_documents_relevant": len(state["relevant_documents"]),
            "processing_time": round(state["processing_time"], 3),
            "retries": state["retry_count"],
            "pipeline_trace": tracer.get_trace_summary(),
            "error": state["error"],
        }


# Singleton
_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAGPipeline instance."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.agents.langgraph_rag import get_langgraph_rag
            _pipeline = get_langgraph_rag()
        except Exception:
            _pipeline = RAGPipeline()
    return _pipeline
