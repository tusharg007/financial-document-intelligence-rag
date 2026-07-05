"""Actual LangGraph-backed financial RAG workflow."""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover - tests can still inspect graceful fallback.
    END = "__end__"
    StateGraph = None

from src.llm.factory import get_llm
from src.retrieval.confidence import compute_confidence, refusal_message
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.retrieval.query_router import classify_query as route_query
from src.retrieval.reranker import get_reranker


class FinancialRAGState(TypedDict, total=False):
    question: str
    query: str
    query_type: str
    filters: Dict[str, Any]
    top_k: int
    use_reranking: bool
    use_multi_query: bool
    llm_provider: str
    debug: bool
    retrieved_documents: List[Dict[str, Any]]
    graded_documents: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    rewritten: bool
    facts: List[Dict[str, Any]]
    calculations: List[str]
    answer: str
    citations: List[Dict[str, Any]]
    refusal: bool
    trace: List[Dict[str, Any]]
    provider_used: str
    latency: float


def _trace(state: FinancialRAGState, node: str, **data: Any) -> None:
    state.setdefault("trace", []).append({"node": node, **data})


class LangGraphRAG:
    """Financial RAG orchestrated with LangGraph StateGraph."""

    def __init__(self):
        self.retriever = get_hybrid_retriever()
        self.reranker = get_reranker()
        self.graph = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            return None
        graph = StateGraph(FinancialRAGState)
        graph.add_node("classify_query", self.classify_query)
        graph.add_node("build_filters", self.build_filters)
        graph.add_node("retrieve_documents", self.retrieve_documents)
        graph.add_node("grade_documents", self.grade_documents)
        graph.add_node("rewrite_query_if_needed", self.rewrite_query_if_needed)
        graph.add_node("extract_financial_facts", self.extract_financial_facts)
        graph.add_node("run_calculations_if_needed", self.run_calculations_if_needed)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("verify_grounding", self.verify_grounding)
        graph.add_node("attach_citations", self.attach_citations)
        graph.add_node("decide_refusal_or_final_answer", self.decide_refusal_or_final_answer)
        graph.add_node("refusal", self.refusal)

        graph.set_entry_point("classify_query")
        graph.add_conditional_edges("classify_query", self._route_scope, {"refusal": "refusal", "continue": "build_filters"})
        graph.add_edge("build_filters", "retrieve_documents")
        graph.add_edge("retrieve_documents", "grade_documents")
        graph.add_conditional_edges("grade_documents", self._route_confidence, {"rewrite": "rewrite_query_if_needed", "facts": "extract_financial_facts", "generate": "generate_answer", "refusal": "refusal"})
        graph.add_edge("rewrite_query_if_needed", "retrieve_documents")
        graph.add_edge("extract_financial_facts", "run_calculations_if_needed")
        graph.add_edge("run_calculations_if_needed", "generate_answer")
        graph.add_edge("generate_answer", "verify_grounding")
        graph.add_edge("verify_grounding", "attach_citations")
        graph.add_edge("attach_citations", "decide_refusal_or_final_answer")
        graph.add_edge("decide_refusal_or_final_answer", END)
        graph.add_edge("refusal", END)
        return graph.compile()

    def _route_scope(self, state: FinancialRAGState) -> str:
        return "refusal" if state.get("query_type") == "out_of_scope" else "continue"

    def _route_confidence(self, state: FinancialRAGState) -> str:
        conf = state.get("confidence", {})
        if conf.get("answerable"):
            if state.get("query_type") in {"numerical", "comparison", "temporal"}:
                return "facts"
            return "generate"
        return "refusal" if state.get("rewritten") else "rewrite"

    def classify_query(self, state: FinancialRAGState) -> FinancialRAGState:
        state["query"] = state.get("query") or state["question"]
        state["query_type"] = route_query(state["query"])
        _trace(state, "classify_query", query_type=state["query_type"])
        return state

    def build_filters(self, state: FinancialRAGState) -> FinancialRAGState:
        filters = dict(state.get("filters") or {})
        q = state["query"].upper()
        for ticker in ["AAPL", "MSFT", "TSLA", "NVDA", "JPM", "AMZN", "GOOGL", "META", "AMD", "NFLX", "F"]:
            if ticker in q:
                filters.setdefault("ticker", ticker)
        for form in ["10-K", "10-Q", "8-K"]:
            if form in q:
                filters.setdefault("form_type", form)
                filters.setdefault("filing_type", form)
        state["filters"] = filters
        _trace(state, "build_filters", filters=filters)
        return state

    def retrieve_documents(self, state: FinancialRAGState) -> FinancialRAGState:
        top_k = int(state.get("top_k") or 5)
        queries = [state["query"]]
        if state.get("use_multi_query"):
            queries.extend([f"{state['query']} risk factors", f"{state['query']} financial statements"])
        docs: List[Dict[str, Any]] = []
        seen = set()
        for query in queries:
            for doc in self.retriever.retrieve(query, top_k=top_k, filters=state.get("filters")):
                if doc["doc_id"] not in seen:
                    seen.add(doc["doc_id"])
                    docs.append(doc)
        docs = docs[: max(top_k * 2, top_k)]
        if state.get("use_reranking", True):
            docs = self.reranker.rerank(state["query"], docs, top_k=top_k)
        else:
            docs = docs[:top_k]
        state["retrieved_documents"] = docs
        _trace(state, "retrieve_documents", count=len(docs), top_k=top_k)
        return state

    def grade_documents(self, state: FinancialRAGState) -> FinancialRAGState:
        docs = state.get("retrieved_documents", [])
        state["confidence"] = compute_confidence(docs)
        state["graded_documents"] = docs
        _trace(state, "grade_documents", confidence=state["confidence"])
        return state

    def rewrite_query_if_needed(self, state: FinancialRAGState) -> FinancialRAGState:
        state["query"] = f"{state['question']} SEC filing disclosure financial data"
        state["rewritten"] = True
        _trace(state, "rewrite_query_if_needed", query=state["query"])
        return state

    def extract_financial_facts(self, state: FinancialRAGState) -> FinancialRAGState:
        facts = []
        for doc in state.get("graded_documents", []):
            for match in re.findall(r"(?:\$[\d,.]+(?:\s*billion|\s*million)?|\d+(?:\.\d+)?%)", doc.get("content", ""), flags=re.I):
                facts.append({"value": match, "doc_id": doc.get("doc_id")})
        state["facts"] = facts[:20]
        _trace(state, "extract_financial_facts", facts=len(state["facts"]))
        return state

    def run_calculations_if_needed(self, state: FinancialRAGState) -> FinancialRAGState:
        state["calculations"] = []
        _trace(state, "run_calculations_if_needed", calculations=0)
        return state

    def generate_answer(self, state: FinancialRAGState) -> FinancialRAGState:
        docs = state.get("graded_documents", [])
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.get("metadata", {})
            context_parts.append(f"[Source {i}] {meta.get('company','Unknown')} {meta.get('form_type') or meta.get('filing_type','')} {meta.get('filing_date','')}: {doc.get('content','')}")
        prompt = (
            "Answer using only the cited filing excerpts. If evidence is insufficient, say so.\n"
            f"Question: {state['question']}\n\n" + "\n\n".join(context_parts)
        )
        llm = get_llm(state.get("llm_provider"))
        response = llm.generate([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=700)
        state["answer"] = str(response.get("text", ""))
        state["provider_used"] = str(response.get("provider", llm.name))
        _trace(state, "generate_answer", provider=state["provider_used"])
        return state

    def verify_grounding(self, state: FinancialRAGState) -> FinancialRAGState:
        answer = state.get("answer", "")
        has_source_ref = "[Source" in answer or bool(state.get("graded_documents"))
        if not has_source_ref:
            state["confidence"] = {"score": 0.0, "label": "low", "answerable": False, "reason": "Answer lacks citation support."}
        _trace(state, "verify_grounding", answerable=state.get("confidence", {}).get("answerable"))
        return state

    def attach_citations(self, state: FinancialRAGState) -> FinancialRAGState:
        citations = []
        for i, doc in enumerate(state.get("graded_documents", []), 1):
            meta = doc.get("metadata", {})
            citations.append({
                "source_num": i,
                "doc_id": doc.get("doc_id"),
                "company": meta.get("company", ""),
                "ticker": meta.get("ticker", ""),
                "filing_type": meta.get("form_type") or meta.get("filing_type", ""),
                "filing_date": meta.get("filing_date", ""),
                "section": meta.get("section", ""),
                "source_url": meta.get("source_url", ""),
                "content_preview": doc.get("content", "")[:240],
                "relevance_score": doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0))),
            })
        state["citations"] = citations
        _trace(state, "attach_citations", citations=len(citations))
        return state

    def decide_refusal_or_final_answer(self, state: FinancialRAGState) -> FinancialRAGState:
        if not state.get("confidence", {}).get("answerable", False):
            state["refusal"] = True
            state["answer"] = refusal_message(state.get("confidence", {}).get("reason", ""))
        else:
            state["refusal"] = False
        _trace(state, "decide_refusal_or_final_answer", refusal=state["refusal"])
        return state

    def refusal(self, state: FinancialRAGState) -> FinancialRAGState:
        state["refusal"] = True
        state["answer"] = refusal_message("The question is outside the indexed financial filing scope or retrieval failed.")
        state["citations"] = []
        state["confidence"] = {"score": 0.0, "label": "low", "answerable": False}
        _trace(state, "refusal")
        return state

    def run(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_multi_query: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        start = time.time()
        state: FinancialRAGState = {
            "question": question,
            "top_k": top_k,
            "use_reranking": use_reranking,
            "use_multi_query": use_multi_query,
            "filters": filters or {},
            "llm_provider": llm_provider or "extractive",
            "debug": debug,
            "rewritten": False,
            "trace": [],
        }
        if self.graph is not None:
            result = self.graph.invoke(state)
        else:
            result = self.decide_refusal_or_final_answer(self.attach_citations(self.verify_grounding(self.generate_answer(self.grade_documents(self.retrieve_documents(self.build_filters(self.classify_query(state))))))))
        result["latency"] = round(time.time() - start, 4)
        result["retrieved_contexts"] = result.get("graded_documents", [])
        if not debug:
            result.pop("trace", None)
        return result


_graph = None


def get_langgraph_rag() -> LangGraphRAG:
    global _graph
    if _graph is None:
        _graph = LangGraphRAG()
    return _graph
