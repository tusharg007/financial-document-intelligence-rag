# Resume Bullets

- Built a financial document intelligence RAG system with SEC EDGAR ingestion, section-aware filing parsing, ChromaDB dense indexing, persisted BM25 sparse indexing, hybrid retrieval, reranking fallback, and citation-preserving answers.
- Implemented a LangGraph `StateGraph` workflow with query classification, metadata filtering, retrieval grading, query rewriting, financial fact extraction, answer generation, grounding verification, citations, confidence scoring, and weak-evidence refusal.
- Added production-facing FastAPI endpoints for health, readiness, Prometheus-style metrics, SEC ingestion jobs, index rebuild jobs, document search, query, comparison, temporal analysis, evaluation, and latest report discovery.
- Created evaluation and LoRA preparation pipelines that write reproducible local reports and explicitly avoid fabricated benchmark or fine-tuning metrics when datasets, API keys, GPU, or adapters are unavailable.

Add metric-specific bullets only after running `scripts/evaluate.py` and verifying `reports/evaluation_latest.json`.
