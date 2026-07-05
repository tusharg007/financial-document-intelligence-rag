# Gap Audit

Generated from local repository inspection. This document records what is actually present in the codebase before the implementation work.

## Summary

The project was a polished demo backed primarily by `src/data/sample_data.py`. It contained useful building blocks for parsing, ChromaDB dense retrieval, BM25 retrieval, reranking, FastAPI, and Streamlit, but many README/UI claims were either demo-only, hardcoded, or not connected to real ingestion/evaluation workflows.

## Evidence

| Question | Finding | File evidence |
|---|---|---|
| Which files use `sample_data.py`? | Production API and tests imported sample data directly. `api.py` seeded the pipeline with sample documents when Chroma was empty and `/evaluate` used sample QA pairs. | `api.py` imports `get_all_documents` in `ensure_initialized`; `api.py` imports `get_evaluation_pairs` in `/evaluate`; `tests/test_pipeline.py` imports sample data throughout. |
| How many hardcoded documents exist? | 40 hardcoded sample document entries were present. | `src/data/sample_data.py` contains 40 `"content":` sample document entries. |
| How many evaluation pairs exist? | 10 hardcoded demo evaluation QA pairs were present. | `src/data/sample_data.py` defines `EVALUATION_QA_PAIRS` with 10 `"question":` entries. |
| Is SEC EDGAR ingestion actually connected to the app/API? | Partially implemented but not connected as a real ingestion job. `src/data/edgar_client.py` could list/download filings, but `api.py` initialized from sample data and had no `/ingest/sec` endpoint. | `src/data/edgar_client.py`; `api.py` `ensure_initialized`. |
| Is Kaggle SEC ingestion actually implemented? | No. No Kaggle loader existed. | No `dataset_loaders.py`; no `data/raw/kaggle` pipeline. |
| Is Financial PhraseBank actually used? | No. It was mentioned in README only. | README dataset table; no loader/use in `src`. |
| Is FiQA actually used? | No. It was mentioned in README only. | README dataset table; no loader/use in `src`. |
| Is FinanceBench/TAT-QA/FinQA used? | No. No code or data paths existed. | No matching loaders or processed eval files. |
| Is LangGraph actually implemented with `StateGraph`, nodes, edges, and conditional routing? | No. `src/agents/rag_agent.py` describes LangGraph but implements a manual Python loop. Search found no `StateGraph`, `add_node`, or conditional edge usage. | `src/agents/rag_agent.py`. |
| Is LoRA fine-tuning actually using a serious dataset? | No. Existing LoRA code was demo-scale and not connected to FinanceBench/TAT-QA/FinQA/FiQA dataset preparation. | `src/finetuning/lora_finetune.py`; `notebooks/colab_lora_finetune.py`. |
| Are evaluation metrics hardcoded anywhere? | Yes. README and Streamlit UI included fixed benchmark values such as Precision@5, Recall@5, MRR, latency, and accuracy claims. | `README.md` evaluation table; `app.py` around evaluation dashboard chart data. |
| Is BM25/sparse retrieval persisted and loaded correctly? | It was persisted to `data/processed/bm25_index.pkl` when built, but no production startup loaded it automatically before search. | `src/embeddings/sparse_embedder.py`; `api.py` only builds sample BM25 on cold start. |
| Are FastAPI request fields such as `top_k`, `use_reranking`, and `use_multi_query` actually used? | No. `QueryRequest` defines them, but `/query` calls `_pipeline.run(request.question)` and drops the values. | `api.py` `/query`. |
| Are Docker, CI, production settings, logging, and health checks implemented? | Logging and `/health` existed in a basic form. Dockerfile, compose, Makefile, CI, readiness, metrics, background jobs, and deployment docs were absent. | `src/utils/logger.py`; `api.py`; no Docker/CI files. |

## Conclusion

The repository had a credible demo skeleton, but its primary production path relied on bundled sample data and hardcoded UI/README metrics. The implementation work must add real ingestion/indexing/evaluation paths, honest fallback behavior, a real LangGraph graph, and production scaffolding without fabricating unavailable external dataset/API/GPU results.
