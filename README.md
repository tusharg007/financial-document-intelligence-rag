# Financial Document Intelligence RAG

> Local-first SEC filing RAG with grounded answers, real source URLs, deterministic evaluation, and developer-facing observability.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](#local-quickstart) [![Streamlit Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/) [![Tests](https://img.shields.io/badge/Pytest-79%20passed-brightgreen.svg)](#testing)

**Live demo:**  
[https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/](https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/)

Financial Document Intelligence RAG is an evidence-preserving question-answering system built on real SEC EDGAR filings. It ingests filings, extracts section-aware chunks, builds dense and sparse indexes, retrieves grounded evidence with metadata filters, and generates cited answers with explicit no-answer handling when the indexed corpus does not support the query.

The repository is designed to be rebuilt locally. Generated SEC artifacts and indexes stay out of Git, while the code, verification scripts, evaluation harness, and developer-facing tooling remain reproducible and reviewable.

## Table of Contents

- [What This Demonstrates](#what-this-demonstrates)
- [Live Demo vs Full Local Run](#live-demo-vs-full-local-run)
- [Verified Metrics](#verified-metrics)
- [Architecture](#architecture)
- [Developer Evaluation Kit](#developer-evaluation-kit)
- [RAG Observability](#rag-observability)
- [Cookbooks](#cookbooks)
- [Local Quickstart](#local-quickstart)
- [Repository Layout](#repository-layout)
- [Known Limitations](#known-limitations)
- [Testing](#testing)

## What This Demonstrates

- SEC filing ingestion over real EDGAR documents
- Section-aware parsing and quality-scored chunk generation
- Hybrid dense plus BM25 retrieval with reranking support
- Grounded citation-based answering with source URLs
- Honest no-answer and insufficient-evidence handling
- A curated evaluation harness for retrieval and answering quality
- Lightweight guardrails and observability for developer workflows
- SDK-style ergonomics for debugging, local checks, and documentation-driven reproducibility

## Live Demo vs Full Local Run

The hosted Streamlit app is a **lightweight interface preview**. The full SEC corpus and generated retrieval artifacts are intentionally excluded from GitHub so the repository stays small, reproducible, and clean.

Intentionally excluded from version control:

- raw SEC filings
- processed chunks
- Chroma dense index
- BM25 sparse index
- LoRA adapters

Because of that, the hosted Streamlit app may show dataset status, evaluation outputs, or demo-style pages instead of performing full live retrieval over the complete local indexes.

This is **intentional engineering hygiene, not a missing feature**. The full RAG workflow is reproducible locally with the commands below.

| Capability | Live Streamlit demo | Full local run |
| --- | --- | --- |
| UI navigation | Supported | Supported |
| Dataset and status views | Supported | Supported |
| Evaluation report display | Supported from saved artifacts | Supported with fresh runs |
| Live retrieval over full indexes | Limited by hosted artifacts | Supported after indexing |
| SEC ingestion | Not intended for hosted demo | Supported |
| Index rebuilding | Not intended for hosted demo | Supported |
| Optional LoRA experiment | UI-visible but not required | Optional, not required |

## Verified Metrics

These figures come from the locally verified pipeline and generated evaluation artifacts, not from the lightweight hosted app runtime.

| Metric | Value |
| --- | ---: |
| Companies | `5` |
| Filings | `40` |
| Chunks | `14,019` |
| Citation coverage (`citation_coverage`) | `0.9259` |
| Source URL coverage (`source_url_coverage`) | `1.000` |
| No-answer handling (`no_answer_handling`) | `1.000` |
| Weak evidence rate (`weak_evidence_rate`) | `0.2222` |
| Keyword hit rate (`keyword_hit_rate`) | `0.500` |
| Latest tests | `79 passed` |

## Architecture

```text
SEC EDGAR filings
  -> parser + section extraction
  -> quality-scored chunks
  -> dense index + BM25 index
  -> hybrid retrieval + reranking
  -> grounded answer generation
  -> evaluation + observability
```

The underlying local pipeline also includes ingestion, manifest generation, index verification, retrieval smoke tests, and grounded-answer verification. For a fuller walkthrough, see [docs/architecture.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/architecture.md).

## Developer Evaluation Kit

This repository includes a lightweight, deterministic package for developer-facing RAG checks: `rag_eval_kit`.

It is useful for:

- quick local response evaluation
- CI-style sanity checks
- debugging unsupported answers
- measuring citation presence and lexical grounding

It is **not** a replacement for richer LLM-as-judge evaluation. The goal is fast, offline, deterministic signal for engineering workflows.

### SDK API

```python
from rag_eval_kit import evaluate_rag_response

result = evaluate_rag_response(
    question="What was the revenue risk?",
    answer=answer,
    contexts=retrieved_chunks,
    ground_truth=None,
)

print(result.faithfulness)
print(result.context_relevance)
print(result.citation_coverage)
print(result.guardrail_status)
```

The result object exposes:

- `faithfulness`
- `context_relevance`
- `citation_coverage`
- `answer_completeness`
- `hallucination_flag`
- `guardrail_status`
- `warnings`

### CLI Usage

```powershell
python -m rag_eval_kit.cli `
  --input examples/rag_eval_cookbook/sample_eval_input.json `
  --output reports/rag_eval_kit_sample_output.json
```

The CLI writes JSON output containing fields such as:

- `faithfulness`
- `context_relevance`
- `citation_coverage`
- `hallucination_flag`
- `guardrail_status`
- `warnings`

## RAG Observability

The kit also includes lightweight query-level observability through `rag_eval_kit.observability.log_rag_observation()`.

It records JSONL events for:

- question text
- answer preview
- retrieved document count
- citation count
- faithfulness and context relevance signals
- citation coverage
- hallucination flag and guardrail status
- latency
- optional user feedback

This is useful for debugging weak retrieval, missing citations, unsupported answers, and slow queries. Logs should remain local and should not be committed.

Example command:

```powershell
python scripts/query_answer_observed.py "What are Apple's main risk factors?" --ticker AAPL
```

That command requires locally built indexes.

## Cookbooks

The repository includes short developer-facing guides for common RAG debugging and instrumentation tasks.

| Cookbook | Focus |
| --- | --- |
| [01_evaluate_rag_pipeline.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/01_evaluate_rag_pipeline.md) | Evaluate a RAG pipeline with deterministic local metrics |
| [02_debug_low_retrieval_quality.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/02_debug_low_retrieval_quality.md) | Debug low retrieval quality and weak evidence |
| [03_add_guardrails_to_rag_answers.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/03_add_guardrails_to_rag_answers.md) | Add simple guardrails to grounded answers |
| [04_prompt_optimization_examples.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/04_prompt_optimization_examples.md) | Compare prompt-shaping patterns for grounded answering |
| [05_observe_rag_queries.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/05_observe_rag_queries.md) | Observe RAG queries with JSONL logging |

Example assets for the evaluation kit live under [examples/rag_eval_cookbook](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/examples/rag_eval_cookbook), including sample input/output JSON, an example script, and a quickstart notebook.

## Local Quickstart

This project is designed to be rebuilt locally on Windows PowerShell.

### Environment setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Minimum `.env` values:

```dotenv
SEC_EDGAR_USER_AGENT=Your Name your_email@example.com
LLM_PROVIDER=extractive
CHROMA_PERSIST_DIR=./data/indexes/chroma
```

### Full local SEC workflow

```powershell
python scripts/ingest_sec.py --tickers AAPL MSFT TSLA NVDA JPM --forms 10-K 10-Q --start-year 2022 --end-year 2024 --limit-per-company 8
python scripts/verify_dataset.py
python scripts/build_indexes.py
python scripts/verify_indexes.py
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/run_evaluation.py
python scripts/verify_evaluation.py
```

### Streamlit locally

```powershell
streamlit run app.py
```

### Evaluation kit CLI

```powershell
python -m rag_eval_kit.cli `
  --input examples/rag_eval_cookbook/sample_eval_input.json `
  --output reports/rag_eval_kit_sample_output.json
```

If you want the fuller command-by-command path, see [docs/quickstart.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/quickstart.md).

## Repository Layout

```text
app.py                      Streamlit entrypoint
api.py                      FastAPI entrypoint
rag_eval_kit/              Lightweight RAG evaluation and observability helpers
scripts/                    Ingestion, indexing, verification, and CLI commands
src/data/                   SEC ingestion, parsing, chunking
src/embeddings/             Dense and sparse indexing
src/retrieval/              Hybrid retrieval, reranking, confidence logic
src/answering/              Grounded answer generation
src/evaluation/             Evaluation harness and metrics
cookbooks/                  Developer-facing workflow guides
examples/                   Sample evaluation-kit inputs, outputs, and notebook
docs/                       Architecture, quickstart, demo, metrics, limitations
reports/                    Generated evaluation and project reports
tests/                      Unit and integration tests
```

## Known Limitations

- SEC filings are hard to parse perfectly across issuers and years.
- Financial tables are not deeply modeled as structured analytical objects.
- The hosted demo does not bundle the generated SEC artifacts or local indexes.
- LoRA is optional and not trained by default.
- Deterministic evaluation heuristics are useful for debugging, but they are not a perfect judge of answer quality.

For more detail, see [docs/known_limitations.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/known_limitations.md).

## Testing

```powershell
python scripts/verify_project_docs.py
python -m pytest -q --basetemp=.pytest-tmp-readme-rag-eval-final
```

Core verification commands:

```powershell
python scripts/verify_dataset.py
python scripts/verify_indexes.py
python scripts/verify_retrieval.py
python scripts/verify_answering.py
python scripts/verify_evaluation.py
```

## Troubleshooting

### Missing indexes in the hosted app

That is expected for the public GitHub and Streamlit deployment. The large generated artifacts are intentionally excluded from version control. Use the local quickstart commands to rebuild the full pipeline.

### Pytest temp-folder locks on Windows

If pytest fails on a stale temp folder, rerun with a fresh base temp:

```powershell
python -m pytest -q --basetemp=.pytest-tmp-readme-rag-eval-final-run2
```
