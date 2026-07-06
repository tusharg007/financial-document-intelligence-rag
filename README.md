# Financial Document Intelligence RAG

> A reproducible SEC filing RAG system for ingestion, chunking, indexing, retrieval, grounded answering, and evaluation.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](#local-reproduction) [![Streamlit Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-red.svg)](https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/) [![Tests](https://img.shields.io/badge/Pytest-72%20passed-brightgreen.svg)](#testing)

**Live demo:**  
[https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/](https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/)

Financial Document Intelligence RAG is a local-first, evidence-preserving question-answering pipeline built on real SEC EDGAR filings. It ingests filings, extracts section-aware chunks, builds dense and sparse indexes, retrieves grounded evidence with metadata filters, and generates cited answers with explicit no-answer handling when the indexed corpus does not support the query.

The project is optimized for engineering honesty: real source URLs, reproducible rebuild commands, verification scripts at each stage, and evaluation metrics that describe both what works and where the system still has room to improve.

## Deployment Notes

The hosted Streamlit app is intended as a **lightweight interface preview**. The full SEC corpus and vector/sparse indexes are generated artifacts and are intentionally excluded from GitHub to keep the repository small, reproducible, and clean.

Intentionally not committed:

- raw SEC filings
- processed chunks
- Chroma dense index
- BM25 sparse index
- LoRA adapters

Because of that, some live app pages may show dataset/index status, saved evaluation outputs, or demo-style content instead of executing the full live retrieval pipeline against the full SEC corpus.

This is **intentional engineering hygiene, not a missing feature**. The complete RAG workflow is supported locally using the documented ingestion, indexing, retrieval, and evaluation commands below. The main evaluated local system runs on:

- `5` companies
- `40` filings
- `14,019` quality-scored chunks

## Verified Metrics

| Metric | Value |
| --- | ---: |
| Companies | `5` |
| Filings | `40` |
| Chunks | `14,019` |
| `citation_coverage` | `0.9259` |
| `source_url_coverage` | `1.000` |
| `no_answer_handling` | `1.000` |
| `weak_evidence_rate` | `0.2222` |
| `keyword_hit_rate` | `0.500` |
| Pytest | `72 passed` |

These figures come from the local verified pipeline and generated evaluation artifacts, not from the lightweight hosted app runtime.

## What This Project Does

- Ingests real SEC EDGAR `10-K` and `10-Q` filings
- Parses filing sections such as `Risk Factors`, `MD&A`, `Business`, and `Notes`
- Produces quality-scored chunks with metadata like:
  - `is_toc_like`
  - `boilerplate_score`
  - `content_quality_score`
  - `section_confidence`
- Builds:
  - Chroma dense index
  - persisted BM25 sparse index
- Retrieves evidence with:
  - dense search
  - BM25
  - reciprocal rank fusion
  - optional reranking
  - metadata filters
- Generates grounded answers with:
  - SEC citations
  - source URLs
  - honest insufficient-evidence handling
- Evaluates retrieval and grounding quality with a curated SEC QA set

## Architecture

```text
SEC EDGAR filings
  -> ingestion + manifest
  -> parser + section extraction
  -> quality-scored chunks
  -> dense Chroma index + BM25 sparse index
  -> hybrid retrieval + reranking
  -> grounded answer generation
  -> evaluation reports
```

For the detailed system walkthrough, see [docs/architecture.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/architecture.md).

## Live Demo vs Full Local Run

| Capability | Live Streamlit demo | Full local run |
| --- | --- | --- |
| UI navigation | Supported | Supported |
| Dataset / status views | Supported | Supported |
| Evaluation report display | Supported from saved artifacts | Supported with fresh evaluation runs |
| Live retrieval over full indexes | Limited by hosted artifacts | Supported after ingestion and indexing |
| SEC ingestion | Not intended for hosted demo | Supported |
| Index rebuilding | Not intended for hosted demo | Supported |
| LoRA experiment | Optional UI preview only | Optional, not required |

## Local Reproduction

This repository is designed to be rebuilt locally on Windows PowerShell.

| Step | Command |
| --- | --- |
| Create venv | `python -m venv .venv` |
| Activate venv | `.\.venv\Scripts\Activate.ps1` |
| Upgrade pip | `python -m pip install --upgrade pip` |
| Install deps | `pip install -r requirements.txt` |
| Create env file | `Copy-Item .env.example .env` |

Minimum `.env` values:

```dotenv
SEC_EDGAR_USER_AGENT=Your Name your_email@example.com
LLM_PROVIDER=extractive
CHROMA_PERSIST_DIR=./data/indexes/chroma
```

### Rebuild the Full Pipeline

```powershell
python scripts/ingest_sec.py --tickers AAPL MSFT TSLA NVDA JPM --forms 10-K 10-Q --start-year 2022 --end-year 2024 --limit-per-company 8
python scripts/verify_dataset.py
python scripts/build_indexes.py
python scripts/verify_indexes.py
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/run_evaluation.py
python scripts/verify_evaluation.py
```

If you want the fuller command-by-command path, see [docs/quickstart.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/quickstart.md).

## Demo Commands

### Retrieval

```powershell
python scripts/query_retrieval.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/query_retrieval.py "What does Microsoft say about revenue?" --ticker MSFT --form-type 10-K --top-k 5
```

### Grounded answers

```powershell
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/query_answer.py "What does Nvidia say about dividend policy in these filings?" --ticker NVDA --top-k 5
```

The second query is a useful sanity check for no-answer behavior. It should return an `insufficient_evidence` style response instead of fabricating unsupported claims.

## Example Output Shape

```text
question: What are Apple's main risk factors?
grounding_status: grounded_with_warnings
used_provider: extractive

answer:
Based on the retrieved SEC filings:
- Apple describes risks tied to competitive pressure, legal and regulatory challenges, and returns on capital. [Source 3]
- The filings also discuss exposures related to liquidity, credit deterioration, market conditions, and political risk. [Source 5]
- Apple cites payment-card security and related operational issues as risks that could materially affect the business. [Source 2]
```

## Tech Stack

- Python 3.11
- pandas / pyarrow
- ChromaDB
- BM25 via `rank_bm25`
- Sentence Transformers embeddings
- Optional cross-encoder reranking
- Deterministic extractive fallback for local no-key runs
- Streamlit UI
- pytest verification

## Developer Evaluation Kit

This repository also includes a lightweight, deterministic developer-facing package: `rag_eval_kit`.

It provides:

- an SDK-style API:
  - `from rag_eval_kit import evaluate_rag_response`
- a CLI for evaluating a single RAG response payload
- cookbook examples for debugging retrieval, prompts, guardrails, and observability
- JSONL observability logging via `log_rag_observation()`
- fully offline heuristic metrics with no external judge dependency

Example CLI:

```powershell
python -m rag_eval_kit.cli --input examples/rag_eval_cookbook/sample_eval_input.json --output reports/rag_eval_kit_sample_output.json
```

See:

- [examples/rag_eval_cookbook/README.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/examples/rag_eval_cookbook/README.md)
- [cookbooks/01_evaluate_rag_pipeline.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/cookbooks/01_evaluate_rag_pipeline.md)

## Repository Layout

```text
app.py                      Streamlit entrypoint
api.py                      FastAPI entrypoint
scripts/                    Ingestion, indexing, verification, and CLI commands
src/data/                   SEC ingestion, parsing, chunking
src/embeddings/             Dense and sparse indexing
src/retrieval/              Hybrid retrieval, reranking, confidence logic
src/answering/              Grounded answer generation
src/evaluation/             Evaluation harness and metrics
docs/                       Architecture, quickstart, demo, metrics, limitations
reports/                    Generated evaluation and project reports
tests/                      Unit and integration tests
```

## Known Limitations

- SEC parsing is difficult and issuer formatting varies.
- Financial tables are not deeply modeled as structured table objects.
- The extractive fallback is weaker than a production-grade external LLM.
- The hosted demo does not bundle large generated indexes.
- LoRA is optional and not required for the main RAG workflow.

See [docs/known_limitations.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/docs/known_limitations.md) for the detailed list.

## Testing

```powershell
python scripts/verify_project_docs.py
python -m pytest -q --basetemp=.pytest-tmp-readme-final
```

Core smoke tests:

```powershell
python scripts/verify_dataset.py
python scripts/verify_indexes.py
python scripts/verify_retrieval.py
python scripts/verify_answering.py
python scripts/verify_evaluation.py
```

## Troubleshooting

### Windows pytest temp-folder locks

If pytest fails on a stale temp folder, rerun with a fresh base temp:

```powershell
python -m pytest -q --basetemp=.pytest-tmp-readme-final-run2
```

### Missing indexes in the hosted app

That is expected for the public GitHub/Streamlit deployment. The large generated artifacts are intentionally excluded from version control. Use the local reproduction commands to rebuild the full pipeline.
