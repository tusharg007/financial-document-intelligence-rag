# Reproducibility

This document lists the exact commands used to rebuild the verified SEC corpus, indexes, retrieval checks, answering checks, and evaluation outputs.

## Environment Setup

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

## Rebuild the SEC Corpus

```powershell
python scripts/ingest_sec.py --tickers AAPL MSFT TSLA NVDA JPM --forms 10-K 10-Q --start-year 2022 --end-year 2024 --limit-per-company 8
python scripts/verify_dataset.py
python scripts/verify_chunk_quality.py
```

Expected outputs:

- `data/processed/filing_manifest.csv`
- `data/processed/filing_manifest.parquet`
- `data/processed/chunks.parquet`
- `data/processed/chunks.jsonl`

## Rebuild the Indexes

```powershell
python scripts/verify_dependencies.py
python scripts/build_indexes.py
python scripts/verify_indexes.py
```

Expected outputs:

- dense index under `data/indexes/chroma/`
- sparse index under `data/indexes/bm25/`

## Re-run Retrieval and Answering Checks

```powershell
python scripts/verify_retrieval.py
python scripts/verify_answering.py
```

Manual spot checks:

```powershell
python scripts/query_retrieval.py "What does Microsoft say about revenue?" --ticker MSFT --form-type 10-K --top-k 5
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/query_answer.py "What does Nvidia say about dividend policy in these filings?" --ticker NVDA --top-k 5
```

## Re-run Evaluation

```powershell
python scripts/run_evaluation.py
python scripts/verify_evaluation.py
```

Generated reports:

- `reports/evaluation_results.json`
- `reports/evaluation_summary.md`
- `reports/evaluation_comparison.md`

## Re-run Documentation Verification and Tests

```powershell
python scripts/verify_project_docs.py
python -m pytest -q --basetemp=.pytest-tmp-project-polish
```

If Windows blocks reuse of an old pytest temp directory:

```powershell
python -m pytest -q --basetemp=.pytest-tmp-project-polish-run2
```

## Intentionally Not Committed

The repository intentionally does not commit the following generated or local-only artifacts:

- raw SEC filings under `data/raw/`
- processed chunk outputs under `data/processed/`
- dense and sparse indexes under `data/indexes/`
- local environment files such as `.env`
- local pytest temp folders such as `.pytest-tmp*`

This behavior is enforced by [.gitignore](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/.gitignore).

