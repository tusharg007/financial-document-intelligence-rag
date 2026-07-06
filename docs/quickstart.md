# Quickstart

This quickstart uses Windows PowerShell because the verified development workflow for this repository has been on Windows.

## 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Configure environment variables

```powershell
Copy-Item .env.example .env
notepad .env
```

Minimum `.env` values:

```dotenv
SEC_EDGAR_USER_AGENT=Your Name your_email@example.com
LLM_PROVIDER=extractive
CHROMA_PERSIST_DIR=./data/indexes/chroma
```

Optional local test setting:

```powershell
$env:RERANKER_MODE="fallback"
```

## 4. Ingest SEC filings

```powershell
python scripts/ingest_sec.py --tickers AAPL MSFT TSLA NVDA JPM --forms 10-K 10-Q --start-year 2022 --end-year 2024 --limit-per-company 8
```

## 5. Verify the dataset

```powershell
python scripts/verify_dataset.py
python scripts/verify_chunk_quality.py
```

## 6. Build indexes

```powershell
python scripts/build_indexes.py
```

## 7. Verify indexes

```powershell
python scripts/verify_dependencies.py
python scripts/verify_indexes.py
```

## 8. Query retrieval

```powershell
python scripts/query_retrieval.py "What does Microsoft say about revenue?" --ticker MSFT --form-type 10-K --top-k 5
python scripts/query_retrieval.py "Tesla risk factors" --ticker TSLA --section "Risk Factors" --top-k 5
```

## 9. Query grounded answers

```powershell
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
python scripts/query_answer.py "What does Nvidia say about dividend policy in these filings?" --ticker NVDA --top-k 5
```

The second command is a no-answer sanity check. It should return an `insufficient_evidence` style response rather than inventing a dividend-policy answer.

## 10. Run evaluation

```powershell
python scripts/run_evaluation.py
python scripts/verify_evaluation.py
```

## 11. Run tests

```powershell
python scripts/verify_project_docs.py
python -m pytest -q --basetemp=.pytest-tmp-project-polish
```

## If pytest hits a stale Windows temp-directory lock

Use a fresh temp folder:

```powershell
python -m pytest -q --basetemp=.pytest-tmp-project-polish-run2
```

