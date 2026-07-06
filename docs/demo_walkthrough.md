# Demo Walkthrough

## Goal

This walkthrough is for a short technical demo of the end-to-end SEC pipeline: ingest, index, retrieve, answer, evaluate, and abstain honestly when the corpus does not support the question.

## 5-Minute Demo Flow

### 1. Show the verified project state

Open:

- [reports/evaluation_summary.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/reports/evaluation_summary.md)
- [reports/final_implementation_summary.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/reports/final_implementation_summary.md)

Call out:

- `5` companies
- `40` filings
- `14,019` chunks
- dense and BM25 index counts both match the corpus
- source URL coverage is `1.000`

### 2. Show retrieval on a grounded question

Run:

```powershell
python scripts/query_retrieval.py "What does Microsoft say about revenue?" --ticker MSFT --form-type 10-K --top-k 5
```

What to point out:

- results include filing metadata
- each result has a `source_url`
- both dense and BM25 scores are visible
- section and filing-year context are preserved

### 3. Show grounded answering

Run:

```powershell
python scripts/query_answer.py "What are Apple's main risk factors?" --ticker AAPL --section "Risk Factors" --top-k 5
```

What to explain technically:

- answer generation uses retrieved SEC chunks only
- citations map back to real SEC URLs
- grounding warnings are explicit
- extractive fallback works without external API keys

### 4. Show honest no-answer handling

Run:

```powershell
python scripts/query_answer.py "What does Nvidia say about dividend policy in these filings?" --ticker NVDA --top-k 5
```

Expected behavior:

- `grounding_status` should indicate insufficient evidence
- warnings should say the retrieved evidence does not directly support the question
- citations may still appear if related filing material was retrieved, but the system should not claim that Nvidia disclosed a supported dividend-policy answer when the evidence is off-topic

### 5. Show evaluation output

Run:

```powershell
python scripts/run_evaluation.py
```

Then open:

- [reports/evaluation_summary.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/reports/evaluation_summary.md)
- [reports/evaluation_comparison.md](/F:/financial-document-intelligence-rag-master/financial-document-intelligence-rag-master/reports/evaluation_comparison.md)

Metrics worth highlighting:

- `citation_coverage`
- `source_url_coverage`
- `no_answer_handling`
- `weak_evidence_rate`
- `keyword_hit_rate`
- `latency_ms_avg`

## Suggested Technical Narrative

- Start with the constraint: SEC filings are difficult to parse cleanly and contain a lot of boilerplate.
- Explain that the chunker now emits quality metadata, which is used later by retrieval and answering.
- Emphasize that both dense and sparse retrieval are built from the same verified corpus.
- Show that answer generation is grounded in citations and can abstain honestly.
- Close with the evaluation harness as evidence that the system is measured rather than hand-waved.

## What Output to Show

- retrieval result rows with scores and metadata
- answer output with `[Source N]` references
- SEC `source_url` values
- evaluation summary metrics
- one honest no-answer example

