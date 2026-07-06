# Observe RAG Queries

## Problem

You want a lightweight log of RAG behavior over time without adding a full observability stack.

## Why It Happens

Developers often need a record of question quality, citation usage, and heuristic scores during local debugging.

## Code

```python
from rag_eval_kit import evaluate_rag_response, log_rag_observation

result = evaluate_rag_response(
    question="What are the main risk factors?",
    answer="Competition and demand are major risks. [Source 1]",
    contexts=["Competition and demand are discussed in the filing."],
)

record = log_rag_observation(
    question="What are the main risk factors?",
    answer="Competition and demand are major risks. [Source 1]",
    contexts=["Competition and demand are discussed in the filing."],
    citations=[{"source_num": 1}],
    eval_result=result,
    latency_ms=42.0,
)
print(record)
```

## Expected Output

- one JSONL line in `logs/rag_observability.jsonl`
- metrics and guardrail status recorded beside the question and answer preview

## Debugging Tips

- use a dedicated temp log path in tests
- include latency when you compare multiple prompt or retrieval variants

## Common Mistakes

- logging full private documents instead of small answer previews
- forgetting that `logs/` should remain uncommitted
