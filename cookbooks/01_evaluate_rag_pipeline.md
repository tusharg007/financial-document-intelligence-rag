# Evaluate a RAG Pipeline

## Problem

You want a quick, offline-friendly way to score a RAG answer without calling an external judge.

## Why It Happens

Most RAG evaluation tools depend on hosted models or heavyweight frameworks. That makes local debugging slower and harder to reproduce.

## Code

```python
from rag_eval_kit import evaluate_rag_response

result = evaluate_rag_response(
    question="What revenue risks are discussed?",
    answer="Revenue may be affected by competition and demand. [Source 1]",
    contexts=["Revenue may be affected by competition and customer demand."],
    expected_keywords=["revenue", "competition", "demand"],
)
print(result.to_dict())
```

## Expected Output

- deterministic JSON-like scores
- `faithfulness`
- `context_relevance`
- `citation_coverage`
- `answer_completeness`
- `guardrail_status`

## Debugging Tips

- Start with short answers and one or two contexts.
- Verify that context text is the same text your pipeline actually retrieved.

## Common Mistakes

- Passing empty contexts and expecting meaningful faithfulness scores
- Treating the heuristics as a substitute for full human evaluation
