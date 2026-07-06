# Add Guardrails to RAG Answers

## Problem

You want a deterministic way to flag unsupported or weakly supported answers.

## Why It Happens

RAG pipelines can still produce polished text even when the supporting contexts are weak or missing.

## Code

```python
from rag_eval_kit import evaluate_rag_response

result = evaluate_rag_response(
    question="What does the filing say about dividend policy?",
    answer="The company has a very strong dividend policy and expects large increases.",
    contexts=["The filing discusses unrelated market risks and customer demand."],
)
print(result.hallucination_flag, result.guardrail_status, result.warnings)
```

## Expected Output

- `hallucination_flag = True`
- `guardrail_status = "fail"`

## Debugging Tips

- inspect sentence-level support from the answer against the contexts
- add citation markers if the answer truly has evidence

## Common Mistakes

- assuming confident tone means grounded content
- allowing answers without any citation markers
