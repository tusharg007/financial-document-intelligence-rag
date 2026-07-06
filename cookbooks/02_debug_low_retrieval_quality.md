# Debug Low Retrieval Quality

## Problem

Relevant answers score poorly because the retrieved contexts are only weakly aligned with the question.

## Why It Happens

- metadata filters are too broad
- chunks are boilerplate-heavy
- retrieval terms and answer terms drift apart

## Code

```python
from rag_eval_kit import evaluate_rag_response

result = evaluate_rag_response(
    question="What does the filing say about cloud demand?",
    answer="The filing discusses demand trends. [Source 1]",
    contexts=["This section focuses on generic operations and legal matters."],
    expected_keywords=["cloud", "demand"],
)
print(result.to_dict())
```

## Expected Output

- lower `context_relevance`
- lower `answer_completeness`
- warning-oriented `guardrail_status`

## Debugging Tips

- inspect the actual retrieved chunk text
- compare query wording with the top chunk wording
- add expected keywords for known target concepts

## Common Mistakes

- debugging only the answer text without checking retrieved contexts
- assuming a citation marker implies the evidence is relevant
