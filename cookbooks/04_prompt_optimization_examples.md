# Prompt Optimization Examples

## Problem

Answer quality is decent, but the output misses expected concepts or citations.

## Why It Happens

- prompts are too broad
- the answer does not explicitly instruct the model to cite sources
- the question does not mention the target section or company clearly

## Code

```python
better_question = "What does Apple's Risk Factors section say about competition and supply chain risk?"
```

```python
prompt_hint = "Answer only from the supplied SEC excerpts and cite them as [Source N]."
```

## Expected Output

- more specific answer wording
- better keyword coverage
- more reliable citation markers

## Debugging Tips

- include section hints when the filing structure matters
- keep prompts narrow enough to match the retrieved evidence

## Common Mistakes

- expecting a generic question to surface a very specific subsection
- optimizing prompts before checking retrieval quality
