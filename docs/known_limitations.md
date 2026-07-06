# Known Limitations

This repository is intentionally explicit about what is still hard or incomplete.

## SEC Parsing Is Difficult

- SEC filings vary by issuer, year, HTML structure, and inline XBRL formatting.
- Section boundaries can still be noisy, especially when issuers embed repeated headers or complex formatting.
- The parser is much stronger than the original scaffold, but it is not a perfect filing-normalization engine.

## Financial Tables Are Not Deeply Modeled

- Tables are preserved primarily as text-bearing chunks.
- The system does not yet build a dedicated table graph, cell-level schema, or robust numeric reasoning layer.
- Some financial-statement questions would benefit from a more structured table representation.

## Extractive Fallback Is Weaker Than a Strong External LLM

- The extractive mode is deterministic and reproducible, which makes it excellent for local verification.
- It is still less fluent and less synthesizing than a high-quality external model.
- This tradeoff is deliberate for reproducibility and honest testing.

## Keyword Hit Rate Still Has Room to Improve

- The current `keyword_hit_rate` is useful but not saturated.
- Harder cases still expose misses in retrieval ranking or answer synthesis.
- Comparison questions and year-specific prompts remain the most likely places for missed expected terms.

## Evaluation Set Is Small

- The SEC evaluation set is curated and intentionally lightweight.
- It is good for regression detection but not a full benchmark.
- A larger and more adversarial set would improve confidence in generalization.

## No Production UI or Deployment Path in This Step

- The repo contains interfaces, but Step 7 focuses on project polish and reproducibility rather than hosting.
- Production serving, authentication, observability, and cost controls are not the focus of this documentation pass.

## Latency Can Be Improved Further

- Dense model load and reranking still dominate local response time.
- Caching, smaller rerank pools, or lighter models could reduce latency.
- The current latency is reasonable for a local technical demo but not fully optimized.

