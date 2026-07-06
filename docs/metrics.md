# Metrics

This project uses a small but practical metric set to evaluate retrieval quality, grounded answering quality, attribution quality, and response latency on the curated SEC evaluation set.

## Current Snapshot

| Metric | Current score | What it suggests |
| --- | ---: | --- |
| `keyword_hit_rate` | `0.500` | Answers recover about half of the expected cue terms across the curated set; useful, but still improvable. |
| `citation_coverage` | `0.926` | Most answers cite the evidence they rely on. |
| `source_url_coverage` | `1.000` | Every cited result keeps a real SEC URL. |
| `weak_evidence_rate` | `0.222` | A minority of cases still land in weak or insufficient evidence territory. |
| `no_answer_handling` | `1.000` | Unsupported questions are handled honestly in the current evaluation set. |
| `latency_ms_avg` | `3944.30` | Local end-to-end latency is acceptable for demos but not yet optimized. |
| `top_k_ticker_match` | `1.000` | Filtered retrieval returns the expected company consistently. |
| `expected_section_match` | `1.000` | Section-aware retrieval behavior is strong on the curated set. |
| `expected_form_type_match` | `1.000` | Form filtering is behaving correctly on annual/quarterly cases. |

## Metric Definitions

### `keyword_hit_rate`

What it measures:

- The fraction of expected keywords recovered in the answer or supporting evidence for a case.

Why it matters:

- It is a rough relevance signal for whether the answer is talking about the expected subject matter.

How to interpret `0.500`:

- The system is finding and surfacing relevant evidence often enough to be useful.
- It still misses some expected thematic terms, especially on harder comparison or year-specific prompts.

### `citation_coverage`

What it measures:

- The extent to which answers include citations tied to retrieved evidence.

Why it matters:

- Financial QA without traceable support is not trustworthy.

How to interpret `0.926`:

- Citation behavior is strong and materially improved over earlier pipeline stages.

### `source_url_coverage`

What it measures:

- Whether cited sources preserve a real SEC `source_url`.

Why it matters:

- This is the final traceability link back to the filing.

How to interpret `1.000`:

- Source attribution is complete on the current evaluation set.

### `weak_evidence_rate`

What it measures:

- The share of evaluation cases that land in a weak or insufficient evidence state.

Why it matters:

- A lower value means the system is finding stronger grounded evidence more consistently.

How to interpret `0.222`:

- The project is doing much better than the earlier regression point, but a non-trivial set of cases still needs stronger retrieval or synthesis.

### `no_answer_handling`

What it measures:

- Whether intentionally unsupported questions are handled honestly rather than answered confidently without support.

Why it matters:

- In financial QA, abstention is often preferable to an unsupported answer.

How to interpret `1.000`:

- The current no-answer cases are being handled correctly.

### `latency_ms_avg`

What it measures:

- Average end-to-end latency per evaluation query, including retrieval and answer generation.

Why it matters:

- Practical systems need acceptable response time as well as accuracy.

How to interpret `3944.30`:

- Local performance is demo-friendly, though reranking and model startup still dominate runtime.

### `top_k_ticker_match`

What it measures:

- Whether the top retrieved results align with the expected company when the query/filter implies one.

Why it matters:

- Company confusion is costly in multi-issuer corpora.

How to interpret `1.000`:

- Ticker targeting is reliable on the curated set.

### `expected_section_match`

What it measures:

- Whether results align with the expected section when the question implies a section such as `Risk Factors`.

Why it matters:

- Section alignment is important for avoiding vague or contextless evidence.

How to interpret `1.000`:

- The current chunk-quality and retrieval integration is doing its job here.

### `expected_form_type_match`

What it measures:

- Whether retrieval/answering is aligned with the intended filing form, such as `10-K`.

Why it matters:

- Annual versus quarterly context changes the meaning of financial disclosures.

How to interpret `1.000`:

- Form-aware filtering and retrieval are currently behaving well.

