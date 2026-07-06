# RAG Evaluation Cookbook Example

This folder contains a minimal, offline-friendly example of the `rag_eval_kit` package.

Contents:

- `sample_eval_input.json` - example question, answer, contexts, and expected keywords
- `sample_eval_output.json` - example metric output
- `evaluate_single_response.py` - small script that evaluates the sample payload
- `rag_eval_quickstart.ipynb` - notebook-style walkthrough for the same flow

Run locally:

```powershell
python -m rag_eval_kit.cli --input examples/rag_eval_cookbook/sample_eval_input.json --output reports/rag_eval_kit_sample_output.json
python examples/rag_eval_cookbook/evaluate_single_response.py
```
