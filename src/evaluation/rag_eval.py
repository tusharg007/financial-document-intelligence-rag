"""RAG answer evaluation helpers."""
from __future__ import annotations

import time
from typing import Any, Dict, List


def token_f1(answer: str, truth: str) -> float:
    a = set((answer or "").lower().split())
    t = set((truth or "").lower().split())
    if not a or not t:
        return 0.0
    common = len(a & t)
    precision = common / len(a)
    recall = common / len(t)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate_cases(cases: List[Dict[str, Any]], pipeline, llm_provider: str = "extractive") -> Dict[str, Any]:
    results = []
    for case in cases:
        started = time.time()
        response = pipeline.run(case["question"], llm_provider=llm_provider, debug=True)
        truth = case.get("answer") or case.get("ground_truth", "")
        results.append({
            "question": case["question"],
            "latency": time.time() - started,
            "token_f1": token_f1(response.get("answer", ""), truth),
            "citation_precision": 1.0 if response.get("citations") else 0.0,
            "faithfulness": 1.0 if response.get("citations") and not response.get("refusal") else 0.0,
            "refusal_correctness": 1.0 if case.get("should_refuse") == response.get("refusal") else 0.0,
        })
    denom = max(len(results), 1)
    return {
        "cases": results,
        "metrics": {
            "answer_correctness": sum(r["token_f1"] for r in results) / denom,
            "citation_precision": sum(r["citation_precision"] for r in results) / denom,
            "faithfulness": sum(r["faithfulness"] for r in results) / denom,
            "refusal_correctness": sum(r["refusal_correctness"] for r in results) / denom,
        },
    }
