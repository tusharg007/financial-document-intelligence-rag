"""Run actual local retrieval/RAG evaluation and write reports."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import PROJECT_ROOT
from src.agents.langgraph_rag import get_langgraph_rag


def _load_eval(eval_set: str) -> List[Dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "processed" / f"eval_{eval_set}.jsonl"
    if not path.exists() and eval_set == "demo":
        from src.data.sample_data import get_evaluation_pairs
        return get_evaluation_pairs()
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set {eval_set} not found at {path}. Run scripts/prepare_datasets.py or provide local data.")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _token_f1(answer: str, truth: str) -> float:
    a = answer.lower().split()
    t = truth.lower().split()
    if not a or not t:
        return 0.0
    common = len(set(a) & set(t))
    precision = common / len(set(a))
    recall = common / len(set(t))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def run_evaluation_cli(eval_set: str, retriever: str, llm: str, output: str | Path) -> Dict[str, Any]:
    cases = _load_eval(eval_set)
    graph = get_langgraph_rag()
    per_case = []
    latencies = []
    for case in cases:
        question = case.get("question") or case.get("query") or ""
        truth = case.get("ground_truth") or case.get("answer") or ""
        started = time.time()
        result = graph.run(
            question,
            top_k=5,
            use_reranking="rerank" in retriever,
            use_multi_query=True,
            llm_provider=llm,
            debug=True,
        )
        latency = time.time() - started
        latencies.append(latency)
        answer = result.get("answer", "")
        per_case.append({
            "question": question,
            "ground_truth": truth,
            "answer": answer,
            "exact_match": int(truth.lower() in answer.lower()) if truth else 0,
            "token_f1": _token_f1(answer, truth) if truth else 0.0,
            "citation_support": int(bool(result.get("citations"))),
            "faithfulness_heuristic": int(not result.get("refusal") and bool(result.get("citations"))),
            "refusal": bool(result.get("refusal")),
            "latency": latency,
            "provider": result.get("provider_used", llm),
        })
    total = max(len(per_case), 1)
    results = {
        "eval_set": eval_set,
        "retriever": retriever,
        "llm": llm,
        "num_cases": len(per_case),
        "metrics": {
            "exact_match": sum(c["exact_match"] for c in per_case) / total,
            "token_f1": sum(c["token_f1"] for c in per_case) / total,
            "citation_precision": sum(c["citation_support"] for c in per_case) / total,
            "faithfulness_heuristic": sum(c["faithfulness_heuristic"] for c in per_case) / total,
            "refusal_rate": sum(1 for c in per_case if c["refusal"]) / total,
            "latency_p50": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "latency_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0,
            "latency_p99": sorted(latencies)[int(0.99 * (len(latencies) - 1))] if latencies else 0,
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (output.with_suffix(".md")).write_text(
        "# Evaluation Report\n\n"
        f"- Eval set: {eval_set}\n- Retriever: {retriever}\n- LLM: {llm}\n- Cases: {len(per_case)}\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in results["metrics"].items()) + "\n",
        encoding="utf-8",
    )
    cases_path = PROJECT_ROOT / "reports" / "rag_eval_cases.jsonl"
    cases_path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in per_case) + ("\n" if per_case else ""), encoding="utf-8")
    ablation_path = PROJECT_ROOT / "reports" / "retrieval_ablation.csv"
    with ablation_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cases", "citation_precision", "token_f1"])
        writer.writeheader()
        writer.writerow({"method": retriever, "cases": len(per_case), "citation_precision": results["metrics"]["citation_precision"], "token_f1": results["metrics"]["token_f1"]})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="demo")
    parser.add_argument("--retriever", default="hybrid_rerank")
    parser.add_argument("--llm", default="extractive")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reports" / "evaluation_latest.json"))
    args = parser.parse_args()
    print(json.dumps(run_evaluation_cli(args.eval_set, args.retriever, args.llm, args.output), indent=2))


if __name__ == "__main__":
    main()
