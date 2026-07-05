"""Retrieval metrics without hardcoded benchmark numbers."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List


def precision_at_k(retrieved: List[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return len(set(retrieved[:k]) & relevant) / max(k, 1)


def recall_at_k(retrieved: List[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return len(set(retrieved[:k]) & relevant) / max(len(relevant), 1)


def mrr(retrieved: List[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for idx, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    dcg = sum((1.0 if doc_id in relevant else 0.0) / math.log2(idx + 2) for idx, doc_id in enumerate(retrieved[:k]))
    ideal = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever(cases: List[Dict], retriever, k: int = 5) -> Dict[str, float]:
    rows = []
    for case in cases:
        results = retriever.retrieve(case["question"], top_k=k)
        retrieved = [r["doc_id"] for r in results]
        relevant = case.get("relevant_doc_ids", [])
        rows.append({
            "precision": precision_at_k(retrieved, relevant, k),
            "recall": recall_at_k(retrieved, relevant, k),
            "mrr": mrr(retrieved, relevant),
            "ndcg": ndcg_at_k(retrieved, relevant, k),
        })
    denom = max(len(rows), 1)
    return {key: sum(row[key] for row in rows) / denom for key in ["precision", "recall", "mrr", "ndcg"]}
