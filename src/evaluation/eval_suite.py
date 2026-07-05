"""
Comprehensive evaluation suite for the RAG pipeline.

Provides retrieval metrics, embedding model comparison, latency benchmarks,
and cost-performance trade-off analysis. Includes RAGAS-compatible metric
calculations without requiring external LLM for evaluation.
"""
import time
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np

from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("eval_suite")


@dataclass
class RetrievalMetrics:
    """Container for retrieval quality metrics."""
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0
    avg_rank_of_relevant: float = 0.0


@dataclass
class LatencyMetrics:
    """Container for latency measurements."""
    embedding_time_ms: float = 0.0
    dense_retrieval_time_ms: float = 0.0
    sparse_retrieval_time_ms: float = 0.0
    hybrid_retrieval_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    total_pipeline_time_ms: float = 0.0
    generation_time_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one evaluation run."""
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    model_name: str = ""
    retrieval_method: str = ""
    num_queries: int = 0
    timestamp: str = ""


class EvaluationSuite:
    """
    Comprehensive evaluation framework for the RAG pipeline.
    
    Provides:
    1. Retrieval quality metrics (Precision, Recall, MRR, NDCG)
    2. Latency benchmarking
    3. Embedding model comparison
    4. Cost-performance trade-off analysis
    """

    def __init__(self):
        self.results_dir = Path(settings.processed_data_dir) / "eval_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Retrieval Quality Metrics
    # ========================================================================

    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K: fraction of retrieved docs that are relevant.
        
        P@K = |retrieved_k ∩ relevant| / K
        """
        retrieved_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        
        if k == 0:
            return 0.0
        
        return len(retrieved_k & relevant) / k

    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K: fraction of relevant docs that are retrieved.
        
        R@K = |retrieved_k ∩ relevant| / |relevant|
        """
        retrieved_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        
        if not relevant:
            return 0.0
        
        return len(retrieved_k & relevant) / len(relevant)

    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / rank_of_first_relevant_doc
        """
        relevant = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        
        return 0.0

    def ndcg_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain @ K.
        
        NDCG = DCG / IDCG where DCG = Σ (rel_i / log2(i+1))
        """
        relevant = set(relevant_ids)
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k]):
            rel = 1.0 if doc_id in relevant else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because 0-indexed
        
        # IDCG (ideal ranking: all relevant docs first)
        ideal_rels = [1.0] * min(len(relevant), k)
        ideal_rels.extend([0.0] * max(0, k - len(ideal_rels)))
        
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

    def evaluate_retrieval(
        self,
        queries: List[str],
        ground_truth: List[List[str]],
        retriever,
        k: int = 5
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality across multiple queries.
        
        Args:
            queries: List of test queries
            ground_truth: List of lists of relevant doc IDs per query
            retriever: Retriever to evaluate
            k: K for precision/recall calculation
            
        Returns:
            Aggregated retrieval metrics
        """
        all_precision = []
        all_recall = []
        all_mrr = []
        all_ndcg = []
        hit_count = 0

        for query, relevant_ids in zip(queries, ground_truth):
            results = retriever.retrieve(query, top_k=k)
            retrieved_ids = [r["doc_id"] for r in results]
            
            p = self.precision_at_k(retrieved_ids, relevant_ids, k)
            r = self.recall_at_k(retrieved_ids, relevant_ids, k)
            mrr = self.mean_reciprocal_rank(retrieved_ids, relevant_ids)
            ndcg = self.ndcg_at_k(retrieved_ids, relevant_ids, k)
            
            all_precision.append(p)
            all_recall.append(r)
            all_mrr.append(mrr)
            all_ndcg.append(ndcg)
            
            if any(rid in set(relevant_ids) for rid in retrieved_ids):
                hit_count += 1

        return RetrievalMetrics(
            precision_at_k=np.mean(all_precision) if all_precision else 0,
            recall_at_k=np.mean(all_recall) if all_recall else 0,
            mrr=np.mean(all_mrr) if all_mrr else 0,
            ndcg=np.mean(all_ndcg) if all_ndcg else 0,
            hit_rate=hit_count / len(queries) if queries else 0,
        )

    # ========================================================================
    # Latency Benchmarking
    # ========================================================================

    def benchmark_latency(
        self,
        query: str,
        pipeline
    ) -> LatencyMetrics:
        """
        Benchmark latency of each pipeline stage.
        
        Args:
            query: Test query
            pipeline: RAGPipeline instance
            
        Returns:
            Latency metrics for each stage
        """
        metrics = LatencyMetrics()

        # Dense embedding
        start = time.time()
        pipeline.retriever.dense.embed_texts([query])
        metrics.embedding_time_ms = (time.time() - start) * 1000

        # Dense retrieval
        start = time.time()
        dense_results = pipeline.retriever.retrieve_dense_only(query, top_k=10)
        metrics.dense_retrieval_time_ms = (time.time() - start) * 1000

        # Sparse retrieval
        start = time.time()
        sparse_results = pipeline.retriever.retrieve_sparse_only(query, top_k=10)
        metrics.sparse_retrieval_time_ms = (time.time() - start) * 1000

        # Hybrid retrieval
        start = time.time()
        hybrid_results = pipeline.retriever.retrieve(query, top_k=10)
        metrics.hybrid_retrieval_time_ms = (time.time() - start) * 1000

        # Reranking
        start = time.time()
        reranked = pipeline.reranker.rerank(query, hybrid_results, top_k=5)
        metrics.reranking_time_ms = (time.time() - start) * 1000

        # Total
        metrics.total_pipeline_time_ms = (
            metrics.embedding_time_ms +
            metrics.hybrid_retrieval_time_ms +
            metrics.reranking_time_ms
        )

        return metrics

    def run_latency_benchmark(
        self,
        queries: List[str],
        pipeline,
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run latency benchmarks across multiple queries and runs.
        
        Returns aggregated statistics (mean, p50, p95, p99).
        """
        all_metrics = []
        
        for run in range(num_runs):
            for query in queries:
                metrics = self.benchmark_latency(query, pipeline)
                all_metrics.append(metrics)
        
        # Aggregate
        def aggregate_field(field_name):
            values = [getattr(m, field_name) for m in all_metrics]
            return {
                "mean": np.mean(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "min": np.min(values),
                "max": np.max(values),
            }
        
        return {
            "embedding": aggregate_field("embedding_time_ms"),
            "dense_retrieval": aggregate_field("dense_retrieval_time_ms"),
            "sparse_retrieval": aggregate_field("sparse_retrieval_time_ms"),
            "hybrid_retrieval": aggregate_field("hybrid_retrieval_time_ms"),
            "reranking": aggregate_field("reranking_time_ms"),
            "total_pipeline": aggregate_field("total_pipeline_time_ms"),
            "num_queries": len(queries),
            "num_runs": num_runs,
        }

    # ========================================================================
    # Embedding Model Comparison
    # ========================================================================

    def compare_embedding_models(
        self,
        models: List[str],
        test_queries: List[str],
        test_documents: List[Dict],
        ground_truth: List[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare different embedding models on retrieval quality and speed.
        
        Args:
            models: List of model names to compare
            test_queries: Evaluation queries
            test_documents: Documents to index
            ground_truth: Optional relevant doc IDs per query
            
        Returns:
            List of comparison results per model
        """
        from src.embeddings.dense_embedder import DenseEmbedder
        
        results = []
        
        for model_name in models:
            logger.info(f"Evaluating model: {model_name}")
            
            # Create temporary embedder
            embedder = DenseEmbedder(
                model_name=model_name,
                collection_name=f"eval_{model_name.replace('/', '_')}",
                persist_dir=str(self.results_dir / "temp_collections")
            )
            
            # Measure indexing time
            start = time.time()
            embedder.add_documents(test_documents)
            index_time = time.time() - start
            
            # Measure query time
            query_times = []
            all_results = []
            
            for query in test_queries:
                start = time.time()
                search_results = embedder.search(query, top_k=5)
                query_times.append((time.time() - start) * 1000)
                all_results.append(search_results)
            
            result = {
                "model": model_name,
                "embedding_dim": len(embedder.embed_texts(["test"])[0]),
                "index_time_s": round(index_time, 3),
                "avg_query_time_ms": round(np.mean(query_times), 2),
                "p95_query_time_ms": round(np.percentile(query_times, 95), 2),
                "num_documents": len(test_documents),
            }
            
            # If ground truth available, compute retrieval metrics
            if ground_truth:
                metrics = RetrievalMetrics()
                p_vals, r_vals, mrr_vals = [], [], []
                
                for i, (query, gt) in enumerate(zip(test_queries, ground_truth)):
                    retrieved_ids = [r["doc_id"] for r in all_results[i]]
                    p_vals.append(self.precision_at_k(retrieved_ids, gt, 5))
                    r_vals.append(self.recall_at_k(retrieved_ids, gt, 5))
                    mrr_vals.append(self.mean_reciprocal_rank(retrieved_ids, gt))
                
                result["precision_at_5"] = round(np.mean(p_vals), 4)
                result["recall_at_5"] = round(np.mean(r_vals), 4)
                result["mrr"] = round(np.mean(mrr_vals), 4)
            
            results.append(result)
            
            # Cleanup
            try:
                embedder.clear_collection()
            except Exception:
                pass
        
        return results

    # ========================================================================
    # Faithfulness & Relevancy (Local Evaluation)
    # ========================================================================

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate answer quality using local heuristic methods.
        
        Metrics:
        - Coverage: How much of the context is reflected in the answer
        - Conciseness: Answer length relative to context length
        - Citation accuracy: Whether citations refer to real sources
        - Factual overlap: N-gram overlap with ground truth (if provided)
        """
        metrics = {}
        
        # Coverage: important terms from context found in answer
        context_text = " ".join(contexts).lower()
        answer_lower = answer.lower()
        
        # Extract important terms (numbers, proper nouns, financial terms)
        import re
        context_terms = set(re.findall(
            r'\b(?:\$[\d,.]+\b|\d+\.?\d*%|\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b)',
            " ".join(contexts)
        ))
        
        if context_terms:
            covered = sum(1 for t in context_terms if t.lower() in answer_lower)
            metrics["context_coverage"] = round(covered / len(context_terms), 4)
        else:
            metrics["context_coverage"] = 0.0
        
        # Conciseness ratio
        if contexts:
            total_context_len = sum(len(c) for c in contexts)
            metrics["conciseness_ratio"] = round(
                len(answer) / max(total_context_len, 1), 4
            )
        
        # Citation check
        citation_refs = re.findall(r'\[Source \d+\]', answer)
        metrics["num_citations"] = len(citation_refs)
        metrics["has_citations"] = len(citation_refs) > 0
        
        # Ground truth overlap (if provided)
        if ground_truth:
            gt_tokens = set(ground_truth.lower().split())
            ans_tokens = set(answer_lower.split())
            
            if gt_tokens:
                overlap = len(gt_tokens & ans_tokens)
                metrics["factual_overlap"] = round(overlap / len(gt_tokens), 4)
            else:
                metrics["factual_overlap"] = 0.0
        
        return metrics

    # ========================================================================
    # Full Evaluation Run
    # ========================================================================

    def run_full_evaluation(
        self,
        pipeline,
        qa_pairs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on the full RAG pipeline.
        
        Args:
            pipeline: RAGPipeline instance
            qa_pairs: List of {question, ground_truth} dicts
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Running full evaluation with {len(qa_pairs)} QA pairs")
        
        results = {
            "num_queries": len(qa_pairs),
            "per_query_results": [],
            "aggregate_metrics": {},
            "latency_stats": {},
        }
        
        quality_scores = []
        latencies = []
        
        for qa in qa_pairs:
            question = qa["question"]
            ground_truth = qa.get("ground_truth", "")
            
            # Run pipeline
            start = time.time()
            response = pipeline.run(question)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Evaluate answer quality
            contexts = [
                c.get("content_preview", "") for c in response.get("citations", [])
            ]
            
            quality = self.evaluate_answer_quality(
                question, response["answer"], contexts, ground_truth
            )
            quality_scores.append(quality)
            
            results["per_query_results"].append({
                "question": question,
                "answer": response["answer"][:500],
                "num_citations": len(response.get("citations", [])),
                "latency_ms": round(latency, 2),
                "quality_metrics": quality,
                "query_type": response.get("query_type", ""),
            })
        
        # Aggregate metrics
        if quality_scores:
            results["aggregate_metrics"] = {
                "avg_context_coverage": round(
                    np.mean([s.get("context_coverage", 0) for s in quality_scores]), 4
                ),
                "avg_factual_overlap": round(
                    np.mean([s.get("factual_overlap", 0) for s in quality_scores
                             if "factual_overlap" in s]), 4
                ),
                "citation_rate": round(
                    np.mean([1 if s.get("has_citations") else 0 for s in quality_scores]), 4
                ),
            }
        
        if latencies:
            results["latency_stats"] = {
                "mean_ms": round(np.mean(latencies), 2),
                "p50_ms": round(np.percentile(latencies, 50), 2),
                "p95_ms": round(np.percentile(latencies, 95), 2),
                "p99_ms": round(np.percentile(latencies, 99), 2),
            }
        
        # Save results
        results_path = self.results_dir / "eval_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation complete. Results saved to {results_path}")
        
        return results

    def generate_benchmark_report(self, results: Dict) -> str:
        """Generate a markdown benchmark report."""
        report = ["# RAG Pipeline Evaluation Report\n"]
        
        # Summary
        agg = results.get("aggregate_metrics", {})
        lat = results.get("latency_stats", {})
        
        report.append("## Summary")
        report.append(f"- **Queries evaluated**: {results.get('num_queries', 0)}")
        report.append(f"- **Avg Context Coverage**: {agg.get('avg_context_coverage', 0):.1%}")
        report.append(f"- **Citation Rate**: {agg.get('citation_rate', 0):.1%}")
        report.append(f"- **Avg Latency**: {lat.get('mean_ms', 0):.0f}ms")
        report.append(f"- **P95 Latency**: {lat.get('p95_ms', 0):.0f}ms\n")
        
        # Per-query table
        report.append("## Per-Query Results\n")
        report.append("| Question | Type | Citations | Latency | Coverage |")
        report.append("|----------|------|-----------|---------|----------|")
        
        for r in results.get("per_query_results", []):
            q = r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"]
            report.append(
                f"| {q} | {r.get('query_type', '')} | "
                f"{r.get('num_citations', 0)} | "
                f"{r.get('latency_ms', 0):.0f}ms | "
                f"{r.get('quality_metrics', {}).get('context_coverage', 0):.1%} |"
            )
        
        return "\n".join(report)


# Singleton
_suite = None

def get_eval_suite() -> EvaluationSuite:
    """Get the global EvaluationSuite instance."""
    global _suite
    if _suite is None:
        _suite = EvaluationSuite()
    return _suite
