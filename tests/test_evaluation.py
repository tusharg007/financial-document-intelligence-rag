def test_token_f1():
    from src.evaluation.rag_eval import token_f1

    assert token_f1("Tesla revenue was high", "Tesla revenue") > 0


def test_retrieval_metrics_bounds():
    from src.evaluation.retrieval_eval import ndcg_at_k, precision_at_k, recall_at_k

    retrieved = ["a", "b", "c"]
    relevant = ["a", "c"]
    assert 0 <= precision_at_k(retrieved, relevant, 3) <= 1
    assert 0 <= recall_at_k(retrieved, relevant, 3) <= 1
    assert 0 <= ndcg_at_k(retrieved, relevant, 3) <= 1
