from fastapi.testclient import TestClient


def test_health_and_ready_endpoints():
    from api import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_query_request_fields_are_forwarded(monkeypatch):
    import api

    captured = {}

    class FakeGraph:
        def run(self, **kwargs):
            captured.update(kwargs)
            return {
                "answer": "ok",
                "citations": [{"source_num": 1}],
                "confidence": {"score": 1, "answerable": True},
                "refusal": False,
                "retrieved_contexts": [{"doc_id": "1"}],
                "trace": [{"node": "test"}],
                "latency": 0.01,
                "provider_used": "extractive",
            }

    monkeypatch.setattr(api, "get_langgraph_rag", lambda: FakeGraph())
    client = TestClient(api.app)
    response = client.post("/query", json={"question": "What was revenue?", "top_k": 2, "use_reranking": False, "use_multi_query": False, "filters": {"ticker": "TSLA"}, "debug": True})
    assert response.status_code == 200
    assert captured["top_k"] == 2
    assert captured["use_reranking"] is False
    assert captured["use_multi_query"] is False
    assert captured["filters"] == {"ticker": "TSLA"}
    assert response.json()["citations"]
