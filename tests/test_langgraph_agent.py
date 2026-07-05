def test_langgraph_trace_for_out_of_scope():
    from src.agents.langgraph_rag import LangGraphRAG

    agent = LangGraphRAG.__new__(LangGraphRAG)
    state = {"question": "What is the best pasta recipe?", "trace": []}
    state = LangGraphRAG.classify_query(agent, state)
    state = LangGraphRAG.refusal(agent, state)
    assert state["refusal"] is True
    assert any(t["node"] == "classify_query" for t in state["trace"])


def test_weak_evidence_refuses():
    from src.agents.langgraph_rag import LangGraphRAG

    agent = LangGraphRAG.__new__(LangGraphRAG)
    state = {"question": "What was revenue?", "confidence": {"answerable": False, "reason": "No evidence"}, "trace": []}
    state = LangGraphRAG.decide_refusal_or_final_answer(agent, state)
    assert state["refusal"] is True
    assert "not have enough grounded evidence" in state["answer"]
