"""Rule-based query router for financial RAG workflows."""


def classify_query(query: str) -> str:
    q = (query or "").lower()
    finance_terms = [
        "revenue", "filing", "risk", "cash", "margin", "income", "sec",
        "10-k", "10-q", "8-k", "company", "debt", "asset", "financial",
    ]
    if not any(term in q for term in finance_terms):
        return "out_of_scope"
    if any(term in q for term in ["summarize", "summary", "overview"]):
        return "document_summary"
    if any(term in q for term in ["compare", "versus", " vs ", "difference"]):
        return "comparison"
    if any(term in q for term in ["trend", "over time", "changed", "year over year", "quarter"]):
        return "temporal"
    if any(term in q for term in ["how much", "percentage", "ratio", "calculate", "increase", "decrease", "$"]):
        return "numerical"
    if "risk" in q:
        return "risk"
    if any(term in q for term in ["sentiment", "positive", "negative", "tone"]):
        return "sentiment"
    return "factual"
