"""Streamlit UI for the Financial Document Intelligence project."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import PROJECT_ROOT
from src.agents.langgraph_rag import get_langgraph_rag
from src.data.chunking import load_chunks
from src.llm.factory import get_llm


REPORTS = PROJECT_ROOT / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def status_badge(ok: bool) -> str:
    return "available" if ok else "missing"


st.set_page_config(page_title="Financial Document Intelligence", layout="wide")
st.title("Financial Document Intelligence")

page = st.sidebar.radio(
    "Page",
    [
        "Chat over filings",
        "Data ingestion dashboard",
        "Dataset explorer",
        "Document/chunk explorer",
        "Company comparison",
        "Temporal trend analysis",
        "Evaluation dashboard",
        "LoRA training/evaluation dashboard",
        "System health/settings",
    ],
)

chunks = load_chunks()
provider = get_llm()

if page == "Chat over filings":
    question = st.text_area("Question", "What revenue risks are discussed in the filings?")
    col1, col2, col3 = st.columns(3)
    top_k = col1.number_input("Top K", min_value=1, max_value=20, value=5)
    use_reranking = col2.checkbox("Reranking", value=True)
    use_multi_query = col3.checkbox("Multi-query", value=True)
    provider_name = st.selectbox("LLM provider", ["extractive", "groq", "huggingface", "lora"])
    if st.button("Ask"):
        result = get_langgraph_rag().run(question, top_k=top_k, use_reranking=use_reranking, use_multi_query=use_multi_query, llm_provider=provider_name, debug=True)
        st.write(result.get("answer"))
        st.json({"confidence": result.get("confidence"), "provider": result.get("provider_used"), "refusal": result.get("refusal")})
        st.subheader("Citations")
        st.dataframe(pd.DataFrame(result.get("citations", [])))
        st.subheader("Trace")
        st.json(result.get("trace", []))

elif page == "Data ingestion dashboard":
    st.write("SEC ingestion is available through `python scripts/ingest_sec.py` or API `POST /ingest/sec`.")
    manifest = PROCESSED / "filing_manifest.csv"
    st.metric("Manifest", status_badge(manifest.exists()))
    if manifest.exists():
        st.dataframe(pd.read_csv(manifest))
    st.code("python scripts/ingest_sec.py --tickers AAPL MSFT --forms 10-K 10-Q --start-year 2023 --end-year 2025 --limit-per-company 5")

elif page == "Dataset explorer":
    card = REPORTS / "dataset_card.md"
    if card.exists():
        st.markdown(card.read_text(encoding="utf-8"))
    else:
        st.info("No dataset card yet. Run `python scripts/build_dataset_card.py`.")
    st.metric("Chunks", len(chunks))
    st.metric("Companies", len({c.get("company") for c in chunks if c.get("company")}))

elif page == "Document/chunk explorer":
    st.write(f"Loaded chunks: {len(chunks)}")
    ticker = st.text_input("Filter ticker")
    rows = [c for c in chunks if not ticker or c.get("ticker", "").upper() == ticker.upper()]
    st.dataframe(pd.DataFrame(rows[:200]))

elif page == "Company comparison":
    companies = st.text_input("Companies", "AAPL, MSFT")
    topic = st.text_input("Topic", "risk factors")
    if st.button("Compare"):
        question = f"Compare {companies} on {topic}"
        st.json(get_langgraph_rag().run(question, llm_provider="extractive", debug=True))

elif page == "Temporal trend analysis":
    company = st.text_input("Company", "TSLA")
    topic = st.text_input("Topic", "revenue")
    if st.button("Analyze trend"):
        question = f"How has {company} changed over time regarding {topic}?"
        st.json(get_langgraph_rag().run(question, llm_provider="extractive", debug=True))

elif page == "Evaluation dashboard":
    latest = read_json(REPORTS / "evaluation_latest.json")
    if latest:
        st.json(latest)
        st.markdown((REPORTS / "evaluation_latest.md").read_text(encoding="utf-8") if (REPORTS / "evaluation_latest.md").exists() else "")
    else:
        st.info("No evaluation report exists. Run `python scripts/evaluate.py --eval-set demo --retriever hybrid_rerank --llm extractive --output reports/evaluation_latest.json`.")

elif page == "LoRA training/evaluation dashboard":
    lora = read_json(REPORTS / "lora_eval_results.json")
    adapter = PROJECT_ROOT / "adapters" / "lora_findoc"
    st.metric("LoRA adapter", status_badge(adapter.exists()))
    if lora:
        st.json(lora)
    else:
        st.info("LoRA not trained yet. No `reports/lora_eval_results.json` file exists.")
    metrics = read_json(REPORTS / "lora_training_metrics.json")
    if metrics:
        st.json(metrics)

elif page == "System health/settings":
    dense_index = PROJECT_ROOT / "data" / "indexes" / "chroma"
    bm25_index = PROJECT_ROOT / "data" / "indexes" / "bm25" / "bm25_index.pkl"
    st.json({
        "provider_health": provider.health_check(),
        "dense_index_status": status_badge(dense_index.exists()),
        "bm25_index_status": status_badge(bm25_index.exists()),
        "chunks": len(chunks),
        "filings": status_badge((PROCESSED / "filing_manifest.csv").exists()),
        "latest_evaluation": status_badge((REPORTS / "evaluation_latest.json").exists()),
        "lora_adapter": status_badge((PROJECT_ROOT / "adapters" / "lora_findoc").exists()),
    })
