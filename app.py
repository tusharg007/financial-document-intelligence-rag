"""Streamlit UI for the Financial Document Intelligence project."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - used for lightweight test imports
    st = None

from config.settings import PROJECT_ROOT
from src.agents.langgraph_rag import get_langgraph_rag
from src.data.chunking import load_chunks
from src.llm.factory import get_llm


REPORTS = PROJECT_ROOT / "reports"
PROCESSED = PROJECT_ROOT / "data" / "processed"
INDEXES = PROJECT_ROOT / "data" / "indexes"

MISSING_INDEX_MESSAGE = (
    "Indexes are not available in this deployment. Run ingestion and indexing locally, "
    "or configure a hosted index artifact."
)
LORA_PAGE_LABEL = "Optional LoRA experiment"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def status_badge(ok: bool) -> str:
    return "available" if ok else "missing"


def project_artifact_status() -> dict:
    chunks_path = PROCESSED / "chunks.parquet"
    chunks_jsonl = PROCESSED / "chunks.jsonl"
    filings_path = PROCESSED / "filing_manifest.csv"
    dense_index_dir = INDEXES / "chroma"
    bm25_dir = INDEXES / "bm25"
    bm25_ready = (
        (bm25_dir / "bm25_documents.jsonl.gz").exists()
        and (bm25_dir / "bm25_tokens.jsonl.gz").exists()
    ) or (bm25_dir / "bm25_index.pkl").exists()
    chunks_ready = chunks_path.exists() or chunks_jsonl.exists()
    dense_ready = dense_index_dir.exists() and any(dense_index_dir.iterdir()) if dense_index_dir.exists() else False
    return {
        "chunks_ready": chunks_ready,
        "filings_ready": filings_path.exists(),
        "dense_ready": dense_ready,
        "bm25_ready": bm25_ready,
    }


def rag_runtime_available(status: dict) -> bool:
    return all([status["chunks_ready"], status["dense_ready"], status["bm25_ready"]])


def show_missing_index_warning(status: dict) -> None:
    st.warning(MISSING_INDEX_MESSAGE)
    st.caption(
        "Current artifact status: "
        f"chunks={status_badge(status['chunks_ready'])}, "
        f"dense={status_badge(status['dense_ready'])}, "
        f"bm25={status_badge(status['bm25_ready'])}, "
        f"filings={status_badge(status['filings_ready'])}"
    )


def maybe_show_demo_output() -> None:
    evaluation = read_json(REPORTS / "evaluation_results.json")
    cases = evaluation.get("cases", []) if isinstance(evaluation, dict) else []
    if not cases:
        return
    st.info("Showing static demo output from the latest saved evaluation report.")
    sample = cases[0]
    st.write(sample.get("answer", ""))
    if sample.get("citations"):
        st.dataframe(pd.DataFrame(sample["citations"]))


def safe_run_rag(question: str, **kwargs):
    try:
        return get_langgraph_rag().run(question, **kwargs)
    except Exception as exc:
        st.error("The RAG workflow could not be executed in this deployment.")
        st.caption(str(exc))
        return None


def lora_runtime_status() -> dict:
    adapter = PROJECT_ROOT / "adapters" / "lora_findoc"
    eval_path = REPORTS / "lora_eval_results.json"
    metrics_path = REPORTS / "lora_training_metrics.json"
    return {
        "adapter_exists": adapter.exists(),
        "adapter_path": str(adapter),
        "eval_results_exists": eval_path.exists(),
        "training_metrics_exists": metrics_path.exists(),
        "status": "available" if adapter.exists() else "not_trained",
        "message": (
            "Optional LoRA fine-tuning artifacts are available."
            if adapter.exists()
            else "LoRA is optional and has not been trained in this deployment."
        ),
    }


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is required to run the app UI.")

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
            LORA_PAGE_LABEL,
            "System health/settings",
        ],
    )

    chunks = load_chunks()
    provider = get_llm()
    artifact_status = project_artifact_status()
    runtime_ready = rag_runtime_available(artifact_status)

    if page == "Chat over filings":
        question = st.text_area("Question", "What revenue risks are discussed in the filings?")
        col1, col2, col3 = st.columns(3)
        top_k = col1.number_input("Top K", min_value=1, max_value=20, value=5)
        use_reranking = col2.checkbox("Reranking", value=True)
        use_multi_query = col3.checkbox("Multi-query", value=True)
        provider_name = st.selectbox("LLM provider", ["extractive", "groq", "huggingface", "lora"])
        if not runtime_ready:
            show_missing_index_warning(artifact_status)
            maybe_show_demo_output()
        if st.button("Ask", disabled=not runtime_ready):
            result = safe_run_rag(
                question,
                top_k=top_k,
                use_reranking=use_reranking,
                use_multi_query=use_multi_query,
                llm_provider=provider_name,
                debug=True,
            )
            if result:
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
        if not runtime_ready:
            show_missing_index_warning(artifact_status)
            maybe_show_demo_output()
        if st.button("Compare", disabled=not runtime_ready):
            question = f"Compare {companies} on {topic}"
            result = safe_run_rag(question, llm_provider="extractive", debug=True)
            if result:
                st.json(result)

    elif page == "Temporal trend analysis":
        company = st.text_input("Company", "TSLA")
        topic = st.text_input("Topic", "revenue")
        if not runtime_ready:
            show_missing_index_warning(artifact_status)
            maybe_show_demo_output()
        if st.button("Analyze trend", disabled=not runtime_ready):
            question = f"How has {company} changed over time regarding {topic}?"
            result = safe_run_rag(question, llm_provider="extractive", debug=True)
            if result:
                st.json(result)

    elif page == "Evaluation dashboard":
        latest = read_json(REPORTS / "evaluation_latest.json")
        if latest:
            st.json(latest)
            st.markdown((REPORTS / "evaluation_latest.md").read_text(encoding="utf-8") if (REPORTS / "evaluation_latest.md").exists() else "")
        else:
            st.info("No evaluation report exists. Run `python scripts/evaluate.py --eval-set demo --retriever hybrid_rerank --llm extractive --output reports/evaluation_latest.json`.")

    elif page == LORA_PAGE_LABEL:
        lora = read_json(REPORTS / "lora_eval_results.json")
        metrics = read_json(REPORTS / "lora_training_metrics.json")
        lora_status = lora_runtime_status()

        st.subheader("Optional LoRA experiment")
        st.caption("The main SEC RAG workflow does not require LoRA fine-tuning.")
        st.info(
            "This deployment uses deterministic extractive answering by default. "
            "LoRA training is optional and is not required for retrieval, citations, grounded answering, or evaluation."
        )
        st.metric("LoRA experiment status", lora_status["status"])

        if lora_status["adapter_exists"] and lora:
            st.success("Optional LoRA artifacts are present for local experimentation.")
            st.json(lora)
        else:
            st.warning("No LoRA adapter is configured for this deployment.")

        with st.expander("Technical status"):
            st.json({
                **lora_status,
                "eval_results": lora,
                "training_metrics": metrics,
            })

    elif page == "System health/settings":
        if not runtime_ready:
            show_missing_index_warning(artifact_status)
        st.json({
            "provider_health": provider.health_check(),
            "dense_index_status": status_badge(artifact_status["dense_ready"]),
            "bm25_index_status": status_badge(artifact_status["bm25_ready"]),
            "chunks": len(chunks),
            "chunks_status": status_badge(artifact_status["chunks_ready"]),
            "filings": status_badge(artifact_status["filings_ready"]),
            "latest_evaluation": status_badge((REPORTS / "evaluation_latest.json").exists()),
            "lora_adapter": status_badge((PROJECT_ROOT / "adapters" / "lora_findoc").exists()),
            "lora_optional": True,
            "rag_runtime_ready": runtime_ready,
            "missing_index_message": "" if runtime_ready else MISSING_INDEX_MESSAGE,
        })


if __name__ == "__main__":
    main()
