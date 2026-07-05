"""FastAPI backend for financial document intelligence."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from config.settings import PROJECT_ROOT
from src.agents.langgraph_rag import get_langgraph_rag
from src.data.chunking import load_chunks
from src.indexing.build_indexes import build_indexes
from src.llm.factory import get_llm


DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
JOBS_DIR = DATA_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    use_reranking: bool = True
    use_multi_query: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    llm_provider: Optional[str] = None
    debug: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    confidence: Dict[str, Any]
    refusal: bool
    retrieved_contexts: List[Dict[str, Any]]
    trace: Optional[List[Dict[str, Any]]] = None
    latency: float
    provider_used: str


class IngestSecRequest(BaseModel):
    tickers: List[str]
    forms: List[str] = ["10-K", "10-Q", "8-K"]
    start_year: int
    end_year: int
    limit_per_company: Optional[int] = 20


class IndexRequest(BaseModel):
    rebuild: bool = False
    skip_dense: bool = False


class CompareRequest(QueryRequest):
    companies: List[str] = Field(default_factory=list)
    topic: str = "risk factors"


class TemporalRequest(QueryRequest):
    company: str = ""
    topic: str = "revenue"


class EvaluateRequest(BaseModel):
    eval_set: str = "demo"
    retriever: str = "hybrid_rerank"
    llm: str = "extractive"


app = FastAPI(title="Financial Document Intelligence API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, **updates: Any) -> None:
    path = _job_path(job_id)
    current = {}
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(updates)
    path.write_text(json.dumps(current, indent=2, default=str), encoding="utf-8")


def _dataset_stats() -> Dict[str, Any]:
    chunks = load_chunks()
    manifest_path = DATA_DIR / "processed" / "filing_manifest.json"
    csv_manifest = DATA_DIR / "processed" / "filing_manifest.csv"
    filings = []
    if csv_manifest.exists():
        import pandas as pd
        filings = pd.read_csv(csv_manifest).fillna("").to_dict(orient="records")
    companies = {c.get("company") for c in chunks if c.get("company")}
    return {
        "chunks": len(chunks),
        "filings": len(filings),
        "companies": len(companies),
        "demo_mode": len(chunks) == 0,
        "manifest_path": str(csv_manifest if csv_manifest.exists() else manifest_path),
    }


@app.get("/")
def root() -> Dict[str, str]:
    return {"name": "Financial Document Intelligence API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health() -> Dict[str, Any]:
    provider = get_llm()
    return {"status": "healthy", "provider": provider.health_check(), "dataset": _dataset_stats()}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    bm25_path = DATA_DIR / "indexes" / "bm25" / "bm25_index.pkl"
    chunks_path = DATA_DIR / "processed" / "chunks.parquet"
    return {"ready": bm25_path.exists() and chunks_path.exists(), "bm25_index": bm25_path.exists(), "chunks": chunks_path.exists()}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    stats = _dataset_stats()
    lines = [
        f"findoc_chunks {stats['chunks']}",
        f"findoc_filings {stats['filings']}",
        f"findoc_companies {stats['companies']}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/dataset/stats")
def dataset_stats() -> Dict[str, Any]:
    return _dataset_stats()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    started = time.time()
    try:
        result = get_langgraph_rag().run(
            question=request.question,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            use_multi_query=request.use_multi_query,
            filters=request.filters,
            llm_provider=request.llm_provider,
            debug=request.debug,
        )
        contexts = result.get("retrieved_contexts", [])[: request.top_k]
        return QueryResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            confidence=result.get("confidence", {}),
            refusal=bool(result.get("refusal", False)),
            retrieved_contexts=contexts,
            trace=result.get("trace") if request.debug else None,
            latency=result.get("latency", round(time.time() - started, 4)),
            provider_used=result.get("provider_used", request.llm_provider or "extractive"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def compare(request: CompareRequest) -> QueryResponse:
    companies = ", ".join(request.companies)
    request.question = f"Compare {companies} on {request.topic}. {request.question}"
    request.filters = request.filters or {}
    return query(request)


@app.post("/temporal")
def temporal(request: TemporalRequest) -> QueryResponse:
    request.question = f"How has {request.company} changed over time regarding {request.topic}? {request.question}"
    return query(request)


def _run_ingest(job_id: str, payload: IngestSecRequest) -> None:
    _write_job(job_id, status="running", started_at=time.time())
    try:
        from src.data.sec_edgar_ingestion import SecEdgarIngestor
        df = SecEdgarIngestor().ingest(payload.tickers, payload.forms, payload.start_year, payload.end_year, payload.limit_per_company)
        _write_job(job_id, status="completed", rows=len(df), completed_at=time.time())
    except Exception as exc:
        _write_job(job_id, status="failed", error=str(exc), completed_at=time.time())


@app.post("/ingest/sec")
def ingest_sec(request: IngestSecRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _write_job(job_id, status="queued", type="ingest_sec", request=request.model_dump())
    background_tasks.add_task(_run_ingest, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    path = _job_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_index(job_id: str, request: IndexRequest) -> None:
    _write_job(job_id, status="running", started_at=time.time())
    try:
        result = build_indexes(rebuild=request.rebuild, skip_dense=request.skip_dense)
        _write_job(job_id, status="completed", result=result, completed_at=time.time())
    except Exception as exc:
        _write_job(job_id, status="failed", error=str(exc), completed_at=time.time())


@app.post("/index/rebuild")
def index_rebuild(request: IndexRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _write_job(job_id, status="queued", type="index_rebuild", request=request.model_dump())
    background_tasks.add_task(_run_index, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.post("/evaluate")
def evaluate(request: EvaluateRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    _write_job(job_id, status="queued", type="evaluate", request=request.model_dump())
    def run_eval() -> None:
        _write_job(job_id, status="running", started_at=time.time())
        try:
            from scripts.evaluate import run_evaluation_cli
            output = REPORTS_DIR / "evaluation_latest.json"
            result = run_evaluation_cli(request.eval_set, request.retriever, request.llm, output)
            _write_job(job_id, status="completed", result=result, completed_at=time.time())
        except Exception as exc:
            _write_job(job_id, status="failed", error=str(exc), completed_at=time.time())
    background_tasks.add_task(run_eval)
    return {"job_id": job_id, "status": "queued"}


@app.get("/reports/latest")
def reports_latest() -> Dict[str, Any]:
    paths = {
        "dataset_card": REPORTS_DIR / "dataset_card.md",
        "evaluation": REPORTS_DIR / "evaluation_latest.json",
        "lora": REPORTS_DIR / "lora_eval_results.json",
        "model_comparison": REPORTS_DIR / "model_comparison.csv",
    }
    return {name: {"exists": path.exists(), "path": str(path)} for name, path in paths.items()}


@app.get("/documents/search")
def documents_search(q: str = "", ticker: str = "", form_type: str = "", top_k: int = 20) -> Dict[str, Any]:
    chunks = load_chunks()
    terms = q.lower().split()
    results = []
    for chunk in chunks:
        if ticker and chunk.get("ticker", "").upper() != ticker.upper():
            continue
        if form_type and (chunk.get("form_type") or chunk.get("filing_type", "")).upper() != form_type.upper():
            continue
        text = chunk.get("content", "").lower()
        if terms and not all(term in text for term in terms):
            continue
        results.append(chunk)
        if len(results) >= top_k:
            break
    return {"results": results, "count": len(results)}


@app.get("/companies")
def companies() -> Dict[str, Any]:
    chunks = load_chunks()
    names = sorted({c.get("company") for c in chunks if c.get("company")})
    tickers = sorted({c.get("ticker") for c in chunks if c.get("ticker")})
    return {"companies": names, "tickers": tickers, "total": len(names), "demo_mode": len(chunks) == 0}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return _dataset_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
