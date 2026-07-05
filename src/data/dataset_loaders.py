"""Dataset loaders for local and HuggingFace financial QA/sentiment datasets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from config.settings import PROJECT_ROOT


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset is not locally available and cannot be fetched."""


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_hf_dataset(name: str, *args: Any, **kwargs: Any):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise DatasetUnavailable(f"Install datasets and download {name}, or place files under data/raw/. Error: {exc}") from exc
    try:
        return load_dataset(name, *args, **kwargs)
    except Exception as exc:
        raise DatasetUnavailable(f"Could not load {name}. If offline, download it manually into data/raw. Error: {exc}") from exc


def load_local_jsonlike(folder: str | Path) -> List[Dict[str, Any]]:
    folder = Path(folder)
    if not folder.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in list(folder.glob("*.jsonl")) + list(folder.glob("*.json")):
        if path.suffix == ".jsonl":
            rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(data if isinstance(data, list) else data.get("data", []))
    for path in list(folder.glob("*.csv")):
        rows.extend(pd.read_csv(path).fillna("").to_dict(orient="records"))
    return rows


def load_kaggle_sec() -> List[Dict[str, Any]]:
    rows = load_local_jsonlike(PROJECT_ROOT / "data" / "raw" / "kaggle")
    if not rows:
        raise DatasetUnavailable("Kaggle/Finnhub SEC files not found. Download them to data/raw/kaggle/ and rerun scripts/prepare_datasets.py.")
    return rows


def prepare_financial_phrasebank(output: Path) -> int:
    ds = _load_hf_dataset("financial_phrasebank", "sentences_allagree", trust_remote_code=True)
    rows = [{"text": r["sentence"], "label": r["label"], "source_dataset": "financial_phrasebank"} for r in ds["train"]]
    return _write_jsonl(rows, output)


def prepare_fiqa(output: Path) -> int:
    try:
        ds = _load_hf_dataset("BeIR/fiqa", "queries")
    except DatasetUnavailable:
        ds = _load_hf_dataset("explodinggradients/fiqa")
    split = next(iter(ds.values()))
    rows = []
    for r in split:
        question = r.get("text") or r.get("question") or r.get("query") or ""
        answer = r.get("answer") or r.get("answers") or ""
        if question:
            rows.append({"question": question, "answer": answer, "source_dataset": "fiqa"})
    return _write_jsonl(rows, output)


def prepare_local_or_hf_eval(dataset_name: str, raw_folder: str, output: Path) -> int:
    local = load_local_jsonlike(PROJECT_ROOT / "data" / "raw" / raw_folder)
    if not local:
        raise DatasetUnavailable(f"{dataset_name} not found locally. Place JSON/JSONL/CSV files under data/raw/{raw_folder}/.")
    rows = []
    for r in local:
        rows.append({
            "question": r.get("question") or r.get("query") or r.get("qa", {}).get("question", ""),
            "answer": r.get("answer") or r.get("ground_truth") or r.get("qa", {}).get("answer", ""),
            "evidence": r.get("evidence") or r.get("context") or r.get("paragraphs", ""),
            "source_dataset": dataset_name,
        })
    rows = [r for r in rows if r["question"]]
    return _write_jsonl(rows, output)


def prepare_all(demo_ok: bool = False) -> Dict[str, Any]:
    out = PROJECT_ROOT / "data" / "processed"
    results: Dict[str, Any] = {}
    loaders = {
        "sentiment_phrasebank": lambda: prepare_financial_phrasebank(out / "sentiment_phrasebank.jsonl"),
        "eval_fiqa": lambda: prepare_fiqa(out / "eval_fiqa.jsonl"),
        "eval_financebench": lambda: prepare_local_or_hf_eval("financebench", "financebench", out / "eval_financebench.jsonl"),
        "eval_tatqa": lambda: prepare_local_or_hf_eval("tatqa", "tatqa", out / "eval_tatqa.jsonl"),
        "eval_finqa": lambda: prepare_local_or_hf_eval("finqa", "finqa", out / "eval_finqa.jsonl"),
    }
    for name, fn in loaders.items():
        try:
            results[name] = {"rows": fn(), "status": "ok"}
        except DatasetUnavailable as exc:
            results[name] = {"rows": 0, "status": "unavailable", "message": str(exc)}
    if demo_ok:
        from src.data.sample_data import get_evaluation_pairs
        results["eval_demo"] = {"rows": _write_jsonl(get_evaluation_pairs(), out / "eval_demo.jsonl"), "status": "demo"}
    return results
