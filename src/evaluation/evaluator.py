"""Evaluation harness for SEC retrieval and grounded answering."""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

from config.settings import PROJECT_ROOT
from src.answering.grounded_answer import GroundedAnswerer, get_grounded_answerer

DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "evaluation" / "sec_eval_questions.jsonl"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "reports" / "evaluation_results.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "evaluation_summary.md"

REQUIRED_CASE_FIELDS = [
    "id",
    "question",
    "expected_keywords",
    "answerable",
    "notes",
]
REQUIRED_METRICS = [
    "retrieval_result_count",
    "top_k_ticker_match",
    "expected_section_match",
    "expected_form_type_match",
    "keyword_hit_rate",
    "citation_coverage",
    "source_url_coverage",
    "answer_non_empty",
    "answer_has_citations",
    "weak_evidence_rate",
    "no_answer_handling",
    "latency_ms",
]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _bool_score(value: bool) -> float:
    return 1.0 if value else 0.0


def _contains_any(text: str, keywords: Iterable[str]) -> List[str]:
    normalized = _normalize_text(text)
    hits = []
    for keyword in keywords:
        if _normalize_text(str(keyword)) in normalized:
            hits.append(str(keyword))
    return hits


def _honest_no_answer(result: Dict[str, Any]) -> bool:
    answer = _normalize_text(result.get("answer", ""))
    grounding_status = str(result.get("grounding_status", "")).lower()
    warnings = " ".join(str(w) for w in result.get("warnings", []))
    warning_text = _normalize_text(warnings)
    return any(
        marker in answer or marker in warning_text or grounding_status in {"weak_evidence", "no_evidence"}
        for marker in [
            "do not have enough grounded evidence",
            "no relevant filing excerpts",
            "evidence strength is low",
            "below the answerable threshold",
            "tentative",
        ]
    )


def load_evaluation_questions(path: Path | str = DEFAULT_EVAL_PATH) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        missing = [field for field in REQUIRED_CASE_FIELDS if field not in item]
        if missing:
            raise ValueError(f"Evaluation question missing fields {missing}: {item}")
        item.setdefault("filters", {})
        item.setdefault("expected_ticker", "")
        item.setdefault("expected_section", "")
        item.setdefault("expected_form_type", "")
        questions.append(item)
    return questions


def evaluate_single_question(
    question_item: Dict[str, Any],
    answerer: GroundedAnswerer,
    top_k: int = 5,
    provider_name: str = "extractive",
) -> Dict[str, Any]:
    result = answerer.answer_question(
        question_item["question"],
        top_k=top_k,
        filters=question_item.get("filters") or None,
        provider_name=provider_name,
    )
    retrieval_results = list(result.get("retrieval_results", []))
    citations = list(result.get("citations", []))
    answer = result.get("answer", "")
    expected_keywords = [str(keyword) for keyword in question_item.get("expected_keywords", [])]
    keyword_hits = _contains_any(answer, expected_keywords)
    ticker = str(question_item.get("expected_ticker", "")).strip()
    section = str(question_item.get("expected_section", "")).strip()
    form_type = str(question_item.get("expected_form_type", "")).strip()
    cited_source_nums = {
        int(match)
        for match in re.findall(r"\[Source\s+(\d+)\]", answer)
        if str(match).isdigit()
    }
    honest_no_answer = _honest_no_answer(result)

    metrics = {
        "retrieval_result_count": len(retrieval_results),
        "top_k_ticker_match": (
            _bool_score(any(str(item.get("ticker", "")) == ticker for item in retrieval_results[:top_k]))
            if ticker else None
        ),
        "expected_section_match": (
            _bool_score(any(str(item.get("section", "")) == section for item in retrieval_results[:top_k]))
            if section else None
        ),
        "expected_form_type_match": (
            _bool_score(any(str(item.get("form_type", "")) == form_type for item in retrieval_results[:top_k]))
            if form_type else None
        ),
        "keyword_hit_rate": (
            len(keyword_hits) / max(len(expected_keywords), 1)
            if expected_keywords else 0.0
        ),
        "citation_coverage": (
            len(cited_source_nums & {citation.get("source_num") for citation in citations if citation.get("source_num") is not None})
            / max(len(citations), 1)
            if citations else 0.0
        ),
        "source_url_coverage": (
            sum(1 for citation in citations if citation.get("source_url")) / max(len(citations), 1)
            if citations else 0.0
        ),
        "answer_non_empty": _bool_score(bool(answer.strip())),
        "answer_has_citations": _bool_score("[Source" in answer),
        "weak_evidence_rate": _bool_score(str(result.get("grounding_status", "")).lower() in {"weak_evidence", "no_evidence"}),
        "no_answer_handling": (
            _bool_score(honest_no_answer) if not bool(question_item.get("answerable", True)) else None
        ),
        "latency_ms": float(result.get("latency_ms", 0.0)),
    }

    return {
        "id": question_item["id"],
        "question": question_item["question"],
        "filters": question_item.get("filters", {}),
        "expected_ticker": ticker,
        "expected_section": section,
        "expected_form_type": form_type,
        "expected_keywords": expected_keywords,
        "answerable": bool(question_item.get("answerable", True)),
        "notes": question_item.get("notes", ""),
        "answer": answer,
        "grounding_status": result.get("grounding_status"),
        "used_provider": result.get("used_provider"),
        "warnings": result.get("warnings", []),
        "metrics": metrics,
        "keyword_hits": keyword_hits,
        "citations": [
            {
                "source_num": citation.get("source_num"),
                "ticker": citation.get("ticker"),
                "company": citation.get("company"),
                "form_type": citation.get("form_type"),
                "filing_date": citation.get("filing_date"),
                "section": citation.get("section"),
                "accession_number": citation.get("accession_number"),
                "source_url": citation.get("source_url"),
            }
            for citation in citations
        ],
        "retrieval_doc_ids": [item.get("doc_id") for item in retrieval_results[:top_k]],
    }


def summarize_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        raise ValueError("No evaluation cases were provided.")

    def _avg_metric(name: str) -> float:
        values = [case["metrics"][name] for case in cases if case["metrics"].get(name) is not None]
        return mean(values) if values else 0.0

    latencies = [float(case["metrics"]["latency_ms"]) for case in cases]
    answerable_cases = [case for case in cases if case["answerable"]]
    no_answer_cases = [case for case in cases if not case["answerable"]]

    summary = {
        "question_count": len(cases),
        "answerable_count": len(answerable_cases),
        "no_answer_count": len(no_answer_cases),
        "metrics": {
            "retrieval_result_count_avg": _avg_metric("retrieval_result_count"),
            "top_k_ticker_match": _avg_metric("top_k_ticker_match"),
            "expected_section_match": _avg_metric("expected_section_match"),
            "expected_form_type_match": _avg_metric("expected_form_type_match"),
            "keyword_hit_rate": _avg_metric("keyword_hit_rate"),
            "citation_coverage": _avg_metric("citation_coverage"),
            "source_url_coverage": _avg_metric("source_url_coverage"),
            "answer_non_empty": _avg_metric("answer_non_empty"),
            "answer_has_citations": _avg_metric("answer_has_citations"),
            "weak_evidence_rate": _avg_metric("weak_evidence_rate"),
            "no_answer_handling": _avg_metric("no_answer_handling"),
            "latency_ms_avg": mean(latencies) if latencies else 0.0,
            "latency_ms_p50": median(latencies) if latencies else 0.0,
            "latency_ms_max": max(latencies) if latencies else 0.0,
        },
    }
    return summary


def write_evaluation_reports(
    evaluation: Dict[str, Any],
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
) -> None:
    results_path = Path(results_path)
    summary_path = Path(summary_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = evaluation["summary"]
    metrics = summary["metrics"]
    lines = [
        "# SEC Evaluation Summary",
        "",
        f"- Questions: `{summary['question_count']}`",
        f"- Answerable questions: `{summary['answerable_count']}`",
        f"- No-answer / weak-evidence questions: `{summary['no_answer_count']}`",
        "",
        "## Headline Metrics",
        "",
        f"- Avg retrieval result count: `{metrics['retrieval_result_count_avg']:.2f}`",
        f"- Top-k ticker match: `{metrics['top_k_ticker_match']:.3f}`",
        f"- Expected section match: `{metrics['expected_section_match']:.3f}`",
        f"- Expected form-type match: `{metrics['expected_form_type_match']:.3f}`",
        f"- Keyword hit rate: `{metrics['keyword_hit_rate']:.3f}`",
        f"- Citation coverage: `{metrics['citation_coverage']:.3f}`",
        f"- Source URL coverage: `{metrics['source_url_coverage']:.3f}`",
        f"- Answer non-empty rate: `{metrics['answer_non_empty']:.3f}`",
        f"- Answer citation rate: `{metrics['answer_has_citations']:.3f}`",
        f"- Weak-evidence rate: `{metrics['weak_evidence_rate']:.3f}`",
        f"- Honest no-answer handling: `{metrics['no_answer_handling']:.3f}`",
        f"- Avg latency (ms): `{metrics['latency_ms_avg']:.2f}`",
        f"- P50 latency (ms): `{metrics['latency_ms_p50']:.2f}`",
        f"- Max latency (ms): `{metrics['latency_ms_max']:.2f}`",
        "",
        "## Case Snapshot",
        "",
    ]
    for case in evaluation["cases"][:8]:
        lines.append(f"- `{case['id']}` | `{case['grounding_status']}` | keyword_hit_rate=`{case['metrics']['keyword_hit_rate']:.3f}` | citations=`{len(case['citations'])}`")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    dataset_path: Path | str = DEFAULT_EVAL_PATH,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    summary_path: Path | str = DEFAULT_SUMMARY_PATH,
    top_k: int = 5,
    provider_name: str = "extractive",
    answerer: Optional[GroundedAnswerer] = None,
) -> Dict[str, Any]:
    questions = load_evaluation_questions(dataset_path)
    answerer = answerer or get_grounded_answerer()
    cases = [
        evaluate_single_question(
            question_item=question,
            answerer=answerer,
            top_k=top_k,
            provider_name=provider_name,
        )
        for question in questions
    ]
    evaluation = {
        "dataset_path": str(Path(dataset_path)),
        "provider_name": provider_name,
        "top_k": top_k,
        "required_metrics": REQUIRED_METRICS,
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    write_evaluation_reports(evaluation, results_path=results_path, summary_path=summary_path)
    return evaluation
