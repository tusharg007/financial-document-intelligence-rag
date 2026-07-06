from __future__ import annotations

import json


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


class FakeAnswerer:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def answer_question(self, query, top_k=5, filters=None, provider_name=None):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "provider_name": provider_name,
        })
        return self.responses[query]


def _grounded_response():
    return {
        "answer": "Based on the retrieved SEC filings:\n- Apple highlights competition and supply chain risks. [Source 1]",
        "citations": [
            {
                "source_num": 1,
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "fiscal_year": 2024,
                "section": "Risk Factors",
                "accession_number": "0000320193-24-000123",
                "source_url": "https://www.sec.gov/example-aapl",
            }
        ],
        "retrieval_results": [
            {
                "doc_id": "aapl-1",
                "ticker": "AAPL",
                "company": "Apple Inc.",
                "form_type": "10-K",
                "filing_date": "2024-11-01",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "section": "Risk Factors",
                "accession_number": "0000320193-24-000123",
                "source_url": "https://www.sec.gov/example-aapl",
            }
        ],
        "grounding_status": "grounded_with_warnings",
        "used_provider": "extractive",
        "warnings": ["Evidence spans multiple sections; check citations for context."],
        "latency_ms": 12.5,
    }


def _no_answer_response():
    return {
        "answer": "I do not have enough grounded evidence in the indexed filings to answer confidently. No relevant filing excerpts were retrieved.",
        "citations": [],
        "retrieval_results": [],
        "grounding_status": "no_evidence",
        "used_provider": "extractive",
        "warnings": ["No retrieved evidence was available."],
        "latency_ms": 8.0,
    }


def test_evaluate_single_question_metrics():
    from src.evaluation.evaluator import evaluate_single_question

    question = {
        "id": "case-1",
        "question": "What are Apple's main risk factors?",
        "filters": {"ticker": "AAPL", "section": "Risk Factors"},
        "expected_ticker": "AAPL",
        "expected_section": "Risk Factors",
        "expected_form_type": "10-K",
        "expected_keywords": ["competition", "supply chain", "regulatory"],
        "answerable": True,
        "notes": "test",
    }
    answerer = FakeAnswerer({question["question"]: _grounded_response()})

    case = evaluate_single_question(question, answerer=answerer, top_k=5, provider_name="extractive")

    assert answerer.calls[0]["filters"] == {"ticker": "AAPL", "section": "Risk Factors"}
    assert case["metrics"]["retrieval_result_count"] == 1
    assert case["metrics"]["top_k_ticker_match"] == 1.0
    assert case["metrics"]["expected_section_match"] == 1.0
    assert case["metrics"]["expected_form_type_match"] == 1.0
    assert case["metrics"]["keyword_hit_rate"] == 2 / 3
    assert case["metrics"]["citation_coverage"] == 1.0
    assert case["metrics"]["source_url_coverage"] == 1.0
    assert case["metrics"]["answer_non_empty"] == 1.0
    assert case["metrics"]["answer_has_citations"] == 1.0


def test_no_answer_handling_metric():
    from src.evaluation.evaluator import evaluate_single_question

    question = {
        "id": "case-2",
        "question": "What did Apple disclose about 2021 risk factors?",
        "filters": {"ticker": "AAPL", "section": "Risk Factors"},
        "expected_ticker": "AAPL",
        "expected_section": "Risk Factors",
        "expected_form_type": "",
        "expected_keywords": ["2021"],
        "answerable": False,
        "notes": "out of range",
    }
    answerer = FakeAnswerer({question["question"]: _no_answer_response()})

    case = evaluate_single_question(question, answerer=answerer, top_k=5, provider_name="extractive")

    assert case["metrics"]["retrieval_result_count"] == 0
    assert case["metrics"]["no_answer_handling"] == 1.0
    assert case["metrics"]["answer_has_citations"] == 0.0
    assert case["metrics"]["source_url_coverage"] == 0.0


def test_summarize_cases_aggregates_keyword_and_source_url_coverage():
    from src.evaluation.evaluator import summarize_cases

    cases = [
        {
            "answerable": True,
            "metrics": {
                "retrieval_result_count": 3,
                "top_k_ticker_match": 1.0,
                "expected_section_match": 1.0,
                "expected_form_type_match": 1.0,
                "keyword_hit_rate": 0.5,
                "citation_coverage": 1.0,
                "source_url_coverage": 1.0,
                "answer_non_empty": 1.0,
                "answer_has_citations": 1.0,
                "weak_evidence_rate": 0.0,
                "no_answer_handling": None,
                "latency_ms": 10.0,
            },
        },
        {
            "answerable": False,
            "metrics": {
                "retrieval_result_count": 0,
                "top_k_ticker_match": 0.0,
                "expected_section_match": 0.0,
                "expected_form_type_match": None,
                "keyword_hit_rate": 0.0,
                "citation_coverage": 0.0,
                "source_url_coverage": 0.0,
                "answer_non_empty": 1.0,
                "answer_has_citations": 0.0,
                "weak_evidence_rate": 1.0,
                "no_answer_handling": 1.0,
                "latency_ms": 20.0,
            },
        },
    ]

    summary = summarize_cases(cases)

    assert summary["question_count"] == 2
    assert summary["no_answer_count"] == 1
    assert summary["metrics"]["keyword_hit_rate"] == 0.25
    assert summary["metrics"]["citation_coverage"] == 0.5
    assert summary["metrics"]["source_url_coverage"] == 0.5
    assert summary["metrics"]["no_answer_handling"] == 1.0


def test_report_writing(tmp_path):
    from src.evaluation.evaluator import write_evaluation_reports

    evaluation = {
        "dataset_path": "data/evaluation/sec_eval_questions.jsonl",
        "provider_name": "extractive",
        "top_k": 5,
        "required_metrics": [],
        "summary": {
            "question_count": 2,
            "answerable_count": 1,
            "no_answer_count": 1,
            "metrics": {
                "retrieval_result_count_avg": 1.5,
                "top_k_ticker_match": 1.0,
                "expected_section_match": 1.0,
                "expected_form_type_match": 1.0,
                "keyword_hit_rate": 0.5,
                "citation_coverage": 1.0,
                "source_url_coverage": 1.0,
                "answer_non_empty": 1.0,
                "answer_has_citations": 1.0,
                "weak_evidence_rate": 0.5,
                "no_answer_handling": 1.0,
                "latency_ms_avg": 15.0,
                "latency_ms_p50": 15.0,
                "latency_ms_max": 20.0,
            },
        },
        "cases": [
            {
                "id": "case-1",
                "grounding_status": "grounded",
                "metrics": {"keyword_hit_rate": 0.5},
                "citations": [{"source_num": 1, "source_url": "https://www.sec.gov/example"}],
            }
        ],
    }
    results_path = tmp_path / "evaluation_results.json"
    summary_path = tmp_path / "evaluation_summary.md"

    write_evaluation_reports(evaluation, results_path=results_path, summary_path=summary_path)

    assert results_path.exists()
    assert summary_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["summary"]["metrics"]["source_url_coverage"] == 1.0
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "SEC Evaluation Summary" in summary_text
    assert "Source URL coverage" in summary_text
