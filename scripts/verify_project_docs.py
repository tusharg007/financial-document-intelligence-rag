"""Verify GitHub-facing documentation artifacts for project polish."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README_PATH = ROOT / "README.md"
DOC_PATHS = {
    "architecture": ROOT / "docs" / "architecture.md",
    "quickstart": ROOT / "docs" / "quickstart.md",
    "demo_walkthrough": ROOT / "docs" / "demo_walkthrough.md",
    "metrics": ROOT / "docs" / "metrics.md",
    "known_limitations": ROOT / "docs" / "known_limitations.md",
    "reproducibility": ROOT / "docs" / "reproducibility.md",
}

BANNED_TERMS = [
    "resume",
    "recruiter",
    "shortlisting",
    "ats",
    "interview prep",
    "interview preparation",
]

REQUIRED_README_SNIPPETS = [
    "Financial Document Intelligence RAG",
    "https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/",
    "keyword_hit_rate",
    "citation_coverage",
    "source_url_coverage",
    "no_answer_handling",
    "python -m venv .venv",
    "pip install -r requirements.txt",
    "python scripts/run_evaluation.py",
    "python scripts/verify_evaluation.py",
    "python -m pytest -q --basetemp=.pytest-tmp-readme-final",
]


def main() -> None:
    failures: list[str] = []

    if not README_PATH.exists():
        failures.append(f"Missing README: {README_PATH}")
        readme_text = ""
    else:
        readme_text = README_PATH.read_text(encoding="utf-8")

    for name, path in DOC_PATHS.items():
        if not path.exists():
            failures.append(f"Missing docs file '{name}': {path}")

    for snippet in REQUIRED_README_SNIPPETS:
        if snippet not in readme_text:
            failures.append(f"README is missing required content: {snippet}")

    if "Live demo" not in readme_text and "Deployment Notes" not in readme_text and "Live Demo" not in readme_text:
        failures.append("README is missing a visible live-demo or deployment-notes section.")

    required_exclusion_phrases = [
        "intentionally excluded",
        "Chroma dense index",
        "BM25 sparse index",
    ]
    for phrase in required_exclusion_phrases:
        if phrase not in readme_text:
            failures.append(f"README is missing deployment/exclusion note: {phrase}")

    lowered = readme_text.lower()
    banned_hits = []
    for term in BANNED_TERMS:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, lowered):
            banned_hits.append(term)
    if banned_hits:
        failures.append(f"README contains banned recruiting-oriented terms: {banned_hits}")

    output = {
        "readme_exists": README_PATH.exists(),
        "docs_exist": {name: path.exists() for name, path in DOC_PATHS.items()},
        "readme_mentions_final_metrics": all(metric in readme_text for metric in [
            "keyword_hit_rate",
            "citation_coverage",
            "source_url_coverage",
            "no_answer_handling",
        ]),
        "readme_mentions_setup_commands": "python -m venv .venv" in readme_text and "pip install -r requirements.txt" in readme_text,
        "readme_mentions_evaluation_commands": "python scripts/run_evaluation.py" in readme_text and "python scripts/verify_evaluation.py" in readme_text,
        "readme_mentions_live_link": "https://financial-document-intelligence-rag-heq3wynx8iusxe5caw32fr.streamlit.app/" in readme_text,
        "readme_mentions_demo_scope": ("Live demo" in readme_text or "Live Demo" in readme_text or "Deployment Notes" in readme_text),
        "readme_mentions_excluded_indexes": "intentionally excluded" in readme_text and "Chroma dense index" in readme_text and "BM25 sparse index" in readme_text,
        "readme_banned_terms_found": banned_hits,
        "failures": failures,
    }
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit("Project docs verification failed.")


if __name__ == "__main__":
    main()
