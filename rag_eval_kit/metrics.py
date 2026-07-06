"""Deterministic, lightweight RAG evaluation metrics."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def overlap_ratio(source_tokens: Iterable[str], target_tokens: Iterable[str]) -> float:
    source = set(source_tokens)
    target = set(target_tokens)
    if not source:
        return 0.0
    return len(source & target) / len(source)


def context_relevance(question: str, contexts: Sequence[str]) -> float:
    question_tokens = unique_tokens(question)
    if not question_tokens or not contexts:
        return 0.0
    context_union: set[str] = set()
    for context in contexts:
        context_union.update(unique_tokens(context))
    return round(overlap_ratio(question_tokens, context_union), 4)


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    sentences = split_sentences(answer)
    if not sentences or not contexts:
        return 0.0
    context_token_sets = [unique_tokens(context) for context in contexts]
    sentence_scores: List[float] = []
    for sentence in sentences:
        sentence_tokens = unique_tokens(sentence)
        if not sentence_tokens:
            continue
        best_support = 0.0
        for context_tokens in context_token_sets:
            best_support = max(best_support, overlap_ratio(sentence_tokens, context_tokens))
        sentence_scores.append(best_support)
    if not sentence_scores:
        return 0.0
    return round(sum(sentence_scores) / len(sentence_scores), 4)


def citation_coverage(answer: str, contexts: Sequence[str]) -> float:
    citations = {int(match) for match in re.findall(r"\[Source\s+(\d+)\]", answer or "")}
    if not contexts:
        return 0.0
    if not citations:
        return 0.0
    expected = min(len(contexts), max(1, len(citations)))
    return round(min(1.0, len(citations) / expected), 4)


def answer_completeness(answer: str, expected_keywords: Sequence[str] | None = None) -> float:
    if not answer.strip():
        return 0.0
    if not expected_keywords:
        return 1.0
    answer_tokens = unique_tokens(answer)
    normalized_keywords = [keyword.lower() for keyword in expected_keywords if keyword]
    if not normalized_keywords:
        return 1.0
    hits = 0
    for keyword in normalized_keywords:
        keyword_tokens = set(tokenize(keyword))
        if keyword_tokens and keyword_tokens <= answer_tokens:
            hits += 1
    return round(hits / len(normalized_keywords), 4)
