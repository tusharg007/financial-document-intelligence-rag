"""Provider selection and explicit extractive fallback."""
from __future__ import annotations

import os
from typing import Dict, List

from src.llm.base import BaseLLM
from src.llm.groq_provider import GroqProvider
from src.llm.huggingface_provider import HuggingFaceProvider
from src.llm.lora_provider import LoRAProvider


class ExtractiveProvider(BaseLLM):
    name = "extractive"

    def health_check(self) -> Dict[str, object]:
        return {"provider": self.name, "ok": True, "message": "Deterministic extractive fallback; no generative model active."}

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, object]:
        context = messages[-1].get("content", "") if messages else ""
        text = "Fallback mode used: no configured LLM provider was available.\n\n"
        lines = [line.strip() for line in context.splitlines() if line.strip()]
        evidence = [line for line in lines if line.startswith("[Source")]
        text += "\n".join(evidence[:5]) if evidence else "\n".join(lines[:8])
        return {"text": text[: max_tokens * 4], "provider": self.name, "model": "extractive"}


def provider_for(name: str) -> BaseLLM:
    name = (name or "extractive").lower()
    if name == "groq":
        return GroqProvider()
    if name == "huggingface":
        return HuggingFaceProvider()
    if name == "lora":
        return LoRAProvider()
    return ExtractiveProvider()


def get_llm(provider: str | None = None) -> BaseLLM:
    requested = provider or os.getenv("LLM_PROVIDER", "extractive")
    if requested != "auto":
        return provider_for(requested)
    for name in os.getenv("LLM_FALLBACK_ORDER", "groq,huggingface,lora,extractive").split(","):
        candidate = provider_for(name.strip())
        if candidate.health_check().get("ok"):
            return candidate
    return ExtractiveProvider()
