"""Groq OpenAI-compatible provider."""
from __future__ import annotations

import os
import time
from typing import Dict, List

from src.llm.base import BaseLLM


class GroqProvider(BaseLLM):
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.timeout = timeout

    def health_check(self) -> Dict[str, object]:
        return {"provider": self.name, "ok": bool(self.api_key), "model": self.model, "message": "GROQ_API_KEY configured" if self.api_key else "GROQ_API_KEY missing"}

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, object]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error = None
        started = time.time()
        for attempt in range(3):
            try:
                import requests
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "provider": self.name,
                    "model": self.model,
                    "latency": time.time() - started,
                    "usage": data.get("usage", {}),
                }
            except Exception as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Groq generation failed: {last_error}")
