"""HuggingFace Inference provider."""
from __future__ import annotations

import os
import time
from typing import Dict, List

from src.llm.base import BaseLLM


class HuggingFaceProvider(BaseLLM):
    name = "huggingface"

    def __init__(self, token: str | None = None, model: str | None = None):
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        self.model = model or os.getenv("HF_GENERATION_MODEL", "HuggingFaceH4/zephyr-7b-beta")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.token:
                raise RuntimeError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN is not configured.")
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(model=self.model, token=self.token)
        return self._client

    def health_check(self) -> Dict[str, object]:
        return {"provider": self.name, "ok": bool(self.token), "model": self.model, "message": "HF token configured" if self.token else "HF token missing"}

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, object]:
        started = time.time()
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        last_error = None
        for attempt in range(3):
            try:
                text = self.client.text_generation(prompt, max_new_tokens=max_tokens, temperature=temperature, return_full_text=False)
                return {"text": text.strip(), "provider": self.name, "model": self.model, "latency": time.time() - started}
            except Exception as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HuggingFace generation failed: {last_error}")
