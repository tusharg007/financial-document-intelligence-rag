"""Local PEFT LoRA provider."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List

from src.llm.base import BaseLLM


class LoRAProvider(BaseLLM):
    name = "lora"

    def __init__(self, base_model: str | None = None, adapter_path: str | None = None):
        self.base_model = base_model or os.getenv("LORA_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.adapter_path = adapter_path or os.getenv("FINETUNED_ADAPTER_PATH", "adapters/lora_findoc")
        self._pipeline = None

    def health_check(self) -> Dict[str, object]:
        exists = Path(self.adapter_path).exists()
        return {
            "provider": self.name,
            "ok": exists,
            "model": self.base_model,
            "adapter_path": self.adapter_path,
            "message": "LoRA adapter available" if exists else "LoRA not trained yet; adapter path missing.",
        }

    @property
    def pipeline(self):
        if not Path(self.adapter_path).exists():
            raise RuntimeError("LoRA adapter missing. Train it first or set FINETUNED_ADAPTER_PATH.")
        if self._pipeline is None:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModelForCausalLM.from_pretrained(self.base_model)
            model = PeftModel.from_pretrained(model, self.adapter_path)
            self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._pipeline

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, object]:
        started = time.time()
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        out = self.pipeline(prompt, max_new_tokens=max_tokens, temperature=temperature)
        return {"text": out[0]["generated_text"][len(prompt):].strip(), "provider": self.name, "model": self.base_model, "latency": time.time() - started}
