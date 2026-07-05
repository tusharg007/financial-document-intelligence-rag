"""Base LLM interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List


class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Dict[str, object]:
        """Generate a response and return provider metadata."""

    def stream(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 512) -> Iterable[str]:
        yield str(self.generate(messages, temperature, max_tokens).get("text", ""))

    @abstractmethod
    def health_check(self) -> Dict[str, object]:
        """Return provider readiness."""
