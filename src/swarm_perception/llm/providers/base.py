"""LLM provider interface for swarm perception and fusion."""

from typing import Any, Protocol


class LLMProvider(Protocol):
    """Minimal contract for text and vision generation backends."""

    def generate_text(self, prompt: str) -> str:
        """Run a text-only completion."""
        ...

    def generate_vision(self, prompt: str, image_bgr: Any) -> str:
        """Run a multimodal completion with an OpenCV BGR image."""
        ...
