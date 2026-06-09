"""Ollama local LLM provider."""

import base64
from typing import Any

import cv2
import requests


class OllamaProvider:
    """Ollama HTTP API backend for text and vision."""

    def __init__(self, llm_config: Any) -> None:
        base_url = getattr(llm_config, "base_url", "http://localhost:11434")
        self.endpoint = f"{base_url.rstrip('/')}/api/generate"
        self.model_name = llm_config.model_name
        self.temperature = getattr(llm_config, "temperature", 0.05)

    def _generate(self, prompt: str, image_bgr: Any | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if image_bgr is not None:
            _, buffer = cv2.imencode(".png", image_bgr)
            payload["images"] = [base64.b64encode(buffer).decode("utf-8")]

        response = requests.post(self.endpoint, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_text(self, prompt: str) -> str:
        return self._generate(prompt)

    def generate_vision(self, prompt: str, image_bgr: Any) -> str:
        return self._generate(prompt, image_bgr=image_bgr)
