"""vLLM OpenAI-compatible async HTTP provider."""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import cv2
import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class VllmProvider:
    """Async client for vLLM's OpenAI-compatible ``/v1/chat/completions`` API."""

    def __init__(self, llm_config: Any) -> None:
        base_url = getattr(llm_config, "base_url", "http://localhost:8080/v1")
        if not str(base_url).startswith("http"):
            base_url = f"http://{base_url}"
        self.endpoint = f"{str(base_url).rstrip('/')}/chat/completions"
        self.model_name = llm_config.model_name
        self.temperature = float(getattr(llm_config, "temperature", 0.05))
        self.max_output_tokens = int(getattr(llm_config, "max_output_tokens", 220))
        api_key_env = getattr(llm_config, "api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(api_key_env, "EMPTY")
        self.request_timeout = float(getattr(llm_config, "request_timeout_seconds", 600))
        max_concurrent = int(getattr(llm_config, "thread_workers", 10))
        self._max_connections = int(getattr(llm_config, "max_connections", 0)) or max_concurrent
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Open a shared async HTTP client with a connection pool sized for parallelism."""
        if self._client is not None:
            return
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_connections,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout),
            limits=limits,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "vllm client started │ endpoint=%s │ max_connections=%s",
            self.endpoint,
            self._max_connections,
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _chat(self, messages: list[dict[str, Any]]) -> str:
        if self._client is None:
            raise RuntimeError("VllmProvider.start() was not called")

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        response = await self._client.post(self.endpoint, json=payload)
        if response.status_code >= 400:
            body = response.text.strip()[:500]
            raise httpx.HTTPStatusError(
                f"{response.status_code} {response.reason_phrase} for {self.endpoint}"
                + (f": {body}" if body else ""),
                request=response.request,
                response=response,
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"vLLM returned no choices: {data}")
        content = choices[0].get("message", {}).get("content")
        if not content:
            raise ValueError("vLLM returned empty content")
        return str(content).strip()

    async def generate_text(self, prompt: str) -> str:
        """Run a text-only chat completion."""
        return await self._chat([{"role": "user", "content": prompt}])

    async def generate_vision(self, prompt: str, image_bgr: Any) -> str:
        """Run a vision chat completion with a base64-encoded PNG crop."""
        _, buffer = cv2.imencode(".png", image_bgr)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]
        return await self._chat(messages)
