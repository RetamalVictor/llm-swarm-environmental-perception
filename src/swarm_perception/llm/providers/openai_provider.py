"""OpenAI LLM provider."""

import base64
import os
from typing import Any

import cv2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIProvider:
    """OpenAI chat-completions backend for text and vision."""

    def __init__(self, llm_config: Any) -> None:
        api_key_env = getattr(llm_config, "api_key_env", "OPENAI_API_KEY")
        base_url = getattr(llm_config, "base_url", None)
        client_kwargs: dict[str, Any] = {"api_key": os.environ[api_key_env]}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.model_name = llm_config.model_name
        self.temperature = getattr(llm_config, "temperature", 0.05)
        self.max_output_tokens = getattr(llm_config, "max_output_tokens", 220)

    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content")
        return content.strip()

    def generate_vision(self, prompt: str, image_bgr: Any) -> str:
        _, buffer = cv2.imencode(".png", image_bgr)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
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
            ],
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("OpenAI returned empty content")
        return content.strip()
