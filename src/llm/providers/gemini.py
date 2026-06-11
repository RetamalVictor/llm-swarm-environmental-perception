"""Google Gemini LLM provider."""

from collections import deque
import logging
import os
import threading
import time
from typing import Any

import cv2
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiProvider:
    """Gemini-backed text and vision generation."""

    def __init__(self, llm_config: Any) -> None:
        api_key_env = getattr(llm_config, "api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ[api_key_env]
        genai.configure(api_key=api_key)  # type: ignore

        self.model = genai.GenerativeModel(llm_config.model_name)

        temperature = getattr(llm_config, "temperature", 0.05)
        max_output_tokens = getattr(llm_config, "max_output_tokens", 220)
        self.generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        self.rate_limit = 3900
        self.rate_window = 60
        self.request_timestamps: deque[float] = deque()
        self.rate_lock = threading.Lock()

    def _enforce_rate_limit(self) -> None:
        with self.rate_lock:
            current_time = time.time()
            while (
                self.request_timestamps
                and current_time - self.request_timestamps[0] > self.rate_window
            ):
                self.request_timestamps.popleft()
            if len(self.request_timestamps) >= self.rate_limit:
                sleep_time = self.rate_window - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    while (
                        self.request_timestamps
                        and current_time - self.request_timestamps[0] > self.rate_window
                    ):
                        self.request_timestamps.popleft()
            self.request_timestamps.append(current_time)

    def generate_text(self, prompt: str) -> str:
        self._enforce_rate_limit()
        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config,
        )
        return response.text.strip()

    def generate_vision(self, prompt: str, image_bgr: Any) -> str:
        self._enforce_rate_limit()
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        response = self.model.generate_content(
            [prompt, pil_image],
            generation_config=self.generation_config,
        )
        return response.text.strip()
