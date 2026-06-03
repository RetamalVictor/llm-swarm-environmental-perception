"""Gemini-backed asynchronous LLM manager for robot perception and fusion."""

from collections import deque
import logging
import os
import queue
import threading
import time
from typing import Any

import cv2
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
logger = logging.getLogger(__name__)

class API_MANAGER:
    """Threaded request manager for photo analysis and inbox synthesis.

    The manager exposes a small async-like API used by each robot:

    - submit photo request (`submit_photo_request`)
    - submit inbox merge request (`submit_inbox_request`)
    - poll for completed results (`get_result`)

    Each request type keeps only the latest outstanding task per robot id.
    Older queued requests are dropped by timestamp to keep the simulation
    responsive when API latency spikes.
    """

    def __init__(self, n_threads: int, config: Any) -> None:
        """Initialize the manager and Gemini client.

        Args:
            n_threads: Number of worker threads that consume request queue jobs.
            config: Loaded swarm configuration namespace (contains model settings
                and prompt templates).
        """
        self.config = config

        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])  # type: ignore
            
        self.MODEL_NAME = self.config.llm.model_name
        self.TEXT_MODEL_NAME = self.config.llm.model_name

        self.model = genai.GenerativeModel(self.MODEL_NAME)

        llm_temperature = getattr(self.config.llm, "temperature", 0.05)
        llm_max_output_tokens = getattr(self.config.llm, "max_output_tokens", 220)
        self.generation_config = genai.GenerationConfig(
            temperature=llm_temperature,
            max_output_tokens=llm_max_output_tokens,
        )

        self.n_threads = n_threads
        self.request_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.latest_request_timestamp = {}
        self.latest_request_lock = threading.Lock()

        # Rate limiting: max 3900 requests per minute.
        self.rate_limit = 3900
        self.rate_window = 60
        self.request_timestamps = deque()
        self.rate_lock = threading.Lock()

    def enforce_rate_limit(self) -> None:
        """Throttle calls to keep aggregate worker throughput under rate limits."""
        with self.rate_lock:
            current_time = time.time()
            while self.request_timestamps and current_time - self.request_timestamps[0] > self.rate_window:
                self.request_timestamps.popleft()
            if len(self.request_timestamps) >= self.rate_limit:
                sleep_time = self.rate_window - (current_time - self.request_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    while self.request_timestamps and current_time - self.request_timestamps[0] > self.rate_window:
                        self.request_timestamps.popleft()
            self.request_timestamps.append(current_time)

    def start(self) -> None:
        """Start background worker threads that process queue requests."""
        print(f"Starting API manager with {self.n_threads} threads ...")
        for i in range(self.n_threads):
            worker_thread = threading.Thread(target=self.worker_per_thread)
            worker_thread.daemon = True
            worker_thread.start()

    def worker_per_thread(self) -> None:
        """Continuously process queue items and store completed results.

        The worker handles both request types:
        - ``photo``: image + current observation text
        - ``inbox``: current observation text + peer observation text

        Any exception falls back to the current observation so robots keep moving
        even when an API call fails.
        """
        while True:
            print("## Tasks left: ", self.request_queue.qsize())
            request_type, request_id, data, timestamp, self_learning = self.request_queue.get()
            with self.latest_request_lock:
                if timestamp < self.latest_request_timestamp.get(request_id, timestamp):
                    self.request_queue.task_done()
                    continue

            try:
                self.enforce_rate_limit()
                started_at = time.time()
                if request_type == "photo":
                    image, observation = data
                    result_text = self.call_photo_api(image, observation, self_learning)
                elif request_type == "inbox":
                    current_observation, inbox = data
                    result_text = self.call_inbox_api(current_observation, inbox)
                elapsed_ms = int((time.time() - started_at) * 1000)
                logger.info(
                    "api request completed: id=%s type=%s latency_ms=%s chars=%s",
                    request_id,
                    request_type,
                    elapsed_ms,
                    len(result_text) if result_text else 0,
                )
            except Exception as e:  # if any errors, return existing result
                print(f"ERROR in worker for {request_id}: {e}")
                logger.exception("api request failed: id=%s type=%s", request_id, request_type)
                if request_type == "photo":
                    image, observation = data
                    result_text = observation
                elif request_type == "inbox":
                    current_observation, inbox = data
                    result_text = current_observation

            with self.results_lock:
                self.results[request_id] = (result_text, data)

            self.request_queue.task_done()

    def submit_photo_request(
        self,
        robot_id: int,
        image_data: Any,
        observation: str,
        self_learning: bool,
    ) -> None:
        """Queue a photo-analysis request for a robot.

        Only the latest pending photo request per robot is considered valid.

        Args:
            robot_id: Sender robot identifier.
            image_data: Cropped BGR image from camera sensor.
            observation: Robot's current textual knowledge base.
            self_learning: Whether the prompt should include prior memory.
        """
        request_id = f"{robot_id}_photo"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)
        
        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time
        request_data = (image_data, observation)
        self.request_queue.put(("photo", request_id, request_data, current_time, self_learning))

    def submit_inbox_request(
        self,
        robot_id: int,
        current_observation: str,
        inbox: str,
    ) -> None:
        """Queue a peer-synthesis request for a robot.

        Args:
            robot_id: Receiver robot identifier.
            current_observation: Robot's current textual knowledge base.
            inbox: Incoming peer message to fuse.
        """
        request_id = f"{robot_id}_inbox"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)
        
        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time

        request_data = (current_observation, inbox)
        self.request_queue.put(("inbox", request_id, request_data, current_time, True))

    def get_result(self, robot_id: int, request_type: str) -> tuple[Any, Any]:
        """Poll and remove a completed result for a robot request.

        Args:
            robot_id: Robot identifier.
            request_type: Request category (``photo`` or ``inbox``).

        Returns:
            Tuple of ``(result_text, request_data)`` if ready, otherwise
            ``(None, None)``.
        """
        request_id = f"{robot_id}_{request_type}"
        with self.results_lock:
            return self.results.pop(request_id, (None, None))

    def call_photo_api(self, image: Any, observation: str, self_learning: bool) -> str:
        """Run Gemini multimodal generation for one camera frame.

        Args:
            image: OpenCV BGR image array from ``CameraSensor.take_photo``.
            observation: Existing robot knowledge base text.
            self_learning: Selects memory-aware vs image-only prompt template.

        Returns:
            Updated observation text generated by Gemini.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        if self_learning:
            prompt_template = self.config.llm.prompts.photo_analysis_self_learning
        else:
            prompt_template = self.config.llm.prompts.photo_analysis_no_self_learning

        prompt = prompt_template.format(observation=observation)

        response = self.model.generate_content(
            [prompt, pil_image],
            generation_config=self.generation_config,
        )

        return response.text.strip()

    def call_inbox_api(self, current_observation: str, inbox: str) -> str:
        """Run Gemini text synthesis for peer knowledge integration.

        Args:
            current_observation: Receiver robot's current knowledge base.
            inbox: Peer message selected by inbox policy.

        Returns:
            Synthesized knowledge base string.
        """
        prompt_template = self.config.llm.prompts.text_synthesis
        prompt = prompt_template.format(current_observation=current_observation, inbox=inbox)

        response = self.model.generate_content(
            prompt,
            generation_config=self.generation_config,
        )

        return response.text.strip()

