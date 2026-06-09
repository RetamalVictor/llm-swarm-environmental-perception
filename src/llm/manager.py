"""Provider-agnostic threaded LLM request manager for robot perception and fusion."""

import logging
import queue
import threading
import time
from typing import Any

from llm.providers.base import LLMProvider

logger = logging.getLogger("swarm.llm")


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

    def __init__(self, n_threads: int, config: Any, provider: LLMProvider) -> None:
        """Initialize the manager with a pluggable LLM provider.

        Args:
            n_threads: Number of worker threads that consume request queue jobs.
            config: Loaded swarm configuration namespace (contains model settings
                and prompt templates).
            provider: Backend that performs text and vision generation.
        """
        self.config = config
        self.provider = provider
        self.n_threads = n_threads
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.results: dict[str, tuple[Any, Any]] = {}
        self.results_lock = threading.Lock()
        self.latest_request_timestamp: dict[str, float] = {}
        self.latest_request_lock = threading.Lock()
        self._active_requests = 0
        self._active_lock = threading.Lock()

    def queue_depth(self) -> int:
        """Return the number of requests waiting in the worker queue."""
        return self.request_queue.qsize()

    def active_request_count(self) -> int:
        """Return the number of requests currently being processed."""
        with self._active_lock:
            return self._active_requests

    def _log_queue_state(self, action: str, request_id: str, request_type: str) -> None:
        logger.info(
            "%s │ id=%s type=%s │ queue=%s active=%s ready=%s",
            action,
            request_id,
            request_type,
            self.queue_depth(),
            self.active_request_count(),
            len(self.results),
        )

    def start(self) -> None:
        """Start background worker threads that process queue requests."""
        logger.info("worker pool started │ threads=%s", self.n_threads)
        for _ in range(self.n_threads):
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
            request_type, request_id, data, timestamp, self_learning = self.request_queue.get()
            with self.latest_request_lock:
                if timestamp < self.latest_request_timestamp.get(request_id, timestamp):
                    logger.debug("skipped stale │ id=%s type=%s", request_id, request_type)
                    self.request_queue.task_done()
                    continue

            with self._active_lock:
                self._active_requests += 1
            logger.info("request started │ id=%s type=%s", request_id, request_type)

            try:
                started_at = time.time()
                if request_type == "photo":
                    image, observation = data
                    result_text = self.call_photo_api(image, observation, self_learning)
                elif request_type == "inbox":
                    current_observation, inbox = data
                    result_text = self.call_inbox_api(current_observation, inbox)
                elapsed_ms = int((time.time() - started_at) * 1000)
                logger.info(
                    "request completed │ id=%s type=%s latency_ms=%s chars=%s",
                    request_id,
                    request_type,
                    elapsed_ms,
                    len(result_text) if result_text else 0,
                )
            except Exception:
                logger.exception("request failed │ id=%s type=%s", request_id, request_type)
                if request_type == "photo":
                    _, observation = data
                    result_text = observation
                elif request_type == "inbox":
                    current_observation, _ = data
                    result_text = current_observation

            with self._active_lock:
                self._active_requests -= 1

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
        """Queue a photo-analysis request for a robot."""
        request_id = f"{robot_id}_photo"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)

        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time
        request_data = (image_data, observation)
        self.request_queue.put(("photo", request_id, request_data, current_time, self_learning))
        self._log_queue_state("photo queued", request_id, "photo")

    def submit_inbox_request(
        self,
        robot_id: int,
        current_observation: str,
        inbox: str,
    ) -> None:
        """Queue a peer-synthesis request for a robot."""
        request_id = f"{robot_id}_inbox"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)

        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time

        request_data = (current_observation, inbox)
        self.request_queue.put(("inbox", request_id, request_data, current_time, True))
        self._log_queue_state("inbox queued", request_id, "inbox")

    def get_result(self, robot_id: int, request_type: str) -> tuple[Any, Any]:
        """Poll and remove a completed result for a robot request."""
        request_id = f"{robot_id}_{request_type}"
        with self.results_lock:
            return self.results.pop(request_id, (None, None))

    def call_photo_api(self, image: Any, observation: str, self_learning: bool) -> str:
        """Run multimodal generation for one camera frame."""
        if self_learning:
            prompt_template = self.config.llm.prompts.photo_analysis_self_learning
        else:
            prompt_template = self.config.llm.prompts.photo_analysis_no_self_learning

        prompt = prompt_template.format(observation=observation)
        return self.provider.generate_vision(prompt, image)

    def call_inbox_api(self, current_observation: str, inbox: str) -> str:
        """Run text synthesis for peer knowledge integration."""
        prompt_template = self.config.llm.prompts.text_synthesis
        prompt = prompt_template.format(
            current_observation=current_observation,
            inbox=inbox,
        )
        return self.provider.generate_text(prompt)
