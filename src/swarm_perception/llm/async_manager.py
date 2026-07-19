"""Asyncio LLM request manager for parallel non-blocking provider calls."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from swarm_perception.llm.manager import build_inbox_prompt, build_photo_prompt
from swarm_perception.llm.providers.vllm import VllmProvider

logger = logging.getLogger("swarm.llm")


class AsyncAPI_MANAGER:
    """Event-loop manager that runs up to ``n_concurrent`` LLM calls in parallel.

    Each submitted request becomes its own asyncio task. A semaphore caps
    in-flight work at ``thread_workers`` so requests are not serialized.
    """

    def __init__(self, n_concurrent: int, config: Any, provider: VllmProvider) -> None:
        self.config = config
        self.provider = provider
        self.n_concurrent = max(1, n_concurrent)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._queue: asyncio.Queue[tuple[Any, ...]] | None = None
        self.results: dict[str, tuple[Any, Any]] = {}
        self.results_lock = threading.Lock()
        self.latest_request_timestamp: dict[str, float] = {}
        self.latest_request_lock = threading.Lock()
        self._active_requests = 0
        self._active_lock = threading.Lock()

    def queue_depth(self) -> int:
        if self._queue is None:
            return 0
        return self._queue.qsize()

    def active_request_count(self) -> int:
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
        if self._loop_thread is not None:
            return

        def run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._async_startup())
            self._ready.set()
            self._loop.run_forever()

        self._loop_thread = threading.Thread(
            target=run_loop,
            daemon=True,
            name="vllm-async-manager",
        )
        self._loop_thread.start()
        self._ready.wait()
        logger.info("async worker pool started │ max_concurrent=%s", self.n_concurrent)

    async def _async_startup(self) -> None:
        await self.provider.start()
        self._queue = asyncio.Queue()
        asyncio.create_task(self._dispatcher())

    def _enqueue(self, item: tuple[Any, ...]) -> None:
        if self._loop is None or self._queue is None:
            raise RuntimeError("AsyncAPI_MANAGER.start() was not called")
        future = asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop)
        future.result()

    async def _dispatcher(self) -> None:
        semaphore = asyncio.Semaphore(self.n_concurrent)
        assert self._queue is not None
        while True:
            request_type, request_id, data, timestamp, self_learning = await self._queue.get()
            asyncio.create_task(
                self._process_request(
                    request_type,
                    request_id,
                    data,
                    timestamp,
                    self_learning,
                    semaphore,
                )
            )

    async def _process_request(
        self,
        request_type: str,
        request_id: str,
        data: Any,
        timestamp: float,
        self_learning: bool,
        semaphore: asyncio.Semaphore,
    ) -> None:
        async with semaphore:
            with self.latest_request_lock:
                if timestamp < self.latest_request_timestamp.get(request_id, timestamp):
                    logger.debug("skipped stale │ id=%s type=%s", request_id, request_type)
                    return

            with self._active_lock:
                self._active_requests += 1
            logger.info("request started │ id=%s type=%s", request_id, request_type)

            try:
                started_at = time.time()
                if request_type == "photo":
                    image, observation = data
                    prompt = build_photo_prompt(self.config, observation, self_learning)
                    result_text = await self.provider.generate_vision(prompt, image)
                elif request_type == "inbox":
                    current_observation, inbox = data
                    prompt = build_inbox_prompt(self.config, current_observation, inbox)
                    result_text = await self.provider.generate_text(prompt)
                else:
                    raise ValueError(f"Unknown request type: {request_type}")

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
                else:
                    result_text = ""

            with self._active_lock:
                self._active_requests -= 1

            with self.results_lock:
                self.results[request_id] = (result_text, data)

    def submit_photo_request(
        self,
        robot_id: int,
        image_data: Any,
        observation: str,
        self_learning: bool,
    ) -> None:
        request_id = f"{robot_id}_photo"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)

        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time
        request_data = (image_data, observation)
        self._enqueue(("photo", request_id, request_data, current_time, self_learning))
        self._log_queue_state("photo queued", request_id, "photo")

    def submit_inbox_request(
        self,
        robot_id: int,
        current_observation: str,
        inbox: str,
    ) -> None:
        request_id = f"{robot_id}_inbox"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)

        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time

        request_data = (current_observation, inbox)
        self._enqueue(("inbox", request_id, request_data, current_time, True))
        self._log_queue_state("inbox queued", request_id, "inbox")

    def get_result(self, robot_id: int, request_type: str) -> tuple[Any, Any]:
        request_id = f"{robot_id}_{request_type}"
        with self.results_lock:
            return self.results.pop(request_id, (None, None))
