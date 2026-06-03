"""Optional local Ollama backend matching the API_MANAGER interface.

This module is kept as an alternate backend for local-network inference.
Production simulations import the Gemini backend from ``llm_api_gemini.py``.
"""

import base64
import queue
import threading
import time
from typing import Any

import cv2
import requests

class API_MANAGER:
    """Threaded request manager for a local Ollama endpoint."""

    def __init__(self, n_threads: int) -> None:
        """Initialize local endpoint, queue state, and worker metadata.

        Args:
            n_threads: Number of worker threads that will process requests.
        """
        self.WINDOWS_PC_IP = "10.228.246.81"
        self.MODEL_NAME = "gemma3:12b"
        self.OLLAMA_ENDPOINT = f"http://{self.WINDOWS_PC_IP}:11434/api/generate"

        self.n_threads = n_threads
        self.request_queue = queue.Queue()
        self.results = {}
        self.results_lock = threading.Lock()
        self.latest_request_timestamp = {}
        self.latest_request_lock = threading.Lock()

    def start(self) -> None:
        """Start background worker threads for queued API calls."""
        print(f"Starting API manager with {self.n_threads} threads ...")
        for i in range(self.n_threads):
            worker_thread = threading.Thread(target=self.worker_per_thread)
            worker_thread.daemon = True
            worker_thread.start()

    def worker_per_thread(self) -> None:
        """Consume queued photo/inbox requests and cache latest results."""
        while True:
            request_type, request_id, data, timestamp = self.request_queue.get()
            # Drop stale queued requests for the same robot/request type.
            with self.latest_request_lock:
                if timestamp < self.latest_request_timestamp.get(request_id, timestamp):
                    print(f"skipped outdated request: {request_id}")
                    self.request_queue.task_done()
                    continue

            try:
                if request_type == "photo":
                    image, observation = data
                    result_text = self.call_photo_api(image, observation)
                elif request_type == "inbox":
                    current_observation, inbox = data
                    result_text = self.call_inbox_api(current_observation, inbox)
            except Exception as e:
                print(f"ERROR in worker for {request_id}: {e}")
                result_text = "ERROR: API call failed"

            with self.results_lock:
                self.results[request_id] = (result_text, data)

            self.request_queue.task_done()

    def submit_photo_request(self, robot_id: int, image_data: Any, observation: str) -> None:
        """Queue photo analysis for one robot.

        Args:
            robot_id: Sender robot identifier.
            image_data: OpenCV BGR image crop.
            observation: Existing memory text to include in prompt.
        """
        request_id = f"{robot_id}_photo"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)
        
        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time
        request_data = (image_data, observation)
        self.request_queue.put(("photo", request_id, request_data, current_time))

    def submit_inbox_request(
        self, robot_id: int, current_observation: str, inbox: str
    ) -> None:
        """Queue text synthesis from current memory plus peer inbox.

        Args:
            robot_id: Receiver robot identifier.
            current_observation: Receiver memory text.
            inbox: Peer-provided memory text.
        """
        request_id = f"{robot_id}_inbox"
        current_time = time.time()
        with self.results_lock:
            if request_id in self.results:
                self.results.pop(request_id)
        
        with self.latest_request_lock:
            self.latest_request_timestamp[request_id] = current_time

        request_data = (current_observation, inbox)
        self.request_queue.put(("inbox", request_id, request_data, current_time))

    def get_result(self, robot_id: int, request_type: str) -> tuple[Any, Any]:
        """Poll and remove one completed request result.

        Args:
            robot_id: Robot identifier.
            request_type: ``photo`` or ``inbox``.

        Returns:
            Tuple of ``(result_text, request_payload)`` or ``(None, None)``.
        """
        request_id = f"{robot_id}_{request_type}"
        with self.results_lock:
            return self.results.pop(request_id, (None, None))

    def call_photo_api(self, image: Any, observation: str) -> str:
        """Call Ollama image-text generation for local perception updates.

        Args:
            image: OpenCV BGR image crop.
            observation: Current memory text.

        Returns:
            Updated memory summary string from the local model.
        """
        _, buffer = cv2.imencode(".png", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "model": self.MODEL_NAME,
            "prompt": f"""
            You are a robotic perception module for a member of a collaborative swarm. Your function is to provide a dense and factual summary of your limited view, which will later be shared and synthesized to build a collective map of the entire environment.            
            
            [My Memory Log So Far]
            {observation}

            [New Visual Input]
            The attached image is my current, cropped view of the world I am trying to create a collective understaind of, with other peers.

            ---
            YOUR TASK:
            First, mentally divide the image into a 3x3 grid (top-left, top-center, top-right; middle-left, center, etc.). Analyze the contents of each grid cell. Identify all primary objects, their attributes (color, texture, shape), and their spatial relationships to each other (e.g., 'a dirt path runs diagonally from the bottom-left to the center').

            After performing this analysis, integrate these new, geometric, and factual details into the [My Memory Log So Far]. Create a single, revised summary that describes the scene's composition and layout, not just a list of items.

            **RULES:**
            -   Be literal. Describe what is visible, not what you infer.
            -   Do not use subjective language ('nice', 'beautiful') or vague assessments ('well-maintained').
            -   The final summary must not exceed 150 words and must be plain text.
            -   YOUR ENTIRE RESPONSE MUST BE ONLY THE FINAL REVISED SUMMARY. DO NOT SHOW YOUR ANALYSIS OR STEPS. DO NOT USE ANY FORMATTING.

            REVISED SUMMARY:
            """,
            "images": [base64_image],
            "stream": False,
        }

        response = requests.post(self.OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()

        response_data = response.json()
        return response_data["response"].strip()

    def call_inbox_api(self, current_observation: str, inbox: str) -> str:
        """Call Ollama text synthesis for peer-message integration.

        Args:
            current_observation: Receiver robot memory.
            inbox: Peer message to merge.

        Returns:
            Merged memory text generated by the local model.
        """
        payload = {
            "model": self.MODEL_NAME,
            "prompt": f"""
            You are a data fusion module for a collaborative robot swarm. Your primary purpose is to help the swarm achieve a collective understanding of its environment. To do this, you will merge your robot's own memory with an observation from a peer to create a more complete part of the larger map.
            [My Current Memory]
            {current_observation}

            [New Observation from Peer]
            {inbox}

            ---
            YOUR TASK:
            Create the most factually dense and spatially coherent summary possible. Treat both logs as partial descriptions of the same world. Identify all unique objects, attributes, and spatial relationships (e.g., 'tree is left of path', 'fence is behind tree') from each source.

            Combine these facts to build a more complete picture. If the logs describe the same object with different levels of detail, use the more specific description in the final summary. Eliminate all redundant information.

            YOUR ENTIRE RESPONSE MUST BE ONLY THE FINAL SYNTHESIZED LOG. IT MUST BE A SINGLE PARAGRAPH OF PLAIN TEXT. DO NOT SHOW YOUR STEPS OR REASONING. DO NOT USE BOLD, LISTS, OR ANY OTHER MARKDOWN. THE SUMMARY CANNOT EXCEED 150 WORDS.

            SYNTHESIZED LOG:
            """,
            "stream": False,
        }
        response = requests.post(self.OLLAMA_ENDPOINT, json=payload)
        response.raise_for_status()
        response_data = response.json()
        return response_data["response"].strip()