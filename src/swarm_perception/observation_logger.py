"""Persist simulation observations and optional artifacts per run."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import pygame as pg

from swarm_perception.utils.paths import OUTPUT_DIR


class ObservationLogger:
    """Write per-run outputs used for analysis and metrics.

    Primary output is ``robots.json``, where each robot id maps to a sequence
    of snapshot observations over time. Optional outputs include:

    - full-frame captures under ``frames/``
    - per-robot camera crops under ``robot_crops/``
    - communication merge events under ``communication_merges.jsonl``
    """

    def __init__(
        self,
        on: bool = False,
        empty_observation: str = "",
        base_dir: Path | str = OUTPUT_DIR,
        external_config: Any = None,
        save_robot_crops: bool = False,
        save_comm_merge_history: bool = False,
    ) -> None:
        """Create a run-scoped logger.

        Args:
            on: Enables or disables all logging writes.
            empty_observation: Initial placeholder text used by robots.
            base_dir: Parent directory for run output folders.
            external_config: Loaded config namespace (used to name run folder).
            save_robot_crops: Whether to store camera crops as PNG files.
            save_comm_merge_history: Whether to store JSONL merge events.
        """
        self.on = on
        self.external_config = external_config
        self.empty_observation = empty_observation
        self.base_dir = Path(base_dir)
        self.save_robot_crops = bool(save_robot_crops)
        self.save_comm_merge_history = bool(save_comm_merge_history)
        self.run_directory: Path | None = None
        self.frames_directory: Path | None = None
        self.robot_crops_directory: Path | None = None
        self.comm_history_filename: Path | None = None
        if self.on:
            self.lock = threading.Lock()

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = self.external_config.config.name
            self.run_directory = self.base_dir / f"{run_name}_{timestamp}"
            self.run_directory.mkdir(parents=True, exist_ok=True)
            self.filename = self.run_directory / "robots.json"
            self.progress_filename = self.run_directory / "robots.json"
            self.frames_directory = self.run_directory / "frames"
            self.robot_crops_directory = self.run_directory / "robot_crops"
            self.comm_history_filename = self.run_directory / "communication_merges.jsonl"
            self._initialize_file()

    def _initialize_file(self) -> None:
        """Ensure required files and optional output folders exist."""
        with self.lock:
            if not os.path.exists(self.progress_filename) or os.path.getsize(self.progress_filename) == 0:
                with open(self.progress_filename, "w", encoding="utf-8") as f:
                    json.dump({}, f)
            if self.save_robot_crops and self.robot_crops_directory is not None:
                self.robot_crops_directory.mkdir(parents=True, exist_ok=True)
            if self.save_comm_merge_history and self.comm_history_filename is not None:
                if (
                    not os.path.exists(self.comm_history_filename)
                    or os.path.getsize(self.comm_history_filename) == 0
                ):
                    with open(self.comm_history_filename, "w", encoding="utf-8"):
                        pass

    def log_observation(self, robot_id: int, observation: str) -> None:
        """Deprecated legacy logger, intentionally left as a no-op.

        Progress tracking is handled by :meth:`log_progress_snapshot`, which
        writes each robot's observation history in ``robots.json`` format.
        """
        _ = (robot_id, observation)
        pass

    def log_progress_snapshot(self, robot_id: int, observation: str) -> None:
        """Append one observation snapshot to ``robots.json``.

        Args:
            robot_id: Robot identifier (stored as JSON object key).
            observation: Observation text to append for this tick/epoch.
        """
        if not self.on:
            return
        with self.lock:
            try:
                with open(self.progress_filename, "r+", encoding="utf-8") as f:
                    data = json.load(f)
                    robot_key = str(robot_id)
                    if robot_key not in data:
                        data[robot_key] = []

                    data[robot_key].append(observation)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            except (json.JSONDecodeError, FileNotFoundError):
                self._initialize_file()
                self.log_progress_snapshot(robot_id, observation)

    def log_frame_capture(self, tick_count: int) -> None:
        """Save the current full simulation frame as a PNG image.

        Args:
            tick_count: Simulation tick used in filename generation.
        """
        if not self.on:
            return
        if self.frames_directory is None:
            return

        screen = pg.display.get_surface()
        if screen is None:
            # In headless mode there may be no display surface to snapshot.
            return

        self.frames_directory.mkdir(parents=True, exist_ok=True)
        filename = f"frame_tick_{int(tick_count):06d}.png"
        frame_path = self.frames_directory / filename
        pg.image.save(screen, frame_path)

    def log_robot_crop(
        self,
        robot_id: int,
        tick_count: int,
        capture_epoch: int,
        cropped_image: Any,
    ) -> None:
        """Save a robot camera crop PNG if crop logging is enabled.

        Args:
            robot_id: Robot identifier used in folder/filename.
            tick_count: Simulation tick used in filename.
            capture_epoch: Capture sequence number for that robot.
            cropped_image: OpenCV image array returned by the camera sensor.
        """
        if not self.on or not self.save_robot_crops:
            return
        if self.robot_crops_directory is None or cropped_image is None:
            return

        robot_dir = self.robot_crops_directory / f"robot_{int(robot_id):02d}"
        robot_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"robot_{int(robot_id):02d}_epoch_{int(capture_epoch):03d}"
            f"_tick_{int(tick_count):06d}.png"
        )
        crop_path = robot_dir / filename
        cv2.imwrite(str(crop_path), cropped_image)

    def log_comm_merge(
        self,
        receiver_robot_id: int,
        sender_robot_id: int,
        sender_tick: int,
        receiver_tick: int,
        capture_epoch: int,
        merge_method: str,
        inbox_policy: str,
    ) -> None:
        """Append one successful communication merge event to JSONL.

        Args:
            receiver_robot_id: Robot that integrated the message.
            sender_robot_id: Robot that sent the merged message.
            sender_tick: Tick at which sender transmitted observation.
            receiver_tick: Tick at which receiver merged observation.
            capture_epoch: Receiver capture epoch for this merge.
            merge_method: ``deterministic`` or ``llm`` merge mode.
            inbox_policy: Policy label used when this merge was executed.
        """
        if not self.on or not self.save_comm_merge_history:
            return
        if self.comm_history_filename is None:
            return

        event = {
            "receiver_robot_id": int(receiver_robot_id),
            "sender_robot_id": int(sender_robot_id),
            "sender_tick": int(sender_tick),
            "receiver_tick": int(receiver_tick),
            "capture_epoch": int(capture_epoch),
            "merge_method": str(merge_method),   # deterministic | llm
            "inbox_policy": str(inbox_policy),   # within_budget | deterministic_after_budget | llm_after_budget
        }
        with self.lock:
            with open(self.comm_history_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")