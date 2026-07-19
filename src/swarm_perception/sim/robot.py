"""Robot agent: correlated random walk, camera capture, and record exchange.

Robots perform a correlated random walk over a shared background image and
periodically capture square crops of it. Each capture produces one provenance
record keyed ``(epoch, robot, crop_idx)``; robots hold these records in a
capped memory and broadcast them to peers in communication range, merging
incoming records by key-union under a per-epoch budget. Embeddings replace
the interim records in a later stage.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from vi import Agent, HeadlessSimulation

from swarm_perception.camera_sensor import CameraSensor
from swarm_perception.config import Config
from swarm_perception.io.run_logger import RunLogger
from swarm_perception.sim.actuator import Actuator


class Robot(Agent):
    """Swarm robot agent with camera sensing and record-based memory.

    Each robot keeps a dict of capture records keyed by
    ``(epoch, robot, crop_idx)`` and grows it through two pathways:

    - individual sensing: one record per own camera capture
    - social exchange: key-union merges of records broadcast by nearby peers

    Configuration is read from the injected ``self.shared.cfg`` (typed
    :class:`~swarm_perception.config.Config`).
    """

    def __init__(
        self,
        images: list[str],
        simulation: HeadlessSimulation,
        pos: Any = None,
        move: Any = None,
    ) -> None:
        """Create one robot with random spawn, sensor, and actuator.

        Args:
            images: Sprite image paths passed to violet ``Agent`` base class.
            simulation: Parent simulation object.
            pos: Optional initial position (overridden by random spawn).
            move: Optional initial movement vector.
        """
        super().__init__(images, simulation, pos, move)
        cfg: Config = self.shared.cfg  # type: ignore[attr-defined]
        self.cfg = cfg
        rng: random.Random = self.shared.rng  # type: ignore[attr-defined]

        self.sense_square = cfg.robot.coverage_side

        # spawning coordinates
        self.pos.x = rng.uniform(cfg.robot.coverage_side, cfg.simulation.width - cfg.robot.coverage_side)
        self.pos.y = rng.uniform(cfg.robot.coverage_side, cfg.simulation.height - cfg.robot.coverage_side)

        self.sensor = CameraSensor(
            agent=self,
            coverage_side=self.sense_square,
            background=self.shared.background,  # type: ignore[attr-defined]
            sensing_radius=cfg.robot.neighbor_radius,
        )
        self.actuator = Actuator(self, rng=rng)
        self.run_logger: RunLogger = self.shared.run_logger  # type: ignore[attr-defined]

        # photo taking related
        self.photo_tick_counter = 0
        self.is_taking_photo = False
        self.flash_duration = 15  # ticks
        self.flash_counter = 0
        self.capture_epoch = 0
        self.tick_count = 0

        # agent's memory: one record per capture key, merged by key-union
        self.memory: dict[tuple[int, int, int], dict[str, Any]] = {}
        self.inbox_merges_this_epoch = 0

        # message transfer: bounded FIFO of (records, sender_tick, sender_id)
        self.inbox_queue: deque[tuple[list[dict[str, Any]], int, int]] = deque(maxlen=8)

    def receive_peer_message(
        self, records: list[dict[str, Any]], sender_tick: int, sender_id: int
    ) -> None:
        """Queue one peer broadcast into the bounded FIFO inbox.

        The inbox holds at most 8 broadcasts; when full, the oldest queued
        broadcast is dropped to make room for the incoming one.

        Args:
            records: Sender's full record list.
            sender_tick: Sender tick when the broadcast was emitted.
            sender_id: Sender robot identifier.
        """
        self.inbox_queue.append((records, sender_tick, sender_id))

    def exchange_with_neighbors(self) -> None:
        """Broadcast this robot's full record list to peers currently in range."""
        if not self.memory or not self.cfg.robot.communication:
            return
        records = list(self.memory.values())
        for neighbor, _ in self.in_proximity_accuracy():
            neighbor.receive_peer_message(records, self.tick_count, int(self.id))  # type: ignore

    def merge_records(self, records: list[dict[str, Any]]) -> None:
        """Merge records into memory by key-union, then enforce the memory cap.

        Records whose key is already present are ignored. When the cap is
        exceeded, the records with the smallest keys (tuple order) are kept —
        a deterministic canonical truncation; k-center selection replaces it
        when embeddings land.
        """
        for record in records:
            key = (int(record["key"][0]), int(record["key"][1]), int(record["key"][2]))
            if key not in self.memory:
                self.memory[key] = record
        cap = self.cfg.robot.memory_cap
        if len(self.memory) > cap:
            self.memory = {key: self.memory[key] for key in sorted(self.memory)[:cap]}

    def process_inbox(self) -> None:
        """Process at most one queued peer broadcast per tick.

        Within the per-epoch budget the broadcast is merged by key-union.
        Over budget, ``robot.inbox_merge_after_budget`` decides: ``"drop"``
        discards the broadcast, ``"deterministic"`` merges it anyway.
        """
        if not self.inbox_queue:
            return
        if self.inbox_merges_this_epoch < self.cfg.robot.max_inbox_merges_per_epoch:
            records, sender_tick, sender_id = self.inbox_queue.popleft()
            self.merge_records(records)
            self.run_logger.log_comm(
                receiver_tick=self.tick_count,
                sender_tick=sender_tick,
                epoch=self.capture_epoch,
                receiver=int(self.id),
                sender=sender_id,
                merge_method="deterministic",
                inbox_policy="within_budget",
            )
            self.inbox_merges_this_epoch += 1
            return
        if self.cfg.robot.inbox_merge_after_budget == "deterministic":
            records, sender_tick, sender_id = self.inbox_queue.popleft()
            self.merge_records(records)
            self.run_logger.log_comm(
                receiver_tick=self.tick_count,
                sender_tick=sender_tick,
                epoch=self.capture_epoch,
                receiver=int(self.id),
                sender=sender_id,
                merge_method="deterministic",
                inbox_policy="deterministic_after_budget",
            )
            return
        # "drop": over budget for this epoch; discard the oldest broadcast.
        self.inbox_queue.popleft()

    def update(self) -> None:
        """Execute one full agent tick for sensing, memory, and exchange.

        The tick loop performs:
        1) sensing overlay and periodic photo capture with record insertion
        2) continuous neighbor broadcast while in proximity
        3) at most one budgeted inbox merge
        """
        cfg = self.cfg
        self.tick_count += 1
        # show the sense rectangle
        if self.is_taking_photo:
            self.sensor.show_outline(color=(255, 255, 0))  # Yellow flash
            self.flash_counter += 1
            if self.flash_counter >= self.flash_duration:
                self.is_taking_photo = False
                self.flash_counter = 0
        else:
            self.sensor.show_outline()

        # take a photo every capture_frequency seconds (real time when fps > 0)
        self.photo_tick_counter += 1
        if self.photo_tick_counter >= cfg.photo_ticks:
            # state variables
            self.photo_tick_counter = 0
            self.is_taking_photo = True
            self.capture_epoch += 1
            self.inbox_merges_this_epoch = 0

            if cfg.simulation.save_photo_frames:
                frame_state = self.shared.photo_frame_capture_state  # type: ignore
                if frame_state["tick"] != self.tick_count:
                    frame_state["tick"] = self.tick_count
                    frame_state["robot_ids"] = set()
                    frame_state["saved"] = False

                frame_state["robot_ids"].add(int(self.id))
                if not frame_state["saved"] and len(frame_state["robot_ids"]) >= cfg.simulation.num_of_robots:
                    # Save one frame once all robots entered photo mode for this tick.
                    self.run_logger.save_frame(self.tick_count)
                    frame_state["saved"] = True

            image, rect = self.sensor.take_photo()
            self.run_logger.log_capture(
                tick=self.tick_count,
                epoch=self.capture_epoch,
                robot=int(self.id),
                bbox=rect,
                pos=(self.pos.x, self.pos.y),
            )
            self.run_logger.save_crop(
                robot_id=int(self.id),
                tick=self.tick_count,
                epoch=self.capture_epoch,
                image=image,
            )
            self.merge_records(
                [
                    {
                        "key": [self.capture_epoch, int(self.id), 0],
                        "bbox": [int(v) for v in rect],
                        "pos": [float(self.pos.x), float(self.pos.y)],
                    }
                ]
            )
            self.run_logger.log_memory(
                tick=self.tick_count,
                epoch=self.capture_epoch,
                robot=int(self.id),
                keys=self.memory,
            )

        # Exchange records whenever peers are currently nearby (not only photo events).
        self.exchange_with_neighbors()
        self.process_inbox()

    def get_velocities(self) -> tuple[float, float]:
        """Return movement command for correlated random walk with edge avoidance.

        Returns:
            Tuple of ``(linear_speed, angular_velocity)`` for this tick.
        """
        linear_speed = self.cfg.robot.linear_speed  # default
        angular_velocity = 0.0  # default

        if self.sensor.detect_edges():
            angular_velocity = self.cfg.robot.angular_velocity  # turn back
        return linear_speed, angular_velocity
