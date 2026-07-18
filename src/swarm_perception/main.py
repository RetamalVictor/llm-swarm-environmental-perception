"""Swarm simulation entrypoint: movement, camera capture, and record exchange.

Robots perform a correlated random walk over a shared background image and
periodically capture square crops of it. Each capture produces one provenance
record keyed ``(epoch, robot, crop_idx)``; robots hold these records in a
capped memory and broadcast them to peers in communication range, merging
incoming records by key-union under a per-epoch budget. Embeddings replace
the interim records in a later stage.

Every run is logged through
:class:`swarm_perception.io.run_logger.RunLogger`: events append to
``events.jsonl`` and the reproducibility artifacts
(``config_resolved.yaml``, ``run_metadata.json``) live in one run directory.

Configuration is loaded inside :func:`main` and injected into the simulation
via violet's shared state (``self.shared.cfg``); there are no import-time
globals.
"""

from __future__ import annotations

import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import pygame as pg
from vi import Agent, Config as ViConfig, HeadlessSimulation, Simulation, Window

from swarm_perception.actuator import Actuator
from swarm_perception.camera_sensor import CameraSensor
from swarm_perception.io.run_logger import RunLogger
from swarm_perception.utils.config import Config, ConfigError, load_config
from swarm_perception.utils.paths import ASSETS_DIR
from swarm_perception.world.background import Background


class Robot(Agent):
    """Swarm robot agent with camera sensing and record-based memory.

    Each robot keeps a dict of capture records keyed by
    ``(epoch, robot, crop_idx)`` and grows it through two pathways:

    - individual sensing: one record per own camera capture
    - social exchange: key-union merges of records broadcast by nearby peers

    Configuration is read from the injected ``self.shared.cfg`` (typed
    :class:`~swarm_perception.utils.config.Config`).
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
        cap = self.cfg.robot.max_facts_per_observation
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


class _EnvironmentMixin:
    """Shared wiring for the windowed and headless simulations.

    Holds everything the two variants have in common: shared service setup
    (config, RNG, run logger), background loading, and the agent
    position-update hook. The only difference between the variants lives in
    the headless tick-pacing ``after_update``.
    """

    def _setup_environment(
        self, cfg: Config, background_path: Any, run_dir: Path | str | None = None
    ) -> None:
        """Construct and register services shared across all agents.

        Args:
            cfg: Typed run configuration, exposed to agents as ``self.shared.cfg``.
            background_path: Optional path to a background image texture.
            run_dir: Run output directory; defaults to
                :func:`swarm_perception.io.run_logger.resolve_run_dir` on ``cfg``.
        """
        # self.shared is shared across all agents and the simulation.
        self.shared.cfg = cfg  # type: ignore[attr-defined]
        self.shared.rng = random.Random(cfg.simulation.seed)  # type: ignore[attr-defined]
        # Load-once world image; every robot crops views of this one array.
        self.shared.background = Background(background_path)  # type: ignore[attr-defined]
        self.shared.run_logger = RunLogger(cfg, run_dir=run_dir)  # type: ignore[attr-defined]
        self.shared.photo_frame_capture_state = {"tick": None, "robot_ids": set(), "saved": False}  # type: ignore[attr-defined]
        self._load_background(background_path)

    def _load_background(self, background_path: Any) -> None:
        """Load and scale the background image when a path is provided."""
        if not background_path:
            return
        try:
            size = self.config.window.as_tuple()  # type: ignore[attr-defined]
            background_image = pg.image.load(background_path)
            if pg.display.get_surface() is not None:
                background_image = background_image.convert()
            self._background = pg.transform.scale(background_image, size)  # type: ignore[attr-defined]
        except pg.error:
            pass  # background not loaded; ignored for now

    def _HeadlessSimulation__update_positions(self) -> None:
        """Update all agent positions using each robot's actuator command."""
        for sprite in self._agents.sprites():  # type: ignore[attr-defined]
            agent: Agent = sprite  # type: ignore
            linear_speed, angular_velocity = agent.get_velocities()  # type: ignore
            agent.actuator.update(linear_speed, angular_velocity)  # type: ignore


class EnvironmentSimulation(_EnvironmentMixin, Simulation):
    """Windowed simulation wiring the shared services."""

    def __init__(
        self,
        vi_config: ViConfig | None = None,
        cfg: Config | None = None,
        background_path: Any = None,
        run_dir: Path | str | None = None,
    ) -> None:
        """Initialize simulation state and globally shared services.

        Args:
            vi_config: Violet simulation config object.
            cfg: Typed run configuration injected into agents.
            background_path: Optional path to background image texture.
            run_dir: Run output directory; defaults to ``resolve_run_dir(cfg)``.
        """
        super().__init__(vi_config)
        self._setup_environment(cfg, background_path, run_dir)


class EnvironmentHeadlessSimulation(_EnvironmentMixin, HeadlessSimulation):
    """Headless variant with the same shared services plus tick pacing."""

    def __init__(
        self,
        vi_config: ViConfig | None = None,
        cfg: Config | None = None,
        background_path: Any = None,
        run_dir: Path | str | None = None,
    ) -> None:
        super().__init__(vi_config)
        self._last_tick_time = time.perf_counter()
        self._setup_environment(cfg, background_path, run_dir)

    def after_update(self) -> None:
        """Pace ticks to simulation.fps."""
        fps = self.shared.cfg.simulation.fps  # type: ignore[attr-defined]
        if fps <= 0:
            return
        tick_interval = 1.0 / fps
        elapsed = time.perf_counter() - self._last_tick_time
        if elapsed < tick_interval:
            time.sleep(tick_interval - elapsed)
        self._last_tick_time = time.perf_counter()


def configure_runtime_mode(headless: bool) -> None:
    """Configure SDL for true no-window operation before simulation startup."""
    if not headless:
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def build_vi_config(cfg: Config) -> ViConfig:
    """Build the violet simulation config from the typed run config."""
    return ViConfig(
        window=Window(cfg.simulation.width, cfg.simulation.height),
        movement_speed=1.0,
        seed=cfg.simulation.seed,
        image_rotation=True,
        radius=cfg.robot.neighbor_radius,
        fps_limit=cfg.simulation.fps,
        duration=cfg.sim_duration,
    )


def main() -> None:
    """Console entrypoint: load config, wire the simulation, and run it."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        cfg = load_config(config_path)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    configure_runtime_mode(cfg.simulation.headless)
    vi_config = build_vi_config(cfg)
    sim_cls = EnvironmentHeadlessSimulation if cfg.simulation.headless else EnvironmentSimulation
    sim = sim_cls(
        vi_config=vi_config,
        cfg=cfg,
        background_path=ASSETS_DIR / cfg.simulation.background_image,
    )
    sim.batch_spawn_agents(
        cfg.simulation.num_of_robots,
        Robot,
        images=[str(ASSETS_DIR / cfg.simulation.robot_image)],
    )
    sim.run()
    sim.shared.run_logger.finalize()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
