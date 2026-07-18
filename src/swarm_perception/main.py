"""Swarm simulation entrypoint implementing perception and communication loops.

This module corresponds to the core methodology in the capstone paper:
robots capture local visual patches, maintain a textual knowledge base, exchange
messages with nearby peers, and integrate information through LLM-assisted
or deterministic merging.

Configuration is loaded inside :func:`main` and injected into the simulation via
violet's shared state (``self.shared.cfg``); there are no import-time globals.

Logging was removed for now; a dedicated logging module will be reintroduced
later.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from collections import deque
from typing import Any

import pygame as pg
from vi import Agent, Config as ViConfig, HeadlessSimulation, Simulation, Window

from swarm_perception.actuator import Actuator
from swarm_perception.camera_sensor import CameraSensor
from swarm_perception.llm.factory import create_api_manager
from swarm_perception.observation_logger import ObservationLogger
from swarm_perception.utils.config import Config, load_config
from swarm_perception.utils.paths import ASSETS_DIR, OUTPUT_DIR

# Run-logging (observations/artifacts) is always on; not a per-run config knob.
LOG_RESULTS = True


def split_facts(paragraph: str) -> list[str]:
    """Split a paragraph into sentence-like fact chunks.

    Args:
        paragraph: Observation text containing one or more fact sentences.

    Returns:
        A cleaned list of sentence fragments split on punctuation boundaries.
    """
    if not paragraph:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def normalize_fact(fact: str) -> str:
    """Normalize a fact sentence into a dedupe key.

    Args:
        fact: Original sentence-like fact text.

    Returns:
        Lowercased alphanumeric text with punctuation removed and normalized
        whitespace.
    """
    lowered = fact.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def join_facts(facts: list[str], max_facts: int) -> str:
    """Join facts into one paragraph while enforcing sentence punctuation.

    Args:
        facts: Candidate facts to join.
        max_facts: Maximum number of fact sentences to retain.

    Returns:
        Single paragraph string capped to ``max_facts`` entries.
    """
    cleaned = [fact.strip() for fact in facts if fact and fact.strip()]
    limited = cleaned[:max_facts]
    normalized = []
    for fact in limited:
        normalized.append(fact if fact.endswith((".", "!", "?")) else f"{fact}.")
    return " ".join(normalized)


def merge_observations(preferred: str, fallback: str, max_facts: int) -> str:
    """Merge two observation strings with deduplication and detail preference.

    Facts from ``preferred`` are considered first, then ``fallback``. Duplicate
    facts are matched by normalized keys; if duplicates differ, the longer
    sentence is kept as the more detailed variant.

    Args:
        preferred: Newer or higher-priority observation text.
        fallback: Older or lower-priority observation text.
        max_facts: Maximum number of retained facts in output.

    Returns:
        Merged paragraph containing unique fact sentences.
    """
    merged: dict[str, str] = {}
    ordered_keys: list[str] = []

    for fact in split_facts(preferred) + split_facts(fallback):
        key = normalize_fact(fact)
        if not key:
            continue
        if key not in merged:
            merged[key] = fact
            ordered_keys.append(key)
        elif len(fact) > len(merged[key]):
            merged[key] = fact

    return join_facts([merged[key] for key in ordered_keys], max_facts)


def iter_robots(simulation: HeadlessSimulation) -> Any:
    """Yield robot agents from a simulation sprite group."""
    for sprite in simulation._agents.sprites():
        yield sprite


def any_robot_llm_pending(simulation: HeadlessSimulation) -> bool:
    """Return True when any robot is still waiting on an in-flight LLM response."""
    use_inbox_synthesis = simulation.shared.cfg.use_llm_inbox_synthesis  # type: ignore[attr-defined]
    for robot in iter_robots(simulation):
        if robot.PHOTO_RESULT_PENDING:  # type: ignore[attr-defined]
            return True
        if use_inbox_synthesis and robot.INBOX_PROCESS_PENDING:  # type: ignore[attr-defined]
            return True
    return False


def freeze_until_llm_batch_done(simulation: HeadlessSimulation) -> None:
    """Block between ticks until all in-flight LLM requests for this epoch finish."""
    if not simulation.shared.cfg.wait_for_llm or not any_robot_llm_pending(simulation):  # type: ignore[attr-defined]
        return
    while any_robot_llm_pending(simulation):
        for robot in iter_robots(simulation):
            robot.poll_photo_result()  # type: ignore[attr-defined]
            robot.poll_inbox_result()  # type: ignore[attr-defined]
        time.sleep(0.02)


class Robot(Agent):
    """Swarm robot agent with camera sensing and asynchronous LLM integration.

    Each robot maintains a local textual knowledge base (`current_observation`)
    and updates it through two pathways:

    - individual learning: camera capture + photo interpretation
    - social learning: peer message intake + inbox synthesis

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
        """Create one robot with random spawn, sensor, actuator, and inbox state.

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

        # spwaning coordinates
        self.pos.x = rng.uniform(cfg.robot.coverage_side, cfg.simulation.width - cfg.robot.coverage_side)
        self.pos.y = rng.uniform(cfg.robot.coverage_side, cfg.simulation.height - cfg.robot.coverage_side)

        self.sensor = CameraSensor(
            agent=self,
            coverage_side=self.sense_square,
            background_image=cfg.simulation.background_image,
            sensing_radius=cfg.robot.neighbor_radius,
        )
        self.actuator = Actuator(self, rng=rng)
        self.llm = self.shared.api_manager  # type: ignore

        # photo taking related
        self.photo_tick_counter = 0
        self.is_taking_photo = False
        self.flash_duration = 15  # ticks
        self.flash_counter = 0

        # agent's memory
        self.current_observation = cfg.robot.empty_observation
        self.EMPTY_OBSERVATION = True
        self.observation_logger = self.shared.observation_logger  # type: ignore
        self.PHOTO_RESULT_PENDING = False
        self.photo_pending_since_tick = 0
        self.capture_epoch = 0
        self.inbox_merges_this_epoch = 0

        # message transfer
        self.inbox_queue = deque()
        self.INBOX_PROCESS_PENDING = False
        self.inbox_pending_since_tick = 0
        self.pending_inbox_sender_id = None
        self.pending_inbox_sender_tick = None
        self.pending_inbox_policy = None
        self.tick_count = 0

    def receive_peer_message(self, message: str, message_tick: int, sender_id: int) -> None:
        """Receive one peer message using a single-slot replacement inbox policy.

        Args:
            message: Sender's current knowledge-base paragraph.
            message_tick: Sender tick when message was emitted.
            sender_id: Sender robot identifier.
        """
        incoming_payload = (message, message_tick, sender_id)
        if not self.inbox_queue:
            self.inbox_queue.append(incoming_payload)
            return

        _, existing_tick, _existing_sender_id = self.inbox_queue[0]
        if message_tick > existing_tick:
            self.inbox_queue[0] = incoming_payload
        # otherwise the pending message is newer; ignore the stale one

    def exchange_with_neighbors(self) -> None:
        """Broadcast current observation to peers currently in communication range."""
        if self.EMPTY_OBSERVATION or not self.cfg.robot.communication:
            return
        for neighbor, _ in self.in_proximity_accuracy():
            neighbor.receive_peer_message(  # type: ignore
                self.current_observation,
                self.tick_count,
                int(self.id),
            )

    def update(self) -> None:
        """Execute one full agent tick for sensing, sharing, and result polling.

        The tick loop performs:
        1) sensing overlay and periodic photo capture
        2) continuous neighbor broadcast while in proximity
        3) inbox merge scheduling under per-epoch budget rules
        4) async polling for photo and inbox LLM results with timeout fallback
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
                    self.observation_logger.log_frame_capture(self.tick_count)
                    frame_state["saved"] = True

            image = self.sensor.take_photo()
            self.observation_logger.log_robot_crop(
                robot_id=self.id,
                tick_count=self.tick_count,
                capture_epoch=self.capture_epoch,
                cropped_image=image,
            )
            self.llm.submit_photo_request(self.id, image, self.current_observation, cfg.robot.self_learning)
            self.PHOTO_RESULT_PENDING = True
            self.photo_pending_since_tick = self.tick_count
            self.observation_logger.log_progress_snapshot(
                robot_id=self.id,
                observation=self.current_observation,
            )

        # Exchange messages whenever peers are currently nearby (not only photo events).
        self.exchange_with_neighbors()

        # process inbox queue (one message at a time)
        inbox_budget_available = self.inbox_merges_this_epoch < cfg.robot.max_inbox_merges_per_epoch
        if self.inbox_queue and not self.INBOX_PROCESS_PENDING and inbox_budget_available:
            next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
            if not cfg.use_llm_inbox_synthesis:
                self.current_observation = merge_observations(
                    next_inbox,
                    self.current_observation,
                    cfg.robot.max_facts_per_observation,
                )
                self.observation_logger.log_comm_merge(
                    receiver_robot_id=self.id,
                    sender_robot_id=incoming_sender_id,
                    sender_tick=incoming_tick,
                    receiver_tick=self.tick_count,
                    capture_epoch=self.capture_epoch,
                    merge_method="deterministic",
                    inbox_policy="within_budget",
                )
                self.inbox_merges_this_epoch += 1
            else:
                self.llm.submit_inbox_request(self.id, self.current_observation, next_inbox)
                self.INBOX_PROCESS_PENDING = True
                self.inbox_pending_since_tick = self.tick_count
                self.pending_inbox_sender_id = incoming_sender_id
                self.pending_inbox_sender_tick = incoming_tick
                self.pending_inbox_policy = "within_budget"
                self.inbox_merges_this_epoch += 1
        elif self.inbox_queue and not self.INBOX_PROCESS_PENDING and not inbox_budget_available:
            budget_policy = cfg.robot.inbox_merge_after_budget
            if budget_policy == "drop":
                pass  # over budget for this epoch; drop the message
            elif budget_policy == "deterministic":
                next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
                self.current_observation = merge_observations(
                    next_inbox,
                    self.current_observation,
                    cfg.robot.max_facts_per_observation,
                )
                self.observation_logger.log_comm_merge(
                    receiver_robot_id=self.id,
                    sender_robot_id=incoming_sender_id,
                    sender_tick=incoming_tick,
                    receiver_tick=self.tick_count,
                    capture_epoch=self.capture_epoch,
                    merge_method="deterministic",
                    inbox_policy="deterministic_after_budget",
                )
            elif budget_policy == "llm":
                next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
                self.llm.submit_inbox_request(self.id, self.current_observation, next_inbox)
                self.INBOX_PROCESS_PENDING = True
                self.inbox_pending_since_tick = self.tick_count
                self.pending_inbox_sender_id = incoming_sender_id
                self.pending_inbox_sender_tick = incoming_tick
                self.pending_inbox_policy = "llm_after_budget"
            else:
                pass  # unknown inbox_merge_after_budget policy; ignore

        self.poll_photo_result()
        self.poll_inbox_result()

    def poll_photo_result(self) -> None:
        """Apply a completed photo LLM result or clear pending state on timeout."""
        if not self.PHOTO_RESULT_PENDING:
            return

        photo_summary_result, _ = self.llm.get_result(self.id, request_type="photo")
        if photo_summary_result:
            self.current_observation = merge_observations(
                photo_summary_result,
                self.current_observation,
                self.cfg.robot.max_facts_per_observation,
            )
            self.PHOTO_RESULT_PENDING = False
            self.EMPTY_OBSERVATION = False
            self.observation_logger.log_observation(self.id, self.current_observation)
            return

        if self.cfg.wait_for_llm:
            return

        if self.tick_count - self.photo_pending_since_tick <= self.cfg.photo_timeout_ticks:
            return

        self.PHOTO_RESULT_PENDING = False

    def poll_inbox_result(self) -> None:
        """Apply a completed inbox LLM result or clear pending state on timeout."""
        if not self.INBOX_PROCESS_PENDING or not self.cfg.use_llm_inbox_synthesis:
            return

        inbox_result, _data = self.llm.get_result(self.id, request_type="inbox")
        if inbox_result:
            self.current_observation = merge_observations(
                inbox_result,
                self.current_observation,
                self.cfg.robot.max_facts_per_observation,
            )
            if self.pending_inbox_sender_id is not None and self.pending_inbox_sender_tick is not None:
                self.observation_logger.log_comm_merge(
                    receiver_robot_id=self.id,
                    sender_robot_id=self.pending_inbox_sender_id,
                    sender_tick=self.pending_inbox_sender_tick,
                    receiver_tick=self.tick_count,
                    capture_epoch=self.capture_epoch,
                    merge_method="llm",
                    inbox_policy=self.pending_inbox_policy or "within_budget",
                )
            self.INBOX_PROCESS_PENDING = False
            self.pending_inbox_sender_id = None
            self.pending_inbox_sender_tick = None
            self.pending_inbox_policy = None
            return

        if self.cfg.wait_for_llm:
            return

        if self.tick_count - self.inbox_pending_since_tick <= self.cfg.inbox_timeout_ticks:
            return

        self.INBOX_PROCESS_PENDING = False
        self.pending_inbox_sender_id = None
        self.pending_inbox_sender_tick = None
        self.pending_inbox_policy = None

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
    (config, RNG, LLM manager, run logger), background loading, and the agent
    position-update hook. The only differences between the variants live in the
    subclasses (base class + the headless tick-pacing ``after_update``).
    """

    def _setup_environment(self, cfg: Config, background_path: Any) -> None:
        """Construct and register services shared across all agents.

        Args:
            cfg: Typed run configuration, exposed to agents as ``self.shared.cfg``.
            background_path: Optional path to a background image texture.
        """
        # self.shared is shared across all agents and the simulation.
        self.shared.cfg = cfg  # type: ignore[attr-defined]
        self.shared.rng = random.Random(cfg.simulation.seed)  # type: ignore[attr-defined]
        self.shared.api_manager = create_api_manager(cfg.llm.thread_workers, cfg)  # type: ignore[attr-defined]
        self.shared.observation_logger = ObservationLogger(  # type: ignore[attr-defined]
            on=LOG_RESULTS,
            empty_observation=cfg.robot.empty_observation,
            base_dir=cfg.simulation.output_dir or OUTPUT_DIR,
            external_config=cfg,
            save_robot_crops=cfg.simulation.save_robot_crops,
            save_comm_merge_history=cfg.robot.save_comm_merge_history,
        )
        self.shared.photo_frame_capture_state = {"tick": None, "robot_ids": set(), "saved": False}  # type: ignore[attr-defined]
        self.shared.api_manager.start()  # type: ignore[attr-defined]
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
    """Windowed simulation wiring shared LLM manager and run logger."""

    def __init__(
        self,
        vi_config: ViConfig | None = None,
        cfg: Config | None = None,
        background_path: Any = None,
    ) -> None:
        """Initialize simulation state and globally shared services.

        Args:
            vi_config: Violet simulation config object.
            cfg: Typed run configuration injected into agents.
            background_path: Optional path to background image texture.
        """
        super().__init__(vi_config)
        self._setup_environment(cfg, background_path)

    def after_update(self) -> None:
        """Render frame, then freeze between epochs when photo LLM sync is enabled."""
        super().after_update()
        freeze_until_llm_batch_done(self)


class EnvironmentHeadlessSimulation(_EnvironmentMixin, HeadlessSimulation):
    """Headless variant with the same shared services plus tick pacing."""

    def __init__(
        self,
        vi_config: ViConfig | None = None,
        cfg: Config | None = None,
        background_path: Any = None,
    ) -> None:
        super().__init__(vi_config)
        self._last_tick_time = time.perf_counter()
        self._setup_environment(cfg, background_path)

    def after_update(self) -> None:
        """Freeze for photo LLM batch completion, then pace ticks to simulation.fps."""
        freeze_until_llm_batch_done(self)
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
    cfg = load_config(config_path)

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


if __name__ == "__main__":
    main()
