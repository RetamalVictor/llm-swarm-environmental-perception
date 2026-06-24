"""Swarm simulation entrypoint implementing perception and communication loops.

This module corresponds to the core methodology in the capstone paper:
robots capture local visual patches, maintain a textual knowledge base, exchange
messages with nearby peers, and integrate information through LLM-assisted
or deterministic merging.
"""

from vi import Agent, Config, HeadlessSimulation, Simulation, Window
import pygame as pg
from swarm_perception.camera_sensor import CameraSensor
from swarm_perception.observation_logger import ObservationLogger
from swarm_perception.actuator import Actuator
from swarm_perception.llm.factory import create_api_manager
import random
import os
import time
from swarm_perception.utils.paths import ASSETS_DIR, LOG_DIR, OUTPUT_DIR
from swarm_perception.utils.config import SwarmConfig
import sys
import re
from collections import deque
from typing import Any
## SETUP LOGS
from swarm_perception.utils.logging_config import setup_logging
from swarm_perception.utils.sim_log import comm_log, config_log, llm_log, log_run_banner, sim_log

setup_logging()

if len(sys.argv) > 1:
    config = SwarmConfig(sys.argv[1]).load_config()
else:
    config = SwarmConfig().load_config()

## CONSTANTS SIMULATION
SEED = config.simulation.seed
WIDTH = config.simulation.width
HEIGHT = config.simulation.height
FPS = config.simulation.fps
BACKGROUND_IMAGE = config.simulation.background_image
ROBOT_IMAGE = config.simulation.robot_image
NUM_OF_ROBOTS = config.simulation.num_of_robots
RUN_LENGTH = config.simulation.run_length

## CONSTANTS ROBOT
LINEAR_SPEED = config.robot.linear_speed
ANGULAR_VELOCITY = config.robot.angular_velocity
COVERAGE_SIDE = config.robot.coverage_side
NEIGHBOR_RADIUS = config.robot.neighbor_radius
CAPTURE_FREQUENCY = config.robot.capture_frequency
LOG_RESULTS = True
COMMUNICATION = config.robot.communication
EMPTY_OBSERVATION = config.robot.empty_observation
SELF_LEARNING = config.robot.self_learning 
USE_LLM_INBOX_SYNTHESIS = bool(
    getattr(config.robot, "use_llm_inbox_synthesis", not getattr(config.robot, "no_inbox_synthesis", False))
)
RUN_OUTPUT_DIR = getattr(config.simulation, "output_dir", OUTPUT_DIR)
SAVE_PHOTO_FRAMES = getattr(config.simulation, "save_photo_frames", False)
SAVE_ROBOT_CROPS = getattr(config.simulation, "save_robot_crops", False)
SAVE_COMM_MERGE_HISTORY = getattr(config.robot, "save_comm_merge_history", False)
HEADLESS = bool(getattr(config.simulation, "headless", False))

## CONSTANTS LLM
NUM_WORKERS = config.llm.thread_workers

PHOTO_TICKS = int(CAPTURE_FREQUENCY * FPS)
MAX_FACTS_PER_OBSERVATION = getattr(config.robot, "max_facts_per_observation", 40)
PHOTO_TIMEOUT_TICKS = getattr(config.robot, "photo_timeout_ticks", PHOTO_TICKS * 2)
INBOX_TIMEOUT_TICKS = getattr(config.robot, "inbox_timeout_ticks", PHOTO_TICKS)
SIM_DURATION = RUN_LENGTH * PHOTO_TICKS
MAX_INBOX_MERGES_PER_EPOCH = max(0, int(getattr(config.robot, "max_inbox_merges_per_epoch", 1)))
INBOX_MERGE_AFTER_BUDGET = str(getattr(config.robot, "inbox_merge_after_budget", "drop")).strip().lower()
WAIT_FOR_LLM = bool(
    getattr(config.robot, "wait_for_llm", False)
    or getattr(config.robot, "wait_for_photo_llm", False)
)
random.seed(SEED)


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
    for robot in iter_robots(simulation):
        if robot.PHOTO_RESULT_PENDING:  # type: ignore[attr-defined]
            return True
        if USE_LLM_INBOX_SYNTHESIS and robot.INBOX_PROCESS_PENDING:  # type: ignore[attr-defined]
            return True
    return False


def freeze_until_llm_batch_done(simulation: HeadlessSimulation) -> None:
    """Block between ticks until all in-flight LLM requests for this epoch finish."""
    if not WAIT_FOR_LLM or not any_robot_llm_pending(simulation):
        return
    sim_log.info("epoch pause │ waiting for LLM batch (photo + inbox)")
    while any_robot_llm_pending(simulation):
        for robot in iter_robots(simulation):
            robot.poll_photo_result()  # type: ignore[attr-defined]
            robot.poll_inbox_result()  # type: ignore[attr-defined]
        time.sleep(0.02)
    sim_log.info("epoch resume │ LLM batch complete")


class Robot(Agent):
    """Swarm robot agent with camera sensing and asynchronous LLM integration.

    Each robot maintains a local textual knowledge base (`current_observation`)
    and updates it through two pathways:

    - individual learning: camera capture + photo interpretation
    - social learning: peer message intake + inbox synthesis
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
        self.sense_square = COVERAGE_SIDE

        # spwaning coordinates
        self.pos.x = random.uniform(COVERAGE_SIDE,WIDTH-COVERAGE_SIDE)
        self.pos.y = random.uniform(COVERAGE_SIDE,HEIGHT-COVERAGE_SIDE)

        self.sensor = CameraSensor(
            agent=self,
            coverage_side=self.sense_square,
            background_image=BACKGROUND_IMAGE,
            sensing_radius=NEIGHBOR_RADIUS,
        )
        self.actuator = Actuator(self)
        self.llm = self.shared.api_manager # type: ignore
        
        # photo taking related
        self.photo_tick_counter = 0
        self.is_taking_photo = False
        self.flash_duration = 15 # ticks
        self.flash_counter = 0

        # agent's memory
        self.current_observation = EMPTY_OBSERVATION
        self.EMPTY_OBSERVATION = True
        self.observation_logger = self.shared.observation_logger # type: ignore
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
            comm_log.debug(
                "message accepted │ robot=%s from=%s tick=%s",
                self.id,
                sender_id,
                message_tick,
            )
            return

        _, existing_tick, existing_sender_id = self.inbox_queue[0]
        if message_tick > existing_tick:
            self.inbox_queue[0] = incoming_payload
            comm_log.debug(
                "message replaced │ robot=%s from=%s→%s tick=%s→%s",
                self.id,
                existing_sender_id,
                sender_id,
                existing_tick,
                message_tick,
            )
        else:
            comm_log.debug(
                "message ignored (stale) │ robot=%s from=%s tick=%s pending_tick=%s",
                self.id,
                sender_id,
                message_tick,
                existing_tick,
            )

    def exchange_with_neighbors(self) -> None:
        """Broadcast current observation to peers currently in communication range."""
        if self.EMPTY_OBSERVATION or not COMMUNICATION:
            return
        for neighbor, _ in self.in_proximity_accuracy():
            neighbor.receive_peer_message(  # type: ignore
                self.current_observation,
                self.tick_count,
                int(self.id),
            )
            comm_log.debug(
                "message sent │ robot=%s to=%s tick=%s",
                self.id,
                neighbor.id,
                self.tick_count,
            )

    def update(self) -> None:
        """Execute one full agent tick for sensing, sharing, and result polling.

        The tick loop performs:
        1) sensing overlay and periodic photo capture
        2) continuous neighbor broadcast while in proximity
        3) inbox merge scheduling under per-epoch budget rules
        4) async polling for photo and inbox LLM results with timeout fallback
        """
        self.tick_count += 1
        # show the sense rectangle
        if self.is_taking_photo:
            self.sensor.show_outline(color=(255, 255, 0)) # Yellow flash
            self.flash_counter += 1
            if self.flash_counter >= self.flash_duration:
                self.is_taking_photo = False
                self.flash_counter = 0
        else:
            self.sensor.show_outline()

        # take a photo every capture_frequency seconds (real time when fps > 0)
        self.photo_tick_counter += 1
        if self.photo_tick_counter >= PHOTO_TICKS:
            # state variables
            self.photo_tick_counter = 0
            self.is_taking_photo = True
            self.capture_epoch += 1
            self.inbox_merges_this_epoch = 0

            if SAVE_PHOTO_FRAMES:
                frame_state = self.shared.photo_frame_capture_state # type: ignore
                if frame_state["tick"] != self.tick_count:
                    frame_state["tick"] = self.tick_count
                    frame_state["robot_ids"] = set()
                    frame_state["saved"] = False

                frame_state["robot_ids"].add(int(self.id))
                if not frame_state["saved"] and len(frame_state["robot_ids"]) >= NUM_OF_ROBOTS:
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
            self.llm.submit_photo_request(self.id, image, self.current_observation, SELF_LEARNING)
            self.PHOTO_RESULT_PENDING = True
            self.photo_pending_since_tick = self.tick_count
            self.observation_logger.log_progress_snapshot(
                robot_id=self.id,
                observation=self.current_observation,
            )
        
        # Exchange messages whenever peers are currently nearby (not only photo events).
        self.exchange_with_neighbors()

        # process inbox queue (one message at a time)
        # print(f"[tick {self.tick_count}] robot {self.id} inbox_queue size: {len(self.inbox_queue)}")
        inbox_budget_available = self.inbox_merges_this_epoch < MAX_INBOX_MERGES_PER_EPOCH
        if self.inbox_queue and not self.INBOX_PROCESS_PENDING and inbox_budget_available:
            next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
            if not USE_LLM_INBOX_SYNTHESIS:
                self.current_observation = merge_observations(
                    next_inbox,
                    self.current_observation,
                    MAX_FACTS_PER_OBSERVATION,
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
                comm_log.info(
                    "inbox merged (deterministic) │ robot=%s from=%s tick=%s",
                    self.id,
                    incoming_sender_id,
                    incoming_tick,
                )
            else:
                self.llm.submit_inbox_request(self.id, self.current_observation, next_inbox)
                self.INBOX_PROCESS_PENDING = True
                self.inbox_pending_since_tick = self.tick_count
                self.pending_inbox_sender_id = incoming_sender_id
                self.pending_inbox_sender_tick = incoming_tick
                self.pending_inbox_policy = "within_budget"
                self.inbox_merges_this_epoch += 1
                llm_log.info(
                    "inbox synthesis queued │ robot=%s from=%s tick=%s",
                    self.id,
                    incoming_sender_id,
                    incoming_tick,
                )
        elif self.inbox_queue and not self.INBOX_PROCESS_PENDING and not inbox_budget_available:
            if INBOX_MERGE_AFTER_BUDGET == "drop":
                comm_log.debug(
                    "inbox skipped (budget) │ robot=%s policy=drop budget=%s",
                    self.id,
                    MAX_INBOX_MERGES_PER_EPOCH,
                )
            elif INBOX_MERGE_AFTER_BUDGET == "deterministic":
                next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
                self.current_observation = merge_observations(
                    next_inbox,
                    self.current_observation,
                    MAX_FACTS_PER_OBSERVATION,
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
                comm_log.info(
                    "inbox merged after budget │ robot=%s from=%s tick=%s",
                    self.id,
                    incoming_sender_id,
                    incoming_tick,
                )
            elif INBOX_MERGE_AFTER_BUDGET == "llm":
                next_inbox, incoming_tick, incoming_sender_id = self.inbox_queue.popleft()
                self.llm.submit_inbox_request(self.id, self.current_observation, next_inbox)
                self.INBOX_PROCESS_PENDING = True
                self.inbox_pending_since_tick = self.tick_count
                self.pending_inbox_sender_id = incoming_sender_id
                self.pending_inbox_sender_tick = incoming_tick
                self.pending_inbox_policy = "llm_after_budget"
                llm_log.info(
                    "inbox synthesis after budget │ robot=%s from=%s tick=%s",
                    self.id,
                    incoming_sender_id,
                    incoming_tick,
                )
            else:
                sim_log.warning(
                    "invalid inbox_merge_after_budget='%s' │ robot=%s fallback=drop",
                    INBOX_MERGE_AFTER_BUDGET,
                    self.id,
                )
            
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
                MAX_FACTS_PER_OBSERVATION,
            )
            self.PHOTO_RESULT_PENDING = False
            self.EMPTY_OBSERVATION = False
            llm_log.info(
                "photo result applied │ robot=%s chars=%s",
                self.id,
                len(self.current_observation),
            )
            self.observation_logger.log_observation(self.id, self.current_observation)
            return

        if WAIT_FOR_LLM:
            return

        if self.tick_count - self.photo_pending_since_tick <= PHOTO_TIMEOUT_TICKS:
            return

        self.PHOTO_RESULT_PENDING = False
        llm_log.warning(
            "photo timeout │ robot=%s waited_ticks=%s",
            self.id,
            self.tick_count - self.photo_pending_since_tick,
        )

    def poll_inbox_result(self) -> None:
        """Apply a completed inbox LLM result or clear pending state on timeout."""
        if not self.INBOX_PROCESS_PENDING or not USE_LLM_INBOX_SYNTHESIS:
            return

        inbox_result, _data = self.llm.get_result(self.id, request_type="inbox")
        if inbox_result:
            self.current_observation = merge_observations(
                inbox_result,
                self.current_observation,
                MAX_FACTS_PER_OBSERVATION,
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
            llm_log.info(
                "inbox result applied │ robot=%s chars=%s",
                self.id,
                len(self.current_observation),
            )
            self.INBOX_PROCESS_PENDING = False
            self.pending_inbox_sender_id = None
            self.pending_inbox_sender_tick = None
            self.pending_inbox_policy = None
            return

        if WAIT_FOR_LLM:
            return

        if self.tick_count - self.inbox_pending_since_tick <= INBOX_TIMEOUT_TICKS:
            return

        self.INBOX_PROCESS_PENDING = False
        self.pending_inbox_sender_id = None
        self.pending_inbox_sender_tick = None
        self.pending_inbox_policy = None
        llm_log.warning(
            "inbox timeout │ robot=%s waited_ticks=%s",
            self.id,
            self.tick_count - self.inbox_pending_since_tick,
        )

    def get_velocities(self) -> tuple[float, float]:
        """Return movement command for correlated random walk with edge avoidance.

        Returns:
            Tuple of ``(linear_speed, angular_velocity)`` for this tick.
        """
        linear_speed = LINEAR_SPEED # default
        angular_velocity = 0.0 # default 
        
        if self.sensor.detect_edges():
            angular_velocity = ANGULAR_VELOCITY # turn back
        return linear_speed, angular_velocity
    
class EnvironmentSimulation(Simulation):
    """Simulation container wiring shared LLM manager and run logger."""

    def __init__(
        self,
        vi_config: Config | None = None,
        background_path: Any = None,
        external_config: Any = None,
    ) -> None:
        """Initialize simulation state and globally shared services.

        Args:
            vi_config: Violet simulation config object.
            background_path: Optional path to background image texture.
            external_config: Loaded config namespace forwarded to dependencies.
        """
        super().__init__(vi_config)

        # self.shared is shared across all agents and the simulation
        # since server is single thread, use one thread for now
        self.shared.api_manager = create_api_manager(NUM_WORKERS, external_config) # type: ignore
        self.shared.observation_logger = ObservationLogger( # type: ignore
            on=LOG_RESULTS,
            empty_observation=EMPTY_OBSERVATION,
            base_dir=RUN_OUTPUT_DIR,
            external_config=external_config,
            save_robot_crops=SAVE_ROBOT_CROPS,
            save_comm_merge_history=SAVE_COMM_MERGE_HISTORY,
        )
        self.shared.photo_frame_capture_state = {"tick": None, "robot_ids": set(), "saved": False} # type: ignore
        self.shared.api_manager.start() # type: ignore

        # Change background image if a path is provided.
        if background_path:
            try:
                size = self.config.window.as_tuple()
                background_image = pg.image.load(background_path)
                if pg.display.get_surface() is not None:
                    background_image = background_image.convert()
                self._background = pg.transform.scale(background_image, size)
            except pg.error as e:
                sim_log.warning("background image not loaded │ %s", e)
        
    def _HeadlessSimulation__update_positions(self) -> None:
        """Update all agent positions using each robot's actuator command."""
        for sprite in self._agents.sprites():
            agent: Agent = sprite  # type: ignore
            
            linear_speed, angular_velocity = agent.get_velocities() # type: ignore
            agent.actuator.update(linear_speed, angular_velocity) # type: ignore

    def after_update(self) -> None:
        """Render frame, then freeze between epochs when photo LLM sync is enabled."""
        super().after_update()
        freeze_until_llm_batch_done(self)


class EnvironmentHeadlessSimulation(HeadlessSimulation):
    """Headless variant with the same shared services and update logic."""

    def __init__(
        self,
        vi_config: Config | None = None,
        background_path: Any = None,
        external_config: Any = None,
    ) -> None:
        super().__init__(vi_config)
        self._last_tick_time = time.perf_counter()

        self.shared.api_manager = create_api_manager(NUM_WORKERS, external_config) # type: ignore
        self.shared.observation_logger = ObservationLogger( # type: ignore
            on=LOG_RESULTS,
            empty_observation=EMPTY_OBSERVATION,
            base_dir=RUN_OUTPUT_DIR,
            external_config=external_config,
            save_robot_crops=SAVE_ROBOT_CROPS,
            save_comm_merge_history=SAVE_COMM_MERGE_HISTORY,
        )
        self.shared.photo_frame_capture_state = {"tick": None, "robot_ids": set(), "saved": False} # type: ignore
        self.shared.api_manager.start() # type: ignore

        if background_path:
            try:
                size = self.config.window.as_tuple()
                background_image = pg.image.load(background_path)
                if pg.display.get_surface() is not None:
                    background_image = background_image.convert()
                self._background = pg.transform.scale(background_image, size)
            except pg.error as e:
                sim_log.warning("background image not loaded │ %s", e)

    def _HeadlessSimulation__update_positions(self) -> None:
        """Update all agent positions using each robot's actuator command."""
        for sprite in self._agents.sprites():
            agent: Agent = sprite  # type: ignore

            linear_speed, angular_velocity = agent.get_velocities() # type: ignore
            agent.actuator.update(linear_speed, angular_velocity) # type: ignore

    def after_update(self) -> None:
        """Freeze for photo LLM batch completion, then pace ticks to simulation.fps."""
        freeze_until_llm_batch_done(self)
        if FPS <= 0:
            return
        tick_interval = 1.0 / FPS
        elapsed = time.perf_counter() - self._last_tick_time
        if elapsed < tick_interval:
            time.sleep(tick_interval - elapsed)
        self._last_tick_time = time.perf_counter()


def configure_runtime_mode() -> None:
    """Configure SDL for true no-window operation before simulation startup."""
    if not HEADLESS:
        return
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    config_log.info("SDL drivers │ headless dummy video/audio")


vi_config = Config(
    window=Window(WIDTH,HEIGHT),
    movement_speed=1.0,
    seed=SEED,
    image_rotation=True,
    radius=NEIGHBOR_RADIUS,
    fps_limit=FPS,
    duration=SIM_DURATION,
)

def main() -> None:
    """Console entrypoint. Config is still loaded at import time (see module top);
    removing import-time globals is deferred to M0 Branch 2."""
    configure_runtime_mode()
    log_run_banner(
        config=config,
        headless=HEADLESS,
        num_robots=NUM_OF_ROBOTS,
        fps=FPS,
        capture_frequency=CAPTURE_FREQUENCY,
        photo_ticks=PHOTO_TICKS,
        sim_duration=SIM_DURATION,
        communication=COMMUNICATION,
        llm_provider=getattr(config.llm, "provider", "gemini"),
        llm_model=config.llm.model_name,
        llm_workers=NUM_WORKERS,
        wait_for_llm=WAIT_FOR_LLM,
    )
    sim_cls = EnvironmentHeadlessSimulation if HEADLESS else EnvironmentSimulation
    sim = sim_cls(background_path=ASSETS_DIR / BACKGROUND_IMAGE, vi_config=vi_config, external_config=config)
    sim.batch_spawn_agents(NUM_OF_ROBOTS, Robot, images=[str(ASSETS_DIR / ROBOT_IMAGE)])
    sim_log.info("simulation started")
    sim.run()
    sim_log.info("simulation finished")


if __name__ == "__main__":
    main()