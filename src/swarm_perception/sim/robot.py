"""Robot agent: policy-driven motion, graded memory, budgeted exchange.

Robots move over a shared background image under the configured movement
policy (``robot.movement_policy``, see :mod:`swarm_perception.sim.policies`).
Capture itself is engine-driven: at every capture epoch the engine embeds all
robots' crops as one batch and hands each robot its
:class:`~swarm_perception.fusion.memory.MemoryRecord` through
:meth:`Robot.incorporate_capture`.

Each robot holds the graded memory ``M = (R, S)``: ``R`` is a bounded
:class:`~swarm_perception.fusion.memory.SetMemory` of embedding records and
``S`` a :class:`~swarm_perception.sim.residue.VisitationResidue` of
fully-observed grid cells. Eviction from ``R`` is demotion, not deletion —
every record a robot ever incorporates marks its rect's cells in ``S`` first.

Exchange runs over the budgeted channel (:mod:`swarm_perception.sim.channel`)
through the per-robot :class:`~swarm_perception.sim.exchange.ExchangeUnit`:
at most ``comms.k`` records per broadcast, quantized once at the source,
subject to seeded packet drop and optional delivery delay, received through a
bounded FIFO inbox with a per-epoch merge budget.
"""

from __future__ import annotations

import random
from typing import Any

from pygame import Surface
from vi import Agent, HeadlessSimulation

from swarm_perception.camera_sensor import CameraSensor
from swarm_perception.config import Config
from swarm_perception.fusion.memory import MemoryRecord, SetMemory
from swarm_perception.io.run_logger import RunLogger
from swarm_perception.sim.actuator import Actuator
from swarm_perception.sim.channel import Channel, Message
from swarm_perception.sim.exchange import ExchangeUnit
from swarm_perception.sim.policies import StepContext, build_policy
from swarm_perception.sim.residue import VisitationResidue


class Robot(Agent):
    """Swarm robot agent with graded memory and budgeted record exchange.

    Configuration is read from the injected ``self.shared.cfg`` (typed
    :class:`~swarm_perception.config.Config`); the shared per-run
    :class:`~swarm_perception.sim.channel.Channel` provides the seeded
    packet-drop stream.
    """

    # Heading in degrees (pygame convention), owned and written by the
    # Actuator; declared here so policy wiring can read it with typing.
    current_angle: float

    def __init__(
        self,
        images: list[Surface],
        simulation: HeadlessSimulation,
        pos: Any = None,
        move: Any = None,
    ) -> None:
        """Create one robot with random spawn, sensor, actuator, and memory.

        Args:
            images: Loaded sprite surfaces passed to violet ``Agent`` base class
                (violet's ``batch_spawn_agents`` loads the image paths first).
            simulation: Parent simulation object.
            pos: Optional initial position (overridden by random spawn).
            move: Optional initial movement vector.
        """
        super().__init__(images, simulation, pos, move)
        cfg: Config = self.shared.cfg  # type: ignore[attr-defined]
        self.cfg = cfg
        rng: random.Random = self.shared.rng  # type: ignore[attr-defined]

        # spawning coordinates
        self.pos.x = rng.uniform(cfg.robot.coverage_side, cfg.simulation.width - cfg.robot.coverage_side)
        self.pos.y = rng.uniform(cfg.robot.coverage_side, cfg.simulation.height - cfg.robot.coverage_side)

        self.sensor = CameraSensor(
            agent=self,
            coverage_side=cfg.robot.coverage_side,
            background=self.shared.background,  # type: ignore[attr-defined]
            sensing_radius=cfg.robot.neighbor_radius,
        )
        self.actuator = Actuator(self, rng=rng)
        # One policy instance per robot: policies may carry per-robot state
        # (e.g. a levy flight in progress). Construction draws no RNG values,
        # so the seed-to-spawn mapping is identical for every policy choice.
        self.rng = rng
        self.policy = build_policy(cfg.robot)
        self.run_logger: RunLogger = self.shared.run_logger  # type: ignore[attr-defined]
        self.channel: Channel = self.shared.channel  # type: ignore[attr-defined]

        # photo flash state (windowed visualization only)
        self.is_taking_photo = False
        self.flash_duration = 15  # ticks
        self.flash_counter = 0
        self.capture_epoch = 0
        self.tick_count = 0

        # graded memory M = (R, S): bounded record set plus visitation residue
        self.memory = SetMemory(
            (), tau_dedup=cfg.fusion.tau_dedup, memory_cap=cfg.fusion.memory_cap
        )
        self.residue = VisitationResidue(
            cfg.simulation.width, cfg.simulation.height, cfg.robot.coverage_side
        )
        # Channel endpoint: inbox, delay queue, payload cache, merge budget.
        self.exchange = ExchangeUnit(self)

    # ------------------------------------------------------------- capture

    def incorporate_capture(self, record: MemoryRecord, tick: int) -> None:
        """Incorporate this robot's own capture record for one epoch.

        Called by the engine's epoch hook after the whole epoch was embedded
        as one batch. Logs the capture, marks the residue (before the memory
        merge decides the record's fate), merges the record, and logs the
        retained memory keys.
        """
        epoch = record.key[0]
        self.run_logger.log_capture(
            tick=tick,
            epoch=epoch,
            robot=int(self.id),
            bbox=record.crop_bbox,
            pos=record.pos,
        )
        if self.cfg.simulation.save_robot_crops:
            image, _ = self.sensor.take_photo()  # same rect: same pos and rounding
            self.run_logger.save_crop(
                robot_id=int(self.id), tick=tick, epoch=epoch, image=image
            )
        self.residue.mark_rect(record.crop_bbox)
        self.memory.add(record)
        self.run_logger.log_memory(
            tick=tick, epoch=epoch, robot=int(self.id), keys=self.memory.keys()
        )

    # ------------------------------------------------------------- exchange

    def exchange_with_neighbors(self) -> None:
        """Broadcast one budgeted message to every peer currently in range."""
        if not self.cfg.comms.enabled or len(self.memory) == 0:
            return
        neighbors = sorted(
            (agent for agent, _ in self.in_proximity_accuracy()),
            key=lambda agent: int(agent.id),
        )
        if neighbors:
            self.exchange.broadcast(neighbors)  # type: ignore[arg-type]

    def deliver_message(self, message: Message) -> None:
        """Accept one transmitted message into this robot's channel endpoint."""
        self.exchange.deliver(message)

    # ----------------------------------------------------------------- tick

    def update(self) -> None:
        """Execute one agent tick: flash state, epoch rollover, and exchange.

        The tick loop performs:
        1) sensing overlay and capture-epoch rollover (the capture itself is
           engine-driven after all agents updated this tick)
        2) delivery of delay-matured messages into the FIFO inbox
        3) continuous neighbor broadcast while in proximity
        4) at most one budgeted inbox merge
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

        # Capture-epoch rollover: the engine's after-update hook performs the
        # actual capture at this same tick (positions do not move in between).
        if cfg.photo_ticks > 0 and self.tick_count % cfg.photo_ticks == 0:
            self.is_taking_photo = True
            self.capture_epoch += 1
            self.exchange.merges_this_epoch = 0

        self.exchange.collect_due_pending()
        # Exchange records whenever peers are currently nearby (not only photo events).
        self.exchange_with_neighbors()
        self.exchange.process_inbox()

    def get_velocities(self) -> tuple[float, float]:
        """Delegate this tick's movement decision to the configured policy.

        Returns:
            Tuple of ``(linear_speed, angular_velocity)`` for this tick.
        """
        area = self._area
        return self.policy.step(
            StepContext(
                pos=(self.pos.x, self.pos.y),
                heading=self.current_angle,
                edge=self.sensor.detect_edges(),
                bounds=(area.left, area.top, area.right, area.bottom),
                cfg=self.cfg.robot,
                rng=self.rng,
            )
        )
