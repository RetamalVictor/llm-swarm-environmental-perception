"""Simulation engine: one public class over violet's two run modes.

:class:`Simulation` is the only public entry to the runtime. It selects the
violet base engine (windowed ``vi.Simulation`` or ``vi.HeadlessSimulation``)
from its ``headless`` flag and wires the shared per-run services (typed
config, seeded RNG, load-once background, :class:`RunLogger`) that every
agent reads through ``self.shared``.

Headless runs execute flat-out: ``simulation.fps`` only derives
``photo_ticks`` (captures per simulated second) and never paces wall-clock
time. Windowed runs keep violet's own display clock for watchability.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import pygame as pg
from vi import Agent, Config as ViConfig, Window
from vi import HeadlessSimulation as ViHeadlessSimulation
from vi import Simulation as ViSimulation
from vi.metrics import Metrics

from swarm_perception.config import Config
from swarm_perception.io.run_logger import RunLogger
from swarm_perception.utils.paths import ASSETS_DIR
from swarm_perception.world.background import Background


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


class _EngineCore:
    """Service wiring mixed into both violet engine variants."""

    def __init__(
        self,
        vi_config: ViConfig,
        cfg: Config,
        background_path: Any,
        run_dir: Path | str | None = None,
    ) -> None:
        """Initialize the violet base engine, then the shared services.

        Args:
            vi_config: Violet simulation config object.
            cfg: Typed run configuration, exposed to agents as ``self.shared.cfg``.
            background_path: Path to the background image texture.
            run_dir: Run output directory; defaults to
                :func:`swarm_perception.io.run_logger.resolve_run_dir` on ``cfg``.
        """
        super().__init__(vi_config)  # continues into the violet base class
        # self.shared is shared across all agents and the simulation.
        self.shared.cfg = cfg  # type: ignore[attr-defined]
        self.shared.rng = random.Random(cfg.simulation.seed)  # type: ignore[attr-defined]
        # Load-once world image; every robot crops views of this one array.
        self.shared.background = Background(background_path)  # type: ignore[attr-defined]
        self.shared.run_logger = RunLogger(cfg, run_dir=run_dir)  # type: ignore[attr-defined]
        self.shared.photo_frame_capture_state = {"tick": None, "robot_ids": set(), "saved": False}  # type: ignore[attr-defined]
        self._load_display_background(background_path)

    def _load_display_background(self, background_path: Any) -> None:
        """Load and scale the display background when a path is provided."""
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

    # The exact name below is load-bearing. Violet's tick() calls
    # ``self.__update_positions()`` from inside its ``HeadlessSimulation``
    # class body, which Python name-mangles at compile time to
    # ``self._HeadlessSimulation__update_positions()``. Overriding the hook
    # therefore requires this literal mangled name; anything else would leave
    # violet's default ``change_position`` update in effect.
    def _HeadlessSimulation__update_positions(self) -> None:
        """Update all agent positions using each robot's actuator command."""
        for sprite in self._agents.sprites():  # type: ignore[attr-defined]
            agent: Agent = sprite  # type: ignore
            linear_speed, angular_velocity = agent.get_velocities()  # type: ignore
            agent.actuator.update(linear_speed, angular_velocity)  # type: ignore


class _WindowedEngine(_EngineCore, ViSimulation):
    """Windowed violet engine with the shared service wiring."""


class _HeadlessEngine(_EngineCore, ViHeadlessSimulation):
    """Headless violet engine with the shared service wiring.

    No per-tick pacing: headless benchmark runs execute as fast as the host
    allows. ``simulation.fps`` only enters the tick-count derivations.
    """


class Simulation:
    """One swarm simulation run, windowed or headless.

    The violet base engine is selected internally from ``headless``; callers
    only ever construct this class. Shared per-run services are reachable via
    :attr:`shared` (``cfg``, ``rng``, ``background``, ``run_logger``).
    """

    def __init__(
        self,
        cfg: Config,
        *,
        headless: bool | None = None,
        background_path: Any = None,
        run_dir: Path | str | None = None,
    ) -> None:
        """Build the engine for one run.

        Args:
            cfg: Typed run configuration.
            headless: Run without a window. Defaults to
                ``cfg.simulation.headless``.
            background_path: Background image path. Defaults to
                ``ASSETS_DIR / cfg.simulation.background_image``.
            run_dir: Run output directory; defaults to
                :func:`swarm_perception.io.run_logger.resolve_run_dir` on ``cfg``.
        """
        if headless is None:
            headless = cfg.simulation.headless
        if background_path is None:
            background_path = ASSETS_DIR / cfg.simulation.background_image
        engine_cls = _HeadlessEngine if headless else _WindowedEngine
        self._engine = engine_cls(build_vi_config(cfg), cfg, background_path, run_dir)

    @property
    def shared(self) -> Any:
        """Violet shared state carrying the per-run services."""
        return self._engine.shared

    @property
    def run_logger(self) -> RunLogger:
        """The run's event and artifact logger."""
        return self._engine.shared.run_logger  # type: ignore[attr-defined]

    def batch_spawn_agents(
        self, count: int, agent_class: type[Agent], images: list[str]
    ) -> Simulation:
        """Spawn ``count`` agents of ``agent_class`` into the run."""
        self._engine.batch_spawn_agents(count, agent_class, images)
        return self

    def run(self) -> Metrics:
        """Run the simulation to completion and return violet's metrics."""
        return self._engine.run()
