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
from pathlib import Path
from typing import Any

import pygame as pg
from vi import Agent, Config as ViConfig, HeadlessSimulation, Simulation, Window

from swarm_perception.io.run_logger import RunLogger
from swarm_perception.sim.robot import Robot
from swarm_perception.config import Config, ConfigError, load_config
from swarm_perception.utils.paths import ASSETS_DIR
from swarm_perception.world.background import Background


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
