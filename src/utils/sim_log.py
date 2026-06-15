"""Categorical logging helpers for swarm simulation runs."""

from __future__ import annotations

import logging
from typing import Any

_BANNER = "═" * 56

config_log = logging.getLogger("swarm.config")
sim_log = logging.getLogger("swarm.sim")
comm_log = logging.getLogger("swarm.comm")
llm_log = logging.getLogger("swarm.llm")


def log_run_banner(
    *,
    config: Any,
    headless: bool,
    num_robots: int,
    fps: int,
    capture_frequency: float,
    photo_ticks: int,
    sim_duration: int,
    communication: bool,
    llm_provider: str,
    llm_model: str,
    llm_workers: int,
    wait_for_photo_llm: bool = False,
) -> None:
    """Print a structured startup summary for the current run."""
    run_name = getattr(config.config, "name", "unnamed")
    mode = "headless" if headless else "windowed"

    sim_log.info(_BANNER)
    sim_log.info("  SWARM SIMULATION")
    sim_log.info(_BANNER)

    config_log.info("run name          %s", run_name)
    config_log.info("display mode      %s", mode)
    config_log.info("robots            %s", num_robots)
    config_log.info("world size        %sx%s", config.simulation.width, config.simulation.height)
    config_log.info("seed              %s", config.simulation.seed)
    config_log.info("run length        %s capture epochs", config.simulation.run_length)

    config_log.info("fps               %s ticks/sec (%s)", fps, "uncapped" if fps <= 0 else "real-time")
    config_log.info(
        "capture interval  %ss (%s ticks)", capture_frequency, photo_ticks
    )
    config_log.info("sim duration      %s ticks", sim_duration)
    config_log.info("communication     %s", communication)
    config_log.info("self learning     %s", config.robot.self_learning)

    config_log.info(
        "llm provider      %s", llm_provider
    )
    config_log.info("llm model         %s", llm_model)
    config_log.info("llm workers       %s", llm_workers)
    config_log.info(
        "photo llm sync    %s",
        "epoch barrier (freeze)" if wait_for_photo_llm else "async (timeout)",
    )
    if getattr(config.llm, "base_url", None):
        config_log.info("llm base url       %s", config.llm.base_url)

    sim_log.info(_BANNER)
