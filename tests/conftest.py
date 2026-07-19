"""Shared test helpers: smoke config loading and a headless run harness."""

import dataclasses
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO / "tests" / "data" / "smoke.yaml"


def load_smoke_config(output_dir: Path):
    """Load the smoke config redirected into ``output_dir``."""
    from swarm_perception.config import load_config

    cfg = load_config(SMOKE_CONFIG)
    sim_cfg = dataclasses.replace(cfg.simulation, output_dir=str(output_dir))
    return dataclasses.replace(cfg, simulation=sim_cfg)


def run_headless(cfg):
    """Run one finalized headless simulation; the sim is natively offline."""
    from swarm_perception.sim.engine import Simulation, configure_runtime_mode
    from swarm_perception.sim.robot import Robot
    from swarm_perception.utils.paths import ASSETS_DIR

    configure_runtime_mode(True)
    sim = Simulation(cfg, headless=True)
    sim.batch_spawn_agents(
        cfg.simulation.num_of_robots,
        Robot,
        images=[str(ASSETS_DIR / cfg.simulation.robot_image)],
    )
    sim.run()
    sim.run_logger.finalize()
    return sim
