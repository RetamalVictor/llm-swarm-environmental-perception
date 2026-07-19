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
    import swarm_perception.main as main
    from swarm_perception.utils.paths import ASSETS_DIR

    main.configure_runtime_mode(True)
    sim = main.EnvironmentHeadlessSimulation(
        vi_config=main.build_vi_config(cfg),
        cfg=cfg,
        background_path=ASSETS_DIR / cfg.simulation.background_image,
    )
    sim.batch_spawn_agents(
        cfg.simulation.num_of_robots,
        main.Robot,
        images=[str(ASSETS_DIR / cfg.simulation.robot_image)],
    )
    sim.run()
    sim.shared.run_logger.finalize()
    return sim
