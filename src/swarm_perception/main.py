"""Console entrypoint wiring config, engine, and robots for one run."""

from __future__ import annotations

import sys

from swarm_perception.config import ConfigError, load_config
from swarm_perception.sim.engine import Simulation, configure_runtime_mode
from swarm_perception.sim.robot import Robot
from swarm_perception.utils.paths import ASSETS_DIR


def main() -> None:
    """Console entrypoint: load config, wire the simulation, and run it."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        cfg = load_config(config_path)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None

    configure_runtime_mode(cfg.simulation.headless)
    sim = Simulation(cfg)
    sim.batch_spawn_agents(
        cfg.simulation.num_of_robots,
        Robot,
        images=[str(ASSETS_DIR / cfg.simulation.robot_image)],
    )
    sim.run()
    sim.run_logger.finalize()


if __name__ == "__main__":
    main()
