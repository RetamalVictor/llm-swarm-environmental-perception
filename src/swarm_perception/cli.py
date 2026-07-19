"""Command-line entrypoint for one simulation run.

Loads the YAML config, applies the override flags, and runs the engine.
Installed as the ``swarm-run`` console script; ``python -m swarm_perception``
reaches the same :func:`main`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from swarm_perception.config import Config, ConfigError, load_config
from swarm_perception.logging_setup import setup_logging
from swarm_perception.sim.engine import Simulation, configure_runtime_mode
from swarm_perception.sim.robot import Robot
from swarm_perception.utils.paths import ASSETS_DIR


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="swarm-run",
        description="Run one swarm perception simulation from a YAML config.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="path to the YAML run config (default: configs/config-debug.yaml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a window (overrides simulation.headless)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="override simulation.seed",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="PATH",
        help="use PATH as the run directory (overrides simulation.output_dir)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v: INFO, -vv: DEBUG; default: WARNING)",
    )
    return parser.parse_args(argv)


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Return ``cfg`` with the CLI override flags folded in."""
    sim = cfg.simulation
    if args.headless:
        sim = dataclasses.replace(sim, headless=True)
    if args.seed is not None:
        sim = dataclasses.replace(sim, seed=args.seed)
    if args.output_dir is not None:
        sim = dataclasses.replace(sim, output_dir=args.output_dir)
    if sim is cfg.simulation:
        return cfg
    return dataclasses.replace(cfg, simulation=sim)


def main(argv: list[str] | None = None) -> None:
    """Console entrypoint: load config, apply overrides, and run the sim.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: With code 1 when the config cannot be loaded.
    """
    args = _parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = _apply_overrides(load_config(args.config), args)
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
