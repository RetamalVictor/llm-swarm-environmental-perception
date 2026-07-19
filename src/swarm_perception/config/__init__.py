"""Typed run configuration: schema dataclasses and the YAML loader.

Public names are re-exported here so call sites can write
``from swarm_perception.config import load_config, ConfigError, Config``.
"""

from swarm_perception.config.loader import load_config
from swarm_perception.config.schema import (
    Config,
    ConfigError,
    RobotCfg,
    RunCfg,
    SimulationCfg,
    build_config,
)

__all__ = [
    "Config",
    "ConfigError",
    "RobotCfg",
    "RunCfg",
    "SimulationCfg",
    "build_config",
    "load_config",
]
