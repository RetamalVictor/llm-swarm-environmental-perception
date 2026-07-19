"""YAML reading for the typed swarm configuration.

The schema (dataclasses + validation) lives in
:mod:`swarm_perception.config.schema`; this module only turns a YAML file
into the parsed mapping handed to :func:`~swarm_perception.config.schema.build_config`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from swarm_perception.config.schema import Config, ConfigError, build_config
from swarm_perception.utils.paths import CONFIG_PATH


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate a YAML config into a typed :class:`Config`.

    Args:
        path: Path to the YAML file. Defaults to ``configs/config-debug.yaml``.

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        ConfigError: If the file is missing, not valid YAML, contains unknown
            keys, or fails validation. The CLI translates this to exit code 1.
    """
    cfg_path = Path(path) if path else CONFIG_PATH / "config-debug.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise ConfigError(f"config not found: {cfg_path}") from None
    except yaml.YAMLError as error:
        raise ConfigError(f"config parse error in {cfg_path}: {error}") from error
    return build_config(data)
