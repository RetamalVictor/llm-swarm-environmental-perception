"""Load YAML swarm configs into attribute-access namespaces."""

import types
from typing import Any

import yaml

from utils.logging_config import logging
from utils.paths import CONFIG_PATH


def dict_to_namespace(data: Any) -> Any:
    """Recursively convert dictionaries/lists into ``SimpleNamespace`` objects.

    Args:
        data: Any nested config structure loaded from YAML.

    Returns:
        A structure with dictionaries replaced by ``types.SimpleNamespace`` and
        list contents converted recursively.
    """
    if isinstance(data, dict):
        namespace = types.SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in data.items()}
        )
        return namespace
    if isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    return data


class SwarmConfig:
    """Configuration loader for simulation and experiment runs.

    The loader defaults to the project debug config, but callers can pass any
    YAML file path. Returned values are converted to nested namespaces so code
    can access fields using dot notation (for example, ``config.robot.coverage_side``).
    """

    def __init__(self, path: str = str(CONFIG_PATH / "config-debug.yaml")):
        """Initialize a loader for a specific YAML config path.

        Args:
            path: Path to a YAML file containing the swarm configuration.
        """
        self.path = path

    def load_config(self) -> Any:
        """Read, validate, and convert the YAML config file.

        Returns:
            A namespace-like object with attribute access to nested config keys.

        Raises:
            SystemExit: If the config file is missing or malformed.
        """
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
            logging.info("Configuration loaded successfully from %s", self.path)
            return dict_to_namespace(config_data)
        except FileNotFoundError:
            logging.error("FATAL: Configuration file not found at %s", self.path)
            raise SystemExit(1)
        except yaml.YAMLError as error:
            logging.error("FATAL: Error parsing configuration file %s: %s", self.path, error)
            raise SystemExit(1)