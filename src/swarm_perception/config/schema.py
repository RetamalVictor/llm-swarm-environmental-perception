"""Typed swarm configuration schema.

Config is parsed into frozen dataclasses with validation; values derived from
other fields (photo_ticks, sim_duration) are exposed as computed properties on
:class:`Config` instead of being baked into module-level globals at import
time. YAML reading lives in :mod:`swarm_perception.config.loader`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


class ConfigError(ValueError):
    """Raised for missing files, malformed YAML, unknown keys, or invalid values."""


@dataclass(frozen=True)
class RunCfg:
    """Top-level run metadata (the YAML ``config:`` section)."""

    name: str = "unnamed"


@dataclass(frozen=True)
class SimulationCfg:
    """World and runtime settings (the YAML ``simulation:`` section)."""

    width: int
    height: int
    fps: int
    num_of_robots: int
    run_length: int
    background_image: str
    robot_image: str
    seed: int = 0
    headless: bool = False
    save_photo_frames: bool = False
    save_robot_crops: bool = False
    output_dir: str | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height", "num_of_robots", "run_length"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"simulation.{name} must be > 0, got {getattr(self, name)}")
        # fps == 0 would make photo_ticks/sim_duration 0 and the run never terminate.
        if self.fps <= 0:
            raise ConfigError(f"simulation.fps must be > 0, got {self.fps}")


@dataclass(frozen=True)
class RobotCfg:
    """Per-robot behavior settings (the YAML ``robot:`` section)."""

    linear_speed: float
    angular_velocity: float
    coverage_side: float
    neighbor_radius: float
    capture_frequency: float
    communication: bool
    self_learning: bool
    memory_cap: int = 40
    max_inbox_merges_per_epoch: int = 1
    inbox_merge_after_budget: str = "drop"

    def __post_init__(self) -> None:
        # Normalize the two fields the old code cleaned at read time.
        object.__setattr__(
            self, "inbox_merge_after_budget", str(self.inbox_merge_after_budget).strip().lower()
        )
        object.__setattr__(
            self, "max_inbox_merges_per_epoch", max(0, int(self.max_inbox_merges_per_epoch))
        )
        for name in ("coverage_side", "capture_frequency", "neighbor_radius"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"robot.{name} must be > 0, got {getattr(self, name)}")
        if self.memory_cap <= 0:
            raise ConfigError(f"robot.memory_cap must be > 0, got {self.memory_cap}")
        allowed_policies = {"drop", "deterministic"}
        if self.inbox_merge_after_budget not in allowed_policies:
            raise ConfigError(
                f"robot.inbox_merge_after_budget must be one of {sorted(allowed_policies)}, "
                f"got {self.inbox_merge_after_budget!r}"
            )


@dataclass(frozen=True)
class Config:
    """Complete, validated run configuration."""

    config: RunCfg
    simulation: SimulationCfg
    robot: RobotCfg

    # --- derived values (formerly module-level globals) ---
    @property
    def photo_ticks(self) -> int:
        """Ticks between photo captures: capture_frequency * fps."""
        return int(self.robot.capture_frequency * self.simulation.fps)

    @property
    def sim_duration(self) -> int:
        """Total simulation ticks across all capture epochs."""
        return self.simulation.run_length * self.photo_ticks


def _section(cls: type, data: Any, name: str) -> Any:
    """Build a config dataclass from a YAML sub-mapping; unknown keys are errors."""
    mapping = dict(data or {})
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(mapping) - valid)
    if unknown:
        raise ConfigError(
            f"config section '{name}' has unknown key(s) {unknown}; valid keys: {sorted(valid)}"
        )
    try:
        return cls(**mapping)
    except TypeError as error:
        raise ConfigError(f"config section '{name}' is invalid: {error}") from error


_ROOT_SECTIONS = {"config", "simulation", "robot"}


def build_config(data: Any) -> Config:
    """Validate a parsed YAML mapping into a typed :class:`Config`.

    Args:
        data: Root object parsed from YAML; must be a mapping of the known
            top-level sections.

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        ConfigError: On a non-mapping root, unknown sections, unknown keys,
            or invalid values.
    """
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    unknown_sections = sorted(set(data) - _ROOT_SECTIONS)
    if unknown_sections:
        raise ConfigError(
            f"unknown top-level config section(s) {unknown_sections}; "
            f"valid sections: {sorted(_ROOT_SECTIONS)}"
        )
    return Config(
        config=_section(RunCfg, data.get("config"), "config"),
        simulation=_section(SimulationCfg, data.get("simulation"), "simulation"),
        robot=_section(RobotCfg, data.get("robot"), "robot"),
    )
