"""Typed swarm configuration schema.

Config is parsed into frozen dataclasses with validation; values derived from
other fields (photo_ticks, sim_duration) are exposed as computed properties on
:class:`Config` instead of being baked into module-level globals at import
time. YAML reading lives in :mod:`swarm_perception.config.loader`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


class ConfigError(ValueError):
    """Raised for missing files, malformed YAML, unknown keys, or invalid values."""


#: Valid ``robot.movement_policy`` values (implementations live in
#: :mod:`swarm_perception.sim.policies`).
MOVEMENT_POLICIES = frozenset({"ballistic", "crw", "levy", "boustrophedon"})

# Per-policy tunables accepted in ``robot.policy_params`` and their defaults:
#   ballistic     — (none)
#   crw           — sigma: 8.0 (per-tick heading noise std, degrees; >= 0)
#                   persistence: 0.7 (directional memory in [0, 1]; 1 = straight)
#   levy          — alpha: 1.5 (Pareto tail exponent; > 0)
#                   clamp: 300.0 (max flight length, pixels; > 0)
#   boustrophedon — lane_spacing: robot.coverage_side (distance between sweep
#                   lanes, pixels; > 0; the default gives gap-free camera coverage)
# ``None`` marks a default resolved from another field at validation time.
_POLICY_PARAM_DEFAULTS: dict[str, dict[str, float | None]] = {
    "ballistic": {},
    "crw": {"sigma": 8.0, "persistence": 0.7},
    "levy": {"alpha": 1.5, "clamp": 300.0},
    "boustrophedon": {"lane_spacing": None},
}


def _validate_policy_params(policy: str, raw: Any, coverage_side: float) -> dict[str, float]:
    """Resolve ``robot.policy_params`` against the selected policy's spec.

    Unknown parameter names are rejected, values must be numeric, and
    documented defaults fill any omitted parameter. Returns the fully
    resolved mapping stored back on the config.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"robot.policy_params must be a mapping, got {type(raw).__name__}")
    defaults = dict(_POLICY_PARAM_DEFAULTS[policy])
    if "lane_spacing" in defaults and defaults["lane_spacing"] is None:
        defaults["lane_spacing"] = float(coverage_side)
    unknown = sorted(set(raw) - set(defaults))
    if unknown:
        raise ConfigError(
            f"robot.policy_params for policy {policy!r} has unknown key(s) {unknown}; "
            f"valid keys: {sorted(defaults)}"
        )
    resolved: dict[str, float] = {}
    for name, default in defaults.items():
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"robot.policy_params.{name} must be a number, got {value!r}"
            )
        resolved[name] = float(value)
    if policy == "crw":
        if resolved["sigma"] < 0:
            raise ConfigError(f"robot.policy_params.sigma must be >= 0, got {resolved['sigma']}")
        if not 0.0 <= resolved["persistence"] <= 1.0:
            raise ConfigError(
                f"robot.policy_params.persistence must be in [0, 1], got {resolved['persistence']}"
            )
    elif policy == "levy":
        for name in ("alpha", "clamp"):
            if resolved[name] <= 0:
                raise ConfigError(f"robot.policy_params.{name} must be > 0, got {resolved[name]}")
    elif policy == "boustrophedon" and resolved["lane_spacing"] <= 0:
        raise ConfigError(
            f"robot.policy_params.lane_spacing must be > 0, got {resolved['lane_spacing']}"
        )
    return resolved


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
    movement_policy: str = "ballistic"
    policy_params: dict[str, float] = field(default_factory=dict)

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
        object.__setattr__(self, "movement_policy", str(self.movement_policy).strip().lower())
        if self.movement_policy not in MOVEMENT_POLICIES:
            raise ConfigError(
                f"robot.movement_policy must be one of {sorted(MOVEMENT_POLICIES)}, "
                f"got {self.movement_policy!r}"
            )
        object.__setattr__(
            self,
            "policy_params",
            _validate_policy_params(self.movement_policy, self.policy_params, self.coverage_side),
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
