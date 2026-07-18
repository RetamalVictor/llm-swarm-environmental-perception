"""Typed swarm configuration loaded from YAML.

Config is parsed into frozen dataclasses with validation; values derived from
other fields (photo_ticks, sim_duration, timeouts, the inbox-synthesis /
wait-for-llm fallbacks) are exposed as computed properties on :class:`Config`
instead of being baked into module-level globals at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from swarm_perception.utils.paths import CONFIG_PATH


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
    empty_observation: str = "first run. no observations so far"
    max_facts_per_observation: int = 40
    max_inbox_merges_per_epoch: int = 1
    inbox_merge_after_budget: str = "drop"
    save_comm_merge_history: bool = False
    use_llm_inbox_synthesis: bool | None = None
    no_inbox_synthesis: bool = False
    wait_for_llm: bool = False
    wait_for_photo_llm: bool = False
    photo_timeout_ticks: int | None = None
    inbox_timeout_ticks: int | None = None

    def __post_init__(self) -> None:
        # Normalize the two fields the old code cleaned at read time.
        object.__setattr__(
            self, "inbox_merge_after_budget", str(self.inbox_merge_after_budget).strip().lower()
        )
        object.__setattr__(
            self, "max_inbox_merges_per_epoch", max(0, int(self.max_inbox_merges_per_epoch))
        )


@dataclass(frozen=True)
class PromptsCfg:
    """LLM prompt templates (the YAML ``llm.prompts:`` section)."""

    photo_analysis_self_learning: str = "{observation}"
    photo_analysis_no_self_learning: str = ""
    text_synthesis: str = "{current_observation} {inbox}"


@dataclass(frozen=True)
class LlmCfg:
    """LLM provider settings (the YAML ``llm:`` section)."""

    model_name: str
    thread_workers: int
    provider: str = "gemini"
    temperature: float = 0.05
    max_output_tokens: int = 220
    base_url: str | None = None
    api_key_env: str | None = None
    prompts: PromptsCfg = field(default_factory=PromptsCfg)


@dataclass(frozen=True)
class Config:
    """Complete, validated run configuration."""

    config: RunCfg
    simulation: SimulationCfg
    robot: RobotCfg
    llm: LlmCfg

    # --- derived values (formerly module-level globals) ---
    @property
    def photo_ticks(self) -> int:
        """Ticks between photo captures: capture_frequency * fps."""
        return int(self.robot.capture_frequency * self.simulation.fps)

    @property
    def sim_duration(self) -> int:
        """Total simulation ticks across all capture epochs."""
        return self.simulation.run_length * self.photo_ticks

    @property
    def photo_timeout_ticks(self) -> int:
        if self.robot.photo_timeout_ticks is not None:
            return self.robot.photo_timeout_ticks
        return self.photo_ticks * 2

    @property
    def inbox_timeout_ticks(self) -> int:
        if self.robot.inbox_timeout_ticks is not None:
            return self.robot.inbox_timeout_ticks
        return self.photo_ticks

    @property
    def use_llm_inbox_synthesis(self) -> bool:
        if self.robot.use_llm_inbox_synthesis is not None:
            return bool(self.robot.use_llm_inbox_synthesis)
        return not self.robot.no_inbox_synthesis

    @property
    def wait_for_llm(self) -> bool:
        return bool(self.robot.wait_for_llm or self.robot.wait_for_photo_llm)


def _section(cls: type, data: Any, name: str) -> Any:
    """Build a config dataclass from a YAML sub-mapping, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in (data or {}).items() if k in valid}
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise ValueError(f"config section '{name}' is invalid: {error}") from error


def _build(data: Any) -> Config:
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    llm_data = dict(data.get("llm") or {})
    prompts = _section(PromptsCfg, llm_data.pop("prompts", None), "llm.prompts")
    return Config(
        config=_section(RunCfg, data.get("config"), "config"),
        simulation=_section(SimulationCfg, data.get("simulation"), "simulation"),
        robot=_section(RobotCfg, data.get("robot"), "robot"),
        llm=_section(LlmCfg, {**llm_data, "prompts": prompts}, "llm"),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Read and validate a YAML config into a typed :class:`Config`.

    Args:
        path: Path to the YAML file. Defaults to ``configs/config-debug.yaml``.

    Returns:
        A validated, frozen :class:`Config`.

    Raises:
        SystemExit: If the file is missing or not valid YAML.
        ValueError: If a section is missing required keys.
    """
    cfg_path = Path(path) if path else CONFIG_PATH / "config-debug.yaml"
    try:
        with open(cfg_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise SystemExit(f"config not found: {cfg_path}")
    except yaml.YAMLError as error:
        raise SystemExit(f"config parse error in {cfg_path}: {error}")
    return _build(data)
