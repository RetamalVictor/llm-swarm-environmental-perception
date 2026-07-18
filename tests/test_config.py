"""Tests for the typed config loader."""

from pathlib import Path

import pytest

from swarm_perception.utils.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]


def test_load_example_config() -> None:
    cfg = load_config(REPO / "examples" / "example1.yaml")
    assert isinstance(cfg, Config)
    assert cfg.simulation.num_of_robots == 10
    # derived values replace the old module-level globals
    assert cfg.photo_ticks == int(cfg.robot.capture_frequency * cfg.simulation.fps)
    assert cfg.sim_duration == cfg.simulation.run_length * cfg.photo_ticks
    assert cfg.use_llm_inbox_synthesis is True


def test_defaults_and_derived() -> None:
    cfg = load_config(REPO / "tests" / "data" / "smoke.yaml")
    assert cfg.photo_ticks == 5
    # timeouts fall back to derived defaults when not set in YAML
    assert cfg.photo_timeout_ticks == cfg.photo_ticks * 2
    assert cfg.inbox_timeout_ticks == cfg.photo_ticks
    # normalized in __post_init__
    assert cfg.robot.inbox_merge_after_budget == "drop"
    # provider/temperature defaults applied
    assert cfg.llm.temperature == 0.05


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("config:\n  name: x\nsimulation:\n  width: 10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)
