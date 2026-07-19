"""Tests for the typed config loader."""

from pathlib import Path

import pytest

from swarm_perception.config import Config, ConfigError, load_config

REPO = Path(__file__).resolve().parents[1]

SMOKE = REPO / "tests" / "data" / "smoke.yaml"


def _write_smoke_variant(tmp_path: Path, old: str, new: str) -> Path:
    """Copy smoke.yaml with one line substituted."""
    text = SMOKE.read_text(encoding="utf-8")
    assert old in text, f"fixture drift: {old!r} not in smoke.yaml"
    out = tmp_path / "variant.yaml"
    out.write_text(text.replace(old, new), encoding="utf-8")
    return out


def test_load_example_config() -> None:
    cfg = load_config(REPO / "examples" / "example1.yaml")
    assert isinstance(cfg, Config)
    assert cfg.simulation.num_of_robots == 10
    # derived values replace the old module-level globals
    assert cfg.photo_ticks == int(cfg.robot.capture_frequency * cfg.simulation.fps)
    assert cfg.sim_duration == cfg.simulation.run_length * cfg.photo_ticks


def test_defaults_and_derived() -> None:
    cfg = load_config(SMOKE)
    assert cfg.photo_ticks == 5
    # memory_cap falls back to its default when not set in YAML
    assert cfg.robot.memory_cap == 40
    # normalized in __post_init__
    assert cfg.robot.inbox_merge_after_budget == "drop"


def test_all_repo_yaml_configs_load() -> None:
    yaml_paths = sorted(
        [
            *(REPO / "examples").glob("*.yaml"),
            *(REPO / "experiments" / "configs").rglob("*.yaml"),
            *(REPO / "tests" / "data").glob("*.yaml"),
        ]
    )
    assert yaml_paths, "expected YAML configs in the repo"
    for path in yaml_paths:
        cfg = load_config(path)
        assert isinstance(cfg, Config), path


def test_missing_required_key_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("config:\n  name: x\nsimulation:\n  width: 10\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_unknown_key_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "seed: 0", "seed: 0\n  tpyo_key: 1")
    with pytest.raises(ConfigError, match="tpyo_key"):
        load_config(variant)


def test_unknown_top_level_section_raises(tmp_path: Path) -> None:
    text = SMOKE.read_text(encoding="utf-8") + "\nmystery:\n  x: 1\n"
    bad = tmp_path / "bad_section.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="mystery"):
        load_config(bad)


def test_llm_section_is_rejected(tmp_path: Path) -> None:
    text = SMOKE.read_text(encoding="utf-8") + "\nllm:\n  model_name: x\n"
    bad = tmp_path / "llm_section.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="llm"):
        load_config(bad)


def test_invalid_value_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "fps: 5", "fps: 0")
    with pytest.raises(ConfigError, match="fps"):
        load_config(variant)


def test_invalid_inbox_policy_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(
        tmp_path, "communication: true", "communication: true\n  inbox_merge_after_budget: bogus"
    )
    with pytest.raises(ConfigError, match="inbox_merge_after_budget"):
        load_config(variant)


def test_llm_inbox_policy_no_longer_allowed(tmp_path: Path) -> None:
    variant = _write_smoke_variant(
        tmp_path, "communication: true", "communication: true\n  inbox_merge_after_budget: llm"
    )
    with pytest.raises(ConfigError, match="inbox_merge_after_budget"):
        load_config(variant)


def test_memory_cap_replaces_max_facts_per_observation(tmp_path: Path) -> None:
    old_key = _write_smoke_variant(
        tmp_path, "communication: true", "communication: true\n  max_facts_per_observation: 40"
    )
    with pytest.raises(ConfigError, match="max_facts_per_observation"):
        load_config(old_key)

    new_key = _write_smoke_variant(
        tmp_path, "communication: true", "communication: true\n  memory_cap: 7"
    )
    assert load_config(new_key).robot.memory_cap == 7


def test_invalid_memory_cap_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(
        tmp_path, "communication: true", "communication: true\n  memory_cap: 0"
    )
    with pytest.raises(ConfigError, match="memory_cap"):
        load_config(variant)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
