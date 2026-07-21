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
    assert cfg.fusion.memory_cap == 40
    assert cfg.comms.over_budget == "drop"
    assert cfg.perception.model == "stub"


def test_omitted_new_sections_fall_back_to_defaults(tmp_path: Path) -> None:
    # A config with only the three legacy sections still loads: perception,
    # fusion, and comms are fully defaulted.
    text = SMOKE.read_text(encoding="utf-8")
    for section in ("perception:", "fusion:", "comms:"):
        head, _, tail = text.partition(section)
        # drop the section body: everything until the next blank line
        _, _, rest = tail.partition("\n\n")
        text = head + rest
    stripped = tmp_path / "stripped.yaml"
    stripped.write_text(text, encoding="utf-8")
    cfg = load_config(stripped)
    assert cfg.perception.model == "stub"
    assert cfg.fusion.tau_dedup == 0.95
    assert cfg.comms.enabled is True
    assert cfg.comms.quantization == "none"


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


def test_invalid_over_budget_policy_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "over_budget: drop", "over_budget: bogus")
    with pytest.raises(ConfigError, match="over_budget"):
        load_config(variant)


def test_llm_over_budget_policy_no_longer_allowed(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "over_budget: drop", "over_budget: llm")
    with pytest.raises(ConfigError, match="over_budget"):
        load_config(variant)


@pytest.mark.parametrize(
    "old_key",
    [
        "self_learning: true",
        "communication: true",
        "memory_cap: 40",
        "max_inbox_merges_per_epoch: 1",
        "inbox_merge_after_budget: drop",
    ],
)
def test_removed_robot_keys_are_rejected(tmp_path: Path, old_key: str) -> None:
    # self_learning is deleted outright; the comm keys moved into comms and
    # memory_cap into fusion. The strict schema rejects any straggler.
    variant = _write_smoke_variant(
        tmp_path, "capture_frequency: 1", f"capture_frequency: 1\n  {old_key}"
    )
    with pytest.raises(ConfigError, match=old_key.split(":")[0]):
        load_config(variant)


def test_invalid_memory_cap_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "memory_cap: 40", "memory_cap: 0")
    with pytest.raises(ConfigError, match="memory_cap"):
        load_config(variant)


def test_invalid_tau_dedup_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "tau_dedup: 0.95", "tau_dedup: 1.5")
    with pytest.raises(ConfigError, match="tau_dedup"):
        load_config(variant)


def test_invalid_drop_p_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(tmp_path, "drop_p: 0.25", "drop_p: 2.0")
    with pytest.raises(ConfigError, match="drop_p"):
        load_config(variant)


def test_invalid_sender_policy_raises(tmp_path: Path) -> None:
    variant = _write_smoke_variant(
        tmp_path, "sender_policy: most_recent", "sender_policy: newest"
    )
    with pytest.raises(ConfigError, match="sender_policy"):
        load_config(variant)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_perception_cfg_validation() -> None:
    from swarm_perception.config import PerceptionCfg

    assert PerceptionCfg().model == "stub"
    assert PerceptionCfg(model="CLIP").model == "clip"  # normalized
    with pytest.raises(ConfigError, match="perception.model"):
        PerceptionCfg(model="resnet")
    with pytest.raises(ConfigError, match="batch_size"):
        PerceptionCfg(batch_size=0)


def test_fusion_cfg_validation() -> None:
    from swarm_perception.config import FusionCfg

    assert FusionCfg().tau_dedup == 0.95
    with pytest.raises(ConfigError, match="tau_dedup"):
        FusionCfg(tau_dedup=0.0)
    with pytest.raises(ConfigError, match="tau_dedup"):
        FusionCfg(tau_dedup=1.5)
    with pytest.raises(ConfigError, match="memory_cap"):
        FusionCfg(memory_cap=0)


def test_comms_cfg_validation() -> None:
    from swarm_perception.config import CommsCfg

    cfg = CommsCfg(sender_policy="Most_Recent", quantization="INT8", over_budget="Drop")
    assert (cfg.sender_policy, cfg.quantization, cfg.over_budget) == (
        "most_recent",
        "int8",
        "drop",
    )
    with pytest.raises(ConfigError, match="comms.k"):
        CommsCfg(k=0)
    with pytest.raises(ConfigError, match="drop_p"):
        CommsCfg(drop_p=1.5)
    with pytest.raises(ConfigError, match="delay_ticks"):
        CommsCfg(delay_ticks=-1)
    with pytest.raises(ConfigError, match="sender_policy"):
        CommsCfg(sender_policy="newest")
    with pytest.raises(ConfigError, match="quantization"):
        CommsCfg(quantization="int4")
    with pytest.raises(ConfigError, match="over_budget"):
        CommsCfg(over_budget="explode")
    with pytest.raises(ConfigError, match="max_inbox_merges_per_epoch"):
        CommsCfg(max_inbox_merges_per_epoch=-1)
