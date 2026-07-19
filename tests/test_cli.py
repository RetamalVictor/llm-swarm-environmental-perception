"""CLI tests: override flags reach the run artifacts and bad configs exit 1."""

import json
from pathlib import Path

import pytest

from conftest import SMOKE_CONFIG
from swarm_perception.cli import main


def _read_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))


def test_seed_override_is_recorded_in_run_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    main([str(SMOKE_CONFIG), "--seed", "123", "--output-dir", str(run_dir)])

    assert _read_metadata(run_dir)["seed"] == 123
    assert (run_dir / "events.jsonl").exists()


def test_headless_flag_overrides_config(tmp_path: Path) -> None:
    # A windowed variant of the smoke config, forced headless from the CLI.
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    assert "headless: true" in text, "fixture drift: smoke.yaml is not headless"
    windowed = tmp_path / "windowed.yaml"
    windowed.write_text(text.replace("headless: true", "headless: false"), encoding="utf-8")

    run_dir = tmp_path / "run"
    main([str(windowed), "--headless", "--output-dir", str(run_dir)])

    assert (run_dir / "events.jsonl").exists()
    resolved = (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    assert "headless: true" in resolved


def test_bad_config_path_exits_1_with_error_on_stderr(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "nope.yaml")])

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err
    assert "nope.yaml" in captured.err
