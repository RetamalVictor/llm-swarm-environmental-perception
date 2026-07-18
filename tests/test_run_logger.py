"""RunLogger contract tests: canonical JSONL events, no wall-clock leakage,
snapshot compaction, run-dir resolution, and same-seed byte-identity of
``events.jsonl`` (D10 T1a, under the mock LLM manager).
"""

import dataclasses
import json
import re

import pytest

from conftest import MockManager, load_smoke_config, run_headless

# Wall-clock values are banned from events (D10); sim "tick" counters are fine.
TIMESTAMP_LIKE = re.compile(r"time|date|stamp|clock|utc", re.IGNORECASE)


@pytest.fixture()
def cfg(tmp_path):
    return load_smoke_config(tmp_path / "run")


@pytest.fixture()
def logger(cfg):
    from swarm_perception.io.run_logger import RunLogger

    return RunLogger(cfg)


def _emit_one_of_each(logger) -> None:
    logger.log_capture(tick=5, epoch=1, robot=0, bbox=(10, 20, 160, 170), pos=(85.5, 95.25))
    logger.log_snapshot(tick=5, robot=0, observation="none")
    logger.log_comm(
        receiver_tick=7,
        sender_tick=6,
        epoch=1,
        receiver=0,
        sender=2,
        merge_method="deterministic",
        inbox_policy="within_budget",
    )


def _read_events(logger) -> list[dict]:
    text = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def test_events_are_one_json_object_per_line_in_order(logger) -> None:
    _emit_one_of_each(logger)
    raw_lines = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3

    events = [json.loads(line) for line in raw_lines]
    assert [e["type"] for e in events] == ["capture", "snapshot", "comm"]

    capture = events[0]
    assert capture["key"] == [1, 0, 0]  # [epoch, robot, crop_idx]
    assert capture["bbox"] == [10, 20, 160, 170]
    assert capture["pos"] == [85.5, 95.25]
    assert capture["tick"] == 5

    # Canonical serialization: sorted keys, no whitespace.
    assert raw_lines[0] == json.dumps(capture, sort_keys=True, separators=(",", ":"))


def test_no_timestamp_like_keys_in_any_event(cfg) -> None:
    from swarm_perception.io.run_logger import RunLogger

    # Enable frame saving so the frame event is exercised too.
    sim_cfg = dataclasses.replace(cfg.simulation, save_photo_frames=True)
    logger = RunLogger(dataclasses.replace(cfg, simulation=sim_cfg))
    _emit_one_of_each(logger)
    logger.save_frame(tick=5)

    events = _read_events(logger)
    assert {e["type"] for e in events} == {"capture", "snapshot", "comm", "frame"}
    for event in events:
        for key in event:
            assert not TIMESTAMP_LIKE.search(key), f"wall-clock-like key {key!r} in {event}"


def test_construction_writes_reproducibility_artifacts(logger) -> None:
    assert (logger.run_dir / "config_resolved.yaml").exists()
    metadata = json.loads((logger.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    for field in ("config_name", "seed", "package_version", "python_version", "platform", "git_sha", "started_at_utc"):
        assert field in metadata


def test_snapshot_compaction_reproduces_robots_json_shape(logger) -> None:
    logger.log_snapshot(tick=5, robot=2, observation="first from 2")
    logger.log_snapshot(tick=5, robot=0, observation="first from 0")
    logger.log_snapshot(tick=10, robot=2, observation="second from 2")
    logger.finalize()

    robots = json.loads((logger.run_dir / "robots.json").read_text(encoding="utf-8"))
    assert robots == {
        "0": ["first from 0"],
        "2": ["first from 2", "second from 2"],
    }


def test_finalize_is_idempotent(logger) -> None:
    logger.log_snapshot(tick=1, robot=0, observation="obs")
    logger.finalize()
    logger.finalize()

    metadata = json.loads((logger.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "finished_at_utc" in metadata
    assert metadata["event_counts"] == {"snapshot": 1}


def test_resolve_run_dir_honors_output_dir(cfg, tmp_path) -> None:
    from swarm_perception.io.run_logger import resolve_run_dir
    from swarm_perception.utils.paths import OUTPUT_DIR

    assert resolve_run_dir(cfg) == tmp_path / "run"

    sim_cfg = dataclasses.replace(cfg.simulation, output_dir=None)
    unset = dataclasses.replace(cfg, simulation=sim_cfg)
    resolved = resolve_run_dir(unset)
    assert resolved.parent == OUTPUT_DIR
    assert resolved.name.startswith(f"{cfg.config.name}-")


def test_same_seed_headless_runs_are_byte_identical(tmp_path, monkeypatch) -> None:
    import swarm_perception.main as main

    monkeypatch.setattr(main, "create_api_manager", lambda *a, **k: MockManager())

    run_dirs = []
    for name in ("run_a", "run_b"):
        run_dir = tmp_path / name
        run_headless(load_smoke_config(run_dir))
        run_dirs.append(run_dir)

    for run_dir in run_dirs:
        for artifact in ("events.jsonl", "config_resolved.yaml", "run_metadata.json", "robots.json"):
            assert (run_dir / artifact).exists(), f"missing {artifact} in {run_dir}"

    events_a = (run_dirs[0] / "events.jsonl").read_bytes()
    events_b = (run_dirs[1] / "events.jsonl").read_bytes()
    assert len(events_a) > 0, "expected a non-empty event log"
    assert events_a == events_b, "same config+seed must produce byte-identical events.jsonl"
