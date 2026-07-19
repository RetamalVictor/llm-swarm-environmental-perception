"""RunLogger contract tests: canonical JSONL events, no wall-clock leakage,
and run-dir resolution. The same-seed byte-identity ratchet (D10 T1a) lives
in ``tests/test_determinism.py``.
"""

import dataclasses
import json
import re

import pytest

from conftest import load_smoke_config

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
    logger.log_memory(tick=5, epoch=1, robot=0, keys=[(1, 2, 0), (1, 0, 0)])
    logger.log_comm(
        receiver_tick=7,
        sender_tick=6,
        epoch=1,
        receiver=0,
        sender=2,
        merge_method="deterministic",
        inbox_policy="within_budget",
    )


def _read_events(run_dir) -> list[dict]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert text.endswith("\n")
    return [json.loads(line) for line in text.splitlines()]


def test_events_are_one_json_object_per_line_in_order(logger) -> None:
    _emit_one_of_each(logger)
    raw_lines = (logger.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3

    events = [json.loads(line) for line in raw_lines]
    assert [e["type"] for e in events] == ["capture", "memory", "comm"]

    capture = events[0]
    assert capture["key"] == [1, 0, 0]  # [epoch, robot, crop_idx]
    assert capture["bbox"] == [10, 20, 160, 170]
    assert capture["pos"] == [85.5, 95.25]
    assert capture["tick"] == 5

    # Canonical serialization: sorted keys, no whitespace.
    assert raw_lines[0] == json.dumps(capture, sort_keys=True, separators=(",", ":"))


def test_memory_event_keys_are_sorted(logger) -> None:
    logger.log_memory(tick=9, epoch=2, robot=1, keys=[(2, 1, 0), (1, 3, 0), (1, 1, 0)])

    memory = _read_events(logger.run_dir)[0]
    assert memory["type"] == "memory"
    assert memory["tick"] == 9
    assert memory["epoch"] == 2
    assert memory["robot"] == 1
    assert memory["keys"] == [[1, 1, 0], [1, 3, 0], [2, 1, 0]]


def test_no_timestamp_like_keys_in_any_event(cfg) -> None:
    from swarm_perception.io.run_logger import RunLogger

    # Enable frame saving so the frame event is exercised too.
    sim_cfg = dataclasses.replace(cfg.simulation, save_photo_frames=True)
    logger = RunLogger(dataclasses.replace(cfg, simulation=sim_cfg))
    _emit_one_of_each(logger)
    logger.save_frame(tick=5)

    events = _read_events(logger.run_dir)
    assert {e["type"] for e in events} == {"capture", "memory", "comm", "frame"}
    for event in events:
        for key in event:
            assert not TIMESTAMP_LIKE.search(key), f"wall-clock-like key {key!r} in {event}"


def test_construction_writes_reproducibility_artifacts(logger) -> None:
    assert (logger.run_dir / "config_resolved.yaml").exists()
    metadata = json.loads((logger.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    for field in ("config_name", "seed", "package_version", "python_version", "platform", "git_sha", "started_at_utc"):
        assert field in metadata


def test_finalize_is_idempotent(logger) -> None:
    logger.log_capture(tick=1, epoch=1, robot=0, bbox=(0, 0, 1, 1), pos=(0.5, 0.5))
    logger.finalize()
    logger.finalize()

    metadata = json.loads((logger.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert "finished_at_utc" in metadata
    assert metadata["event_counts"] == {"capture": 1}


def test_resolve_run_dir_honors_output_dir(cfg, tmp_path) -> None:
    from swarm_perception.io.run_logger import resolve_run_dir
    from swarm_perception.utils.paths import OUTPUT_DIR

    assert resolve_run_dir(cfg) == tmp_path / "run"

    sim_cfg = dataclasses.replace(cfg.simulation, output_dir=None)
    unset = dataclasses.replace(cfg, simulation=sim_cfg)
    resolved = resolve_run_dir(unset)
    assert resolved.parent == OUTPUT_DIR
    assert resolved.name.startswith(f"{cfg.config.name}-")
