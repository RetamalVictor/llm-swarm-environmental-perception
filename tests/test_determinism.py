"""T1a determinism ratchet: same config + seed must reproduce the run exactly.

Two independent headless simulations from the identical config and seed are
compared at the byte level on ``events.jsonl``. The smoke config runs the
FULL channel — comms on, ``drop_p`` 0.25, int8 quantization, residue sharing
— so any nondeterminism in the sim loop, the RNG plumbing, the channel
(drop stream, quantize-once payloads, byte accounting), or the event
serialization fails this test.
"""

import json
from pathlib import Path

from conftest import load_smoke_config, run_headless


def _read_events(run_dir: Path) -> list[dict]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_same_seed_runs_are_byte_identical_t1a(tmp_path: Path) -> None:
    run_dirs = []
    for name in ("run_a", "run_b"):
        run_dir = tmp_path / name
        run_headless(load_smoke_config(run_dir))
        run_dirs.append(run_dir)

    for run_dir in run_dirs:
        for artifact in ("events.jsonl", "config_resolved.yaml", "run_metadata.json"):
            assert (run_dir / artifact).exists(), f"missing {artifact} in {run_dir}"

    events_a = (run_dirs[0] / "events.jsonl").read_bytes()
    events_b = (run_dirs[1] / "events.jsonl").read_bytes()
    assert len(events_a) > 0, "expected a non-empty event log"
    assert events_a == events_b, "same config+seed must produce byte-identical events.jsonl"

    events = _read_events(run_dirs[0])

    # The native sim emits one memory event per robot per capture epoch,
    # with keys already in sorted tuple order.
    memory_events = [e for e in events if e["type"] == "memory"]
    assert memory_events, "expected memory events from the native run"
    for event in memory_events:
        assert event["keys"] == sorted(event["keys"])

    # The determinism claim must actually cover the channel: the smoke run
    # produces priced comm traffic, seeded packet drops, and k-capped sends.
    comm_events = [e for e in events if e["type"] == "comm"]
    assert comm_events, "smoke config must exercise the channel"
    assert any(e["dropped"] for e in comm_events), "drop_p > 0 must drop something"
    assert any(not e["dropped"] for e in comm_events), "some messages must merge"
    for event in comm_events:
        assert event["bytes"] > 0
        assert 0 < event["k_sent"] <= 2  # comms.k in smoke.yaml
