"""End-to-end smoke test: a short headless run completes natively offline.

Guards the sim wiring — proves the package still spawns, loops, logs, and
writes the run artifacts with no network access or external services.
"""

import json

from conftest import load_smoke_config, run_headless


def test_headless_run_completes(tmp_path) -> None:
    run_headless(load_smoke_config(tmp_path))

    for artifact in ("events.jsonl", "config_resolved.yaml", "run_metadata.json"):
        assert (tmp_path / artifact).exists(), f"expected {artifact} in the run dir"

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    types = {event["type"] for event in events}
    assert "capture" in types
    assert "memory" in types
