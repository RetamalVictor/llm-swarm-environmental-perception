"""End-to-end smoke test: a short headless run completes with a mock LLM.

Guards the config/globals refactor — proves the moved, de-globaled package
still spawns, loops, logs, and writes the run artifacts without a live LLM.
"""

from conftest import MockManager, load_smoke_config, run_headless


def test_headless_run_completes(tmp_path, monkeypatch) -> None:
    import swarm_perception.main as main

    cfg = load_smoke_config(tmp_path)
    monkeypatch.setattr(main, "create_api_manager", lambda *a, **k: MockManager())

    run_headless(cfg)

    for artifact in ("events.jsonl", "config_resolved.yaml", "run_metadata.json", "robots.json"):
        assert (tmp_path / artifact).exists(), f"expected {artifact} in the run dir"
