"""End-to-end smoke test: a short headless run completes with a mock LLM.

Guards the config/globals refactor — proves the moved, de-globaled package
still spawns, loops, logs, and writes output without a live LLM.
"""

import dataclasses
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class MockManager:
    """Instant in-memory stand-in for the threaded/async LLM managers."""

    def __init__(self, *args, **kwargs) -> None:
        self._results: dict[str, tuple] = {}

    def start(self) -> None:
        pass

    def queue_depth(self) -> int:
        return 0

    def active_request_count(self) -> int:
        return 0

    def submit_photo_request(self, robot_id, image, observation, self_learning) -> None:
        self._results[f"{robot_id}_photo"] = ("A tree is present.", None)

    def submit_inbox_request(self, robot_id, current_observation, inbox) -> None:
        self._results[f"{robot_id}_inbox"] = ("A tree is present.", None)

    def get_result(self, robot_id, request_type):
        return self._results.pop(f"{robot_id}_{request_type}", (None, None))


def test_headless_run_completes(tmp_path, monkeypatch) -> None:
    import swarm_perception.main as main
    from swarm_perception.utils.config import load_config
    from swarm_perception.utils.paths import ASSETS_DIR

    cfg = load_config(REPO / "tests" / "data" / "smoke.yaml")
    # redirect run output into the test's temp dir
    sim_cfg = dataclasses.replace(cfg.simulation, output_dir=str(tmp_path))
    cfg = dataclasses.replace(cfg, simulation=sim_cfg)

    monkeypatch.setattr(main, "create_api_manager", lambda *a, **k: MockManager())
    main.configure_runtime_mode(True)

    sim = main.EnvironmentHeadlessSimulation(
        vi_config=main.build_vi_config(cfg),
        cfg=cfg,
        background_path=ASSETS_DIR / cfg.simulation.background_image,
    )
    sim.batch_spawn_agents(
        cfg.simulation.num_of_robots,
        main.Robot,
        images=[str(ASSETS_DIR / cfg.simulation.robot_image)],
    )
    sim.run()

    outputs = list(Path(tmp_path).rglob("robots.json"))
    assert outputs, "expected robots.json to be written under the run output dir"
