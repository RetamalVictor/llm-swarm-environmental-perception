"""Shared test helpers: mock LLM manager and a headless run harness."""

import dataclasses
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = REPO / "tests" / "data" / "smoke.yaml"


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


def load_smoke_config(output_dir: Path):
    """Load the smoke config redirected into ``output_dir``."""
    from swarm_perception.utils.config import load_config

    cfg = load_config(SMOKE_CONFIG)
    sim_cfg = dataclasses.replace(cfg.simulation, output_dir=str(output_dir))
    return dataclasses.replace(cfg, simulation=sim_cfg)


def run_headless(cfg):
    """Run one finalized headless simulation under the mock LLM manager.

    The caller must have patched ``swarm_perception.main.create_api_manager``
    (e.g. with :class:`MockManager`) before calling.
    """
    import swarm_perception.main as main
    from swarm_perception.utils.paths import ASSETS_DIR

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
    sim.shared.run_logger.finalize()
    return sim
