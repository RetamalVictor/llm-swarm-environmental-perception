"""Tests for movement policies: determinism, coverage, config, extraction.

The trajectory tests roll policies forward with the same integration math as
:class:`~swarm_perception.sim.actuator.Actuator` (heading += angular, then a
polar step) and the same edge margins as the camera sensor, without spinning
up the full pygame simulation.
"""

import dataclasses
import math
import random
from pathlib import Path
from typing import cast

import pytest
import yaml

from conftest import run_headless
from swarm_perception.config import ConfigError, RobotCfg, build_config, load_config
from swarm_perception.sim.policies import (
    Ballistic,
    Boustrophedon,
    CorrelatedRandomWalk,
    LevyWalk,
    MovementPolicy,
    StepContext,
    build_policy,
)

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "tests" / "data" / "smoke.yaml"

Bounds = tuple[float, float, float, float]


def _robot_cfg(**overrides) -> RobotCfg:
    base = dict(
        linear_speed=4.0,
        angular_velocity=5.0,
        coverage_side=40.0,
        neighbor_radius=50.0,
        capture_frequency=1.0,
        communication=False,
        self_learning=False,
    )
    base.update(overrides)
    return RobotCfg(**base)


def _edge(x: float, y: float, bounds: Bounds, margin: float) -> str | bool:
    """Mirror CameraSensor.detect_edges for the policy rollout harness."""
    left, top, right, bottom = bounds
    if x < left + margin:
        return "left"
    if x > right - margin:
        return "right"
    if y < top + margin:
        return "top"
    if y > bottom - margin:
        return "bottom"
    return False


def _rollout(
    policy: MovementPolicy,
    cfg: RobotCfg,
    *,
    rng: random.Random,
    ticks: int,
    start: tuple[float, float],
    heading: float = 0.0,
    bounds: Bounds = (0.0, 0.0, 2000.0, 2000.0),
) -> list[tuple[float, float]]:
    """Integrate a policy exactly like the actuator does; return positions."""
    x, y = start
    margin = cfg.coverage_side // 2  # same floor as CameraSensor.detect_edges
    positions: list[tuple[float, float]] = []
    for _ in range(ticks):
        ctx = StepContext(
            pos=(x, y),
            heading=heading,
            edge=_edge(x, y, bounds, margin),
            bounds=bounds,
            cfg=cfg,
            rng=rng,
        )
        speed, angular = policy.step(ctx)
        heading = (heading + angular) % 360.0
        x += speed * math.cos(math.radians(heading))
        y += speed * math.sin(math.radians(heading))
        positions.append((x, y))
    return positions


# ------------------------------------------------------------- stochastic

STOCHASTIC_FACTORIES = {
    "crw": lambda: CorrelatedRandomWalk(sigma=8.0, persistence=0.7),
    "levy": lambda: LevyWalk(alpha=1.5, clamp=300.0),
}


@pytest.mark.parametrize("name", sorted(STOCHASTIC_FACTORIES))
def test_stochastic_policy_seed_determinism(name: str) -> None:
    make = STOCHASTIC_FACTORIES[name]
    cfg = _robot_cfg()
    kwargs = dict(ticks=500, start=(1000.0, 1000.0))
    run_a = _rollout(make(), cfg, rng=random.Random(3), **kwargs)
    run_b = _rollout(make(), cfg, rng=random.Random(3), **kwargs)
    run_c = _rollout(make(), cfg, rng=random.Random(4), **kwargs)
    assert run_a == run_b, f"{name}: same seed must give an identical trajectory"
    assert run_a != run_c, f"{name}: different seeds must give different trajectories"


def test_stochastic_policies_stay_near_arena() -> None:
    """Edge handling must pull wanderers back instead of letting them escape.

    A policy with no edge handling drifts arbitrarily far (a straight escape
    covers speed * ticks = 12000 px here). With inward corrections the worst
    excursion is a few turn circles of loitering past the boundary — the CRW
    turns back at angular_velocity deg/tick, i.e. with turn radius
    ``r = speed / radians(angular_velocity)`` (~46 px) — so an envelope of
    ``margin + 8 r`` (~387 px, measured worst over 20 seeds is ~200) cleanly
    separates bounded excursions from a true walk-off.
    """
    bounds: Bounds = (0.0, 0.0, 400.0, 400.0)
    cfg = _robot_cfg()
    margin = cfg.coverage_side // 2
    slack = margin + 8 * cfg.linear_speed / math.radians(cfg.angular_velocity)
    for name, make in STOCHASTIC_FACTORIES.items():
        for seed in (1, 2, 3):
            positions = _rollout(
                make(), cfg, rng=random.Random(seed), ticks=3000, start=(200.0, 200.0),
                bounds=bounds,
            )
            for x, y in positions:
                assert -slack < x < 400 + slack and -slack < y < 400 + slack, (
                    f"{name} seed {seed} escaped to ({x:.1f}, {y:.1f})"
                )


# --------------------------------------------------------- boustrophedon


class _ForbiddenRng:
    """Any RNG access fails the test: boustrophedon ignores the RNG by design."""

    def __getattr__(self, name: str):
        raise AssertionError(f"boustrophedon must not touch the RNG (called .{name})")


def test_boustrophedon_covers_every_lane_band() -> None:
    # Arena 400x400, coverage_side 40 -> margin 20, so the usable sweep area
    # is [20, 380] x [20, 380]. With lane_spacing 40 the usable height splits
    # into (380 - 20) / 40 = 9 lane bands. Tick bound: one sweep crosses
    # 380 - 20 = 360 px at 4 px/tick = 90 ticks (+1 to trigger the turn); one
    # lane shift is 40 / 4 = 10 ticks (+1). Starting at the top-left corner,
    # visiting all 9 bands takes 9 sweeps + 8 shifts <= 9*91 + 8*11 = 907
    # ticks; 2000 ticks gives >2x slack for heading snaps and rounding.
    bounds: Bounds = (0.0, 0.0, 400.0, 400.0)
    cfg = _robot_cfg()  # linear_speed 4.0, coverage_side 40.0
    positions = _rollout(
        Boustrophedon(lane_spacing=40.0),
        cfg,
        rng=cast(random.Random, _ForbiddenRng()),
        ticks=2000,
        start=(20.0, 20.0),
        heading=0.0,
        bounds=bounds,
    )
    ys = [p[1] for p in positions]
    for band in range(9):
        low, high = 20.0 + 40.0 * band, 60.0 + 40.0 * band
        assert any(low <= y < high for y in ys), f"band {band} [{low}, {high}) never visited"
    # The sweep must also span the arena horizontally, not hug one side.
    xs = [p[0] for p in positions]
    assert min(xs) <= 24.0 and max(xs) >= 376.0


# ------------------------------------------------- ballistic extraction


def test_ballistic_matches_legacy_velocities() -> None:
    """The extracted policy reproduces the pre-policy get_velocities exactly."""
    cfg = _robot_cfg(linear_speed=2.3, angular_velocity=5.0)
    policy = Ballistic()

    def ctx(edge: str | bool) -> StepContext:
        return StepContext(
            pos=(100.0, 100.0),
            heading=45.0,
            edge=edge,
            bounds=(0.0, 0.0, 2500.0, 2500.0),
            cfg=cfg,
            rng=cast(random.Random, _ForbiddenRng()),  # ballistic draws nothing
        )

    assert policy.step(ctx(False)) == (2.3, 0.0)
    for edge in ("left", "right", "top", "bottom"):
        assert policy.step(ctx(edge)) == (2.3, 5.0)


def test_ballistic_key_absent_and_explicit_are_byte_identical(tmp_path: Path) -> None:
    """Omitting movement_policy must equal the explicit ballistic default.

    The cross-commit gate (byte-identity of events.jsonl against the
    pre-extraction main) is covered by this default-equivalence plus the
    T1a determinism ratchet in test_determinism.py.
    """
    text = SMOKE.read_text(encoding="utf-8")
    assert "movement_policy: ballistic" in text, "fixture drift: smoke.yaml lost the key"
    stripped = "".join(
        line for line in text.splitlines(keepends=True) if "movement_policy" not in line
    )

    events: list[bytes] = []
    for name, content in (("explicit", text), ("absent", stripped)):
        cfg_path = tmp_path / f"{name}.yaml"
        cfg_path.write_text(content, encoding="utf-8")
        run_dir = tmp_path / f"run_{name}"
        cfg = load_config(cfg_path)
        cfg = dataclasses.replace(
            cfg, simulation=dataclasses.replace(cfg.simulation, output_dir=str(run_dir))
        )
        run_headless(cfg)
        events.append((run_dir / "events.jsonl").read_bytes())

    assert events[0], "expected a non-empty event log"
    assert events[0] == events[1], "default movement_policy must be byte-equivalent"


# ------------------------------------------------------- config plumbing


def _smoke_data(**robot_overrides):
    data = yaml.safe_load(SMOKE.read_text(encoding="utf-8"))
    data["robot"].update(robot_overrides)
    return data


def test_bad_policy_name_raises() -> None:
    with pytest.raises(ConfigError, match="movement_policy"):
        build_config(_smoke_data(movement_policy="zigzag"))


def test_unknown_policy_param_raises() -> None:
    with pytest.raises(ConfigError, match="bogus"):
        build_config(_smoke_data(movement_policy="crw", policy_params={"sigma": 5.0, "bogus": 1}))


def test_param_from_other_policy_rejected() -> None:
    # Strictness is per selected policy: levy's alpha is unknown to crw.
    with pytest.raises(ConfigError, match="alpha"):
        build_config(_smoke_data(movement_policy="crw", policy_params={"alpha": 1.5}))


def test_out_of_range_policy_param_raises() -> None:
    with pytest.raises(ConfigError, match="persistence"):
        build_config(_smoke_data(movement_policy="crw", policy_params={"persistence": 1.5}))


def test_policy_param_defaults_are_resolved() -> None:
    cfg = build_config(_smoke_data(movement_policy="boustrophedon"))
    # Default lane_spacing is the camera coverage side: gap-free sweep.
    assert cfg.robot.policy_params == {"lane_spacing": cfg.robot.coverage_side}
    crw = build_config(_smoke_data(movement_policy="crw", policy_params={"sigma": 12}))
    assert crw.robot.policy_params == {"sigma": 12.0, "persistence": 0.7}


def test_build_policy_selects_configured_class() -> None:
    cases = {
        "ballistic": Ballistic,
        "crw": CorrelatedRandomWalk,
        "levy": LevyWalk,
        "boustrophedon": Boustrophedon,
    }
    for name, cls in cases.items():
        cfg = build_config(_smoke_data(movement_policy=name))
        assert isinstance(build_policy(cfg.robot), cls)
