"""Movement policies: per-tick heading/speed decisions for robots.

Movement is part of the benchmark task tuple: each robot owns one
:class:`MovementPolicy` instance selected by ``robot.movement_policy`` and
parameterized by ``robot.policy_params`` (validated in
:mod:`swarm_perception.config.schema`). Every tick the robot hands the policy
a read-only :class:`StepContext` and receives a ``(linear_speed,
angular_velocity)`` command that the actuator integrates into the pose.

All stochastic policies draw exclusively from the injected per-run
``random.Random`` (design decision D8): no ``random`` module globals and no
numpy RNG, so a run is fully reproduced by its config + seed.

Angles follow the pygame convention used by the actuator: degrees, ``0`` is
+x (east) and ``90`` is +y (down, screen coordinates).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from swarm_perception.config.schema import RobotCfg

# Heading (degrees) pointing into the arena from each sensor edge label.
_INWARD_HEADING = {"left": 0.0, "right": 180.0, "top": 90.0, "bottom": 270.0}


@dataclass(frozen=True)
class StepContext:
    """Read-only inputs a movement policy receives each tick.

    Attributes:
        pos: Robot position ``(x, y)`` in world pixels.
        heading: Current heading in degrees (pygame convention).
        edge: Sensor edge label (``"left"``/``"right"``/``"top"``/``"bottom"``)
            when the sensing square touches an arena boundary, else ``False``.
        bounds: Arena rectangle ``(left, top, right, bottom)`` in pixels.
        cfg: The run's robot configuration (speed, turn rate, coverage side).
        rng: The single seeded per-run RNG (D8). Policies must use only this.
    """

    pos: tuple[float, float]
    heading: float
    edge: str | bool
    bounds: tuple[float, float, float, float]
    cfg: RobotCfg
    rng: random.Random


class MovementPolicy(ABC):
    """Per-tick movement decision interface.

    Implementations may keep internal state (e.g. remaining flight length),
    so one instance must be constructed per robot.
    """

    @abstractmethod
    def step(self, ctx: StepContext) -> tuple[float, float]:
        """Decide this tick's movement command.

        Args:
            ctx: Current pose, edge status, arena bounds, config, and run RNG.

        Returns:
            ``(linear_speed, angular_velocity)`` — forward speed in pixels per
            tick and heading change in degrees per tick, integrated by
            :class:`~swarm_perception.sim.actuator.Actuator`.
        """


class Ballistic(MovementPolicy):
    """Straight-line motion with constant-rate turns away from arena edges.

    Exactly the pre-policy robot behavior: drive at ``robot.linear_speed`` and
    turn by ``robot.angular_velocity`` degrees per tick while the sensing
    square touches an arena boundary. Draws nothing from the RNG per tick, so
    a ballistic run reproduces the pre-extraction event stream byte for byte.
    """

    def step(self, ctx: StepContext) -> tuple[float, float]:
        linear_speed = ctx.cfg.linear_speed
        angular_velocity = 0.0
        if ctx.edge:
            angular_velocity = ctx.cfg.angular_velocity
        return linear_speed, angular_velocity


class CorrelatedRandomWalk(MovementPolicy):
    """Correlated random walk: persistent heading with Gaussian perturbation.

    Heading update per tick: ``theta <- theta + (1 - persistence) *
    N(0, sigma^2)`` via ``rng.gauss``. ``persistence`` in ``[0, 1]`` weights
    directional memory — ``1`` degenerates to ballistic motion, ``0`` is an
    uncorrelated random walk with per-tick turn std ``sigma``. At an arena
    edge the ballistic avoidance turn is added on top so robots re-enter.
    """

    def __init__(self, sigma: float, persistence: float) -> None:
        """Args: sigma — turn noise std in degrees; persistence — in [0, 1]."""
        self.sigma = float(sigma)
        self.persistence = float(persistence)

    def step(self, ctx: StepContext) -> tuple[float, float]:
        angular_velocity = (1.0 - self.persistence) * ctx.rng.gauss(0.0, self.sigma)
        if ctx.edge:
            angular_velocity += ctx.cfg.angular_velocity
        return ctx.cfg.linear_speed, angular_velocity


class LevyWalk(MovementPolicy):
    """Levy walk: straight flights with power-law lengths, redrawn directions.

    Each flight draws a fresh uniform direction and a Pareto flight length via
    inverse-CDF sampling, ``length = linear_speed * (1 - u) ** (-1 / alpha)``
    with ``u = rng.random()`` (scale ``x_min`` = one tick of travel), clamped
    at ``clamp`` pixels. The robot then flies straight until the length is
    exhausted. Touching an arena edge truncates the current flight and the
    next direction is drawn from the half-plane pointing back into the arena.
    """

    def __init__(self, alpha: float, clamp: float) -> None:
        """Args: alpha — Pareto tail exponent (> 0); clamp — max flight, px."""
        self.alpha = float(alpha)
        self.clamp = float(clamp)
        self._remaining = 0.0  # flight distance left, in pixels

    def step(self, ctx: StepContext) -> tuple[float, float]:
        speed = ctx.cfg.linear_speed
        if ctx.edge:
            self._remaining = 0.0  # truncate the flight at the boundary
        angular_velocity = 0.0
        if self._remaining <= 0.0:
            if isinstance(ctx.edge, str):
                # Redraw within the inward half-plane to leave the edge zone.
                heading = _INWARD_HEADING[ctx.edge] + ctx.rng.uniform(-90.0, 90.0)
            else:
                heading = ctx.rng.uniform(0.0, 360.0)
            length = speed * (1.0 - ctx.rng.random()) ** (-1.0 / self.alpha)
            self._remaining = min(length, self.clamp)
            angular_velocity = heading - ctx.heading
        self._remaining -= speed
        return speed, angular_velocity


class Boustrophedon(MovementPolicy):
    """Deterministic lawnmower sweep over horizontal lanes.

    The robot sweeps east/west along a lane; when it reaches the side margin
    it shifts vertically by ``lane_spacing`` pixels onto the next lane and
    sweeps back (serpentine). At the top/bottom margin the vertical direction
    reverses, so the sweep ping-pongs over the arena indefinitely. Turns are
    instantaneous heading snaps. Margins keep the camera square
    (``coverage_side / 2``) inside the arena.

    Ignores the RNG by design: the trajectory is a pure function of the spawn
    pose and the arena geometry, giving a deterministic coverage baseline
    against which the stochastic policies are compared.
    """

    def __init__(self, lane_spacing: float) -> None:
        """Args: lane_spacing — vertical distance between sweep lanes, px."""
        self.lane_spacing = float(lane_spacing)
        self._x_dir = 1  # +1 sweeps east, -1 sweeps west
        self._y_dir = 1  # +1 shifts down, -1 shifts up
        self._target_y: float | None = None  # shift target; None while sweeping

    def step(self, ctx: StepContext) -> tuple[float, float]:
        left, top, right, bottom = ctx.bounds
        margin = ctx.cfg.coverage_side / 2.0
        x, y = ctx.pos

        if self._target_y is None:
            at_end = x >= right - margin if self._x_dir > 0 else x <= left + margin
            if at_end:
                target = y + self._y_dir * self.lane_spacing
                if target > bottom - margin or target < top + margin:
                    self._y_dir = -self._y_dir  # serpentine bounce at top/bottom
                    target = y + self._y_dir * self.lane_spacing
                self._target_y = min(max(target, top + margin), bottom - margin)
                self._x_dir = -self._x_dir  # next lane sweeps the other way
        else:
            done = y >= self._target_y if self._y_dir > 0 else y <= self._target_y
            if done:
                self._target_y = None

        if self._target_y is None:
            desired = 0.0 if self._x_dir > 0 else 180.0
        else:
            desired = 90.0 if self._y_dir > 0 else 270.0
        return ctx.cfg.linear_speed, desired - ctx.heading


def build_policy(cfg: RobotCfg) -> MovementPolicy:
    """Construct one movement policy instance for one robot.

    ``cfg.policy_params`` arrives fully resolved from schema validation
    (unknown keys rejected, defaults filled in), so lookups here are total.

    Args:
        cfg: The run's robot configuration.

    Returns:
        A fresh policy instance (policies may carry per-robot state).
    """
    params = cfg.policy_params
    if cfg.movement_policy == "crw":
        return CorrelatedRandomWalk(sigma=params["sigma"], persistence=params["persistence"])
    if cfg.movement_policy == "levy":
        return LevyWalk(alpha=params["alpha"], clamp=params["clamp"])
    if cfg.movement_policy == "boustrophedon":
        return Boustrophedon(lane_spacing=params["lane_spacing"])
    return Ballistic()
