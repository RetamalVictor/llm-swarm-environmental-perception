"""Actuator module for velocity-to-motion integration in the simulator."""

import random
from typing import Any

random.seed(43)


class Actuator:
    """Apply movement commands to a robot pose each simulation tick."""

    def __init__(self, agent: Any) -> None:
        """Attach the actuator and initialize heading.

        Args:
            agent: Violet agent instance whose pose is mutated in place.
        """
        self.agent = agent
        self.agent.current_angle = random.uniform(0, 360)

    def update(self, linear_speed: float, angular_velocity: float) -> None:
        """Integrate one control step using polar movement.

        Args:
            linear_speed: Forward speed in simulator units per tick.
            angular_velocity: Heading change in degrees per tick.
        """
        self.agent.current_angle += angular_velocity
        self.agent.current_angle %= 360

        self.agent.move.from_polar((linear_speed, self.agent.current_angle))
        self.agent.pos += self.agent.move
