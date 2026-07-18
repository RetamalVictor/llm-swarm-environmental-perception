"""Camera sensor utilities for local perception and visualization."""

import pygame as pg
from typing import Any

from swarm_perception.world.background import Background


class CameraSensor:
    """Simulated onboard camera that crops the shared background image."""

    def __init__(
        self,
        agent: Any,
        coverage_side: int | float,
        background: Background,
        sensing_radius: int | None = None,
    ) -> None:
        """Initialize camera and drawing parameters.

        Args:
            agent: Robot agent whose position defines the crop center.
            coverage_side: Width and height of the square observation window.
            background: Shared load-once world image (one instance per run).
            sensing_radius: Optional communication/proximity radius for overlay.
        """
        self._agent = agent
        self.coverage_side = coverage_side
        self._background = background
        self.sensing_radius = sensing_radius
        self._label_font = None
        # Scale label with sensing window so visual size follows config changes.
        self._label_font_size = max(16, int(self.coverage_side * 0.25))

    def _coverage_bounds(self) -> tuple[int, int, int, int]:
        """Return coverage square bounds centered at robot position."""
        half_side = self.coverage_side / 2
        start_x = int(round(self._agent.pos.x - half_side))
        start_y = int(round(self._agent.pos.y - half_side))
        end_x = int(round(self._agent.pos.x + half_side))
        end_y = int(round(self._agent.pos.y + half_side))
        return start_x, start_y, end_x, end_y

    def take_photo(self) -> Any:
        """Capture and return the current local image crop.

        Returns:
            Zero-copy array view of the visible patch around the robot,
            clipped to image bounds (BGR, same as the legacy behavior).
        """
        crop, _rect = self._background.crop(
            (self._agent.pos.x, self._agent.pos.y), self.coverage_side
        )
        return crop

    def show_outline(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        """Draw sensor overlay and robot id label on the active display surface.

        Args:
            color: RGB color of the sensing square border.
        """
        start_x, start_y, end_x, end_y = self._coverage_bounds()
        bounding_rect = pg.Rect(
            start_x,
            start_y,
            end_x - start_x,
            end_y - start_y,
        )
        screen = pg.display.get_surface()
        if screen is None:
            return

        if self.sensing_radius is not None:
            pg.draw.circle(
                screen,
                (255, 255, 0),
                (int(self._agent.pos.x), int(self._agent.pos.y)),
                int(self.sensing_radius),
                width=2,
            )
        
        pg.draw.rect(screen, color, bounding_rect, width=2)

        # Draw a clear robot id at the center of the sensing rectangle.
        if not pg.font.get_init():
            pg.font.init()

        if self._label_font is None:
            self._label_font = pg.font.SysFont("arial", self._label_font_size, bold=True)

        label_text = str(int(self._agent.id))
        label_surface = self._label_font.render(label_text, True, (255, 255, 255))
        center_x, center_y = bounding_rect.center
        text_offset = max(6, int(self.coverage_side * 0.07))
        label_rect = label_surface.get_rect(center=(center_x, center_y + text_offset))

        # Add a thin dark backing to improve contrast on bright backgrounds.
        backing = label_rect.inflate(
            max(6, int(self.coverage_side * 0.04)),
            max(4, int(self.coverage_side * 0.03)),
        )
        pg.draw.rect(screen, (0, 0, 0), backing, border_radius=4)
        screen.blit(label_surface, label_rect)

    def detect_edges(self) -> str | bool:
        """Detect whether the sensing square touches any environment boundary.

        Returns:
            One of ``left``, ``right``, ``top``, ``bottom`` when near an edge;
            otherwise ``False``.
        """
        pos = self._agent.pos
        area = self._agent._area
        margin = self.coverage_side // 2

        if pos.x < area.left + margin:
            return "left"
        elif pos.x > area.right - margin:
            return "right"
        elif pos.y < area.top + margin:
            return "top"
        elif pos.y > area.bottom - margin:
            return "bottom"
        else:
            return False