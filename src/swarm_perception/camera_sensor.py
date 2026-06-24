"""Camera sensor utilities for local perception and visualization."""

import cv2
import pygame as pg
from typing import Any

from swarm_perception.utils.paths import ASSETS_DIR


class CameraSensor:
    """Simulated onboard camera that crops the global background image."""

    def __init__(
        self,
        agent: Any,
        coverage_side: int | float,
        background_image: str,
        sensing_radius: int | None = None,
    ) -> None:
        """Initialize camera and drawing parameters.

        Args:
            agent: Robot agent whose position defines the crop center.
            coverage_side: Width and height of the square observation window.
            background_image: File name of the global map image in assets.
            sensing_radius: Optional communication/proximity radius for overlay.
        """
        self._agent = agent
        self.coverage_side = coverage_side
        self.background_image = background_image
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

        The method clips the crop to image bounds, then appends center
        coordinates to ``logs/camera_captures.txt`` for debugging.

        Returns:
            OpenCV image array for the visible patch around the robot.
        """
        img = cv2.imread(str(ASSETS_DIR / self.background_image))
        img_h, img_w, _ = img.shape

        # Crop using the same config-driven coverage square used for visualization.
        start_x, start_y, end_x, end_y = self._coverage_bounds()

        safe_start_x = max(0, start_x)
        safe_start_y = max(0, start_y)
        safe_end_x = min(img_w, end_x)
        safe_end_y = min(img_h, end_y)

        cropped_image = img[safe_start_y:safe_end_y, safe_start_x:safe_end_x]
        # cv2.imwrite(f'robot_{self._agent.id}_{pg.time.get_ticks()}.png', cropped_image)
        
        # Log the capture coordinates to a file
        log_file_path = ASSETS_DIR.parent / "logs" / "camera_captures.txt"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file_path, "a", encoding="utf-8") as f:
            center_x = (safe_start_x + safe_end_x) / 2
            center_y = (safe_start_y + safe_end_y) / 2
            f.write(f"{self._agent.id}, {center_x}, {center_y}\n")

        return cropped_image

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