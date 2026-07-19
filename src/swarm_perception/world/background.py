"""Load-once background image serving crops as array views.

Replaces the per-photo ``cv2.imread`` in the camera sensor (profiled at ~90%
of sim-core cost): the PNG is decoded exactly once and every crop is a numpy
view into the same array.

The pixel array keeps OpenCV's BGR channel order so behavior is identical for
every current consumer; the one-time BGR->RGB conversion happens at the
perception boundary (``perception/crops.py``, design decision D11).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

Rect = tuple[int, int, int, int]


class Background:
    """The composed world image, loaded once and shared by all robots."""

    def __init__(self, path: str | Path) -> None:
        """Decode the background PNG into memory.

        Args:
            path: Image file readable by OpenCV.

        Raises:
            FileNotFoundError: If the file is missing or not decodable.
        """
        self._path = Path(path)
        pixels = cv2.imread(str(self._path), cv2.IMREAD_COLOR)
        if pixels is None:
            raise FileNotFoundError(f"background image not found or unreadable: {self._path}")
        self._bgr: np.ndarray = pixels
        self._mean_color: tuple[int, int, int] | None = None

    @property
    def width(self) -> int:
        return int(self._bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self._bgr.shape[0])

    def crop(self, center: tuple[float, float], side: float) -> tuple[np.ndarray, Rect]:
        """Return the square view under ``center``, clipped to image bounds.

        Bounds replicate the legacy ``take_photo`` exactly: rounded square
        around the center, then clamped, so crops at the world edge are
        smaller than ``side``. The logged ground-truth bbox is always this
        clipped rect; padding for the encoder happens downstream.

        Args:
            center: ``(x, y)`` crop center in image pixels.
            side: Square edge length in pixels.

        Returns:
            ``(view, (x1, y1, x2, y2))`` — a zero-copy view into the
            background array and the clipped rect it covers.
        """
        half = side / 2
        x1 = max(0, int(round(center[0] - half)))
        y1 = max(0, int(round(center[1] - half)))
        x2 = min(self.width, int(round(center[0] + half)))
        y2 = min(self.height, int(round(center[1] + half)))
        return self._bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    def mean_color(self) -> tuple[int, int, int]:
        """Mean BGR color of the whole image (cached); the edge-pad color."""
        if self._mean_color is None:
            b, g, r = self._bgr.reshape(-1, 3).mean(axis=0)
            self._mean_color = (int(round(b)), int(round(g)), int(round(r)))
        return self._mean_color
