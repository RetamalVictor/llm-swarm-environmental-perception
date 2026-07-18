"""Crop correctness for the load-once background (world/background.py)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from swarm_perception.world.background import Background


@pytest.fixture()
def bg(tmp_path: Path) -> tuple[Background, np.ndarray]:
    """A 100x80 gradient image where pixel value encodes its coordinates."""
    height, width = 80, 100
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]  # B channel = x
    img[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]  # G channel = y
    path = tmp_path / "bg.png"
    assert cv2.imwrite(str(path), img)
    return Background(path), img


def test_center_crop_pixels_and_rect(bg) -> None:
    background, img = bg
    crop, rect = background.crop((50, 40), 10)
    assert rect == (45, 35, 55, 45)
    assert crop.shape == (10, 10, 3)
    assert np.array_equal(crop, img[35:45, 45:55])


def test_edge_clip_top_left(bg) -> None:
    background, _ = bg
    crop, rect = background.crop((2, 3), 10)
    # Unclamped square would start at (-3, -2); clipped to image bounds.
    assert rect == (0, 0, 7, 8)
    assert crop.shape == (8, 7, 3)


def test_edge_clip_bottom_right(bg) -> None:
    background, _ = bg
    crop, rect = background.crop((99, 79), 10)
    assert rect == (94, 74, 100, 80)
    assert crop.shape == (6, 6, 3)


def test_crop_is_a_view_not_a_copy(bg) -> None:
    background, _ = bg
    crop, _ = background.crop((50, 40), 10)
    assert np.shares_memory(crop, background._bgr)


def test_rounding_matches_legacy_take_photo(bg) -> None:
    background, img = bg
    # Legacy used int(round(center +/- side/2)); fractional centers must agree.
    crop, rect = background.crop((10.5, 10.5), 5)
    assert rect == (8, 8, 13, 13)
    assert np.array_equal(crop, img[8:13, 8:13])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Background(tmp_path / "nope.png")


def test_mean_color(tmp_path: Path) -> None:
    img = np.full((10, 10, 3), (10, 20, 30), np.uint8)
    path = tmp_path / "flat.png"
    assert cv2.imwrite(str(path), img)
    assert Background(path).mean_color() == (10, 20, 30)
