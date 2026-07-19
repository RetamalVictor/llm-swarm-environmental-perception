"""Crop extraction correctness (perception/crops.py) — runs without torch.

Covers the codebase's single BGR->RGB boundary, mean-color padding geometry
for all four edges plus a corner, and exact rect passthrough.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from swarm_perception.perception.crops import extract_crops
from swarm_perception.world.background import Background


@pytest.fixture()
def bg(tmp_path: Path) -> tuple[Background, np.ndarray]:
    """A 50x40 gradient image where pixel value encodes its coordinates."""
    height, width = 40, 50
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]  # B channel = x
    img[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]  # G channel = y
    img[:, :, 2] = 200  # constant R so all three channels are distinct
    path = tmp_path / "bg.png"
    assert cv2.imwrite(str(path), img)
    return Background(path), img


def pad_rgb(background: Background) -> tuple[int, int, int]:
    b, g, r = background.mean_color()
    return (r, g, b)


def test_bgr_to_rgb_flip(tmp_path: Path) -> None:
    """A pixel saved as pure blue must come out blue in RGB, not red.

    cv2 stores blue as (255, 0, 0) in channel order B, G, R. Without the
    boundary flip that byte pattern read as RGB would be pure red — so this
    single pixel proves the conversion happened.
    """
    img = np.zeros((20, 20, 3), np.uint8)
    img[6, 8] = (255, 0, 0)  # pure blue in BGR
    path = tmp_path / "blue.png"
    assert cv2.imwrite(str(path), img)
    background = Background(path)

    crops, rects = extract_crops(background, [(0, (10.0, 10.0))], 12)
    assert rects == [(4, 4, 16, 16)]
    # Blue pixel at image (x=8, y=6) -> output (row 2, col 4).
    assert crops[0][2, 4].tolist() == [0, 0, 255]  # RGB blue
    assert crops[0][2, 4].tolist() != [255, 0, 0]  # would mean no flip


def test_interior_crop_content_and_rect(bg) -> None:
    background, img = bg
    crops, rects = extract_crops(background, [(0, (25.0, 20.0))], 10)
    assert rects == [(20, 15, 30, 25)]
    assert crops[0].shape == (10, 10, 3)
    assert crops[0].dtype == np.uint8
    # Full content equality against the channel-flipped source window.
    assert np.array_equal(crops[0], img[15:25, 20:30, ::-1])


@pytest.mark.parametrize(
    ("center", "rect", "pad_mask_fn"),
    [
        # Left edge: ideal x1 = -3, so output columns 0..2 are padding.
        ((2.0, 20.0), (0, 15, 7, 25), lambda m: m[:, :3].fill(True)),
        # Right edge: ideal x2 = 53 clips to 50, so columns 7..9 are padding.
        ((48.0, 20.0), (43, 15, 50, 25), lambda m: m[:, 7:].fill(True)),
        # Top edge: ideal y1 = -3, so rows 0..2 are padding.
        ((25.0, 2.0), (20, 0, 30, 7), lambda m: m[:3, :].fill(True)),
        # Bottom edge: ideal y2 = 43 clips to 40, so rows 7..9 are padding.
        ((25.0, 38.0), (20, 33, 30, 40), lambda m: m[7:, :].fill(True)),
    ],
)
def test_edge_padding_geometry(bg, center, rect, pad_mask_fn) -> None:
    background, img = bg
    crops, rects = extract_crops(background, [(0, center)], 10)
    assert rects == [rect]
    crop = crops[0]
    assert crop.shape == (10, 10, 3)

    pad_mask = np.zeros((10, 10), dtype=bool)
    pad_mask_fn(pad_mask)
    # Padded region: exactly the mean color, flipped to RGB.
    assert np.array_equal(
        crop[pad_mask], np.tile(pad_rgb(background), (pad_mask.sum(), 1))
    )
    # Content region: the clipped source window, channel-flipped, anchored so
    # output (0, 0) corresponds to the ideal rect's top-left.
    x1, y1, x2, y2 = rect
    assert np.array_equal(
        crop[~pad_mask].reshape(y2 - y1, x2 - x1, 3), img[y1:y2, x1:x2, ::-1]
    )


def test_corner_padding_geometry(bg) -> None:
    """Top-left corner clip pads both the top rows and the left columns."""
    background, img = bg
    crops, rects = extract_crops(background, [(0, (2.0, 3.0))], 10)
    assert rects == [(0, 0, 7, 8)]
    crop = crops[0]
    # Ideal rect starts at (-3, -2): 3 pad columns, 2 pad rows.
    assert np.array_equal(crop[2:10, 3:10], img[0:8, 0:7, ::-1])
    expected_pad = np.array(pad_rgb(background), np.uint8)
    assert (crop[:2, :] == expected_pad).all()
    assert (crop[:, :3] == expected_pad).all()


def test_rect_passthrough_is_exact(bg) -> None:
    """Returned rects are Background.crop's rects, not re-derived geometry."""
    background, _ = bg
    centers = [(2.0, 3.0), (25.5, 20.5), (48.0, 38.0)]
    captures = [(i, c) for i, c in enumerate(centers)]
    _, rects = extract_crops(background, captures, 10)
    assert rects == [background.crop(c, 10)[1] for c in centers]


def test_output_is_an_owned_copy(bg) -> None:
    background, _ = bg
    crops, _ = extract_crops(background, [(0, (25.0, 20.0))], 10)
    assert not np.shares_memory(crops[0], background._bgr)


def test_unsorted_or_duplicate_robot_ids_raise(bg) -> None:
    background, _ = bg
    with pytest.raises(ValueError, match="strictly sorted"):
        extract_crops(background, [(1, (25.0, 20.0)), (0, (25.0, 20.0))], 10)
    with pytest.raises(ValueError, match="strictly sorted"):
        extract_crops(background, [(0, (25.0, 20.0)), (0, (26.0, 20.0))], 10)


def test_tiny_coverage_side_raises(bg) -> None:
    background, _ = bg
    with pytest.raises(ValueError, match="coverage_side"):
        extract_crops(background, [(0, (25.0, 20.0))], 0.4)


def test_empty_captures(bg) -> None:
    background, _ = bg
    assert extract_crops(background, [], 10) == ([], [])
