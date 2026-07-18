"""Layout ground-truth API: visibility queries and canonical round-trip.

The visibility math is exercised on hand-built layouts with tiny known masks
(no generator involved), so every expected pixel count below is checkable by
eye. Only the round-trip test generates a real world.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swarm_perception.world.generate import generate_world
from swarm_perception.world.layout import Layout, LayoutObject
from swarm_perception.world.rle import encode_mask

REPO = Path(__file__).resolve().parents[1]
PNG_SET = REPO / "pre_assets" / "pngs" / "5"

# 4x4 L-shape: left column and bottom row set (7 foreground pixels).
# The top-right 3x3 corner is fully transparent.
L_MASK = np.array(
    [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
    ],
    dtype=bool,
)

# 4x4 asymmetric mask: exactly two pixels, at local (row=1, col=2) and
# (row=3, col=1). Any off-by-one in the crop-to-local translation moves
# onto a zero pixel.
DOTS_MASK = np.zeros((4, 4), dtype=bool)
DOTS_MASK[1, 2] = True
DOTS_MASK[3, 1] = True


def make_object(oid: int, bbox: tuple[int, int, int, int], mask: np.ndarray) -> LayoutObject:
    x1, y1, x2, y2 = bbox
    assert mask.shape == (y2 - y1, x2 - x1)
    return LayoutObject(
        id=oid,
        label=f"label{oid}",
        png=f"{oid}.png",
        bbox=bbox,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_rle=encode_mask(mask),
    )


def make_layout(*objects: LayoutObject, width: int = 100, height: int = 100) -> Layout:
    return Layout(
        background_image="bg.png",
        width=width,
        height=height,
        generator={},
        objects=list(objects),
    )


@pytest.fixture()
def l_layout() -> Layout:
    """Single L-mask object with bbox (10, 10, 14, 14)."""
    return make_layout(make_object(0, (10, 10, 14, 14), L_MASK))


@pytest.fixture()
def dots_layout() -> Layout:
    """Single two-pixel object with bbox NOT at the origin: (10, 20, 14, 24).

    Global foreground pixels: (x=12, y=21) and (x=11, y=23).
    """
    return make_layout(make_object(7, (10, 20, 14, 24), DOTS_MASK))


# --------------------------------------------------------------------------
# bbox mode
# --------------------------------------------------------------------------


def test_bbox_mode_overlap_visible(l_layout) -> None:
    # Intersection (12, 12, 14, 14): area 4.
    assert l_layout.visible_in((12, 12, 30, 30), mode="bbox") == [0]


def test_bbox_mode_touching_edges_share_no_pixels(l_layout) -> None:
    # Half-open rects: adjacency along an edge or at a corner is not overlap.
    assert l_layout.visible_in((14, 10, 20, 14), mode="bbox") == []  # right edge
    assert l_layout.visible_in((10, 14, 14, 20), mode="bbox") == []  # bottom edge
    assert l_layout.visible_in((0, 0, 10, 10), mode="bbox") == []  # corner point


def test_bbox_mode_disjoint(l_layout) -> None:
    assert l_layout.visible_in((50, 50, 60, 60), mode="bbox") == []


def test_bbox_mode_min_visible_px_threshold(l_layout) -> None:
    # Intersection (10, 10, 12, 13): 2 x 3 = 6 pixels.
    crop = (10, 10, 12, 13)
    assert l_layout.visible_in(crop, min_visible_px=6, mode="bbox") == [0]
    assert l_layout.visible_in(crop, min_visible_px=7, mode="bbox") == []


# --------------------------------------------------------------------------
# mask mode
# --------------------------------------------------------------------------


def test_mask_mode_transparent_corner_not_visible(l_layout) -> None:
    # Crop covers local rows 0..2, cols 1..3: inside the bbox but entirely in
    # the transparent corner of the L.
    crop = (11, 10, 14, 13)
    assert l_layout.visible_in(crop, mode="bbox") == [0]
    assert l_layout.visible_in(crop, mode="mask") == []


def test_mask_mode_exact_pixel_threshold(l_layout) -> None:
    # Crop covers local col 0, rows 0..3: exactly the 4 left-column pixels.
    crop = (10, 10, 11, 14)
    assert l_layout.visible_in(crop, min_visible_px=4, mode="mask") == [0]
    assert l_layout.visible_in(crop, min_visible_px=5, mode="mask") == []


def test_mask_mode_full_containment(l_layout) -> None:
    full = (0, 0, 100, 100)
    assert l_layout.visible_in(full, mode="mask") == [0]
    # The L has exactly 7 foreground pixels.
    assert l_layout.visible_in(full, min_visible_px=7, mode="mask") == [0]
    assert l_layout.visible_in(full, min_visible_px=8, mode="mask") == []


def test_mask_mode_local_frame_translation(dots_layout) -> None:
    # Exactly the global pixel (12, 21) == local (1, 2).
    assert dots_layout.visible_in((12, 21, 13, 22), mode="mask") == [7]
    # One pixel to the left: local (1, 1) is transparent.
    assert dots_layout.visible_in((11, 21, 12, 22), mode="mask") == []
    # Left half of the bbox contains only the (3, 1) pixel: count is 1.
    left_half = (10, 20, 12, 24)
    assert dots_layout.visible_in(left_half, min_visible_px=1, mode="mask") == [7]
    assert dots_layout.visible_in(left_half, min_visible_px=2, mode="mask") == []


def test_mask_mode_crop_clipped_outside_canvas(dots_layout) -> None:
    # Crop starts far outside the canvas; the in-canvas part reaches (12, 21).
    assert dots_layout.visible_in((-100, -100, 13, 22), mode="mask") == [7]
    # Crop extends far beyond the canvas; the in-canvas part reaches (11, 23).
    assert dots_layout.visible_in((11, 23, 500, 500), mode="mask") == [7]
    # Entirely outside the canvas.
    assert dots_layout.visible_in((-50, -50, 0, 0), mode="mask") == []


def test_visible_ids_sorted_ascending() -> None:
    # Objects deliberately stored out of id order.
    layout = make_layout(
        make_object(5, (10, 20, 14, 24), DOTS_MASK),
        make_object(1, (10, 10, 14, 14), L_MASK),
    )
    assert layout.visible_in((0, 0, 100, 100), mode="mask") == [1, 5]
    assert layout.visible_in((0, 0, 100, 100), mode="bbox") == [1, 5]


# --------------------------------------------------------------------------
# argument validation and conveniences
# --------------------------------------------------------------------------


def test_invalid_mode_raises(l_layout) -> None:
    with pytest.raises(ValueError, match="invalid mode"):
        l_layout.visible_in((0, 0, 100, 100), mode="alpha")


def test_min_visible_px_below_one_raises(l_layout) -> None:
    with pytest.raises(ValueError, match="min_visible_px"):
        l_layout.visible_in((0, 0, 100, 100), min_visible_px=0)


def test_objects_by_id_and_len() -> None:
    a = make_object(1, (10, 10, 14, 14), L_MASK)
    b = make_object(5, (10, 20, 14, 24), DOTS_MASK)
    layout = make_layout(a, b)
    assert len(layout) == 2
    assert layout.objects_by_id == {1: a, 5: b}


def test_mask_is_cached_and_read_only(l_layout) -> None:
    obj = l_layout.objects[0]
    mask = obj.mask()
    np.testing.assert_array_equal(mask, L_MASK)
    assert obj.mask() is mask
    assert not mask.flags.writeable


# --------------------------------------------------------------------------
# round-trip against a real generated world
# --------------------------------------------------------------------------


def test_generator_roundtrip_and_full_canvas_visibility(tmp_path) -> None:
    base = tmp_path / "base.png"
    Image.new("RGBA", (32, 32), (120, 140, 90, 255)).save(base)
    _, layout_path = generate_world(
        seed=123,
        pngs_dir=PNG_SET,
        num_objects=2,
        base=base,
        out_dir=tmp_path / "world",
        size=(300, 300),
        padding=8,
        object_sizes=[48],
        candidates_per_object=150,
        layout_attempts=4,
    )

    layout = Layout.load(layout_path)
    assert len(layout) == 2
    assert (layout.width, layout.height) == (300, 300)
    assert sorted(layout.objects_by_id) == [0, 1]

    resaved = tmp_path / "resaved.layout.json"
    layout.save(resaved)
    assert resaved.read_bytes() == layout_path.read_bytes()

    full = (0, 0, layout.width, layout.height)
    assert layout.visible_in(full, mode="mask") == [obj.id for obj in layout.objects]
    # padding=8 keeps every bbox at least 8 px from the canvas edge, so a
    # 1x1 crop in the corner overlaps nothing.
    assert layout.visible_in((0, 0, 1, 1), mode="mask") == []
    assert layout.visible_in((0, 0, 1, 1), mode="bbox") == []


def test_mask_shape_mismatch_raises() -> None:
    """A mask whose RLE header disagrees with bbox must raise, not clip."""
    obj = LayoutObject(
        id=3,
        label="bad",
        png="bad.png",
        bbox=(10, 10, 14, 14),  # 4x4 extent
        center=(12.0, 12.0),
        mask_rle=encode_mask(np.ones((3, 3), dtype=bool)),
    )
    with pytest.raises(ValueError, match="mask shape"):
        obj.mask()


def test_negative_extent_crop_raises(l_layout) -> None:
    with pytest.raises(ValueError, match="negative extent"):
        l_layout.visible_in((30, 30, 10, 10), mode="bbox")


def test_degenerate_crop_is_empty_not_error(l_layout) -> None:
    assert l_layout.visible_in((12, 12, 12, 20), mode="mask") == []


def test_mask_is_immutable_even_via_base(l_layout) -> None:
    mask = l_layout.objects[0].mask()
    assert mask.base is None  # owning copy: no writable array reachable
    with pytest.raises(ValueError):
        mask[0, 0] = False


def test_objects_by_id_cached(l_layout) -> None:
    assert l_layout.objects_by_id is l_layout.objects_by_id
