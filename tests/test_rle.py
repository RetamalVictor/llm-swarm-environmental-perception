"""RLE mask codec: round trips, documented format, and input validation."""

from pathlib import Path

import numpy as np
import pytest

from swarm_perception.world.rle import decode_mask, encode_mask

REPO = Path(__file__).resolve().parents[1]
PNG_SET = REPO / "pre_assets" / "pngs" / "5"


def test_round_trip_random_masks() -> None:
    rng = np.random.default_rng(1234)
    for shape in [(1, 1), (1, 9), (9, 1), (3, 7), (16, 16), (33, 47)]:
        for density in (0.1, 0.5, 0.9):
            mask = rng.random(shape) < density
            decoded = decode_mask(encode_mask(mask))
            assert decoded.dtype == np.bool_
            assert decoded.shape == shape
            np.testing.assert_array_equal(decoded, mask)


def test_round_trip_all_zero_and_all_one() -> None:
    zero = np.zeros((4, 6), dtype=bool)
    one = np.ones((4, 6), dtype=bool)
    assert encode_mask(zero) == "6,4|24"
    assert encode_mask(one) == "6,4|0 24"
    np.testing.assert_array_equal(decode_mask(encode_mask(zero)), zero)
    np.testing.assert_array_equal(decode_mask(encode_mask(one)), one)


def test_documented_format_known_pattern() -> None:
    # Row-major flatten: 0 1 1 1 0 0 -> runs of 1 zero, 3 ones, 2 zeros.
    mask = np.array([[0, 1, 1], [1, 0, 0]], dtype=bool)
    rle = encode_mask(mask)
    assert rle == "3,2|1 3 2"
    np.testing.assert_array_equal(decode_mask(rle), mask)


def test_encode_accepts_nonbool_arrays() -> None:
    alpha = np.array([[0, 128], [255, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(decode_mask(encode_mask(alpha)), alpha != 0)


def test_round_trip_real_png_alpha() -> None:
    Image = pytest.importorskip("PIL.Image")
    with Image.open(PNG_SET / "16.png") as im:
        thumb = im.convert("RGBA")
        thumb.thumbnail((64, 64), Image.Resampling.LANCZOS)
        alpha = np.asarray(thumb.getchannel("A")) > 0
    assert alpha.any() and not alpha.all()  # a real sprite, not a filled square
    np.testing.assert_array_equal(decode_mask(encode_mask(alpha)), alpha)


def test_encode_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        encode_mask(np.zeros((2, 2, 4), dtype=np.uint8))


@pytest.mark.parametrize(
    "bad",
    [
        "3,2",  # missing separator
        "3|6",  # malformed header
        "a,b|4",  # non-integer dims
        "2,2|1 1",  # counts do not sum to W*H
        "2,2|5 -1",  # negative run
        "2,2|x y",  # non-integer counts
    ],
)
def test_decode_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        decode_mask(bad)
