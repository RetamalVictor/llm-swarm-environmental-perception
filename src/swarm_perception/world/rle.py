"""Run-length encoding for binary object masks.

Format
------
``"W,H|c0 c1 c2 ..."`` where ``W`` and ``H`` are the mask width and height in
pixels, and ``c0 c1 c2 ...`` are space-separated run lengths over the
row-major (C-order) flattened mask. Runs alternate between 0-pixels and
1-pixels and always start with a 0-run: if the first pixel is 1, the leading
0-run has length 0. The counts sum to ``W * H``.

Examples: an all-zero 3x2 mask is ``"3,2|6"``; an all-one 3x2 mask is
``"3,2|0 6"``; a zero-pixel mask (``W*H == 0``) has an empty counts section.
"""

from __future__ import annotations

import numpy as np


def encode_mask(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask (nonzero == foreground) to an RLE string."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {arr.shape}")
    height, width = arr.shape
    flat = (arr != 0).ravel(order="C")

    if flat.size == 0:
        counts: list[int] = []
    else:
        boundaries = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
        edges = np.concatenate(([0], boundaries, [flat.size]))
        counts = np.diff(edges).tolist()
        if flat[0]:
            # First run is foreground: prepend the zero-length 0-run.
            counts.insert(0, 0)

    return f"{width},{height}|" + " ".join(str(c) for c in counts)


def decode_mask(rle: str) -> np.ndarray:
    """Decode an RLE string to a 2-D boolean mask of shape (H, W)."""
    header, sep, body = rle.partition("|")
    if not sep:
        raise ValueError("invalid RLE: missing '|' separator")
    try:
        width_s, height_s = header.split(",")
        width, height = int(width_s), int(height_s)
    except ValueError as exc:
        raise ValueError(f"invalid RLE header {header!r}, expected 'W,H'") from exc
    if width < 0 or height < 0:
        raise ValueError(f"invalid RLE dimensions {width}x{height}")

    try:
        counts = np.array([int(c) for c in body.split()], dtype=np.int64)
    except ValueError as exc:
        raise ValueError("invalid RLE counts: not all integers") from exc
    if counts.size and counts.min() < 0:
        raise ValueError("invalid RLE counts: negative run length")
    if counts.sum() != width * height:
        raise ValueError(
            f"invalid RLE: counts sum to {counts.sum()}, expected {width * height}"
        )

    values = np.arange(counts.size, dtype=np.int64) % 2  # 0-run, 1-run, 0-run, ...
    flat = np.repeat(values, counts).astype(bool)
    return flat.reshape(height, width)
