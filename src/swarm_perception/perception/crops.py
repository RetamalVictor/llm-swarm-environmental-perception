"""Epoch-batched crop extraction feeding the perception encoder (D11).

THE SINGLE BGR->RGB BOUNDARY OF THE WHOLE CODEBASE LIVES IN THIS MODULE.
Everything upstream — :class:`~swarm_perception.world.background.Background`,
the camera sensor, anything touched by OpenCV — stores pixels in BGR channel
order. Everything downstream — the CLIP encoder, committed fixture PNGs,
anything saved through PIL — is RGB. The conversion happens exactly once, in
:func:`extract_crops`, via the ``[..., ::-1]`` channel flip when a crop view
is pasted onto its output canvas. No other module may reorder channels.

Edge handling: crops whose ideal square sticks out of the world are PADDED to
the full coverage square with the background's mean color — never resized or
squashed. Padding placement mirrors which side got clipped: pixel ``(0, 0)``
of every output corresponds to the ideal (unclipped) rect's top-left corner,
so the crop stays geometrically aligned with the world.

Geometry is logged, not re-derived: the rects returned here are exactly the
clipped rects reported by :meth:`Background.crop`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from swarm_perception.world.background import Background, Rect


def extract_crops(
    background: Background,
    captures: Sequence[tuple[int, tuple[float, float]]],
    coverage_side: float,
) -> tuple[list[np.ndarray], list[Rect]]:
    """Extract one padded RGB crop per robot for a capture epoch.

    Args:
        background: The shared load-once world image (BGR internally).
        captures: ``(robot_id, (x, y))`` pairs, strictly sorted by robot id.
            Sorted order is enforced because the encoder embeds each epoch as
            a single batch in exactly this order (determinism contract, D11).
        coverage_side: Side length of the coverage square in pixels.

    Returns:
        ``(crops_rgb, rects)`` — parallel lists in input order. Each crop is
        an owned uint8 RGB array of exactly ``side x side`` pixels (``side =
        round(coverage_side)``); each rect is the clipped ``(x1, y1, x2, y2)``
        exactly as reported by :meth:`Background.crop` — the ground-truth
        geometry to log, never recomputed here.

    Raises:
        ValueError: If ``coverage_side`` rounds below 1 pixel or robot ids
            are not strictly increasing.
    """
    out_side = int(round(coverage_side))
    if out_side < 1:
        raise ValueError(f"coverage_side must round to >= 1 pixel, got {coverage_side}")
    ids = [robot_id for robot_id, _ in captures]
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError(
            f"captures must be strictly sorted by robot id (batch order is part "
            f"of the determinism contract, D11); got ids {ids}"
        )

    # Pad color: the background's cached mean, flipped BGR -> RGB.
    mean_b, mean_g, mean_r = background.mean_color()
    pad_rgb = (mean_r, mean_g, mean_b)

    half = coverage_side / 2
    crops_rgb: list[np.ndarray] = []
    rects: list[Rect] = []
    for _, (cx, cy) in captures:
        view, rect = background.crop((cx, cy), coverage_side)
        x1, y1, _, _ = rect

        # Ideal (unclipped) top-left, same rounding as Background.crop; the
        # clip only ever moves x1/y1 inward, so the paste offset is >= 0.
        ideal_x1 = int(round(cx - half))
        ideal_y1 = int(round(cy - half))
        off_x = x1 - ideal_x1
        off_y = y1 - ideal_y1

        canvas = np.empty((out_side, out_side, 3), dtype=np.uint8)
        canvas[:] = pad_rgb

        # For odd coverage sides, banker's rounding can make the ideal rect
        # one pixel larger than ``out_side``; trim that overflow at the
        # bottom/right so the top-left anchor is preserved. Every shipped
        # config uses an even side, where the ideal rect is exactly square.
        paste_h = min(view.shape[0], out_side - off_y)
        paste_w = min(view.shape[1], out_side - off_x)
        if paste_h > 0 and paste_w > 0:
            # THE BGR->RGB BOUNDARY: the one channel flip in the codebase.
            canvas[off_y : off_y + paste_h, off_x : off_x + paste_w] = view[
                :paste_h, :paste_w, ::-1
            ]

        crops_rgb.append(canvas)
        rects.append(rect)
    return crops_rgb, rects
