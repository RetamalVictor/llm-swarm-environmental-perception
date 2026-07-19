"""Geometric ground-truth query API over the generator's ``.layout.json``.

:class:`Layout` mirrors the sidecar JSON written by
:mod:`swarm_perception.world.generate` and answers the benchmark's core
question exactly: which objects are visible inside a crop rect, judged either
by bounding-box overlap or through each object's alpha mask.

Coordinate conventions
----------------------
All coordinates are pixels with the origin at the canvas top-left, ``x``
growing right and ``y`` growing down. Every rect — object ``bbox`` and crop
``bbox`` alike — is half-open: ``(x1, y1, x2, y2)`` covers ``x1 <= x < x2``
and ``y1 <= y < y2``, matching numpy slicing ``arr[y1:y2, x1:x2]`` and the
``Background.crop`` rects. Consequently two rects that merely touch along an
edge share no pixels.

Object masks are stored in the object's *local* bbox frame: ``mask()[r, c]``
corresponds to canvas pixel ``(x1 + c, y1 + r)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from swarm_perception.world.rle import decode_mask

Rect = tuple[int, int, int, int]

_VISIBILITY_MODES = ("bbox", "mask")


@dataclass(frozen=True)
class LayoutObject:
    """One placed object from a ``.layout.json`` ``objects`` entry."""

    id: int
    label: str
    png: str
    bbox: Rect  # half-open canvas rect (x1, y1, x2, y2)
    center: tuple[float, float]
    mask_rle: str

    def mask(self) -> np.ndarray:
        """Boolean alpha mask in the object's local bbox frame (H, W).

        The RLE decode is cached on first call. The dataclass is frozen, so
        the cache is stashed in ``__dict__`` via ``object.__setattr__``; the
        cached array is an owning, read-only copy so sharing it is safe.

        Raises:
            ValueError: If the decoded mask's shape disagrees with ``bbox`` —
                numpy would otherwise clip mismatched slices silently and
                corrupt every visibility answer.
        """
        cached = self.__dict__.get("_mask")
        if cached is None:
            cached = np.array(decode_mask(self.mask_rle))
            x1, y1, x2, y2 = self.bbox
            if cached.shape != (y2 - y1, x2 - x1):
                raise ValueError(
                    f"object {self.id}: mask shape {cached.shape} does not match "
                    f"bbox extent {(y2 - y1, x2 - x1)}"
                )
            cached.flags.writeable = False
            object.__setattr__(self, "_mask", cached)
        return cached


@dataclass(frozen=True)
class Layout:
    """Parsed ``.layout.json``: canvas metadata plus placed objects."""

    background_image: str
    width: int
    height: int
    generator: dict
    objects: list[LayoutObject]

    @classmethod
    def load(cls, path: str | Path) -> Layout:
        """Read a ``.layout.json`` file written by the world generator."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        objects = [
            LayoutObject(
                id=entry["id"],
                label=entry["label"],
                png=entry["png"],
                bbox=(*entry["bbox"],),
                center=(*entry["center"],),
                mask_rle=entry["mask_rle"],
            )
            for entry in data["objects"]
        ]
        return cls(
            background_image=data["background_image"],
            width=data["width"],
            height=data["height"],
            generator=data["generator"],
            objects=objects,
        )

    def save(self, path: str | Path) -> None:
        """Write ``.layout.json`` using the generator's canonical dumping.

        Same serialization as :func:`world.generate.generate_world` —
        ``sort_keys``, 2-space indent, fixed separators, ``ensure_ascii=False``,
        UTF-8, LF newlines, trailing newline — so ``load(...)`` followed by
        ``save(...)`` reproduces the generator's file byte for byte.
        """
        payload = {
            "background_image": self.background_image,
            "width": self.width,
            "height": self.height,
            "generator": self.generator,
            "objects": [
                {
                    "id": obj.id,
                    "label": obj.label,
                    "png": obj.png,
                    "bbox": list(obj.bbox),
                    "center": list(obj.center),
                    "mask_rle": obj.mask_rle,
                }
                for obj in self.objects
            ],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                       separators=(",", ": "))
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @property
    def objects_by_id(self) -> dict[int, LayoutObject]:
        """Id -> object mapping, built once and cached (Layout is frozen)."""
        cached = self.__dict__.get("_objects_by_id")
        if cached is None:
            cached = {obj.id: obj for obj in self.objects}
            object.__setattr__(self, "_objects_by_id", cached)
        return cached

    def __len__(self) -> int:
        return len(self.objects)

    def visible_in(
        self,
        crop_bbox: Rect,
        min_visible_px: int = 1,
        mode: str = "mask",
    ) -> list[int]:
        """Ids of objects visible inside ``crop_bbox``, sorted ascending.

        ``mode="bbox"`` counts an object as visible iff the intersection area
        of ``crop_bbox`` and the object's bbox is at least ``min_visible_px``.

        ``mode="mask"`` is exact visibility through the alpha mask: within the
        rect intersection, at least ``min_visible_px`` of the object's mask
        pixels must be set. An empty rect intersection is never visible.

        Note the unit difference: at the same ``min_visible_px``, bbox mode
        thresholds intersection *area* while mask mode thresholds *opaque
        pixels* — for sprites with transparent margins, bbox mode can pass a
        threshold that mask mode fails. Do not compare modes at a fixed
        threshold without accounting for this.

        ``crop_bbox`` may extend beyond the canvas; the out-of-canvas part
        simply cannot overlap any object. A degenerate rect (``x1 == x2`` or
        ``y1 == y2``) is legitimately empty; a *negative* extent is always a
        caller bug and raises.
        """
        if mode not in _VISIBILITY_MODES:
            raise ValueError(
                f"invalid mode {mode!r}, expected one of {_VISIBILITY_MODES}"
            )
        if min_visible_px < 1:
            raise ValueError(
                f"min_visible_px must be at least 1, got {min_visible_px}"
            )

        cx1, cy1, cx2, cy2 = crop_bbox
        if cx2 < cx1 or cy2 < cy1:
            raise ValueError(
                f"crop_bbox has negative extent: {crop_bbox} (expected half-open "
                "(x1, y1, x2, y2) with x1 <= x2 and y1 <= y2)"
            )
        visible: list[int] = []
        for obj in self.objects:
            ox1, oy1, ox2, oy2 = obj.bbox
            ix1, iy1 = max(cx1, ox1), max(cy1, oy1)
            ix2, iy2 = min(cx2, ox2), min(cy2, oy2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue
            if mode == "bbox":
                count = (ix2 - ix1) * (iy2 - iy1)
            else:
                # Translate the intersection into the object's local frame.
                local = obj.mask()[iy1 - oy1 : iy2 - oy1, ix1 - ox1 : ix2 - ox1]
                count = int(np.count_nonzero(local))
            if count >= min_visible_px:
                visible.append(obj.id)
        return sorted(visible)
