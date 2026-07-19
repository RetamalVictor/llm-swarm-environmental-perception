"""Deterministic world generator: sprites composited onto a base texture.

Port of ``pre_assets/scripts/generate_background.py`` with two contract
changes: the seed is mandatory (all randomness flows from one
``random.Random(seed)``), and every placed object's geometry — bounding box,
center, and alpha mask — is persisted to a ``.layout.json`` sidecar instead of
being discarded after compositing.

Outputs, written to ``--out``::

    background-<setname>-<n>obj-seed<seed>.png
    background-<setname>-<n>obj-seed<seed>.layout.json

Same arguments + same seed produce byte-identical files.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from swarm_perception.world.rle import encode_mask

# Layout search effort. More candidates/attempts spread objects further apart
# at the cost of generation time. Function arguments override these defaults.
CANDIDATES_PER_OBJECT = 800
LAYOUT_ATTEMPTS = 40

DEFAULT_SIZE = (2500, 2500)
DEFAULT_PADDING = 100
DEFAULT_OBJECT_SIZE = 300

LABELS_FILENAME = "labels.json"

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class _PreparedObject:
    """One object sprite, thumbnailed to its final placed size."""

    index: int  # position in sorted filename order; becomes the object id
    name: str  # source PNG filename
    image: Image.Image  # RGBA thumbnail at placed size


def _sorted_png_paths(pngs_dir: Path) -> list[Path]:
    """Sort set PNGs numerically by stem (1.png, 2.png, ...), then by name."""

    def numeric_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        return (int(stem), path.name) if stem.isdigit() else (10**9, path.name)

    return sorted(pngs_dir.glob("*.png"), key=numeric_key)


def _resolve_sizes(object_sizes: list[int], count: int) -> list[int]:
    """Expand a single size to all objects, or match sizes one-to-one."""
    if not object_sizes:
        raise ValueError("object_sizes must contain at least one value")
    if any(size <= 0 for size in object_sizes):
        raise ValueError(f"object_sizes must be positive, got {object_sizes}")
    if len(object_sizes) == 1:
        return [object_sizes[0]] * count
    if len(object_sizes) != count:
        raise ValueError(
            f"got {len(object_sizes)} object sizes for {count} objects; "
            "give one size for all objects or exactly one per object"
        )
    return list(object_sizes)


def _load_labels(pngs_dir: Path) -> dict[str, str]:
    """Read the optional ``labels.json`` mapping in the PNG set directory."""
    labels_path = pngs_dir / LABELS_FILENAME
    if not labels_path.exists():
        return {}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{labels_path} must map PNG filenames/stems to labels")
    return data


def _prepare_objects(paths: list[Path], sizes: list[int]) -> list[_PreparedObject]:
    prepared = []
    for index, (path, target_size) in enumerate(zip(paths, sizes)):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        prepared.append(_PreparedObject(index=index, name=path.name, image=image))
    return prepared


def _rects_overlap(a: Rect, b: Rect, padding: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + padding <= bx1
        or bx2 + padding <= ax1
        or ay2 + padding <= by1
        or by2 + padding <= ay1
    )


def _rect_center(rect: Rect) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _placement_score(rect: Rect, placed_rects: list[Rect]) -> float:
    """Max-min spread with a light average-distance tie-breaker."""
    if not placed_rects:
        return 0.0
    cx, cy = _rect_center(rect)
    distances = [math.dist((cx, cy), _rect_center(other)) for other in placed_rects]
    return min(distances) + (sum(distances) / len(distances)) * 0.05


def _random_rect(
    rng: random.Random,
    item_width: int,
    item_height: int,
    width: int,
    height: int,
    padding: int,
) -> Rect:
    max_x = width - item_width - padding
    max_y = height - item_height - padding
    if max_x < padding or max_y < padding:
        raise ValueError(
            f"object sized {item_width}x{item_height} does not fit inside the "
            f"{width}x{height} canvas with padding {padding}"
        )
    x = rng.randint(padding, max_x)
    y = rng.randint(padding, max_y)
    return (x, y, x + item_width, y + item_height)


class _PlacementFailed(RuntimeError):
    """One layout attempt could not place every object without overlap."""


def _build_layout(
    objects: list[_PreparedObject],
    width: int,
    height: int,
    padding: int,
    rng: random.Random,
    candidates_per_object: int,
) -> list[tuple[_PreparedObject, Rect]]:
    # Place larger sprites first to make packing easier; stable tie-break on
    # filename keeps the order deterministic.
    ordered = sorted(
        objects,
        key=lambda obj: (-max(obj.image.width, obj.image.height), obj.name),
    )

    placed: list[tuple[_PreparedObject, Rect]] = []
    placed_rects: list[Rect] = []

    for obj in ordered:
        if not placed_rects:
            # Keep the first placement fully random to avoid center bias.
            rect = _random_rect(rng, obj.image.width, obj.image.height, width, height, padding)
            placed.append((obj, rect))
            placed_rects.append(rect)
            continue

        best_rect: Rect | None = None
        best_score = -1.0
        for _ in range(candidates_per_object):
            candidate = _random_rect(
                rng, obj.image.width, obj.image.height, width, height, padding
            )
            if any(_rects_overlap(candidate, other, padding) for other in placed_rects):
                continue
            score = _placement_score(candidate, placed_rects)
            if score > best_score:
                best_score = score
                best_rect = candidate

        if best_rect is None:
            raise _PlacementFailed(
                "could not place all objects without overlap; try smaller "
                "object sizes, less padding, or a larger canvas"
            )
        placed.append((obj, best_rect))
        placed_rects.append(best_rect)

    return placed


def _layout_quality(placed: list[tuple[_PreparedObject, Rect]]) -> float:
    """Minimum pairwise center distance (0.0 for fewer than two objects)."""
    centers = [_rect_center(rect) for _, rect in placed]
    if len(centers) < 2:
        return 0.0
    return min(
        math.dist(a, b) for i, a in enumerate(centers) for b in centers[i + 1 :]
    )


def _choose_best_layout(
    objects: list[_PreparedObject],
    width: int,
    height: int,
    padding: int,
    rng: random.Random,
    candidates_per_object: int,
    layout_attempts: int,
) -> list[tuple[_PreparedObject, Rect]]:
    """Best of ``layout_attempts`` layouts, all drawn from the single rng."""
    best_layout: list[tuple[_PreparedObject, Rect]] | None = None
    best_quality = -1.0
    failures: list[str] = []

    for _ in range(layout_attempts):
        try:
            layout = _build_layout(objects, width, height, padding, rng, candidates_per_object)
        except _PlacementFailed as exc:
            failures.append(str(exc))
            continue
        quality = _layout_quality(layout)
        if quality > best_quality:
            best_layout = layout
            best_quality = quality

    if best_layout is None:
        raise RuntimeError(
            f"all {layout_attempts} layout attempts failed: {failures[-1]}"
        )
    return best_layout


def generate_world(
    *,
    seed: int,
    pngs_dir: str | Path,
    num_objects: int,
    base: str | Path,
    out_dir: str | Path,
    size: tuple[int, int] = DEFAULT_SIZE,
    padding: int = DEFAULT_PADDING,
    object_sizes: list[int] | None = None,
    candidates_per_object: int = CANDIDATES_PER_OBJECT,
    layout_attempts: int = LAYOUT_ATTEMPTS,
) -> tuple[Path, Path]:
    """Generate one world PNG plus its ``.layout.json`` ground truth.

    Places the first ``num_objects`` PNGs (sorted filename order) from
    ``pngs_dir`` onto ``base`` resized to ``size``, with at least ``padding``
    pixels between object rects and from the canvas edge. Every placed
    object's rect, center, and pre-composite alpha mask (RLE) are recorded.

    Returns ``(png_path, layout_path)``. Same arguments + same seed produce
    byte-identical outputs.
    """
    if seed is None:
        raise ValueError("seed is required: worlds must be reproducible")
    rng = random.Random(seed)

    pngs_dir = Path(pngs_dir)
    base = Path(base)
    out_dir = Path(out_dir)
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"canvas size must be positive, got {width}x{height}")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    png_paths = _sorted_png_paths(pngs_dir)
    if not png_paths:
        raise FileNotFoundError(f"no PNGs found in {pngs_dir}")
    if num_objects < 1:
        raise ValueError(f"num_objects must be at least 1, got {num_objects}")
    if num_objects > len(png_paths):
        raise ValueError(
            f"requested {num_objects} objects but {pngs_dir} has only "
            f"{len(png_paths)} PNGs"
        )
    png_paths = png_paths[:num_objects]

    sizes = _resolve_sizes(
        list(object_sizes) if object_sizes else [DEFAULT_OBJECT_SIZE], num_objects
    )
    labels = _load_labels(pngs_dir)
    objects = _prepare_objects(png_paths, sizes)

    if not base.exists():
        raise FileNotFoundError(f"base texture not found: {base}")
    base_image = Image.open(base).convert("RGBA")
    base_image = base_image.resize((width, height), Image.Resampling.LANCZOS)
    # Fresh canvas so no ancillary metadata (ICC profile, dpi, ...) from the
    # base texture leaks into the output PNG.
    canvas = Image.new("RGBA", (width, height))
    canvas.paste(base_image, (0, 0))

    placed = _choose_best_layout(
        objects, width, height, padding, rng, candidates_per_object, layout_attempts
    )

    for obj, rect in placed:
        canvas.alpha_composite(obj.image, (rect[0], rect[1]))

    objects_json = []
    for obj, rect in sorted(placed, key=lambda item: item[0].index):
        x1, y1, x2, y2 = rect
        alpha = np.asarray(obj.image.getchannel("A"))
        stem = Path(obj.name).stem
        objects_json.append(
            {
                "id": obj.index,
                "label": labels.get(obj.name) or labels.get(stem) or stem,
                "png": obj.name,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                "mask_rle": encode_mask(alpha > 0),
            }
        )

    set_name = pngs_dir.name
    out_stem = f"background-{set_name}-{num_objects}obj-seed{seed}"
    png_path = out_dir / f"{out_stem}.png"
    layout_path = out_dir / f"{out_stem}.layout.json"

    layout = {
        "background_image": png_path.name,
        "width": width,
        "height": height,
        "generator": {
            "seed": seed,
            "padding": padding,
            "object_sizes": sizes,
            "png_set": set_name,
            "num_objects": num_objects,
        },
        "objects": objects_json,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(png_path, format="PNG")
    layout_path.write_text(
        json.dumps(layout, ensure_ascii=False, sort_keys=True, indent=2,
                   separators=(",", ": "))
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return png_path, layout_path


def _parse_size(text: str) -> tuple[int, int]:
    try:
        width_s, _, height_s = text.lower().partition("x")
        width, height = int(width_s), int(height_s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid size {text!r}, expected WxH (e.g. 2500x2500)"
        ) from None
    return (width, height)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-gen",
        description=(
            "Generate a deterministic world PNG plus a .layout.json recording "
            "each placed object's bbox, center, and alpha mask."
        ),
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="rng seed (required; same seed reproduces the world byte-for-byte)",
    )
    parser.add_argument(
        "--pngs", type=Path, required=True,
        help="directory of RGBA object PNGs; its name is the set name",
    )
    parser.add_argument(
        "--num-objects", type=int, required=True,
        help="number of objects to place (first N PNGs in sorted filename order)",
    )
    parser.add_argument(
        "--size", type=_parse_size, default=DEFAULT_SIZE, metavar="WxH",
        help=f"canvas size (default: {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]})",
    )
    parser.add_argument(
        "--padding", type=int, default=DEFAULT_PADDING,
        help=f"min pixels between objects and from the edge (default: {DEFAULT_PADDING})",
    )
    parser.add_argument(
        "--object-size", type=int, action="append", dest="object_sizes",
        metavar="PX",
        help=(
            "thumbnail max dimension; give once for all objects or repeat "
            f"once per object in sorted filename order (default: {DEFAULT_OBJECT_SIZE})"
        ),
    )
    parser.add_argument(
        "--base", type=Path, required=True,
        help="background texture image, resized to the canvas size",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="output directory for the PNG and its .layout.json",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    png_path, layout_path = generate_world(
        seed=args.seed,
        pngs_dir=args.pngs,
        num_objects=args.num_objects,
        base=args.base,
        out_dir=args.out,
        size=args.size,
        padding=args.padding,
        object_sizes=args.object_sizes,
    )
    print(f"wrote {png_path}")
    print(f"wrote {layout_path}")


if __name__ == "__main__":
    main()
