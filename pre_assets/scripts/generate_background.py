from __future__ import annotations

import math
import random
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Pillow is required to run this script. Install project dependencies first, "
        "then rerun `python3 pre_assets/scripts/generate_background.py`."
    ) from exc


# Final output image size.
OUTPUT_WIDTH = 2500
OUTPUT_HEIGHT = 2500

# PNG sizes matched by sorted filename order: 1.png, 2.png, 3.png, ...
# If you provide a single value, it is reused for every PNG.
# Each value is treated as the max dimension for that PNG while preserving aspect ratio.
PNG_SIZES = [300]
# PNG_SIZES = [200, 200, 800, 200, 200, 300, 600, 200, 200, 250, 600, 250, 200, 600, 300]
#           [ 1,  2,    3,   4,   5,   6,   7,   8,   9,  10,  11, 12, 13,   14,  15]
# Extra space between placed PNGs and from the image edge.
PADDING = 100

# More candidates/attempts usually gives a better spread, but is slower.
CANDIDATES_PER_IMAGE = 800
LAYOUT_ATTEMPTS = 40

# Set to an integer for repeatable results, or None for a fresh random layout each run.
RANDOM_SEED = None


SCRIPT_DIR = Path(__file__).resolve().parent
PRE_ASSETS_DIR = SCRIPT_DIR.parent
BACKGROUND_PATH = PRE_ASSETS_DIR / "background" / "background.png"
PNGS_DIR = PRE_ASSETS_DIR / "pngs" / "20"
OUTPUT_DIR = PRE_ASSETS_DIR / "background"


def sorted_png_paths() -> list[Path]:
    def numeric_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        return (int(stem), path.name) if stem.isdigit() else (10**9, path.name)

    return sorted(PNGS_DIR.glob("*.png"), key=numeric_key)


def resolve_sizes(count: int) -> list[int]:
    if not PNG_SIZES:
        raise ValueError("PNG_SIZES must contain at least one integer.")

    if len(PNG_SIZES) == 1:
        return [int(PNG_SIZES[0])] * count

    if len(PNG_SIZES) != count:
        raise ValueError(
            f"PNG_SIZES has {len(PNG_SIZES)} values, but found {count} PNGs in {PNGS_DIR}."
        )

    return [int(size) for size in PNG_SIZES]


def load_and_resize_images(paths: list[Path], sizes: list[int]) -> list[dict]:
    prepared = []

    for path, target_size in zip(paths, sizes):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        prepared.append(
            {
                "name": path.name,
                "image": image,
                "width": image.width,
                "height": image.height,
            }
        )

    # Place larger images first to make packing easier.
    prepared.sort(key=lambda item: max(item["width"], item["height"]), reverse=True)
    return prepared


def rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + padding <= bx1
        or bx2 + padding <= ax1
        or ay2 + padding <= by1
        or by2 + padding <= ay1
    )


def rect_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def placement_score(
    rect: tuple[int, int, int, int],
    placed_rects: list[tuple[int, int, int, int]],
) -> float:
    if not placed_rects:
        return 0.0

    cx, cy = rect_center(rect)
    distances = []
    for other in placed_rects:
        ox, oy = rect_center(other)
        distances.append(math.dist((cx, cy), (ox, oy)))

    # Prioritize the minimum distance (max-min spread), with a light tie-breaker for average distance.
    return min(distances) + (sum(distances) / len(distances)) * 0.05


def random_rect(rng: random.Random, item_width: int, item_height: int, width: int, height: int) -> tuple[int, int, int, int]:
    max_x = width - item_width - PADDING
    max_y = height - item_height - PADDING
    if max_x < PADDING or max_y < PADDING:
        raise ValueError(
            f"Image sized {item_width}x{item_height} does not fit inside the {width}x{height} background."
        )

    x = rng.randint(PADDING, max_x)
    y = rng.randint(PADDING, max_y)
    return (x, y, x + item_width, y + item_height)


def build_layout(images: list[dict], width: int, height: int, rng: random.Random) -> list[dict]:
    placed: list[dict] = []
    placed_rects: list[tuple[int, int, int, int]] = []

    for item in images:
        if not placed_rects:
            # Keep first placement fully random to avoid center bias.
            first_rect = random_rect(rng, item["width"], item["height"], width, height)
            placed_item = dict(item)
            placed_item["rect"] = first_rect
            placed.append(placed_item)
            placed_rects.append(first_rect)
            continue

        best_rect = None
        best_score = -1.0

        for _ in range(CANDIDATES_PER_IMAGE):
            candidate = random_rect(rng, item["width"], item["height"], width, height)
            if any(rects_overlap(candidate, other, PADDING) for other in placed_rects):
                continue

            score = placement_score(candidate, placed_rects)
            if score > best_score:
                best_score = score
                best_rect = candidate

        if best_rect is None:
            raise RuntimeError(
                "Could not place all PNGs without overlap. Try smaller PNG_SIZES, "
                "less PADDING, or a larger output size."
            )

        placed_item = dict(item)
        placed_item["rect"] = best_rect
        placed.append(placed_item)
        placed_rects.append(best_rect)

    return placed


def layout_quality(layout: list[dict], width: int, height: int) -> float:
    rects = [item["rect"] for item in layout]
    if not rects:
        return 0.0

    min_gap = float("inf")
    for i, rect in enumerate(rects):
        cx, cy = rect_center(rect)
        for other in rects[i + 1 :]:
            ox, oy = rect_center(other)
            min_gap = min(min_gap, math.dist((cx, cy), (ox, oy)))

    if len(rects) == 1:
        return 0.0

    return min_gap


def choose_best_layout(images: list[dict], width: int, height: int) -> list[dict]:
    best_layout = None
    best_quality = -1.0

    for attempt in range(LAYOUT_ATTEMPTS):
        seed = RANDOM_SEED if RANDOM_SEED is not None else random.randrange(10**9)
        rng = random.Random(seed + attempt)
        layout = build_layout(images, width, height, rng)
        quality = layout_quality(layout, width, height)

        if quality > best_quality:
            best_layout = layout
            best_quality = quality

    if best_layout is None:
        raise RuntimeError("Failed to generate a valid layout.")

    return best_layout


def main() -> None:
    png_paths = sorted_png_paths()
    if not png_paths:
        raise FileNotFoundError(f"No PNGs found in {PNGS_DIR}")

    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError(f"Background image not found: {BACKGROUND_PATH}")

    sizes = resolve_sizes(len(png_paths))
    images = load_and_resize_images(png_paths, sizes)

    background = Image.open(BACKGROUND_PATH).convert("RGBA")
    background = background.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)

    layout = choose_best_layout(images, OUTPUT_WIDTH, OUTPUT_HEIGHT)

    for item in layout:
        x1, y1, _, _ = item["rect"]
        background.alpha_composite(item["image"], (x1, y1))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"background-{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}.png"
    background.save(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
