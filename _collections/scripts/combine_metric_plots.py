#!/usr/bin/env python3
"""
Combine recall/precision/F1 PNG plots into a single-row PDF.

Example:
  python3 scripts/combine_metric_plots.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image


def resolve_inputs(input_dir: Path) -> List[Path]:
    """Find the three expected files, with fallback for common precision typo."""
    recall = input_dir / "recall_main.png"
    precision = input_dir / "precision_main.png"
    precision_typo = input_dir / "precisoin_main.png"
    f1 = input_dir / "f1_main.png"

    if not recall.exists():
        raise FileNotFoundError(f"Missing file: {recall}")
    if not f1.exists():
        raise FileNotFoundError(f"Missing file: {f1}")

    if precision.exists():
        precision_path = precision
    elif precision_typo.exists():
        precision_path = precision_typo
    else:
        raise FileNotFoundError(f"Missing file: {precision} (or typo variant {precision_typo.name})")

    return [recall, precision_path, f1]


def combine_images_row(
    image_paths: List[Path],
    output_pdf: Path,
    target_height: int | None = None,
    gap_px: int = 20,
    margin_px: int = 0,
) -> None:
    """Resize images to equal height and combine them in one aligned row."""
    opened = [Image.open(p).convert("RGB") for p in image_paths]
    try:
        # Use the smallest height by default to avoid upscaling artifacts.
        common_height = target_height if target_height is not None else min(img.height for img in opened)
        if common_height <= 0:
            raise ValueError("target_height must be > 0")

        resized = []
        for img in opened:
            new_w = round(img.width * (common_height / img.height))
            resized.append(img.resize((new_w, common_height), Image.Resampling.LANCZOS))

        total_w = sum(img.width for img in resized) + gap_px * (len(resized) - 1) + 2 * margin_px
        total_h = common_height + 2 * margin_px

        canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

        x = margin_px
        y = margin_px  # Same y gives equal top and bottom alignment.
        for img in resized:
            canvas.paste(img, (x, y))
            x += img.width + gap_px

        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_pdf, "PDF", resolution=600.0)
    finally:
        for img in opened:
            img.close()


def parse_args() -> argparse.Namespace:
    default_dir = Path("experiments/metrics/outputs")
    default_output = default_dir / "main_metrics_row.pdf"

    parser = argparse.ArgumentParser(description="Combine main metric PNGs into one PDF row.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_dir,
        help="Directory containing recall_main.png, precision_main.png, and f1_main.png.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Optional common pixel height (default: smallest input height).",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=20,
        help="Gap between plots in pixels.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Outer page margin in pixels (default: 0 for tight crop).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = resolve_inputs(args.input_dir)
    combine_images_row(
        image_paths=input_paths,
        output_pdf=args.output,
        target_height=args.target_height,
        gap_px=args.gap,
        margin_px=args.margin,
    )
    print(f"Saved combined PDF: {args.output}")


if __name__ == "__main__":
    main()
