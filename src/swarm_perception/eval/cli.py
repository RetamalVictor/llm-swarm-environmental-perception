"""Command-line entrypoint for offline run evaluation.

Installed as the ``swarm-eval`` console script. One subcommand per metric;
``geometric`` is the benchmark's primary metric::

    swarm-eval geometric <run_dir> --layout <world.layout.json>

The layout path is always explicit — a run directory does not record which
world file it ran on, and guessing would be worse than asking. Results are
written to ``coverage.json`` inside the run directory and a one-line summary
is printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from swarm_perception.eval.geometry import DEFAULT_TIME_TO_LEVELS, compute_coverage
from swarm_perception.world.layout import Layout


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swarm-eval",
        description="Evaluate a finished simulation run from its run directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    geometric = subparsers.add_parser(
        "geometric",
        help="geometric object coverage (primary metric)",
        description=(
            "Compute geometric object coverage from the run's events.jsonl "
            "and the world's .layout.json ground truth; writes coverage.json "
            "into the run directory."
        ),
    )
    geometric.add_argument(
        "run_dir",
        help="run directory containing events.jsonl",
    )
    geometric.add_argument(
        "--layout",
        required=True,
        metavar="PATH",
        help="path to the .layout.json the run's world was generated with",
    )
    geometric.add_argument(
        "--min-visible-px",
        type=int,
        default=1,
        metavar="N",
        help="visibility threshold in pixels (default: 1)",
    )
    geometric.add_argument(
        "--mode",
        choices=("mask", "bbox"),
        default="mask",
        help="visibility mode: exact alpha mask or bbox overlap (default: mask)",
    )
    geometric.add_argument(
        "--time-to",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIME_TO_LEVELS),
        metavar="PCT",
        help="coverage levels (percent) for the time-to-level summary "
        "(default: 50 90)",
    )
    geometric.set_defaults(func=_run_geometric)
    return parser


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _run_geometric(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    events_path = run_dir / "events.jsonl"
    layout_path = Path(args.layout)
    if not run_dir.is_dir():
        _fail(f"run directory not found: {run_dir}")
    if not events_path.is_file():
        _fail(f"no events.jsonl in {run_dir} (is this a run directory?)")
    if not layout_path.is_file():
        _fail(f"layout file not found: {layout_path}")

    try:
        result = compute_coverage(
            events_path,
            Layout.load(layout_path),
            min_visible_px=args.min_visible_px,
            mode=args.mode,
            time_to_levels=args.time_to,
        )
    except ValueError as error:
        _fail(str(error))

    out_path = run_dir / "coverage.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    if result["per_epoch"]:
        final = result["per_epoch"][-1]
        print(
            f"epoch {final['epoch']}: mean={final['mean']:.4f} "
            f"min={final['min']:.4f} union={final['union']:.4f}"
        )
    else:
        print("no capture/memory events found; coverage is empty")
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> None:
    """Console entrypoint: dispatch to the selected metric subcommand.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]``.

    Raises:
        SystemExit: With code 1 on missing inputs or malformed run artifacts.
    """
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
