"""Geometric object coverage computed from ``events.jsonl`` + layout ground truth.

This is the benchmark's PRIMARY metric (design decisions D1/D2): coverage is
judged against the exact placed geometry persisted by the world generator,
with no model in the loop.

Definition
----------
An object counts as **covered for robot i at epoch t** iff robot i's retained
memory — per the last ``memory`` event for robot i at or before epoch t —
contains a capture key whose crop rect makes the object visible per
:meth:`~swarm_perception.world.layout.Layout.visible_in` (mask mode by
default). ``capture`` events map each key ``(epoch, robot, crop_idx)`` to its
crop bbox; a robot's memory may hold keys captured by *other* robots — that
is exactly what communication contributes, and what this metric rewards.

Reported per epoch:

- ``per_robot`` — each robot's covered fraction of the layout's objects.
- ``mean`` — arithmetic mean over robots.
- ``min`` — minimum over robots: the headline number. The benchmark scores
  the worst-informed robot, so hoarding coverage in one robot does not pay.
- ``union`` — fraction covered by the union of all robots' covered sets.
  **Diagnostic only**: under key-union merging, communication merely
  redistributes existing capture keys between robots, so the pooled union is
  invariant to communication (exactly so absent memory-cap evictions, which
  can drop keys from every holder). It measures how much of the world the
  swarm sensed, never how well that information spread.

``time_to`` reports, per requested percentage level, the first epoch at which
each aggregate (``mean`` / ``min`` / ``union``) reaches the level, or ``None``
if it never does.

This module never imports the simulation engine: it reads finished run
artifacts only (see :mod:`swarm_perception.eval`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from swarm_perception.world.layout import Layout, Rect

Key = tuple[int, int, int]  # (epoch, robot, crop_idx)

DEFAULT_TIME_TO_LEVELS: tuple[float, ...] = (50.0, 90.0)

_AGGREGATES = ("mean", "min", "union")


@dataclass
class _RunEvents:
    """The slice of one ``events.jsonl`` that geometric coverage consumes."""

    capture_bbox: dict[Key, Rect] = field(default_factory=dict)
    # memory[robot][epoch] -> that robot's full retained key set at that epoch
    # (last event wins if an epoch is logged twice).
    memory: dict[int, dict[int, list[Key]]] = field(default_factory=dict)
    robots: set[int] = field(default_factory=set)
    epochs: set[int] = field(default_factory=set)


def _parse_events(events_path: Path) -> _RunEvents:
    """Read the ``capture`` and ``memory`` events out of one ``events.jsonl``."""
    run = _RunEvents()
    with open(events_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{events_path}:{line_no}: invalid JSON event line"
                ) from exc
            kind = event.get("type")
            if kind == "capture":
                e, r, c = event["key"]
                run.capture_bbox[(int(e), int(r), int(c))] = tuple(event["bbox"])
                run.robots.add(int(event["robot"]))
                run.epochs.add(int(event["epoch"]))
            elif kind == "memory":
                robot, epoch = int(event["robot"]), int(event["epoch"])
                keys = [(int(e), int(r), int(c)) for e, r, c in event["keys"]]
                run.memory.setdefault(robot, {})[epoch] = keys
                run.robots.add(robot)
                run.epochs.add(epoch)
    return run


def compute_coverage(
    events_path: str | Path,
    layout: Layout,
    *,
    min_visible_px: int = 1,
    mode: str = "mask",
    time_to_levels: Sequence[float] = DEFAULT_TIME_TO_LEVELS,
) -> dict[str, Any]:
    """Compute geometric coverage per epoch from one run's event log.

    Args:
        events_path: Path to the run's ``events.jsonl``.
        layout: Ground-truth world layout the run was executed on.
        min_visible_px: Visibility threshold forwarded to
            :meth:`Layout.visible_in` (mask pixels in mask mode, intersection
            area in bbox mode).
        mode: Visibility mode, ``"mask"`` (exact, default) or ``"bbox"``.
        time_to_levels: Percentage levels for the time-to-level summary.

    Returns:
        A JSON-serializable dict::

            {
              "params": {"min_visible_px": ..., "mode": ...},
              "num_objects": ..., "num_robots": ..., "robots": [...],
              "epochs": [...],
              "per_epoch": [{"epoch": t, "per_robot": {"<robot>": frac, ...},
                             "mean": ..., "min": ..., "union": ...}, ...],
              "time_to": {"<level>": {"mean": epoch|None, "min": ...,
                                      "union": ...}, ...},
            }

    Raises:
        ValueError: If the layout has no objects, the event log is malformed,
            or a memory event references a key with no matching capture.
    """
    events_path = Path(events_path)
    if len(layout) == 0:
        raise ValueError("layout has no objects: coverage is undefined")

    run = _parse_events(events_path)
    robots = sorted(run.robots)
    epochs = sorted(run.epochs)
    num_objects = len(layout)

    # Each key's visible-object set, resolved once per distinct crop rect.
    visible_by_rect: dict[Rect, frozenset[int]] = {}

    def visible_objects(key: Key) -> frozenset[int]:
        bbox = run.capture_bbox.get(key)
        if bbox is None:
            raise ValueError(
                f"memory event references key {list(key)} with no matching "
                f"capture event in {events_path} (truncated or corrupt log?)"
            )
        cached = visible_by_rect.get(bbox)
        if cached is None:
            cached = frozenset(layout.visible_in(bbox, min_visible_px, mode))
            visible_by_rect[bbox] = cached
        return cached

    # Per robot: snapshot epochs in ascending order, walked with a cursor so
    # "last memory event at or before t" is O(total snapshots) overall.
    snapshot_epochs = {r: sorted(run.memory.get(r, {})) for r in robots}
    cursor = dict.fromkeys(robots, -1)

    per_epoch: list[dict[str, Any]] = []
    series: dict[str, list[float]] = {name: [] for name in _AGGREGATES}
    for epoch in epochs:
        per_robot: dict[str, float] = {}
        fractions: list[float] = []
        union_covered: set[int] = set()
        for robot in robots:
            available = snapshot_epochs[robot]
            while cursor[robot] + 1 < len(available) and available[cursor[robot] + 1] <= epoch:
                cursor[robot] += 1
            covered: set[int] = set()
            if cursor[robot] >= 0:
                for key in run.memory[robot][available[cursor[robot]]]:
                    covered |= visible_objects(key)
            fraction = len(covered) / num_objects
            per_robot[str(robot)] = fraction
            fractions.append(fraction)
            union_covered |= covered

        entry: dict[str, Any] = {
            "epoch": epoch,
            "per_robot": per_robot,
            "mean": sum(fractions) / len(fractions) if fractions else 0.0,
            "min": min(fractions) if fractions else 0.0,
            "union": len(union_covered) / num_objects,
        }
        per_epoch.append(entry)
        for name in _AGGREGATES:
            series[name].append(entry[name])

    time_to: dict[str, dict[str, int | None]] = {}
    for level in time_to_levels:
        threshold = level / 100.0
        time_to[f"{level:g}"] = {
            name: next(
                (epochs[i] for i, v in enumerate(series[name]) if v >= threshold),
                None,
            )
            for name in _AGGREGATES
        }

    return {
        "params": {"min_visible_px": min_visible_px, "mode": mode},
        "num_objects": num_objects,
        "num_robots": len(robots),
        "robots": robots,
        "epochs": epochs,
        "per_epoch": per_epoch,
        "time_to": time_to,
    }
