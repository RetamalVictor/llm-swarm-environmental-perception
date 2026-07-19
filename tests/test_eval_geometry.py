"""Geometric coverage metric: hand-computed cases over synthetic event logs.

Every expected value below is computable by eye: layouts use small solid
rectangular masks on a 100x100 canvas, and the event logs are written
directly, so per-robot / min / union fractions and time-to-level epochs are
exact fractions of the object count.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from swarm_perception.eval.geometry import compute_coverage
from swarm_perception.world.layout import Layout, LayoutObject
from swarm_perception.world.rle import encode_mask

# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def solid_object(oid: int, bbox: tuple[int, int, int, int]) -> LayoutObject:
    """Object whose alpha mask is fully opaque over its bbox."""
    x1, y1, x2, y2 = bbox
    return LayoutObject(
        id=oid,
        label=f"obj{oid}",
        png=f"{oid}.png",
        bbox=bbox,
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        mask_rle=encode_mask(np.ones((y2 - y1, x2 - x1), dtype=bool)),
    )


def make_layout(*objects: LayoutObject) -> Layout:
    return Layout(
        background_image="bg.png", width=100, height=100, generator={}, objects=list(objects)
    )


def capture(tick: int, epoch: int, robot: int, bbox, crop_idx: int = 0) -> dict:
    return {
        "type": "capture",
        "tick": tick,
        "epoch": epoch,
        "robot": robot,
        "key": [epoch, robot, crop_idx],
        "bbox": list(bbox),
        "pos": [0.0, 0.0],
    }


def memory(tick: int, epoch: int, robot: int, keys) -> dict:
    return {
        "type": "memory",
        "tick": tick,
        "epoch": epoch,
        "robot": robot,
        "keys": [list(k) for k in keys],
    }


def write_events(path: Path, events: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(e, sort_keys=True, separators=(",", ":")) + "\n" for e in events),
        encoding="utf-8",
        newline="\n",
    )
    return path


# ---------------------------------------------------------------------------
# hand-computed two-robot run, including the comm case
# ---------------------------------------------------------------------------

# Three solid 10x10 objects, far apart.
THREE_OBJECTS = make_layout(
    solid_object(0, (10, 10, 20, 20)),
    solid_object(1, (40, 40, 50, 50)),
    solid_object(2, (70, 70, 80, 80)),
)

# Epoch 1: robot 0 sees object 0; robot 1 sees object 2.
# Epoch 2: robot 0 also sees object 1. Robot 1's own epoch-2 crop sees
# nothing, but its memory now holds robot 0's key (1, 0, 0) received over
# comms — that key alone adds object 0 to robot 1's coverage.
TWO_ROBOT_EVENTS = [
    capture(5, 1, 0, (5, 5, 25, 25)),
    memory(5, 1, 0, [(1, 0, 0)]),
    capture(5, 1, 1, (60, 60, 90, 90)),
    memory(5, 1, 1, [(1, 1, 0)]),
    capture(10, 2, 0, (35, 35, 55, 55)),
    memory(10, 2, 0, [(1, 0, 0), (2, 0, 0)]),
    capture(10, 2, 1, (0, 0, 5, 5)),
    memory(10, 2, 1, [(1, 0, 0), (1, 1, 0), (2, 1, 0)]),
]


def test_two_robot_run_exact_values(tmp_path) -> None:
    events = write_events(tmp_path / "events.jsonl", TWO_ROBOT_EVENTS)
    result = compute_coverage(events, THREE_OBJECTS)

    assert result["num_objects"] == 3
    assert result["robots"] == [0, 1]
    assert result["epochs"] == [1, 2]

    epoch1, epoch2 = result["per_epoch"]
    assert epoch1["per_robot"] == pytest.approx({"0": 1 / 3, "1": 1 / 3})
    assert epoch1["mean"] == pytest.approx(1 / 3)
    assert epoch1["min"] == pytest.approx(1 / 3)
    assert epoch1["union"] == pytest.approx(2 / 3)

    # Robot 1 covers objects {0, 2}: 2 via its own epoch-1 capture, 0 only
    # via the received key (1, 0, 0) — communication raised its coverage.
    assert epoch2["per_robot"] == pytest.approx({"0": 2 / 3, "1": 2 / 3})
    assert epoch2["mean"] == pytest.approx(2 / 3)
    assert epoch2["min"] == pytest.approx(2 / 3)
    assert epoch2["union"] == pytest.approx(1.0)

    assert result["time_to"] == {
        "50": {"mean": 2, "min": 2, "union": 1},
        "90": {"mean": None, "min": None, "union": 2},
    }


def test_received_key_is_what_adds_coverage(tmp_path) -> None:
    """Counterfactual: drop the received key and robot 1 stays at 1/3."""
    no_comm = [
        e
        for e in TWO_ROBOT_EVENTS
        if not (e["type"] == "memory" and e["robot"] == 1 and e["epoch"] == 2)
    ]
    no_comm.append(memory(10, 2, 1, [(1, 1, 0), (2, 1, 0)]))  # own keys only
    events = write_events(tmp_path / "events.jsonl", no_comm)

    result = compute_coverage(events, THREE_OBJECTS)
    epoch2 = result["per_epoch"][-1]
    assert epoch2["per_robot"]["1"] == pytest.approx(1 / 3)
    assert epoch2["min"] == pytest.approx(1 / 3)
    # The pooled union is unchanged: comms never adds captures (diagnostic).
    assert epoch2["union"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# oracle property
# ---------------------------------------------------------------------------


def test_oracle_crops_on_every_object_reach_full_coverage(tmp_path) -> None:
    """A robot whose crops center on every object ends at coverage 1.0."""
    events: list[dict] = []
    keys: list[tuple[int, int, int]] = []
    for epoch, obj in enumerate(THREE_OBJECTS.objects, start=1):
        x1, y1, x2, y2 = obj.bbox
        crop = (x1 - 5, y1 - 5, x2 + 5, y2 + 5)  # centered on the object
        events.append(capture(epoch * 5, epoch, 0, crop))
        keys.append((epoch, 0, 0))
        events.append(memory(epoch * 5, epoch, 0, keys))
    path = write_events(tmp_path / "events.jsonl", events)

    result = compute_coverage(path, THREE_OBJECTS)
    final = result["per_epoch"][-1]
    assert final["per_robot"]["0"] == pytest.approx(1.0)
    assert final["min"] == pytest.approx(1.0)
    assert final["union"] == pytest.approx(1.0)
    assert result["time_to"]["90"] == {"mean": 3, "min": 3, "union": 3}


# ---------------------------------------------------------------------------
# visibility threshold
# ---------------------------------------------------------------------------


def test_overlap_below_min_visible_px_does_not_count(tmp_path) -> None:
    layout = make_layout(solid_object(0, (10, 10, 20, 20)))
    # Crop (18, 18, 30, 30) overlaps exactly 2x2 = 4 mask pixels.
    events = write_events(
        tmp_path / "events.jsonl",
        [capture(5, 1, 0, (18, 18, 30, 30)), memory(5, 1, 0, [(1, 0, 0)])],
    )

    at_4 = compute_coverage(events, layout, min_visible_px=4)
    at_5 = compute_coverage(events, layout, min_visible_px=5)
    assert at_4["per_epoch"][0]["per_robot"]["0"] == pytest.approx(1.0)
    assert at_5["per_epoch"][0]["per_robot"]["0"] == pytest.approx(0.0)
    assert at_5["per_epoch"][0]["union"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# malformed logs and CLI
# ---------------------------------------------------------------------------


def test_memory_key_without_capture_raises(tmp_path) -> None:
    events = write_events(
        tmp_path / "events.jsonl", [memory(5, 1, 0, [(1, 0, 0)])]
    )
    with pytest.raises(ValueError, match=r"no matching capture"):
        compute_coverage(events, THREE_OBJECTS)


def test_cli_writes_coverage_json(tmp_path) -> None:
    from swarm_perception.eval.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_events(run_dir / "events.jsonl", TWO_ROBOT_EVENTS)
    layout_path = tmp_path / "world.layout.json"
    THREE_OBJECTS.save(layout_path)

    main(["geometric", str(run_dir), "--layout", str(layout_path)])

    result = json.loads((run_dir / "coverage.json").read_text(encoding="utf-8"))
    assert result["per_epoch"][-1]["min"] == pytest.approx(2 / 3)
    assert result["params"] == {"min_visible_px": 1, "mode": "mask"}


def test_cli_missing_inputs_exit_politely(tmp_path, capsys) -> None:
    from swarm_perception.eval.cli import main

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    layout_path = tmp_path / "world.layout.json"
    THREE_OBJECTS.save(layout_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["geometric", str(run_dir), "--layout", str(layout_path)])
    assert excinfo.value.code == 1
    assert "events.jsonl" in capsys.readouterr().err

    write_events(run_dir / "events.jsonl", TWO_ROBOT_EVENTS)
    with pytest.raises(SystemExit) as excinfo:
        main(["geometric", str(run_dir), "--layout", str(tmp_path / "missing.layout.json")])
    assert excinfo.value.code == 1
    assert "layout" in capsys.readouterr().err
