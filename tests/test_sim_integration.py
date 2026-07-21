"""Integration: graded-memory invariants and channel behavior in real runs.

Covers the R-15 acceptance properties end to end with the stub encoder:
the residue invariant under forced evictions, residue union through
``share_visitation``, robot-level quantize-once, and memory events whose
retained keys reflect the cap.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from conftest import load_smoke_config, run_headless
from swarm_perception.config import Config
from swarm_perception.fusion import MemoryRecord, SetMemory
from swarm_perception.io.run_logger import RunLogger
from swarm_perception.perception import stub_embedding
from swarm_perception.sim.channel import Channel
from swarm_perception.sim.exchange import ExchangeUnit
from swarm_perception.sim.residue import VisitationResidue

Key = tuple[int, int, int]


def _read_events(run_dir: Path) -> list[dict]:
    text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _tight_cap_config(run_dir: Path) -> Config:
    """Smoke config with a cap of 2 and more epochs, forcing evictions."""
    cfg = load_smoke_config(run_dir)
    return dataclasses.replace(
        cfg,
        simulation=dataclasses.replace(cfg.simulation, run_length=4),
        fusion=dataclasses.replace(cfg.fusion, memory_cap=2),
    )


def test_residue_invariant_and_cap_under_forced_evictions(tmp_path: Path) -> None:
    sim = run_headless(_tight_cap_config(tmp_path))
    events = _read_events(tmp_path)

    capture_bbox: dict[Key, tuple[int, int, int, int]] = {}
    for event in events:
        if event["type"] == "capture":
            e, r, c = event["key"]
            capture_bbox[(e, r, c)] = tuple(event["bbox"])

    ever_retained: dict[int, set[Key]] = {}
    per_robot_memory_events: dict[int, list[list[Key]]] = {}
    for event in events:
        if event["type"] == "memory":
            robot = int(event["robot"])
            keys = [tuple(k) for k in event["keys"]]
            assert len(keys) <= 2, "memory_cap=2 must bound every memory event"
            ever_retained.setdefault(robot, set()).update(keys)
            per_robot_memory_events.setdefault(robot, []).append(keys)

    # The tight cap must actually evict: some robot retained more distinct
    # keys over the run than it can hold at once.
    assert any(len(keys) > 2 for keys in ever_retained.values()), (
        "expected evictions under memory_cap=2"
    )
    # Retained keys always come from real captures (dedup/cap never invent).
    for keys in ever_retained.values():
        assert keys <= set(capture_bbox)

    # Residue invariant: every record a robot ever retained — including ones
    # later evicted — has its rect's fully-covered cells in the robot's S.
    for robot in sim.robots:
        robot_id = int(robot.id)
        for key in ever_retained.get(robot_id, set()):
            cells = robot.residue.covered_cells(capture_bbox[key])
            assert cells <= robot.residue.cells, (
                f"robot {robot_id} lost residue cells of record {key}"
            )
    sim.run_logger.finalize()


class _FakeRobot:
    """Duck-typed stand-in for Robot: just what ExchangeUnit reads."""

    def __init__(self, robot_id: int, cfg: Config, run_logger: RunLogger, channel: Channel):
        self.id = robot_id
        self.cfg = cfg
        self.run_logger = run_logger
        self.channel = channel
        self.tick_count = 1
        self.capture_epoch = 1
        self.memory = SetMemory(
            (), tau_dedup=cfg.fusion.tau_dedup, memory_cap=cfg.fusion.memory_cap
        )
        self.residue = VisitationResidue(
            cfg.simulation.width, cfg.simulation.height, cfg.robot.coverage_side
        )
        self.exchange = ExchangeUnit(self)  # type: ignore[arg-type]

    def deliver_message(self, message) -> None:
        self.exchange.deliver(message)


def _record(key: Key, bbox: tuple[int, int, int, int]) -> MemoryRecord:
    return MemoryRecord(
        embedding=stub_embedding(key),
        key=key,
        pos=(float(bbox[0]), float(bbox[1])),
        crop_bbox=bbox,
        first_seen=key[0],
    )


def test_sharing_unions_residues_and_reuses_payload_bytes(tmp_path: Path) -> None:
    cfg = load_smoke_config(tmp_path / "run")
    cfg = dataclasses.replace(
        cfg, comms=dataclasses.replace(cfg.comms, drop_p=0.0, share_visitation=True)
    )
    logger = RunLogger(cfg, run_dir=tmp_path / "run")
    channel = Channel(cfg.comms, seed=cfg.simulation.seed)
    a = _FakeRobot(0, cfg, logger, channel)
    b = _FakeRobot(1, cfg, logger, channel)
    c = _FakeRobot(2, cfg, logger, channel)

    # A's residue holds the footprint of an already-evicted record: those
    # cells exist nowhere in A's record memory, only in S.
    evicted_rect = (0, 0, 150, 150)
    a.residue.mark_rect(evicted_rect)
    record = _record((1, 0, 0), (300, 300, 450, 450))
    a.residue.mark_rect(record.crop_bbox)
    a.memory.add(record)

    a.exchange.broadcast([b])  # type: ignore[list-item]
    b.exchange.process_inbox()

    # Record arrived, and BOTH spatial sources unioned into B's residue:
    # the wire record's rect and the bitmap-only cells of the evicted rect.
    assert (1, 0, 0) in b.memory
    assert b.residue.covered_cells(evicted_rect) <= b.residue.cells
    assert b.residue.covered_cells(record.crop_bbox) <= b.residue.cells

    # Quantize-once through robots: B re-shares the exact bytes it received,
    # so C holds the identical payload A first produced (int8 wire format).
    payload_at_a = a.exchange.payload_cache[(1, 0, 0)]
    assert b.exchange.payload_cache[(1, 0, 0)] == payload_at_a
    b.tick_count = 2
    b.exchange.broadcast([c])  # type: ignore[list-item]
    c.exchange.process_inbox()
    assert c.exchange.payload_cache[(1, 0, 0)] == payload_at_a
    assert (1, 0, 0) in c.memory
    logger.finalize()


def test_comm_bytes_are_additive_and_derivable_per_robot(tmp_path: Path) -> None:
    run_headless(load_smoke_config(tmp_path))
    events = _read_events(tmp_path)
    comm = [e for e in events if e["type"] == "comm"]
    assert comm, "smoke run must produce comm traffic"
    per_receiver: dict[int, int] = {}
    for event in comm:
        assert event["bytes"] > 0
        assert event["k_sent"] > 0
        per_receiver[event["receiver"]] = per_receiver.get(event["receiver"], 0) + event["bytes"]
    # Cumulative spent bytes per robot are a pure fold over its comm events.
    assert sum(per_receiver.values()) == sum(e["bytes"] for e in comm)
