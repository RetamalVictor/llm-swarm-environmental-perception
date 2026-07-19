"""Append-only per-run event log and reproducibility artifacts.

One :class:`RunLogger` owns one run directory and writes:

- ``events.jsonl`` — one JSON object per line, append-only, UTF-8, ``\\n``
  newlines. Serialized with ``json.dumps(obj, sort_keys=True,
  separators=(",", ":"))`` so the byte stream is canonical. **No wall-clock
  values ever appear in events** (design decision D10): timestamps live only
  in ``run_metadata.json``, so two runs with the same config and seed produce
  byte-identical event logs.
- ``config_resolved.yaml`` — the fully resolved typed config, written at
  construction.
- ``run_metadata.json`` — seed, config name, package version, python/platform,
  git SHA, and timestamps; rewritten by :meth:`RunLogger.finalize` with the
  end timestamp and per-type event counts.
- optional PNG artifacts: full frames under ``frames/`` and camera crops under
  ``robot_crops/``, gated by the ``simulation.save_photo_frames`` /
  ``simulation.save_robot_crops`` config flags.

Event vocabulary (the contract downstream eval consumes):

- ``capture`` — ``{type, tick, epoch, robot, key: [epoch, robot, crop_idx],
  bbox: [x1, y1, x2, y2], pos: [x, y]}``. ``bbox`` is the clipped crop rect in
  image pixels; ``pos`` is the robot position at capture time.
- ``memory`` — ``{type, tick, epoch, robot, keys: [[epoch, robot, crop_idx],
  ...]}``. One robot's full memory key set at its capture epoch; ``keys`` is
  sorted in tuple order.
- ``comm`` — ``{type, receiver_tick, sender_tick, epoch, receiver, sender,
  merge_method, inbox_policy, bytes, k_sent, dropped}``. One transmitted peer
  message at the moment its fate is decided. ``bytes`` prices the message per
  the byte model in :mod:`swarm_perception.sim.channel` (the single byte-model
  authority); ``k_sent`` counts records on the wire. Messages that never help
  the receiver still log their cost with ``dropped: true`` and an
  ``inbox_policy`` naming the fate (``channel_drop``, ``inbox_overflow``,
  ``drop_after_budget``); merged messages log ``within_budget`` or
  ``deterministic_after_budget``. Summing ``bytes`` over a robot's comm
  events yields its cumulative spent channel bytes.
- ``frame`` — ``{type, tick}``. A synchronized full-frame capture point (the
  PNG itself exists only when a display surface does).

Hot path: each event is written with open-append-write under a lock — cheap
at sim event rates and crash-durable. No read-back and no ``os.stat``
anywhere on the event path.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import threading
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import cv2
import pygame as pg
import yaml

from swarm_perception.config import Config
from swarm_perception.utils.paths import OUTPUT_DIR

_JSON_SEPARATORS = (",", ":")


def resolve_run_dir(cfg: Config) -> Path:
    """Return the run directory for a config.

    ``cfg.simulation.output_dir`` is used verbatim when set (batch runners own
    the layout and pass a unique directory per run). Otherwise a fresh
    directory ``OUTPUT_DIR / "<config name>-<timestamp>"`` is chosen. The
    timestamp is computed **here and only here** — it names the directory for
    ad-hoc runs and never enters event data (D10).
    """
    if cfg.simulation.output_dir:
        return Path(cfg.simulation.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return OUTPUT_DIR / f"{cfg.config.name}-{timestamp}"


def _git_sha() -> str:
    """Current git commit SHA, or ``"unknown"`` outside a repo / without git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _package_version() -> str:
    try:
        return importlib_metadata.version("swarm-perception")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunLogger:
    """Write one run's event log and artifacts into one run directory."""

    def __init__(self, cfg: Config, run_dir: Path | str | None = None) -> None:
        """Create the run directory and its reproducibility artifacts.

        Args:
            cfg: Fully resolved run configuration.
            run_dir: Run directory to own; defaults to
                :func:`resolve_run_dir` on ``cfg``.
        """
        self._cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else resolve_run_dir(cfg)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._events_path = self.run_dir / "events.jsonl"
        self._metadata_path = self.run_dir / "run_metadata.json"
        self._frames_dir = self.run_dir / "frames"
        self._crops_dir = self.run_dir / "robot_crops"

        self._lock = threading.Lock()
        self._event_counts: Counter[str] = Counter()
        self._finalized = False

        # Start a fresh event log for this run (writes are append-only after this).
        self._events_path.write_bytes(b"")

        self._write_config_resolved()
        self._metadata: dict[str, Any] = {
            "config_name": cfg.config.name,
            "seed": cfg.simulation.seed,
            "package_version": _package_version(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "git_sha": _git_sha(),
            "started_at_utc": _utc_now_iso(),
        }
        self._write_metadata()

    # ------------------------------------------------------------------ events

    def _emit(self, event: dict[str, Any]) -> None:
        """Append one event as a single canonical JSON line.

        Open-append-write per event: cheap at sim event rates, safe under the
        worker threads that exist until PR-04, and crash-durable. No wall-clock
        values may be added here (D10).
        """
        line = json.dumps(event, sort_keys=True, separators=_JSON_SEPARATORS)
        with self._lock:
            self._event_counts[event["type"]] += 1
            with open(self._events_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    def log_capture(
        self,
        *,
        tick: int,
        epoch: int,
        robot: int,
        bbox: tuple[int, int, int, int],
        pos: tuple[float, float],
        crop_idx: int = 0,
    ) -> None:
        """Log one camera capture with its provenance key and clipped rect.

        Args:
            tick: Simulation tick of the capture.
            epoch: Capture epoch of the robot.
            robot: Robot identifier.
            bbox: Clipped crop rect ``(x1, y1, x2, y2)`` in image pixels.
            pos: Robot position ``(x, y)`` at capture time.
            crop_idx: Crop index within the epoch (0 until multi-crop epochs).
        """
        self._emit(
            {
                "type": "capture",
                "tick": int(tick),
                "epoch": int(epoch),
                "robot": int(robot),
                "key": [int(epoch), int(robot), int(crop_idx)],
                "bbox": [int(v) for v in bbox],
                "pos": [float(pos[0]), float(pos[1])],
            }
        )

    def log_memory(self, *, tick: int, epoch: int, robot: int, keys: Any) -> None:
        """Log one robot's memory key set at its capture epoch.

        Args:
            tick: Simulation tick of the capture.
            epoch: Capture epoch of the robot.
            robot: Robot identifier.
            keys: Iterable of record keys ``(epoch, robot, crop_idx)``;
                emitted sorted in tuple order.
        """
        self._emit(
            {
                "type": "memory",
                "tick": int(tick),
                "epoch": int(epoch),
                "robot": int(robot),
                "keys": [[int(e), int(r), int(i)] for e, r, i in sorted(keys)],
            }
        )

    def log_comm(
        self,
        *,
        receiver_tick: int,
        sender_tick: int,
        epoch: int,
        receiver: int,
        sender: int,
        merge_method: str,
        inbox_policy: str,
        bytes_size: int,
        k_sent: int,
        dropped: bool,
    ) -> None:
        """Log one transmitted peer message at the moment its fate is decided.

        Args:
            receiver_tick: Receiver tick when the fate resolved (for channel
                drops, the tick delivery would have happened).
            sender_tick: Sender tick when the broadcast was emitted.
            epoch: Capture epoch the event belongs to.
            receiver: Receiver robot identifier.
            sender: Sender robot identifier.
            merge_method: ``"deterministic"`` for merged messages, ``"none"``
                for messages that were never merged.
            inbox_policy: The fate: ``within_budget`` /
                ``deterministic_after_budget`` for merges; ``channel_drop`` /
                ``inbox_overflow`` / ``drop_after_budget`` for losses.
            bytes_size: Spent channel bytes per the byte model in
                :mod:`swarm_perception.sim.channel` (emitted as ``bytes``).
            k_sent: Number of records on the wire.
            dropped: True when the message never merged into the receiver.
        """
        self._emit(
            {
                "type": "comm",
                "receiver_tick": int(receiver_tick),
                "sender_tick": int(sender_tick),
                "epoch": int(epoch),
                "receiver": int(receiver),
                "sender": int(sender),
                "merge_method": str(merge_method),
                "inbox_policy": str(inbox_policy),
                "bytes": int(bytes_size),
                "k_sent": int(k_sent),
                "dropped": bool(dropped),
            }
        )

    # ---------------------------------------------------------- PNG artifacts

    def save_frame(self, tick: int) -> None:
        """Log a ``frame`` event and save the current display frame as PNG.

        Gated by ``simulation.save_photo_frames``. The event is emitted
        whenever the flag is on so the event stream does not depend on display
        availability; the PNG is skipped when there is no surface (headless).
        """
        if not self._cfg.simulation.save_photo_frames:
            return
        self._emit({"type": "frame", "tick": int(tick)})

        screen = pg.display.get_surface()
        if screen is None:
            return
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = self._frames_dir / f"frame_tick_{int(tick):06d}.png"
        pg.image.save(screen, frame_path)

    def save_crop(self, *, robot_id: int, tick: int, epoch: int, image: Any) -> None:
        """Save one robot camera crop PNG (gated by ``simulation.save_robot_crops``).

        File naming is unchanged from the legacy logger:
        ``robot_XX/robot_XX_epoch_YYY_tick_ZZZZZZ.png``.
        """
        if not self._cfg.simulation.save_robot_crops or image is None:
            return
        robot_dir = self._crops_dir / f"robot_{int(robot_id):02d}"
        robot_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"robot_{int(robot_id):02d}_epoch_{int(epoch):03d}"
            f"_tick_{int(tick):06d}.png"
        )
        cv2.imwrite(str(robot_dir / filename), image)

    # -------------------------------------------------------------- finalize

    def finalize(self) -> None:
        """Seal the run: end timestamp and per-type event counts.

        Rewrites ``run_metadata.json`` with ``finished_at_utc`` and per-type
        event counts. Idempotent — the second and later calls are no-ops, so
        both ``main()`` and test harnesses may call it.
        """
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._metadata["finished_at_utc"] = _utc_now_iso()
            self._metadata["event_counts"] = dict(sorted(self._event_counts.items()))
            self._write_metadata()

    # -------------------------------------------------------------- internals

    def _write_config_resolved(self) -> None:
        with open(self.run_dir / "config_resolved.yaml", "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(dataclasses.asdict(self._cfg), f, sort_keys=True)

    def _write_metadata(self) -> None:
        with open(self._metadata_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self._metadata, f, indent=2)
