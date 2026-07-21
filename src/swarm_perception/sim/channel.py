"""Budgeted, lossy, quantized record channel (design decision D7).

THE BYTE MODEL IS DEFINED HERE AND ONLY HERE. A message carrying ``k``
records costs, in accounted channel bytes::

    bytes = k * (HEADER_BYTES + payload) + residue

- ``HEADER_BYTES`` = 36 per record: key as 3x int32 (12) + pos as 2x
  float32 (8) + crop bbox as 4x int32 (16).
- ``payload`` = embedding bytes under the wire quantization: 4 bytes/dim
  (``"none"``, fp32), 2 (``"fp16"``), 1 (``"int8"``) — 2048 / 1024 / 512
  bytes at the pinned 512-dim.
- ``residue`` = the packed visitation bitmap length (ceil(n_cells / 8))
  when ``comms.share_visitation`` is on, else 0.

Every transmission logs exactly one comm event carrying these bytes,
INCLUDING messages that never help the receiver: channel drops (Bernoulli
``drop_p``), inbox-overflow evictions, and over-budget discards log their
cost with ``dropped: true``. The accounting measures spent channel
capacity, not useful throughput; a robot's cumulative bytes is the sum of
``bytes`` over its comm events.

QUANTIZE-ONCE: a record's wire payload is created the first time the
record is shared and cached by key; a receiver caches the received payload
verbatim and re-shares those exact bytes. Together with embeddings being
canonical per key (stub: a pure key function; clip: one embedding per
capture, never re-encoded), the payload is a canonical function of the
record key on every gossip path.

Channel randomness (packet drop) draws from a DEDICATED seeded stream,
derived from the run seed but separate from the motion RNG, so changing
``drop_p`` never perturbs trajectories.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import MutableMapping, Sequence

import numpy as np

from swarm_perception.config.schema import CommsCfg
from swarm_perception.fusion.memory import Key, MemoryRecord, k_center_select

#: Fixed per-record wire header: key (3x int32) + pos (2x float32) + bbox (4x int32).
HEADER_BYTES = 36

_PAYLOAD_BYTES_PER_DIM = {"none": 4, "fp16": 2, "int8": 1}


def payload_nbytes(quantization: str, dim: int) -> int:
    """Embedding payload size in bytes for one record."""
    return _PAYLOAD_BYTES_PER_DIM[quantization] * dim


def encode_embedding(embedding: np.ndarray, quantization: str) -> bytes:
    """Encode one L2-normalized fp32 embedding into its wire payload.

    Little-endian dtypes throughout. ``"int8"`` uses symmetric per-vector
    max-abs scaling to codes in [-127, 127]; the scale is NOT transmitted —
    decoding renormalizes, which cancels it exactly.
    """
    vec = np.asarray(embedding, dtype=np.float32)
    if quantization == "none":
        return vec.astype("<f4").tobytes()
    if quantization == "fp16":
        return vec.astype("<f2").tobytes()
    if quantization == "int8":
        scale = np.float32(np.max(np.abs(vec)))
        codes = np.round(vec / scale * np.float32(127.0)).astype("<i1")
        return codes.tobytes()
    raise ValueError(f"unknown quantization {quantization!r}")


def decode_embedding(payload: bytes, quantization: str, dim: int) -> np.ndarray:
    """Decode a wire payload back to a unit-norm fp32 embedding.

    ``"none"`` round-trips exactly; ``"fp16"`` and ``"int8"`` renormalize in
    float64 so the result is unit-norm (as :class:`MemoryRecord` validates).
    For ``"int8"`` the renormalization makes re-encoding bit-identical to
    the original payload: quantization is scale-invariant, so the unknown
    scale cancels.
    """
    expected = payload_nbytes(quantization, dim)
    if len(payload) != expected:
        raise ValueError(
            f"payload is {len(payload)} bytes, expected {expected} for "
            f"{quantization!r} at dim {dim}"
        )
    if quantization == "none":
        return np.frombuffer(payload, dtype="<f4").astype(np.float32)
    if quantization == "fp16":
        vec = np.frombuffer(payload, dtype="<f2").astype(np.float64)
    elif quantization == "int8":
        vec = np.frombuffer(payload, dtype="<i1").astype(np.float64)
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def select_records(
    records: Sequence[MemoryRecord], k: int, sender_policy: str
) -> list[MemoryRecord]:
    """Pick at most ``k`` records to share, deterministically, in key order.

    ``"most_recent"`` keeps the ``k`` largest keys (tuple order);
    ``"coverage_greedy"`` runs greedy max-min cosine distance selection
    (:func:`~swarm_perception.fusion.memory.k_center_select` — the same
    seed-smallest-key, tie-by-smallest-key rules as the memory cap).
    """
    ordered = sorted(records, key=lambda r: r.key)
    if len(ordered) <= k:
        return ordered
    if sender_policy == "most_recent":
        return ordered[-k:]
    if sender_policy == "coverage_greedy":
        return k_center_select(ordered, k)
    raise ValueError(f"unknown sender_policy {sender_policy!r}")


@dataclasses.dataclass(frozen=True)
class WireRecord:
    """One record as transmitted: header fields plus the quantized payload."""

    key: Key
    payload: bytes
    pos: tuple[float, float]
    bbox: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class Message:
    """One broadcast as it leaves the sender (receiver-independent)."""

    sender: int
    sender_tick: int
    epoch: int
    delivery_tick: int
    records: tuple[WireRecord, ...]
    residue_bitmap: bytes | None
    nbytes: int

    @property
    def k_sent(self) -> int:
        """Number of records on the wire."""
        return len(self.records)


def build_message(
    *,
    sender: int,
    sender_tick: int,
    epoch: int,
    records: Sequence[MemoryRecord],
    payload_cache: MutableMapping[Key, bytes],
    cfg: CommsCfg,
    residue_bitmap: bytes | None,
) -> Message:
    """Assemble one broadcast: select, quantize-once, and price it.

    Payloads come from ``payload_cache`` when present (a record first seen
    on the wire keeps its exact received bytes); otherwise they are encoded
    here and cached, so every record is quantized exactly once per run.
    """
    selected = select_records(records, cfg.k, cfg.sender_policy)
    wire: list[WireRecord] = []
    for rec in selected:
        payload = payload_cache.get(rec.key)
        if payload is None:
            payload = encode_embedding(rec.embedding, cfg.quantization)
            payload_cache[rec.key] = payload
        wire.append(
            WireRecord(key=rec.key, payload=payload, pos=rec.pos, bbox=rec.crop_bbox)
        )
    residue_len = len(residue_bitmap) if residue_bitmap is not None else 0
    nbytes = sum(HEADER_BYTES + len(w.payload) for w in wire) + residue_len
    return Message(
        sender=sender,
        sender_tick=sender_tick,
        epoch=epoch,
        delivery_tick=sender_tick + cfg.delay_ticks,
        records=tuple(wire),
        residue_bitmap=residue_bitmap,
        nbytes=nbytes,
    )


def reconstruct_records(message: Message, quantization: str, dim: int) -> list[MemoryRecord]:
    """Rebuild memory records from a received message.

    Only header fields travel: ``first_seen`` restarts at the key's epoch
    and ``merged_from_count`` at 1 — fold provenance is local to a robot
    and never transmitted.
    """
    return [
        MemoryRecord(
            embedding=decode_embedding(w.payload, quantization, dim),
            key=w.key,
            pos=w.pos,
            crop_bbox=w.bbox,
            first_seen=w.key[0],
        )
        for w in message.records
    ]


class Channel:
    """Per-run channel state: the dedicated, seeded packet-drop stream."""

    def __init__(self, cfg: CommsCfg, seed: int) -> None:
        """Derive the channel RNG from the run seed (separate stream).

        Args:
            cfg: The run's ``comms`` config section.
            seed: The run seed (``simulation.seed``).
        """
        self.cfg = cfg
        self._rng = random.Random(f"comms:{seed}")

    def drops_message(self) -> bool:
        """Draw one Bernoulli(drop_p) drop decision.

        Draws nothing when ``drop_p`` is 0, so drop-free runs consume no
        channel randomness at all.
        """
        if self.cfg.drop_p <= 0.0:
            return False
        return self._rng.random() < self.cfg.drop_p
