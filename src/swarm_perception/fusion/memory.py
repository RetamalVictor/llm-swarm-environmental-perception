"""Bounded embedding memory with a canonical, order-independent merge (D6).

A :class:`SetMemory` holds at most ``memory_cap`` :class:`MemoryRecord`\\ s and
is always stored in canonical form. Canonicalizing one batch of records runs
three steps:

a. **Key-equality collapse** — records with an identical ``key`` are the same
   capture, so this step is exact-duplicate removal: exactly one record
   survives and nothing is summed. The survivor is the one with the smallest
   ``(first_seen, merged_from_count)``, with embedding bytes as the final
   tie-break, so the collapse is a symmetric function of the duplicate set.
b. **Greedy cosine dedup** — survivors are concatenated and sorted by ``key``
   (tuple order), then scanned once: a record is kept iff its cosine
   similarity to every already-kept record is ``< tau_dedup``; otherwise it
   folds into the kept record it is most similar to (on exact similarity
   ties, the kept record with the smallest key). Folding adds the loser's
   ``merged_from_count`` to the keeper and takes the min ``first_seen``; the
   keeper's embedding is unchanged. The earlier key survives by construction.
c. **Capacity cap** — if more than ``memory_cap`` records remain, greedy
   k-center selection over cosine distance keeps ``memory_cap`` of them:
   start from the record with the smallest key, then repeatedly add the
   record maximizing the min distance to the selected set; ties in the
   distance comparison are broken by smallest key using an absolute
   tolerance of 1e-6 on the distance.

Properties this construction guarantees:

- merge is COMMUTATIVE and IDEMPOTENT and deterministic under permuted
  insertion order.
- It is NOT associative (greedy dedup discards records; different merge
  trees can differ) — never claim associativity. :meth:`SetMemory.add` is a
  merge with a single record, so a *sequence* of adds is one particular
  merge tree and is order-sensitive; only batch canonicalization of one
  record multiset is permutation-invariant.
- Cost is O(n²) in the number of records (pairwise cosine similarities
  drive both the dedup scan and the k-center selection).

Pure numpy; no torch.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator

import numpy as np

Key = tuple[int, int, int]

_TIE_TOLERANCE = 1e-6  # absolute tolerance on k-center distance comparisons
_NORM_TOLERANCE = 1e-3  # allowed |L2 norm - 1| at record construction


@dataclasses.dataclass(frozen=True, eq=False)
class MemoryRecord:
    """One capture's embedding plus provenance.

    Attributes:
        embedding: 1-D float32, L2-normalized (validated; stored read-only).
        key: ``(capture_epoch, robot_id, crop_idx)`` — capture identity.
        pos: Robot position ``(x, y)`` at capture time.
        crop_bbox: Clipped crop rect ``(x1, y1, x2, y2)`` in image pixels.
        first_seen: Smallest epoch among this record and every record that
            folded into it.
        merged_from_count: Number of raw captures folded into this record.
    """

    embedding: np.ndarray
    key: Key
    pos: tuple[float, float]
    crop_bbox: tuple[int, int, int, int]
    first_seen: int
    merged_from_count: int = 1

    def __post_init__(self) -> None:
        emb = np.asarray(self.embedding, dtype=np.float32)
        if emb.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {emb.shape}")
        norm = float(np.linalg.norm(emb.astype(np.float64)))
        if abs(norm - 1.0) > _NORM_TOLERANCE:
            raise ValueError(f"embedding must be L2-normalized, got norm {norm:.6f}")
        if self.merged_from_count < 1:
            raise ValueError(
                f"merged_from_count must be >= 1, got {self.merged_from_count}"
            )
        emb = emb.copy()
        emb.setflags(write=False)
        object.__setattr__(self, "embedding", emb)


def record_rank(record: MemoryRecord) -> tuple[Key, int, int, bytes]:
    """Total deterministic order over records, independent of input order.

    Sorts by ``key`` first, then ``(first_seen, merged_from_count, embedding
    bytes)`` so any selection among records — including exact duplicates — is
    a symmetric function of the record set.
    """
    return (
        record.key,
        record.first_seen,
        record.merged_from_count,
        record.embedding.tobytes(),
    )


def _collapse_equal_keys(records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
    """Step (a): exact-duplicate removal — one survivor per key, nothing summed."""
    by_key: dict[Key, MemoryRecord] = {}
    for record in records:
        held = by_key.get(record.key)
        if held is None or record_rank(record) < record_rank(held):
            by_key[record.key] = record
    return sorted(by_key.values(), key=lambda r: r.key)


def _greedy_cosine_dedup(
    records: list[MemoryRecord], tau_dedup: float
) -> list[MemoryRecord]:
    """Step (b): one pass over key-sorted, key-unique records.

    Each similarity is one ``np.dot`` over the same float32 vectors, so a
    given pair always yields the identical value regardless of how many
    records surround it — this keeps merges bit-deterministic across
    different merge contexts.
    """
    kept: list[MemoryRecord] = []
    first_seen: list[int] = []
    counts: list[int] = []
    for record in records:
        if kept:
            sims = [float(np.dot(k.embedding, record.embedding)) for k in kept]
            best = int(np.argmax(sims))  # first max: smallest kept key on ties
            if sims[best] >= tau_dedup:
                counts[best] += record.merged_from_count
                first_seen[best] = min(first_seen[best], record.first_seen)
                continue
        kept.append(record)
        first_seen.append(record.first_seen)
        counts.append(record.merged_from_count)

    merged: list[MemoryRecord] = []
    for record, seen, count in zip(kept, first_seen, counts, strict=True):
        if seen != record.first_seen or count != record.merged_from_count:
            record = dataclasses.replace(
                record, first_seen=seen, merged_from_count=count
            )
        merged.append(record)
    return merged


def _k_center_cap(records: list[MemoryRecord], memory_cap: int) -> list[MemoryRecord]:
    """Step (c): greedy k-center over cosine distance on key-sorted records."""
    if len(records) <= memory_cap:
        return records
    embeddings = np.stack([r.embedding for r in records]).astype(np.float64)
    distance = 1.0 - embeddings @ embeddings.T

    chosen = np.zeros(len(records), dtype=bool)
    chosen[0] = True  # seed: the record with the smallest key
    selected = [0]
    min_dist = distance[0].copy()
    while len(selected) < memory_cap:
        best = -1
        best_dist = -np.inf
        for i in range(len(records)):  # key order: smallest key wins ties
            if chosen[i]:
                continue
            if best < 0 or min_dist[i] > best_dist + _TIE_TOLERANCE:
                best = i
                best_dist = float(min_dist[i])
        selected.append(best)
        chosen[best] = True
        min_dist = np.minimum(min_dist, distance[best])
    return [records[i] for i in sorted(selected)]


def _canonical_records(
    records: Iterable[MemoryRecord], tau_dedup: float, memory_cap: int
) -> tuple[MemoryRecord, ...]:
    """Run the full canonical construction (steps a, b, c) over a batch."""
    survivors = _collapse_equal_keys(records)
    dims = {r.embedding.shape[0] for r in survivors}
    if len(dims) > 1:
        raise ValueError(f"embedding dims differ across records: {sorted(dims)}")
    deduped = _greedy_cosine_dedup(survivors, tau_dedup)
    return tuple(_k_center_cap(deduped, memory_cap))


class SetMemory:
    """Bounded record set, always stored in canonical form.

    Records are canonicalized at construction (see the module docstring), so
    two memories built from the same record multiset — in any order — are
    equal, and :meth:`merge` is commutative and idempotent.
    """

    def __init__(
        self,
        records: Iterable[MemoryRecord] = (),
        *,
        tau_dedup: float,
        memory_cap: int,
    ) -> None:
        if not 0.0 < tau_dedup <= 1.0:
            raise ValueError(f"tau_dedup must be in (0, 1], got {tau_dedup}")
        if memory_cap < 1:
            raise ValueError(f"memory_cap must be >= 1, got {memory_cap}")
        self._tau_dedup = float(tau_dedup)
        self._memory_cap = int(memory_cap)
        self._records = _canonical_records(records, self._tau_dedup, self._memory_cap)

    @property
    def tau_dedup(self) -> float:
        """Cosine-similarity threshold at which records fold together."""
        return self._tau_dedup

    @property
    def memory_cap(self) -> int:
        """Hard cap on stored records."""
        return self._memory_cap

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        """Stored records, sorted by key, keys unique."""
        return self._records

    def keys(self) -> tuple[Key, ...]:
        """Stored record keys in tuple order."""
        return tuple(r.key for r in self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[MemoryRecord]:
        return iter(self._records)

    def __contains__(self, key: object) -> bool:
        return any(r.key == key for r in self._records)

    def add(self, record: MemoryRecord) -> None:
        """Merge one record in place.

        Equivalent to merging with a singleton memory. A sequence of adds is
        one particular merge tree and therefore order-sensitive (merge is not
        associative); batch construction is the permutation-invariant path.
        """
        self._records = _canonical_records(
            self._records + (record,), self._tau_dedup, self._memory_cap
        )

    def merge(self, other: SetMemory) -> SetMemory:
        """Canonical merge of two memories; commutative and idempotent."""
        if (other.tau_dedup, other.memory_cap) != (self._tau_dedup, self._memory_cap):
            raise ValueError(
                "cannot merge memories with different parameters: "
                f"({self._tau_dedup}, {self._memory_cap}) vs "
                f"({other.tau_dedup}, {other.memory_cap})"
            )
        return SetMemory(
            self._records + other._records,
            tau_dedup=self._tau_dedup,
            memory_cap=self._memory_cap,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetMemory):
            return NotImplemented
        if (self._tau_dedup, self._memory_cap) != (other._tau_dedup, other._memory_cap):
            return False
        if len(self._records) != len(other._records):
            return False
        return all(
            a.key == b.key
            and a.first_seen == b.first_seen
            and a.merged_from_count == b.merged_from_count
            and a.pos == b.pos
            and a.crop_bbox == b.crop_bbox
            and np.array_equal(a.embedding, b.embedding)
            for a, b in zip(self._records, other._records, strict=True)
        )

    def __repr__(self) -> str:
        return (
            f"SetMemory(n={len(self._records)}, tau_dedup={self._tau_dedup}, "
            f"memory_cap={self._memory_cap})"
        )
