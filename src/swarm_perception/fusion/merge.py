"""Peer-to-peer memory merging with a per-message record budget."""

from __future__ import annotations

from collections.abc import Sequence

from swarm_perception.fusion.memory import MemoryRecord, SetMemory, record_rank


def peer_merge(
    mine: SetMemory, incoming: Sequence[MemoryRecord], budget: int | None
) -> SetMemory:
    """Merge at most ``budget`` incoming records into ``mine``.

    Incoming records are taken in sorted-key order (full deterministic rank
    :func:`swarm_perception.fusion.memory.record_rank`, so duplicate keys in
    ``incoming`` cannot make the budget cut order-dependent); the first
    ``budget`` of them are merged through the same canonical construction as
    :meth:`SetMemory.merge`. ``budget=None`` means unlimited; ``budget=0``
    returns a canonical copy of ``mine``.
    """
    if budget is not None and budget < 0:
        raise ValueError(f"budget must be >= 0 or None, got {budget}")
    taken = sorted(incoming, key=record_rank)
    if budget is not None:
        taken = taken[:budget]
    return SetMemory(
        mine.records + tuple(taken),
        tau_dedup=mine.tau_dedup,
        memory_cap=mine.memory_cap,
    )
