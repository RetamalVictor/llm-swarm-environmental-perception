"""Embedding-memory fusion (design decision D6).

Bounded per-robot embedding memories — :class:`SetMemory` holding
:class:`MemoryRecord` entries — with a canonical merge construction that is
commutative, idempotent, and deterministic under permuted insertion order
(and explicitly NOT associative; see :mod:`swarm_perception.fusion.memory`),
plus budgeted peer-to-peer merging via :func:`peer_merge`. Pure numpy; no
torch.
"""

from swarm_perception.fusion.memory import MemoryRecord, SetMemory
from swarm_perception.fusion.merge import peer_merge

__all__ = ["MemoryRecord", "SetMemory", "peer_merge"]
