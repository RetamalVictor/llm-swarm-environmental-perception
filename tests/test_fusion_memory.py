"""Fusion memory: merge properties, non-associativity pin, exact hand merges.

Hypothesis runs with ``derandomize=True`` and ``deadline=None`` so CI is
stable and reproducible; embeddings come from ``np.random.default_rng`` on a
drawn seed, unit-normalized, dims 8-16, set sizes up to 20.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarm_perception.fusion import MemoryRecord, SetMemory, peer_merge

SETTINGS = settings(max_examples=60, deadline=None, derandomize=True)
TAUS = st.sampled_from((0.7, 0.9, 0.98))
CAPS = st.integers(min_value=1, max_value=12)


def unit(*components: float, dim: int = 8) -> np.ndarray:
    """A unit-normalized float32 vector with the given leading components."""
    vec = np.zeros(dim, dtype=np.float64)
    vec[: len(components)] = components
    vec /= np.linalg.norm(vec)
    return vec.astype(np.float32)


def record(
    key: tuple[int, int, int],
    embedding: np.ndarray,
    *,
    first_seen: int = 0,
    count: int = 1,
) -> MemoryRecord:
    return MemoryRecord(
        embedding=embedding,
        key=key,
        pos=(0.0, 0.0),
        crop_bbox=(0, 0, 8, 8),
        first_seen=first_seen,
        merged_from_count=count,
    )


def signature(memory: SetMemory) -> tuple:
    """Comparable content: key, first_seen, merged_from_count, embedding."""
    return tuple(
        (r.key, r.first_seen, r.merged_from_count, r.embedding.tobytes())
        for r in memory.records
    )


@st.composite
def record_pools(draw: st.DrawFn, pools: int = 1, max_size: int = 20):
    """``pools`` record lists sharing one embedding dim; keys may collide."""
    dim = draw(st.integers(min_value=8, max_value=16))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    meta = st.tuples(
        st.integers(0, 4),  # capture epoch
        st.integers(0, 3),  # robot id
        st.integers(0, 2),  # crop idx
        st.integers(0, 9),  # first_seen
    )
    out: list[list[MemoryRecord]] = []
    for _ in range(pools):
        records: list[MemoryRecord] = []
        for epoch, robot, crop, first_seen in draw(st.lists(meta, max_size=max_size)):
            vec = rng.normal(size=dim)
            vec /= np.linalg.norm(vec)
            records.append(
                record(
                    (epoch, robot, crop), vec.astype(np.float32), first_seen=first_seen
                )
            )
        out.append(records)
    return out


# --------------------------------------------------------------- properties


@SETTINGS
@given(pools=record_pools(pools=2), tau=TAUS, cap=CAPS)
def test_merge_commutative(pools, tau: float, cap: int) -> None:
    recs_a, recs_b = pools
    a = SetMemory(recs_a, tau_dedup=tau, memory_cap=cap)
    b = SetMemory(recs_b, tau_dedup=tau, memory_cap=cap)
    ab = a.merge(b)
    ba = b.merge(a)
    assert ab == ba
    assert signature(ab) == signature(ba)


@SETTINGS
@given(pools=record_pools(pools=1), tau=TAUS, cap=CAPS)
def test_merge_idempotent(pools, tau: float, cap: int) -> None:
    (recs,) = pools
    a = SetMemory(recs, tau_dedup=tau, memory_cap=cap)  # a == canonical(recs)
    assert a.merge(a) == a
    assert signature(a.merge(a)) == signature(a)
    # Canonicalization itself is idempotent too.
    assert SetMemory(a.records, tau_dedup=tau, memory_cap=cap) == a


@SETTINGS
@given(pools=record_pools(pools=1), tau=TAUS, cap=CAPS, data=st.data())
def test_construction_permutation_invariant(pools, tau, cap, data) -> None:
    (recs,) = pools
    permuted = data.draw(st.permutations(recs))
    assert SetMemory(permuted, tau_dedup=tau, memory_cap=cap) == SetMemory(
        recs, tau_dedup=tau, memory_cap=cap
    )


@SETTINGS
@given(pools=record_pools(pools=1), tau=TAUS, cap=CAPS)
def test_canonical_invariants(pools, tau: float, cap: int) -> None:
    (recs,) = pools
    mem = SetMemory(recs, tau_dedup=tau, memory_cap=cap)
    assert len(mem) <= cap
    keys = mem.keys()
    assert list(keys) == sorted(keys)
    assert len(set(keys)) == len(keys)
    stored = mem.records
    for i in range(len(stored)):
        for j in range(i + 1, len(stored)):
            sim = float(np.dot(stored[i].embedding, stored[j].embedding))
            # Small slack: k-center math must not push a pair over tau.
            assert sim < tau + 1e-5


# NO associativity test on purpose: merge is NOT associative. The regression
# test below pins that documented behavior with a hand-built counterexample.


def test_merge_not_associative_regression() -> None:
    tau = 0.9
    a = record((0, 0, 0), unit(1.0, 0.0), first_seen=0)
    b = record(
        (0, 0, 1),
        unit(math.cos(math.radians(20)), math.sin(math.radians(20))),
        first_seen=1,
    )
    c = record(
        (0, 0, 2),
        unit(math.cos(math.radians(40)), math.sin(math.radians(40))),
        first_seen=2,
    )
    # Counterexample shape: sim(a,b) >= tau, sim(b,c) >= tau, sim(a,c) < tau.
    assert float(np.dot(a.embedding, b.embedding)) >= tau
    assert float(np.dot(b.embedding, c.embedding)) >= tau
    assert float(np.dot(a.embedding, c.embedding)) < tau

    mem_a = SetMemory([a], tau_dedup=tau, memory_cap=10)
    mem_b = SetMemory([b], tau_dedup=tau, memory_cap=10)
    mem_c = SetMemory([c], tau_dedup=tau, memory_cap=10)

    left = mem_a.merge(mem_b).merge(mem_c)  # (A + B) + C
    right = mem_a.merge(mem_b.merge(mem_c))  # A + (B + C)

    # (A + B): b folds into a; c then survives because sim(a,c) < tau.
    assert left.keys() == ((0, 0, 0), (0, 0, 2))
    assert left.records[0].merged_from_count == 2
    assert left.records[0].first_seen == 0
    assert left.records[1].merged_from_count == 1
    # (B + C): c folds into b; the folded b then folds into a with c's count.
    assert right.keys() == ((0, 0, 0),)
    assert right.records[0].merged_from_count == 3
    assert right.records[0].first_seen == 0
    # Different merge trees, different memories — the documented behavior.
    assert left != right


# ------------------------------------------------------------ k-center cap


def test_k_center_cap_two_of_four_tie_breaks_by_smallest_key() -> None:
    r1 = record((0, 0, 0), unit(1.0, 0.0, 0.0))
    r2 = record((0, 0, 1), unit(0.0, 1.0, 0.0))
    r3 = record((0, 0, 2), unit(0.0, 0.0, 1.0))
    r4 = record((0, 0, 3), unit(0.5, math.sqrt(3.0) / 2.0, 0.0))
    # All pairwise sims < tau=0.95 (max is sim(r2,r4)=0.866): dedup keeps all.
    mem = SetMemory([r1, r2, r3, r4], tau_dedup=0.95, memory_cap=2)
    # Seed is the smallest key (r1). Cosine distances to r1: r2 -> 1.0,
    # r3 -> 1.0, r4 -> 0.5. r2 and r3 tie within 1e-6; smallest key wins.
    assert mem.keys() == ((0, 0, 0), (0, 0, 1))


def test_k_center_prefers_farther_record_over_smaller_key() -> None:
    r1 = record((0, 0, 0), unit(1.0, 0.0))
    r2 = record((0, 0, 1), unit(0.0, 1.0))
    r3 = record((0, 0, 2), unit(-1.0, 0.0))
    mem = SetMemory([r1, r2, r3], tau_dedup=0.95, memory_cap=2)
    # Distances to r1: r2 -> 1.0, r3 -> 2.0. No tie: r3 wins despite its key.
    assert mem.keys() == ((0, 0, 0), (0, 0, 2))


# ------------------------------------------------------- exact hand merges


def test_merge_folds_near_duplicate_exact() -> None:
    tau = 0.9
    a = record((0, 0, 0), unit(1.0, 0.0), first_seen=5, count=1)
    b = record(
        (1, 0, 0),
        unit(math.cos(math.radians(10)), math.sin(math.radians(10))),
        first_seen=2,
        count=3,
    )
    assert float(np.dot(a.embedding, b.embedding)) >= tau  # cos(10 deg) ~ 0.985
    merged = SetMemory([a], tau_dedup=tau, memory_cap=10).merge(
        SetMemory([b], tau_dedup=tau, memory_cap=10)
    )
    assert len(merged) == 1
    (kept,) = merged.records
    assert kept.key == (0, 0, 0)  # earlier key survives
    assert kept.merged_from_count == 4  # 1 + 3
    assert kept.first_seen == 2  # min(5, 2)
    assert np.array_equal(kept.embedding, a.embedding)  # keeper unchanged
    assert kept.pos == a.pos and kept.crop_bbox == a.crop_bbox


def test_merge_keeps_distinct_records() -> None:
    a = record((0, 0, 0), unit(1.0, 0.0), first_seen=1)
    b = record((1, 0, 0), unit(0.0, 1.0), first_seen=2)
    merged = SetMemory([b], tau_dedup=0.9, memory_cap=10).merge(
        SetMemory([a], tau_dedup=0.9, memory_cap=10)
    )
    assert merged.keys() == ((0, 0, 0), (1, 0, 0))  # sorted by key
    assert [r.merged_from_count for r in merged] == [1, 1]
    assert [r.first_seen for r in merged] == [1, 2]


def test_key_equality_collapse_is_exact_duplicate_removal() -> None:
    emb = unit(1.0)
    low = record((2, 1, 0), emb, first_seen=3, count=2)
    high = record((2, 1, 0), emb, first_seen=7, count=5)
    merged = SetMemory([high], tau_dedup=0.9, memory_cap=10).merge(
        SetMemory([low], tau_dedup=0.9, memory_cap=10)
    )
    assert len(merged) == 1
    (kept,) = merged.records
    # Smaller first_seen wins; counts are NOT summed for identical keys.
    assert kept.first_seen == 3
    assert kept.merged_from_count == 2


def test_fold_targets_most_similar_keeper() -> None:
    k1 = record((0, 0, 0), unit(1.0, 0.0))
    k2 = record((0, 0, 1), unit(0.0, 1.0))
    # 80 degrees from k1, 10 degrees from k2: folds into k2, not key-first k1.
    c = record(
        (0, 0, 2), unit(math.cos(math.radians(80)), math.sin(math.radians(80)))
    )
    mem = SetMemory([k1, k2, c], tau_dedup=0.9, memory_cap=10)
    assert mem.keys() == ((0, 0, 0), (0, 0, 1))
    by_key = {r.key: r for r in mem}
    assert by_key[(0, 0, 0)].merged_from_count == 1
    assert by_key[(0, 0, 1)].merged_from_count == 2


# --------------------------------------------------------------- peer_merge


def _peer_fixture() -> tuple[SetMemory, list[MemoryRecord]]:
    mine = SetMemory(
        [record((0, 0, 0), unit(1.0))], tau_dedup=0.9, memory_cap=10
    )
    incoming = [  # deliberately unsorted; all mutually orthogonal
        record((1, 1, 0), unit(0.0, 0.0, 0.0, 1.0)),
        record((1, 0, 1), unit(0.0, 0.0, 1.0)),
        record((1, 0, 0), unit(0.0, 1.0)),
    ]
    return mine, incoming


def test_peer_merge_budget_takes_smallest_keys() -> None:
    mine, incoming = _peer_fixture()
    merged = peer_merge(mine, incoming, budget=2)
    assert merged.keys() == ((0, 0, 0), (1, 0, 0), (1, 0, 1))


def test_peer_merge_unlimited_and_zero_budget() -> None:
    mine, incoming = _peer_fixture()
    assert peer_merge(mine, incoming, budget=None).keys() == (
        (0, 0, 0),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
    )
    assert peer_merge(mine, incoming, budget=0) == mine


def test_peer_merge_negative_budget_raises() -> None:
    mine, _ = _peer_fixture()
    with pytest.raises(ValueError):
        peer_merge(mine, [], budget=-1)


# --------------------------------------------------- validation & accessors


def test_memory_record_validation() -> None:
    with pytest.raises(ValueError):
        record((0, 0, 0), np.zeros((2, 2), dtype=np.float32))  # not 1-D
    with pytest.raises(ValueError):
        record((0, 0, 0), np.full(8, 0.5, dtype=np.float32))  # norm != 1
    with pytest.raises(ValueError):
        record((0, 0, 0), unit(1.0), count=0)
    rec = record((0, 0, 0), unit(1.0))
    with pytest.raises(ValueError):
        rec.embedding[0] = 0.0  # stored embedding is read-only


def test_set_memory_parameter_validation() -> None:
    with pytest.raises(ValueError):
        SetMemory(tau_dedup=0.0, memory_cap=5)
    with pytest.raises(ValueError):
        SetMemory(tau_dedup=1.5, memory_cap=5)
    with pytest.raises(ValueError):
        SetMemory(tau_dedup=0.9, memory_cap=0)
    a = SetMemory(tau_dedup=0.9, memory_cap=5)
    b = SetMemory(tau_dedup=0.8, memory_cap=5)
    with pytest.raises(ValueError):
        a.merge(b)


def test_mixed_embedding_dims_raise() -> None:
    with pytest.raises(ValueError):
        SetMemory(
            [record((0, 0, 0), unit(1.0, dim=8)), record((0, 0, 1), unit(1.0, dim=9))],
            tau_dedup=0.9,
            memory_cap=5,
        )


def test_add_and_accessors() -> None:
    mem = SetMemory(tau_dedup=0.9, memory_cap=4)
    rec = record((0, 0, 0), unit(1.0))
    mem.add(rec)
    assert len(mem) == 1
    assert (0, 0, 0) in mem
    mem.add(rec)  # exact duplicate: key collapse, nothing summed
    assert len(mem) == 1
    assert mem.records[0].merged_from_count == 1
    mem.add(record((0, 0, 1), unit(0.0, 1.0)))
    assert [r.key for r in mem] == [(0, 0, 0), (0, 0, 1)]
    assert mem.tau_dedup == 0.9
    assert mem.memory_cap == 4
