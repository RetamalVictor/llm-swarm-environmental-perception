"""Channel model: codecs, sender policies, byte model, quantize-once gossip."""

from __future__ import annotations

import numpy as np
import pytest

from swarm_perception.config import CommsCfg
from swarm_perception.fusion import MemoryRecord
from swarm_perception.perception import stub_embedding
from swarm_perception.perception.encoder import EMBED_DIM
from swarm_perception.sim.channel import (
    HEADER_BYTES,
    Channel,
    build_message,
    decode_embedding,
    encode_embedding,
    payload_nbytes,
    reconstruct_records,
    select_records,
)

DIM = EMBED_DIM


def record(key: tuple[int, int, int], embedding: np.ndarray | None = None) -> MemoryRecord:
    return MemoryRecord(
        embedding=stub_embedding(key) if embedding is None else embedding,
        key=key,
        pos=(float(key[0]), float(key[1])),
        crop_bbox=(0, 0, 10, 10),
        first_seen=key[0],
    )


def unit(*components: float) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float64)
    vec[: len(components)] = components
    return (vec / np.linalg.norm(vec)).astype(np.float32)


# ------------------------------------------------------------------- codecs


@pytest.mark.parametrize(
    ("quantization", "nbytes"), [("none", 4 * DIM), ("fp16", 2 * DIM), ("int8", DIM)]
)
def test_payload_sizes(quantization: str, nbytes: int) -> None:
    emb = stub_embedding((1, 0, 0))
    payload = encode_embedding(emb, quantization)
    assert len(payload) == nbytes == payload_nbytes(quantization, DIM)


@pytest.mark.parametrize("quantization", ["none", "fp16", "int8"])
def test_decode_returns_unit_norm_and_is_deterministic(quantization: str) -> None:
    emb = stub_embedding((2, 1, 0))
    payload = encode_embedding(emb, quantization)
    decoded = decode_embedding(payload, quantization, DIM)
    assert decoded.dtype == np.float32
    assert abs(float(np.linalg.norm(decoded.astype(np.float64))) - 1.0) < 1e-3
    assert decoded.tobytes() == decode_embedding(payload, quantization, DIM).tobytes()
    # Quantization keeps the direction: cosine to the original stays high.
    assert float(np.dot(decoded, emb)) > 0.99


def test_none_round_trips_exactly() -> None:
    emb = stub_embedding((3, 0, 0))
    assert decode_embedding(encode_embedding(emb, "none"), "none", DIM).tobytes() == emb.tobytes()


def test_int8_requantization_is_idempotent() -> None:
    # Re-encoding a decoded int8 embedding must reproduce the payload bit for
    # bit: max-abs scaling is scale-invariant and decode renormalizes.
    for key in [(1, 0, 0), (5, 3, 0), (40, 9, 0)]:
        payload = encode_embedding(stub_embedding(key), "int8")
        decoded = decode_embedding(payload, "int8", DIM)
        assert encode_embedding(decoded, "int8") == payload


def test_decode_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="payload"):
        decode_embedding(b"\x00" * 10, "int8", DIM)


# ------------------------------------------------------------ sender policies


def test_most_recent_takes_largest_keys_in_key_order() -> None:
    records = [record((e, r, 0)) for e in (1, 2, 3) for r in (0, 1)]
    picked = select_records(list(reversed(records)), 3, "most_recent")
    assert [r.key for r in picked] == [(2, 1, 0), (3, 0, 0), (3, 1, 0)]


def test_coverage_greedy_differs_from_most_recent_on_crafted_memory() -> None:
    # Three nearly-parallel recent records plus one old orthogonal one: the
    # coverage sender must spend a slot on the odd direction, the recency
    # sender must not.
    records = [
        record((1, 0, 0), unit(0.0, 1.0)),  # old, orthogonal
        record((5, 0, 0), unit(1.0)),
        record((6, 0, 0), unit(0.999, 0.045)),
        record((7, 0, 0), unit(0.999, -0.045)),
    ]
    recent = [r.key for r in select_records(records, 2, "most_recent")]
    coverage = [r.key for r in select_records(records, 2, "coverage_greedy")]
    assert recent == [(6, 0, 0), (7, 0, 0)]
    assert (1, 0, 0) in coverage
    assert recent != coverage


def test_selection_is_deterministic_under_permutation_and_caps_at_k() -> None:
    records = [record((e, 0, 0)) for e in range(10)]
    for policy in ("most_recent", "coverage_greedy"):
        a = [r.key for r in select_records(records, 4, policy)]
        b = [r.key for r in select_records(list(reversed(records)), 4, policy)]
        assert a == b
        assert len(a) == 4
    # k >= n shares everything, key-sorted.
    assert [r.key for r in select_records(records, 99, "most_recent")] == [
        (e, 0, 0) for e in range(10)
    ]


# ------------------------------------------------- messages and quantize-once


def _cfg(**overrides) -> CommsCfg:
    base = dict(k=4, quantization="int8", delay_ticks=0)
    base.update(overrides)
    return CommsCfg(**base)


def test_build_message_prices_by_the_byte_model() -> None:
    records = [record((e, 0, 0)) for e in (1, 2, 3)]
    residue = b"\x00" * 40
    msg = build_message(
        sender=0,
        sender_tick=17,
        epoch=3,
        records=records,
        payload_cache={},
        cfg=_cfg(),
        residue_bitmap=residue,
    )
    assert msg.k_sent == 3
    assert msg.nbytes == 3 * (HEADER_BYTES + DIM) + 40
    assert msg.delivery_tick == 17

    no_residue = build_message(
        sender=0,
        sender_tick=17,
        epoch=3,
        records=records,
        payload_cache={},
        cfg=_cfg(quantization="fp16", delay_ticks=5),
        residue_bitmap=None,
    )
    assert no_residue.nbytes == 3 * (HEADER_BYTES + 2 * DIM)
    assert no_residue.delivery_tick == 22


def test_quantize_once_payload_is_canonical_across_gossip_paths() -> None:
    # A shares to B; B re-shares to C from its receive-populated cache. The
    # payload bytes must equal what a direct A -> C share produces.
    rec = record((4, 2, 0))
    cache_a: dict[tuple[int, int, int], bytes] = {}
    cfg = _cfg(k=1)

    msg_ab = build_message(
        sender=2, sender_tick=1, epoch=4, records=[rec],
        payload_cache=cache_a, cfg=cfg, residue_bitmap=None,
    )
    payload_ab = msg_ab.records[0].payload

    # B stores the record (decoded embedding) and caches the wire bytes.
    cache_b = {w.key: w.payload for w in msg_ab.records}
    [rec_at_b] = reconstruct_records(msg_ab, cfg.quantization, DIM)
    msg_bc = build_message(
        sender=5, sender_tick=9, epoch=5, records=[rec_at_b],
        payload_cache=cache_b, cfg=cfg, residue_bitmap=None,
    )
    assert msg_bc.records[0].payload == payload_ab

    # Direct A -> C (fresh cache at A: re-encoding its canonical embedding).
    msg_ac = build_message(
        sender=2, sender_tick=30, epoch=6, records=[rec],
        payload_cache={}, cfg=cfg, residue_bitmap=None,
    )
    assert msg_ac.records[0].payload == payload_ab

    # And even without B's cache, int8 re-encoding of the decoded embedding
    # reproduces the same bytes (scale-invariance backstop).
    assert encode_embedding(rec_at_b.embedding, "int8") == payload_ab


def test_reconstruct_records_restarts_local_provenance() -> None:
    rec = record((7, 1, 0))
    msg = build_message(
        sender=1, sender_tick=3, epoch=7, records=[rec],
        payload_cache={}, cfg=_cfg(quantization="none"), residue_bitmap=None,
    )
    [rebuilt] = reconstruct_records(msg, "none", DIM)
    assert rebuilt.key == rec.key
    assert rebuilt.pos == rec.pos
    assert rebuilt.crop_bbox == rec.crop_bbox
    assert rebuilt.first_seen == 7
    assert rebuilt.merged_from_count == 1
    assert rebuilt.embedding.tobytes() == rec.embedding.tobytes()


# ------------------------------------------------------------------- channel


def test_channel_drop_stream_is_seeded_and_separate() -> None:
    a = Channel(_cfg(drop_p=0.5), seed=7)
    b = Channel(_cfg(drop_p=0.5), seed=7)
    draws_a = [a.drops_message() for _ in range(64)]
    draws_b = [b.drops_message() for _ in range(64)]
    assert draws_a == draws_b, "same seed must give the same drop sequence"
    assert any(draws_a) and not all(draws_a)

    different = Channel(_cfg(drop_p=0.5), seed=8)
    assert [different.drops_message() for _ in range(64)] != draws_a


def test_channel_draws_nothing_at_drop_p_zero() -> None:
    channel = Channel(_cfg(drop_p=0.0), seed=7)
    state = channel._rng.getstate()
    assert not any(channel.drops_message() for _ in range(16))
    assert channel._rng.getstate() == state
