"""Stub epoch encoder: canonical-function-of-key embeddings, no torch."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from swarm_perception.perception import StubEncoder, build_epoch_encoder, stub_embedding
from swarm_perception.perception.encoder import EMBED_DIM


def _crop() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_stub_embedding_is_canonical_function_of_key() -> None:
    a = stub_embedding((3, 1, 0))
    b = stub_embedding((3, 1, 0))
    assert a.dtype == np.float32
    assert a.shape == (EMBED_DIM,)
    assert a.tobytes() == b.tobytes(), "same key must give bit-identical bytes"
    assert abs(float(np.linalg.norm(a.astype(np.float64))) - 1.0) < 1e-3


def test_stub_embedding_depends_on_every_key_component() -> None:
    base = stub_embedding((3, 1, 0)).tobytes()
    assert stub_embedding((4, 1, 0)).tobytes() != base
    assert stub_embedding((3, 2, 0)).tobytes() != base
    assert stub_embedding((3, 1, 1)).tobytes() != base


def test_stub_encoder_ignores_pixels_and_preserves_order() -> None:
    keys = [(1, 0, 0), (1, 2, 0), (1, 5, 0)]
    noise = np.full((8, 8, 3), 255, dtype=np.uint8)
    batch_a = StubEncoder().embed_epoch([_crop(), _crop(), _crop()], keys)
    batch_b = StubEncoder().embed_epoch([noise, noise, noise], keys)
    assert batch_a.shape == (3, EMBED_DIM)
    assert batch_a.tobytes() == batch_b.tobytes(), "stub must not read pixels"
    for row, key in zip(batch_a, keys, strict=True):
        assert row.tobytes() == stub_embedding(key).tobytes()


def test_stub_encoder_rejects_mismatched_lengths_and_handles_empty() -> None:
    enc = StubEncoder()
    with pytest.raises(ValueError, match="parallel"):
        enc.embed_epoch([_crop()], [(1, 0, 0), (1, 1, 0)])
    assert enc.embed_epoch([], []).shape == (0, EMBED_DIM)


def test_build_epoch_encoder_stub_never_imports_torch() -> None:
    before = set(sys.modules)
    encoder = build_epoch_encoder("stub", "cpu", 64)
    encoder.embed_epoch([_crop()], [(1, 0, 0)])
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] == "torch" for name in imported)
    assert isinstance(encoder, StubEncoder)


def test_build_epoch_encoder_clip_reports_missing_extra() -> None:
    torch_missing = "torch" not in sys.modules
    try:
        import torch  # noqa: F401

        torch_missing = False
    except ImportError:
        torch_missing = True
    if not torch_missing:
        pytest.skip("perception extra installed; missing-extra path not testable")
    with pytest.raises(ImportError, match="perception"):
        build_epoch_encoder("clip", "cpu", 64)
