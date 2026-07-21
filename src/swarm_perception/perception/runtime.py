"""Epoch encoders: the deterministic stub and the CLIP adapter (D11).

The engine embeds every capture epoch through exactly one
:class:`EpochEncoder` call: all robots' crops, assembled in sorted robot-id
order, become one batch of L2-normalized float32 embeddings. Two
implementations exist:

- :class:`StubEncoder` (``perception.model: stub``, the default and the CI
  path) — pure numpy, no torch. Each embedding is a CANONICAL FUNCTION OF
  THE RECORD KEY ALONE: a unit vector drawn from a Generator seeded by a
  stable hash of ``(epoch, robot, crop_idx)``. Because the vector depends on
  nothing but the key, every robot that ever materializes a record produces
  bit-identical embedding bytes, so quantize-once wire payloads and T1a
  byte-identity hold with no model in the loop.
- :class:`ClipEpochEncoder` (``perception.model: clip``) — wraps the pinned
  :class:`~swarm_perception.perception.encoder.CLIPEncoder`; import stays
  lazy so the ``perception`` extra is only needed when selected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from swarm_perception.perception.encoder import EMBED_DIM

Key = tuple[int, int, int]


class EpochEncoder(Protocol):
    """One capture epoch in, one embedding batch out (order-preserving)."""

    def embed_epoch(
        self, crops_rgb: Sequence[np.ndarray], keys: Sequence[Key]
    ) -> np.ndarray:
        """Embed one epoch's crops; row i corresponds to input i."""
        ...


def stub_embedding(key: Key) -> np.ndarray:
    """The stub's embedding for one record key.

    A unit-normalized float32 vector of dim :data:`EMBED_DIM`, drawn from
    ``np.random.default_rng`` seeded with a SHA-256 hash of the key text.
    Deterministic and a canonical function of the key: independent of crop
    pixels, call order, platform process state, and everything else.
    """
    epoch, robot, crop_idx = key
    digest = hashlib.sha256(f"{epoch}:{robot}:{crop_idx}".encode("ascii")).digest()
    seed = int.from_bytes(digest[:8], "big")
    vec = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
    return vec / np.float32(np.linalg.norm(vec))


class StubEncoder:
    """Deterministic pure-numpy encoder: embeddings from keys, never pixels."""

    def embed_epoch(
        self, crops_rgb: Sequence[np.ndarray], keys: Sequence[Key]
    ) -> np.ndarray:
        """Embed one epoch from its record keys; ``crops_rgb`` is ignored."""
        if len(crops_rgb) != len(keys):
            raise ValueError(
                f"crops and keys must be parallel, got {len(crops_rgb)} crops "
                f"for {len(keys)} keys"
            )
        if not keys:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.stack([stub_embedding(key) for key in keys])


class ClipEpochEncoder:
    """CLIP adapter: sorted-order crops through the pinned encoder.

    ``batch_size`` chunks one epoch when the swarm outgrows encoder memory;
    chunk boundaries are a pure function of the config, so runs stay
    reproducible. The reference configuration keeps ``batch_size >=
    simulation.num_of_robots`` so each epoch is exactly one batch (D11).
    """

    def __init__(self, device: str, batch_size: int) -> None:
        """Load the pinned CLIP model (needs the ``perception`` extra)."""
        from swarm_perception.perception.encoder import CLIPEncoder

        self._encoder = CLIPEncoder(device=device)
        self._batch_size = int(batch_size)

    def embed_epoch(
        self, crops_rgb: Sequence[np.ndarray], keys: Sequence[Key]
    ) -> np.ndarray:
        """Embed one epoch's crops in order; ``keys`` only checks parallelism."""
        if len(crops_rgb) != len(keys):
            raise ValueError(
                f"crops and keys must be parallel, got {len(crops_rgb)} crops "
                f"for {len(keys)} keys"
            )
        if not crops_rgb:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        chunks = [
            self._encoder.embed_images(crops_rgb[start : start + self._batch_size])
            for start in range(0, len(crops_rgb), self._batch_size)
        ]
        return np.concatenate(chunks, axis=0)


def build_epoch_encoder(model: str, device: str, batch_size: int) -> EpochEncoder:
    """Build the epoch encoder selected by the ``perception`` config section.

    Args:
        model: ``"stub"`` or ``"clip"`` (validated by the config schema).
        device: torch device string; only meaningful for ``"clip"``.
        batch_size: Max crops per encoder call; only meaningful for ``"clip"``.

    Raises:
        ImportError: For ``"clip"`` without the ``perception`` extra; the
            message names the missing extra.
        ValueError: For an unknown model name (unreachable behind the schema).
    """
    if model == "stub":
        return StubEncoder()
    if model == "clip":
        try:
            return ClipEpochEncoder(device=device, batch_size=batch_size)
        except ImportError as exc:
            raise ImportError(
                "perception.model 'clip' needs torch and open_clip; install "
                "the perception extra: uv sync --extra perception"
            ) from exc
    raise ValueError(f"unknown perception model {model!r}")
