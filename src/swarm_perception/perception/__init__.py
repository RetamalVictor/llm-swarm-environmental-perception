"""Frozen perception stage: crop extraction and the epoch encoders (D11).

:mod:`swarm_perception.perception.crops` owns the codebase's single BGR->RGB
boundary and pads edge-clipped crops to the full coverage square.
:mod:`swarm_perception.perception.runtime` selects the epoch encoder: the
deterministic pure-numpy :class:`StubEncoder` (CI path) or the pinned
:class:`~swarm_perception.perception.encoder.CLIPEncoder`; torch and
open_clip are imported lazily, so this package imports cleanly without the
``perception`` extra.
"""

from swarm_perception.perception.crops import extract_crops
from swarm_perception.perception.encoder import CLIPEncoder
from swarm_perception.perception.runtime import (
    EpochEncoder,
    StubEncoder,
    build_epoch_encoder,
    stub_embedding,
)

__all__ = [
    "CLIPEncoder",
    "EpochEncoder",
    "StubEncoder",
    "build_epoch_encoder",
    "extract_crops",
    "stub_embedding",
]
