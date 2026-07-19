"""Frozen perception stage: crop extraction and the CLIP encoder (D11).

:mod:`swarm_perception.perception.crops` owns the codebase's single BGR->RGB
boundary and pads edge-clipped crops to the full coverage square.
:class:`~swarm_perception.perception.encoder.CLIPEncoder` wraps the pinned
OpenCLIP model; torch and open_clip are imported lazily inside the class, so
this package imports cleanly without the ``perception`` extra.
"""

from swarm_perception.perception.crops import extract_crops
from swarm_perception.perception.encoder import CLIPEncoder

__all__ = ["CLIPEncoder", "extract_crops"]
