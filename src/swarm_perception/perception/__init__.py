"""Frozen perception stage: crop extraction and the CLIP encoder (D11).

:mod:`swarm_perception.perception.crops` owns the codebase's single BGR->RGB
boundary and pads edge-clipped crops to the full coverage square.
"""

from swarm_perception.perception.crops import extract_crops

__all__ = ["extract_crops"]
