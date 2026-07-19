"""Frozen CLIP image encoder: the perception model contract as code (D11).

The encoder is pinned to OpenCLIP ViT-B/32 with the ``laion2b_s34b_b79k``
pretrained weights (:data:`MODEL_NAME` / :data:`PRETRAINED_TAG`). Image
preprocessing comes from ``open_clip.create_model_and_transforms`` — it is
never hand-rolled, so the resize / center-crop / normalization pipeline is
exactly the one the weights were trained with. Embeddings are float32,
L2-normalized in fp32, matching what
:class:`~swarm_perception.fusion.memory.MemoryRecord` validates.

torch and open_clip are imported lazily inside :class:`CLIPEncoder`, so
``swarm_perception`` imports cleanly without the ``perception`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# The frozen model identity. Changing either string invalidates every stored
# embedding and the golden test; treat them as part of the benchmark spec.
MODEL_NAME = "ViT-B-32"
PRETRAINED_TAG = "laion2b_s34b_b79k"
EMBED_DIM = 512


class CLIPEncoder:
    """Pinned OpenCLIP image encoder producing L2-normalized fp32 embeddings.

    CPU fp32 is the reference platform; ``device`` exists for local speedups
    only and reported numbers must come from the reference configuration.
    """

    def __init__(self, device: str = "cpu") -> None:
        """Load the pinned model in eval mode.

        Args:
            device: torch device string; the default CPU is the reference.

        Raises:
            ImportError: If the ``perception`` extra is not installed.
        """
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - exercised without extra
            raise ImportError(
                "CLIPEncoder needs torch and open_clip; install the perception "
                "extra: uv sync --extra perception"
            ) from exc
        self._torch = torch
        self._open_clip = open_clip
        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED_TAG
        )
        model.eval()  # inference mode is part of the determinism contract
        self._model = model.to(device)
        self._preprocess = preprocess
        self._tokenizer: Any = None  # built lazily; debug-only text path

    def embed_images(self, crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Embed one epoch's crops as a single batch, in the given order.

        The caller passes crops sorted by robot id (:func:`extract_crops`
        enforces that order). Batch composition is part of the determinism
        contract (D11): all crops of an epoch are assembled into ONE batch in
        sorted robot-id order, so a given (config, seed) always presents the
        model with byte-identical batches.

        Args:
            crops_rgb: uint8 RGB arrays of shape ``(H, W, 3)`` — the output
                of :func:`~swarm_perception.perception.crops.extract_crops`.

        Returns:
            float32 array of shape ``(N, 512)``, L2-normalized in fp32.

        Raises:
            ValueError: If any crop is not a uint8 ``(H, W, 3)`` array.
        """
        torch = self._torch
        if len(crops_rgb) == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        from PIL import Image

        tensors = []
        for i, crop in enumerate(crops_rgb):
            if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3:
                raise ValueError(
                    f"crop {i} must be uint8 (H, W, 3) RGB, got "
                    f"dtype={crop.dtype}, shape={crop.shape}"
                )
            tensors.append(self._preprocess(Image.fromarray(crop)))
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch)
        features = features.float()  # normalize in fp32, never half
        features = features / features.norm(dim=-1, keepdim=True)
        result: np.ndarray = features.cpu().numpy()
        return result.astype(np.float32, copy=False)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """DEBUG-ONLY: embed text prompts with the matching text tower.

        Never used in any reported metric — the benchmark's ground truth is
        geometric and its similarity space is image-image. This exists solely
        for interactive sanity checks (e.g. "does this crop embed near the
        word 'tree'?") during development.

        Args:
            texts: Prompt strings.

        Returns:
            float32 array of shape ``(N, 512)``, L2-normalized in fp32.
        """
        torch = self._torch
        if len(texts) == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        if self._tokenizer is None:
            self._tokenizer = self._open_clip.get_tokenizer(MODEL_NAME)
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        features = features.float()
        features = features / features.norm(dim=-1, keepdim=True)
        result: np.ndarray = features.cpu().numpy()
        return result.astype(np.float32, copy=False)
