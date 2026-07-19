"""Golden regression test for the frozen CLIP encoder (CPU fp32 reference).

Skipped entirely when open_clip is not installed, so the plain dev CI stays
green; a pinned-platform CI job that always runs this for real is a later
step (T1b).

Fixture provenance: the three PNGs under tests/data/crops/ were generated
ONCE by tests/data/crops/make_fixtures.py from the committed background
src/assets/background-2500.png via extract_crops (side 64) at centers
(150.0, 150.0), (1250.0, 600.0), and (20.0, 2480.0) — the last one clipped
at the bottom-left corner, baking the mean-color padding into the fixture.
The golden embeddings were produced by the same script on the reference
platform (CPU, fp32) and committed as golden_embeddings.npy.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("open_clip")

from PIL import Image  # noqa: E402

from swarm_perception.perception.encoder import EMBED_DIM, CLIPEncoder  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "crops"
FIXTURES = ["crop0.png", "crop1.png", "crop2.png"]


@pytest.fixture(scope="module")
def encoder() -> CLIPEncoder:
    return CLIPEncoder(device="cpu")


def load_fixture_crops() -> list[np.ndarray]:
    # PIL loads PNGs directly as RGB — fixtures never pass through cv2/BGR.
    return [np.asarray(Image.open(DATA_DIR / name).convert("RGB")) for name in FIXTURES]


def test_embeddings_match_golden(encoder: CLIPEncoder) -> None:
    """Embeddings of the committed crops must match the committed golden file.

    Mechanism: np.allclose against a committed .npy (chosen over a bytes
    checksum so a failure shows the numeric drift instead of a bare hash
    mismatch). Tolerance: atol=1e-4 with rtol=0. On one platform, repeated
    CPU fp32 runs reproduce components to ~1e-7; across BLAS builds drift
    stays below ~1e-5. Real regressions — wrong pretrained tag, altered
    preprocessing, missing normalization — move components of these unit
    vectors by >= 1e-2, several orders above the tolerance.
    """
    golden = np.load(DATA_DIR / "golden_embeddings.npy")
    assert golden.shape == (len(FIXTURES), EMBED_DIM)
    assert golden.dtype == np.float32

    embeddings = encoder.embed_images(load_fixture_crops())
    assert embeddings.shape == golden.shape
    assert embeddings.dtype == np.float32
    np.testing.assert_allclose(embeddings, golden, rtol=0, atol=1e-4)


def test_embeddings_are_unit_norm(encoder: CLIPEncoder) -> None:
    embeddings = encoder.embed_images(load_fixture_crops())
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_batch_matches_singletons(encoder: CLIPEncoder) -> None:
    """Batching must not change results beyond fp32 noise on CPU."""
    crops = load_fixture_crops()
    batched = encoder.embed_images(crops)
    singles = np.vstack([encoder.embed_images([c]) for c in crops])
    np.testing.assert_allclose(batched, singles, rtol=0, atol=1e-5)


def test_empty_batch(encoder: CLIPEncoder) -> None:
    out = encoder.embed_images([])
    assert out.shape == (0, EMBED_DIM)
    assert out.dtype == np.float32


def test_rejects_non_uint8(encoder: CLIPEncoder) -> None:
    bad = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        encoder.embed_images([bad])
