"""One-shot generator for the encoder golden-test fixtures. Run manually.

Produces crop0..2.png and golden_embeddings.npy next to this file, from the
committed background src/assets/background-2500.png via extract_crops.
Regenerate ONLY when the encoder contract deliberately changes — the whole
point of the committed outputs is that they never drift silently.

    uv run --extra dev --extra perception python tests/data/crops/make_fixtures.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from swarm_perception.perception.crops import extract_crops
from swarm_perception.perception.encoder import CLIPEncoder
from swarm_perception.world.background import Background

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BACKGROUND = REPO / "src" / "assets" / "background-2500.png"

SIDE = 64.0
# (robot_id, center): two interior patches plus one clipped at the
# bottom-left corner so the mean-color padding is baked into a fixture.
CAPTURES = [
    (0, (150.0, 150.0)),
    (1, (1250.0, 600.0)),
    (2, (20.0, 2480.0)),
]


def main() -> None:
    crops_rgb, rects = extract_crops(Background(BACKGROUND), CAPTURES, SIDE)
    for (robot_id, center), crop, rect in zip(CAPTURES, crops_rgb, rects, strict=True):
        out = HERE / f"crop{robot_id}.png"
        Image.fromarray(crop).save(out)  # PIL: saved as RGB, no cv2/BGR involved
        print(f"{out.name}: center={center} rect={rect}")

    embeddings = CLIPEncoder(device="cpu").embed_images(crops_rgb)
    np.save(HERE / "golden_embeddings.npy", embeddings)
    print(f"golden_embeddings.npy: shape={embeddings.shape} dtype={embeddings.dtype}")


if __name__ == "__main__":
    main()
