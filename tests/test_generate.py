"""World generator: determinism, layout.json ground truth, and CLI contract.

Worlds here are tiny (400x400, 3 objects, 48 px sprites) with reduced layout
search effort so the suite stays fast; determinism only requires identical
arguments, not the production defaults.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from swarm_perception.world.generate import generate_world, main
from swarm_perception.world.rle import decode_mask

REPO = Path(__file__).resolve().parents[1]
PNG_SET = REPO / "pre_assets" / "pngs" / "5"

SIZE = (400, 400)
PADDING = 8
OBJECT_SIZE = 48
NUM_OBJECTS = 3
SEED = 42
SEARCH = {"candidates_per_object": 150, "layout_attempts": 4}


@pytest.fixture(scope="module")
def base_texture(tmp_path_factory) -> Path:
    """Small deterministic RGBA texture standing in for the base background."""
    path = tmp_path_factory.mktemp("base") / "base.png"
    yy, xx = np.mgrid[0:64, 0:64]
    arr = np.empty((64, 64, 4), dtype=np.uint8)
    arr[..., 0] = (xx * 3) % 256
    arr[..., 1] = (yy * 5) % 256
    arr[..., 2] = ((xx + yy) * 2) % 256
    arr[..., 3] = 255
    Image.fromarray(arr, "RGBA").save(path)
    return path


@pytest.fixture(scope="module")
def world(base_texture, tmp_path_factory) -> tuple[Path, dict]:
    """One generated world shared by the read-only assertions."""
    out_dir = tmp_path_factory.mktemp("world")
    png_path, layout_path = generate_world(
        seed=SEED,
        pngs_dir=PNG_SET,
        num_objects=NUM_OBJECTS,
        base=base_texture,
        out_dir=out_dir,
        size=SIZE,
        padding=PADDING,
        object_sizes=[OBJECT_SIZE],
        **SEARCH,
    )
    return png_path, json.loads(layout_path.read_text(encoding="utf-8"))


def _placed_thumbnail(png_name: str) -> Image.Image:
    """Recompute the placed thumbnail exactly as the generator does."""
    with Image.open(PNG_SET / png_name) as im:
        thumb = im.convert("RGBA")
    thumb.thumbnail((OBJECT_SIZE, OBJECT_SIZE), Image.Resampling.LANCZOS)
    return thumb


def test_output_naming_and_schema(world) -> None:
    png_path, layout = world
    assert png_path.name == f"background-5-{NUM_OBJECTS}obj-seed{SEED}.png"
    assert layout["background_image"] == png_path.name
    assert (layout["width"], layout["height"]) == SIZE
    assert layout["generator"] == {
        "seed": SEED,
        "padding": PADDING,
        "object_sizes": [OBJECT_SIZE] * NUM_OBJECTS,
        "png_set": "5",
        "num_objects": NUM_OBJECTS,
    }
    assert [obj["id"] for obj in layout["objects"]] == list(range(NUM_OBJECTS))
    # First N of the set in sorted filename order; labels fall back to stems.
    assert [obj["png"] for obj in layout["objects"]] == ["16.png", "17.png", "18.png"]
    assert [obj["label"] for obj in layout["objects"]] == ["16", "17", "18"]


def test_bbox_within_canvas_and_consistent_with_mask(world) -> None:
    _, layout = world
    width, height = SIZE
    for obj in layout["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        assert PADDING <= x1 < x2 <= width - PADDING
        assert PADDING <= y1 < y2 <= height - PADDING
        assert obj["center"] == [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        mask = decode_mask(obj["mask_rle"])
        assert mask.shape == (y2 - y1, x2 - x1)
        assert mask.any()
        # The mask's nonzero extent, shifted by the paste offset, stays inside
        # both the bbox and the canvas.
        ys, xs = np.nonzero(mask)
        assert x1 + int(xs.max()) < x2 <= width
        assert y1 + int(ys.max()) < y2 <= height


def test_objects_respect_padding_separation(world) -> None:
    _, layout = world
    rects = [obj["bbox"] for obj in layout["objects"]]
    for i, (ax1, ay1, ax2, ay2) in enumerate(rects):
        for bx1, by1, bx2, by2 in rects[i + 1 :]:
            separated = (
                ax2 + PADDING <= bx1
                or bx2 + PADDING <= ax1
                or ay2 + PADDING <= by1
                or by2 + PADDING <= ay1
            )
            assert separated


def test_mask_rle_equals_placed_thumbnail_alpha(world) -> None:
    _, layout = world
    for obj in layout["objects"]:
        thumb = _placed_thumbnail(obj["png"])
        expected = np.asarray(thumb.getchannel("A")) > 0
        np.testing.assert_array_equal(decode_mask(obj["mask_rle"]), expected)


def test_composited_pixels_match_thumbnail_at_opaque_pixels(world) -> None:
    png_path, layout = world
    composite = np.asarray(Image.open(png_path).convert("RGBA"))
    for obj in layout["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        thumb = _placed_thumbnail(obj["png"])
        thumb_arr = np.asarray(thumb)
        solid = thumb_arr[..., 3] == 255
        assert solid.any()
        region = composite[y1:y2, x1:x2]
        np.testing.assert_array_equal(
            region[solid][:, :3], thumb_arr[solid][:, :3]
        )


def test_same_seed_byte_identical_outputs(base_texture, tmp_path) -> None:
    outputs = []
    for run in ("a", "b"):
        png_path, layout_path = generate_world(
            seed=SEED,
            pngs_dir=PNG_SET,
            num_objects=NUM_OBJECTS,
            base=base_texture,
            out_dir=tmp_path / run,
            size=SIZE,
            padding=PADDING,
            object_sizes=[OBJECT_SIZE],
            **SEARCH,
        )
        outputs.append((png_path.read_bytes(), layout_path.read_bytes()))
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]


def test_different_seed_changes_layout(base_texture, tmp_path) -> None:
    layouts = []
    for seed in (1, 2):
        _, layout_path = generate_world(
            seed=seed,
            pngs_dir=PNG_SET,
            num_objects=NUM_OBJECTS,
            base=base_texture,
            out_dir=tmp_path,
            size=SIZE,
            padding=PADDING,
            object_sizes=[OBJECT_SIZE],
            **SEARCH,
        )
        layouts.append(json.loads(layout_path.read_text(encoding="utf-8")))
    assert [o["bbox"] for o in layouts[0]["objects"]] != [
        o["bbox"] for o in layouts[1]["objects"]
    ]


def test_layout_json_is_utf8_lf_sorted(base_texture, tmp_path) -> None:
    _, layout_path = generate_world(
        seed=SEED,
        pngs_dir=PNG_SET,
        num_objects=1,
        base=base_texture,
        out_dir=tmp_path,
        size=SIZE,
        padding=PADDING,
        object_sizes=[OBJECT_SIZE],
        **SEARCH,
    )
    raw = layout_path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert list(data) == sorted(data)


def test_seed_none_rejected(base_texture, tmp_path) -> None:
    with pytest.raises(ValueError, match="seed is required"):
        generate_world(
            seed=None,  # type: ignore[arg-type]
            pngs_dir=PNG_SET,
            num_objects=1,
            base=base_texture,
            out_dir=tmp_path,
            size=SIZE,
            padding=PADDING,
        )


def test_labels_json_used_when_present(base_texture, tmp_path) -> None:
    set_dir = tmp_path / "tinyset"
    set_dir.mkdir()
    shutil.copy(PNG_SET / "16.png", set_dir / "16.png")
    shutil.copy(PNG_SET / "17.png", set_dir / "17.png")
    (set_dir / "labels.json").write_text(
        json.dumps({"16.png": "skyscraper", "17": "beach"}), encoding="utf-8"
    )
    _, layout_path = generate_world(
        seed=SEED,
        pngs_dir=set_dir,
        num_objects=2,
        base=base_texture,
        out_dir=tmp_path / "out",
        size=SIZE,
        padding=PADDING,
        object_sizes=[OBJECT_SIZE],
        **SEARCH,
    )
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    # Filename key and stem key both resolve.
    assert [o["label"] for o in layout["objects"]] == ["skyscraper", "beach"]
    assert layout["generator"]["png_set"] == "tinyset"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_objects": 0}, "at least 1"),
        ({"num_objects": 99}, "only"),
        ({"num_objects": 3, "object_sizes": [10, 20]}, "object sizes"),
        ({"num_objects": 1, "object_sizes": [0]}, "positive"),
        ({"num_objects": 1, "padding": -1}, "non-negative"),
        ({"num_objects": 1, "size": (0, 400)}, "positive"),
    ],
)
def test_invalid_arguments_rejected(base_texture, tmp_path, kwargs, match) -> None:
    args = dict(
        seed=SEED,
        pngs_dir=PNG_SET,
        base=base_texture,
        out_dir=tmp_path,
        size=SIZE,
        padding=PADDING,
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        generate_world(**args)


def test_object_too_big_for_canvas_rejected(base_texture, tmp_path) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        generate_world(
            seed=SEED,
            pngs_dir=PNG_SET,
            num_objects=1,
            base=base_texture,
            out_dir=tmp_path,
            size=(40, 40),
            padding=PADDING,
            object_sizes=[OBJECT_SIZE],
            **SEARCH,
        )


def test_cli_requires_seed(base_texture, tmp_path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--pngs", str(PNG_SET),
                "--num-objects", "1",
                "--base", str(base_texture),
                "--out", str(tmp_path),
            ]
        )
    assert excinfo.value.code == 2


def test_cli_generates_world(base_texture, tmp_path, capsys) -> None:
    main(
        [
            "--seed", "7",
            "--pngs", str(PNG_SET),
            "--num-objects", "2",
            "--size", "400x400",
            "--padding", str(PADDING),
            "--object-size", "32",
            "--object-size", "48",
            "--base", str(base_texture),
            "--out", str(tmp_path),
        ]
    )
    png_path = tmp_path / "background-5-2obj-seed7.png"
    layout_path = tmp_path / "background-5-2obj-seed7.layout.json"
    assert png_path.exists() and layout_path.exists()
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    assert layout["generator"]["object_sizes"] == [32, 48]
    out = capsys.readouterr().out
    assert png_path.name in out and layout_path.name in out
