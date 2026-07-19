# Environmental Perception in Robot Swarms

A deterministic simulator for studying how a swarm of simple mobile robots
collectively perceives a 2D environment. Robots capture local image crops at
synchronized epochs, keep a bounded memory of what they have seen, and exchange
memories when they come within communication range. Every world is procedurally
generated together with exact, pixel-level ground truth, and every run is fully
seeded: the same config and seed reproduce the event log byte for byte.

## How a run works

1. Robots move continuously through the world (bounce-off-edges motion).
2. Every capture epoch (`robot.capture_frequency` seconds of sim time), all
   robots simultaneously photograph the square patch under them
   (`robot.coverage_side` pixels). Captures are discrete and synchronized on
   purpose — each one becomes a record with the globally unique key
   `(epoch, robot, crop_index)`, which makes memories mergeable and runs
   replayable.
3. Each robot stores records in its memory, capped at `robot.memory_cap`.
4. When two robots are within `robot.neighbor_radius`, they exchange records
   through a bounded inbox with a per-epoch merge budget.
5. Everything observable is appended to `events.jsonl`: every capture with its
   exact crop rectangle, every memory state, every merge.

In the windowed mode you can watch this live: squares are camera footprints
(flashing yellow on a capture epoch), yellow circles are communication radii.

## Repository layout

- `src/swarm_perception/cli.py` — the `swarm-run` entrypoint (config path plus
  `--headless`, `--seed`, `--output-dir` overrides).
- `src/swarm_perception/sim/` — simulation engine, robot control loop, actuator.
- `src/swarm_perception/world/` — world generation (`swarm-gen`), background
  image access, layout ground truth with visibility queries.
- `src/swarm_perception/io/run_logger.py` — append-only `events.jsonl` plus
  per-run reproducibility artifacts.
- `src/swarm_perception/config/` — typed, strictly validated YAML configs.
- `experiments/` — batch runners for repeated seeded experiments.
- `legacy/` — quarantined artifacts from an earlier iteration of the project;
  not used by the current pipeline.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --extra dev

# Headless run: writes output/<name>-<timestamp>/ with the full event log
uv run swarm-run examples/example1.yaml

# Communication vs no-communication baselines
uv run swarm-run experiments/configs/bg2500-big_comm.yaml
uv run swarm-run experiments/configs/bg2500-big_noncomm.yaml
```

To watch a run in a window, set `simulation.headless: false` in your config
(and pick a `simulation.width`/`height` that fits your screen).

## Generating worlds

`swarm-gen` composes RGBA object sprites onto a base texture and writes the
world image together with its exact layout:

```bash
uv run swarm-gen --seed 7 --pngs pre_assets/pngs/20 --num-objects 12 \
  --base pre_assets/background/background.png --out output/worlds
```

This produces `background-<set>-<n>obj-seed<seed>.png` and a matching
`.layout.json` holding, for every placed object: its label, bounding box,
center, and a run-length-encoded alpha mask. The same seed reproduces both
files byte for byte. To simulate on a generated world, copy it into
`src/assets/` and point `simulation.background_image` at it.

## Run outputs

Each run directory (`output/<name>-<timestamp>/`) contains:

- `events.jsonl` — append-only log of `capture`, `memory`, and `comm` events;
  compact sorted-key JSON, LF line endings, no wall-clock timestamps.
- `config_resolved.yaml` — the exact configuration the run used.
- `run_metadata.json` — seed, package version, platform, git commit.

Determinism is enforced in CI: two runs with the same config and seed must
produce byte-identical `events.jsonl`.

## Perception encoder

Captured crops are embedded by a frozen encoder
(`src/swarm_perception/perception/`), installed via the `perception` extra
(`uv sync --extra perception`; the `eval` extra includes it). The contract:

- **Model**: OpenCLIP ViT-B/32 with the `laion2b_s34b_b79k` pretrained
  weights, pinned as constants in `perception/encoder.py`. Changing either
  string invalidates every stored embedding and the golden test.
- **Transforms**: taken from `open_clip.create_model_and_transforms`, never
  hand-rolled — preprocessing is exactly what the weights were trained with.
- **Color**: the codebase is BGR (OpenCV) everywhere upstream; the single
  BGR→RGB conversion lives in `perception/crops.py`. Edge-clipped crops are
  padded to the full coverage square with the background's mean color —
  never resized — anchored so the output stays geometrically aligned with
  the world; the logged rect is exactly what `Background.crop` reported.
- **Embeddings**: float32 `(N, 512)`, L2-normalized in fp32, computed with
  `model.eval()` under `torch.no_grad()` on CPU (the reference platform).
- **Batch order**: each capture epoch is embedded as one batch in sorted
  robot-id order; batch composition is part of the determinism contract.
- **Text tower**: `embed_text` is debug-only and never used in any reported
  metric.

A golden test (`tests/test_encoder_golden.py`) locks the whole pipeline to
committed fixture crops and embeddings; it is skipped when the `perception`
extra is not installed.

## Configuration

Every YAML key is documented inline in `examples/example1.yaml`. Notes:

- `robot.communication` toggles peer exchange entirely
  (`*_comm.yaml` / `*_noncomm.yaml` config pairs).
- `robot.memory_cap` bounds how many records a robot may hold.
- `robot.max_inbox_merges_per_epoch` and `robot.inbox_merge_after_budget`
  (`drop` or `deterministic`) control the merge budget.
- `simulation.fps` only derives the capture cadence; headless runs are never
  wall-clock paced.

## Batch experiments

```bash
python experiments/run_experiments.py \
  --variations bg2500-big bg2500-small \
  --repeats 10 \
  --seed-start 1 \
  --max-workers 1
```

Creates a batch directory under `experiments/runs/` with per-run logs and a
batch summary.

## Development

```bash
uv run pytest -q        # full suite
uv run ruff check .     # lint
uv run mypy src/        # types
```

All three run on every push and pull request.
