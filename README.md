# Environmental Perception in a Swarm of Conversational Agents

The main workflow is centered on three folders:
- `src/`: simulation runtime and agent logic
- `experiments/`: repeated runs + evaluation/plots
- `pre_assets/`: environment and ground-truth generation

## What Each Core Module Does

- `src/main.py`: entrypoint for one simulation run, robot control loop, LLM calls, and neighbor communication.
- `src/camera_sensor.py`: takes local image crops from the environment image.
- `src/observation_logger.py`: writes per-robot observation history and optional artifacts (frames, crops, merge logs).
- `experiments/run_experiments.py`: runs paired communication/non-communication experiments across seeds.
- `experiments/metrics/plot_cosine_experiment_averages.py`: computes recall/precision/F1 against ground truth and generates plots.
- `pre_assets/scripts/generate_background.py`: creates synthetic backgrounds by composing object PNGs.
- `pre_assets/ground_truth/build_ground_truth.py`: builds ground-truth fact JSON using the same LLM family.

## Quick Start

### 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env` in the repository root:

```bash
GOOGLE_API_KEY=your_key_here
```

### 2) Run a Single Simulation

From repo root:

```bash
python src/main.py experiments/configs/bg2500-big_comm.yaml
```

Try non-communication baseline:

```bash
python src/main.py experiments/configs/bg2500-big_noncomm.yaml
```

## Documentation

Generate browsable docs (narrative pages + auto API reference from docstrings):

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open `http://127.0.0.1:8000`.

Build static docs:

```bash
mkdocs build --strict
```

### Publish to GitHub Pages

Published site (after CI runs on `main`): **https://absera.github.io/swarm-robotics/**

One-time repository setup:

1. Open **Settings → Pages** on GitHub.
2. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).

CI workflow: [`.github/workflows/docs.yml`](.github/workflows/docs.yml)

- Runs on pushes to `main` that touch docs-related paths (`docs/`, `mkdocs.yml`, `src/`, `experiments/`, etc.).
- You can also trigger it manually: **Actions → Deploy documentation → Run workflow**.

Merge your docs branch into `main` and push; the workflow deploys the `site/` artifact automatically.

## Reproduce Core Experiment Results

### 1) Run repeated experiments

```bash
python experiments/run_experiments.py \
  --variations bg2500-big bg2500-small \
  --repeats 10 \
  --seed-start 1 \
  --max-workers 1
```

This creates a new batch directory under `experiments/runs/` with per-run logs and summaries.

### 2) Compute metrics and plots

Replace `<batch_id>` with the generated folder name (for example `batch_20260603_104500`):

```bash
python experiments/metrics/plot_cosine_experiment_averages.py \
  --batch-summary-csv experiments/runs/<batch_id>/batch_summary.csv \
  --ground-truth-json experiments/metrics/ground_truth_148.json \
  --threshold 0.60 \
  --model all-MiniLM-L6-v2 \
  --out-dir experiments/metrics/outputs
```

## Generate Assets and Ground Truth

### Background generation (`pre_assets`)

`pre_assets/scripts/generate_background.py` uses constants at the top of the file (`PNGS_DIR`, `PNG_SIZES`, `OUTPUT_WIDTH`, etc.).  
After adjusting those values:

```bash
python3 pre_assets/scripts/generate_background.py
```

If you use a newly generated background in simulation, copy it to `src/assets/` and update `background_image` in your YAML config.

### Ground-truth generation (`pre_assets`)

`pre_assets/ground_truth/build_ground_truth.py` is configured via top-level constants (`CONFIG_PATH`, `PNG_DIR`, `OUTPUT_PATH`, etc.).  
After setting those values:

```bash
python pre_assets/ground_truth/build_ground_truth.py
```

## Outputs You Should Expect

- Single run outputs:
  - `output/<config_name>_<timestamp>/robots.json`
  - Optional: `frames/`, `robot_crops/`, `communication_merges.jsonl`
- Batch experiment outputs:
  - `experiments/runs/<batch_id>/batch_summary.csv`
  - `experiments/runs/<batch_id>/batch_summary.json`
  - Per-seed folders with `run.log`, `run_metadata.json`, `config_resolved.yaml`
- Metrics outputs:
  - `experiments/metrics/outputs/per_robot_per_snapshot_cosine_metrics.csv`
  - `experiments/metrics/outputs/per_run_final_summary_cosine_metrics.csv`
  - `experiments/metrics/outputs/cosine_knowledge_progression_dashboard.png`
  - `experiments/metrics/outputs/cosine_metrics_progression_grid.png`

## Configuration Notes

- `*_comm.yaml` enables peer communication (`robot.communication: true`).
- `*_noncomm.yaml` disables peer communication (`robot.communication: false`).
- `bg2500-big_*` uses larger sensing/communication radius (`coverage_side` and `neighbor_radius` 200).
- `bg2500-small_*` uses smaller sensing/communication radius (150).

## Troubleshooting

- **Missing API key**: simulation and ground-truth generation require `GOOGLE_API_KEY`.
- **No config passed**: always pass a config path to `src/main.py` 
- **Background not found**: simulation reads from `src/assets/`; ensure the YAML `background_image` file exists there.
- **Headless flag caveat**: current runtime path uses the windowed simulation loop
