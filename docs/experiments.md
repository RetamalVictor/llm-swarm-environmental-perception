# Running Experiments

This project compares communication-enabled and non-communication swarms using repeated seeds and evaluates semantic quality against a ground-truth fact set.

## 1) Run Batched Simulations

Use `experiments/run_experiments.py`:

```bash
python experiments/run_experiments.py \
  --variations bg2500-big bg2500-small \
  --repeats 10 \
  --seed-start 1 \
  --max-workers 1
```

### What It Produces

- A batch directory: `experiments/runs/batch_<timestamp>/`
- Per-run folders with:
  - `config_resolved.yaml`
  - `run.log`
  - `run_metadata.json`
  - simulation artifacts (including `robots.json`)
- Batch summaries:
  - `batch_summary.json`
  - `batch_summary.csv`

## 2) Aggregate Metrics and Plots

Use `experiments/metrics/plot_cosine_experiment_averages.py`:

```bash
python experiments/metrics/plot_cosine_experiment_averages.py \
  --batch-summary-csv experiments/runs/<batch_id>/batch_summary.csv \
  --ground-truth-json experiments/metrics/ground_truth_148.json \
  --threshold 0.60 \
  --model all-MiniLM-L6-v2 \
  --out-dir experiments/metrics/outputs
```

### Evaluation Outputs

- `per_robot_per_snapshot_cosine_metrics.csv`
- `per_run_final_summary_cosine_metrics.csv`
- `cosine_knowledge_progression_dashboard.png`
- `cosine_metrics_progression_grid.png`

## Metric Definitions

- **Recall**: fraction of ground-truth facts matched by at least one observation sentence.
- **Precision**: fraction of observation sentences that match at least one ground-truth fact.
- **F1**: harmonic mean of recall and precision.

Each match uses cosine similarity over sentence embeddings with threshold `tau` (default `0.60`).
