# Configuration Reference

Simulation behavior is controlled by a single YAML file passed to `swarm-run` (or `python -m swarm_perception`). The loader (`swarm_perception.utils.config.SwarmConfig`) converts nested keys into dot-access namespaces (for example, `config.robot.coverage_side`).

Start from the ready-made samples in `examples/`.

## Top-Level Structure

```yaml
config:
  name: "my_run"

simulation:
  # world size, timing, assets, output

robot:
  # motion, sensing, communication, memory
```

---

## `config`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | *(required)* | Short label used in output directory names and experiment logs. |

---

## `simulation`

Controls the pygame world, run duration, and artifact saving.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run_length` | int | — | Number of **capture epochs** per run. Each epoch is one photo round for all robots; this value drives plot progression length in metrics. |
| `headless` | bool | `false` | When `true`, runs without a visible UI window. Robot crops are still captured. |
| `width` | int | — | Simulation world width in pixels. |
| `height` | int | — | Simulation world height in pixels. |
| `fps` | int | — | Simulation ticks per real second. Set `0` for uncapped headless speed. |
| `num_of_robots` | int | — | Number of robots spawned at startup. |
| `background_image` | string | — | Filename under `src/assets/` for the environment image. |
| `robot_image` | string | — | Filename under `src/assets/` for the robot sprite. |
| `seed` | int | — | Random seed for reproducible placement and motion. |
| `save_photo_frames` | bool | `false` | Save one full-scene PNG per capture epoch to `run_output/frames/`. Requires an active pygame display surface (typically **not** headless). |
| `save_robot_crops` | bool | `false` | Save each robot's cropped camera image per epoch to `run_output/robot_crops/`. Works in headless and windowed modes. |
| `output_dir` | string | `output/` | Override the run output root. `experiments/run_experiments.py` sets this per job automatically. |

### Derived timing

The runtime computes:

- `PHOTO_TICKS = capture_frequency * fps` — ticks between capture rounds
- `SIM_DURATION = run_length * PHOTO_TICKS` — total simulation length in ticks

---

## `robot`

Controls movement, sensing geometry, peer communication, and record memory.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `linear_speed` | float | — | Forward movement speed per simulation tick. |
| `angular_velocity` | float | — | Turn rate used for edge-avoidance corrections. |
| `coverage_side` | int | — | Side length (pixels) of the square camera crop. Maps to paper sensing radius **R**. |
| `neighbor_radius` | int | — | Distance threshold for detecting neighboring robots. Maps to paper communication range **C**. |
| `capture_frequency` | float | — | Simulated **seconds** of movement between capture rounds (when `fps > 0`). |
| `communication` | bool | — | Enable (`true`) or disable (`false`) peer-to-peer record exchange. |
| `self_learning` | bool | — | Reserved for memory-aware capture; unused by the interim record path. |
| `memory_cap` | int | `40` | Hard cap on stored capture records. On overflow, the records with the smallest keys (tuple order) are kept. |
| `max_inbox_merges_per_epoch` | int | `1` | Per-robot budget of inbox merges per capture epoch. |
| `inbox_merge_after_budget` | string | `"drop"` | Behavior when the merge budget is exhausted: `"drop"` (discard queued broadcasts) or `"deterministic"` (merge anyway). |

---

## Related Files

| Path | Purpose |
|------|---------|
| `examples/example1.yaml` | Annotated sample config |
| `experiments/configs/` | Experiment-scale configs (comm/noncomm pairs) |
