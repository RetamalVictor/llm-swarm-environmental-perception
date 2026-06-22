# Methodology Mapping (Paper to Code)

This page maps Section 3 of the capstone paper to the implementation.

## Terminology Mapping

| Paper term | Code implementation |
|---|---|
| Knowledge Base (`T_s`) | `Robot.current_observation` in `src/main.py` |
| Interpretation (individual learning) | `API_MANAGER.submit_photo_request()` → `provider.generate_vision()` via `src/llm/manager.py` |
| Integration (social learning) | `submit_inbox_request()` → `provider.generate_text()` or deterministic fallback merge in `src/main.py` |
| Camera patch (`R x R`) | `CameraSensor.take_photo()` with `coverage_side` in `src/camera_sensor.py` |
| Communication range (`C`) | `neighbor_radius` and proximity lookup in `Robot.exchange_with_neighbors()` |
| Snapshot history | `ObservationLogger.log_progress_snapshot()` in `src/observation_logger.py` |
| Recall/Precision/F1 metrics | `score_observation_metrics()` in `experiments/metrics/plot_cosine_experiment_averages.py` |

## Runtime Loop

```mermaid
flowchart LR
  perceive[Perceive local image patch] --> processSelf[Interpret with LLM]
  processSelf --> updateTs[Update local T_s]
  updateTs --> broadcast[Broadcast to neighbors]
  broadcast --> synthesize[Integrate peer message]
  synthesize --> updateTs
```

## Important Implementation Notes

- The inbox is effectively single-slot: newer pending messages replace older pending ones.
- Robots attempt neighbor communication every tick when they are in range.
- Inbox merges are budgeted per capture epoch by `max_inbox_merges_per_epoch`.
- After budget is reached, behavior is controlled by `inbox_merge_after_budget` (`drop`, `deterministic`, or `llm`).
- A deterministic text merge helper (`merge_observations`) always runs before final storage to de-duplicate and cap facts.
- `simulation.headless: true` runs without a visible UI window while keeping robot crop capture and photo LLM submission active.
- `simulation.save_photo_frames` depends on an active pygame display surface; `simulation.save_robot_crops` works in both headless and non-headless runs.

## Experiment Configuration Knobs

Primary keys live in the YAML files under `experiments/configs/` and `examples/`. See the full [Configuration Reference](configuration.md) for defaults and behavior.

Key experiment levers:

- `simulation.run_length`, `simulation.fps`, `simulation.num_of_robots`
- `robot.coverage_side`, `robot.neighbor_radius`, `robot.capture_frequency`
- `robot.communication`, `robot.self_learning`, `robot.use_llm_inbox_synthesis`
- `robot.wait_for_llm`, `robot.photo_timeout_ticks`, `robot.inbox_timeout_ticks`
- `llm.provider`, `llm.model_name`, `llm.thread_workers`, `llm.temperature`, `llm.max_output_tokens`
