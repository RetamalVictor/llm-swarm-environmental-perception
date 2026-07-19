# Architecture

The runtime follows a decentralized swarm model. Each robot maintains its own language-based knowledge base and updates it from local sensing and nearby peer communication.

## Agent Loop

```mermaid
flowchart LR
  subgraph robotTick [Robot Tick]
    takePhoto[CameraSensor.take_photo]
    selfPrompt[API_MANAGER.call_photo_api]
    mergeSelf[merge_observations]
    broadcast[exchange_with_neighbors]
    inboxMerge[submit_inbox_request or deterministic merge]
    takePhoto --> selfPrompt --> mergeSelf --> broadcast --> inboxMerge
  end
```

## Main Components

- **Simulation runtime**
  - `src/main.py` initializes config, constants, the simulation object, and spawns robots.
  - `Robot.update()` executes sensing, communication, inbox budgeting, and async result polling.
- **Perception and movement**
  - `src/camera_sensor.py` crops a local view and renders sensing overlays.
  - `src/actuator.py` applies linear and angular commands to robot pose.
- **Knowledge and logging**
  - `src/llm/factory.py` selects a provider from `llm.provider` and returns either `API_MANAGER` (threaded) or `AsyncAPI_MANAGER` (vLLM).
  - `src/llm/providers/` implements `gemini`, `openai`, `ollama`, and `vllm` backends behind a shared `LLMProvider` protocol.
  - `src/observation_logger.py` stores `robots.json` snapshots and optional artifacts.
- **Experiment and evaluation**
  - `experiments/run_experiments.py` executes comm/noncomm config pairs across seeds.
  - `experiments/metrics/plot_cosine_experiment_averages.py` computes recall, precision, F1, then writes CSV/plot outputs.

## Shared Data Flow

```mermaid
flowchart TB
  configs[experiments/configs/*.yaml] --> runExperiments[run_experiments.py]
  runExperiments --> mainExec[src/main.py]
  mainExec --> runDir[run output directory]
  runDir --> robotsJson[robots.json]
  runDir --> mergeHistory[communication_merges.jsonl]
  robotsJson --> plotMetrics[plot_cosine_experiment_averages.py]
  groundTruth[ground_truth_*.json] --> plotMetrics
  plotMetrics --> metricsCsv[metrics CSV files]
  plotMetrics --> metricsPng[dashboard PNG files]
```

## LLM Layer

```mermaid
flowchart TB
  robot[Robot.submit_photo_request / submit_inbox_request]
  robot --> mgr{llm.provider}
  mgr -->|gemini openai ollama| threaded[API_MANAGER thread pool]
  mgr -->|vllm| asyncMgr[AsyncAPI_MANAGER asyncio]
  threaded --> prov[providers/gemini openai ollama]
  asyncMgr --> vllm[providers/vllm]
  prov --> ext[External API]
  vllm --> ext
  threaded --> poll[get_result per tick]
  asyncMgr --> poll
```

| Module | Responsibility |
|--------|----------------|
| `llm/factory.py` | `create_api_manager(n_threads, config)` — wires provider to manager |
| `llm/manager.py` | Threaded queue, stale-request dropping, prompt building |
| `llm/async_manager.py` | Non-blocking parallel HTTP for vLLM |
| `llm/providers/base.py` | `generate_text` / `generate_vision` protocol |
| `llm/providers/*.py` | Provider-specific API clients |

See [Configuration](configuration.md) for all YAML keys.
