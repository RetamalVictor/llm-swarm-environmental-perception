# Environmental Perception Docs

This documentation covers the implementation in `src/` and `experiments/` for the capstone project, *Environmental Perception in a Swarm of Conversational Agents*.

## What This Site Contains

- Architecture and data flow from simulation to evaluation metrics
- **[Configuration reference](configuration.md)** — every YAML key explained
- Methodology mapping between paper concepts and concrete code modules
- Experiment orchestration and metrics pipeline usage
- Auto-generated API reference from in-code docstrings

## Quick Links

- Project README: `README.md`
- Main runtime entrypoint: `src/main.py`
- LLM factory: `src/llm/factory.py`
- Example configs: `examples/example1_*.yaml`
- Batch runner: `experiments/run_experiments.py`
- Metrics script: `experiments/metrics/plot_cosine_experiment_averages.py`
- Paper PDF: [ABSERA-YIHUNIE-CAPSTONE-SPRING-2026.pdf](file:///Users/absera/Downloads/ABSERA-YIHUNIE-CAPSTONE-SPRING-2026.pdf)

## High-Level Workflow

```mermaid
flowchart LR
  configYaml[Config YAML] --> mainPy[src/main.py]
  mainPy --> runOutput[output or experiments/runs]
  runOutput --> robotsJson[robots.json snapshots]
  robotsJson --> metricsPy[plot_cosine_experiment_averages.py]
  gtJson[Ground truth JSON] --> metricsPy
  metricsPy --> plotsAndCsv[CSV summaries and PNG dashboards]
```
