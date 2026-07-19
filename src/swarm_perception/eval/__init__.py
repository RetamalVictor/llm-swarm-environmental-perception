"""Offline evaluation over finished run artifacts.

Every metric in this package is computed *after* a run, from the artifacts a
run directory already contains (``events.jsonl``, ``config_resolved.yaml``,
``run_metadata.json``) plus world ground truth (``.layout.json``). Nothing
here may import the simulation engine or agents: evaluation must stay
runnable on a bare run directory, on machines without a display or the sim
stack, and must never be able to perturb — or depend on — live sim state.
Pure-data modules (:mod:`swarm_perception.io`, :mod:`swarm_perception.world`)
are fine.

Modules:

- :mod:`swarm_perception.eval.geometry` — geometric object coverage, the
  benchmark's primary metric.
- :mod:`swarm_perception.eval.cli` — the ``swarm-eval`` console script.
"""
