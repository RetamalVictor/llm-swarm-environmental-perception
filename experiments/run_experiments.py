"""Batch runner for communication vs non-communication experiment pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RunJob:
    """Immutable run specification for one variation/mode/seed combination."""

    variation: str
    mode: str
    config_path: Path
    seed: int
    run_dir: Path
    run_id: str


@dataclass
class RunResult:
    """Execution result metadata for one completed run."""

    run_id: str
    variation: str
    mode: str
    seed: int
    status: str
    returncode: int | None
    duration_sec: float
    run_dir: str
    config_path: str
    command: str
    started_at: str
    finished_at: str
    error: str


def now_utc() -> datetime:
    """Return timezone-aware current UTC timestamp."""
    return datetime.now(timezone.utc)


def ts_for_name(ts: datetime) -> str:
    """Format timestamp to a file-system-friendly batch id fragment."""
    return ts.strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    """Parse CLI options for repeated experiment execution."""
    parser = argparse.ArgumentParser(
        description="Run repeated swarm simulations for *_comm.yaml and *_noncomm.yaml configs."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("experiments/configs"),
        help="Directory containing YAML config files.",
    )
    parser.add_argument(
        "--variations",
        nargs="*",
        default=None,
        help="Variation prefixes to run (e.g. baseline variation1). If omitted, all discovered pairs are run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Explicit seeds to use (e.g. --seeds 1 2 3 4 5). Overrides --repeats/--seed-start.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeated runs when --seeds is not provided.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Start seed when generating repeats.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel runs. Use >1 only if your machine/API quota can handle it.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory where run folders and summary files are stored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs only. Do not execute simulations.",
    )
    return parser.parse_args()


def discover_variation_pairs(config_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover matching ``*_comm.yaml`` and ``*_noncomm.yaml`` config pairs.

    Args:
        config_dir: Directory containing experiment config files.

    Returns:
        Mapping of variation name to a dict with optional ``comm`` and
        ``noncomm`` config paths.
    """
    pairs: dict[str, dict[str, Path]] = {}
    for cfg in sorted(config_dir.glob("*.yaml")):
        name = cfg.stem
        if name.endswith("_comm"):
            variation = name[: -len("_comm")]
            pairs.setdefault(variation, {})["comm"] = cfg
        elif name.endswith("_noncomm"):
            variation = name[: -len("_noncomm")]
            pairs.setdefault(variation, {})["noncomm"] = cfg
    return pairs


def load_yaml(path: Path) -> dict:
    """Load one YAML config and ensure it deserializes to a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data


def build_jobs(
    variation_pairs: dict[str, dict[str, Path]],
    selected_variations: list[str],
    seeds: list[int],
    batch_dir: Path,
) -> list[RunJob]:
    """Construct all run jobs for selected variations and seeds.

    Args:
        variation_pairs: Output from :func:`discover_variation_pairs`.
        selected_variations: Variations requested by the user.
        seeds: Seed values to run for each variation/mode.
        batch_dir: Batch root folder for run outputs.

    Returns:
        Ordered list of run specifications.

    Raises:
        ValueError: If any selected variation is missing comm/noncomm configs.
    """
    jobs: list[RunJob] = []
    for variation in selected_variations:
        entry = variation_pairs.get(variation, {})
        missing = [mode for mode in ("comm", "noncomm") if mode not in entry]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Variation '{variation}' missing config(s): {missing_str}")

        for seed in seeds:
            for mode in ("comm", "noncomm"):
                run_id = f"{variation}_{mode}_seed{seed:03d}"
                run_dir = batch_dir / variation / mode / f"seed_{seed:03d}"
                jobs.append(
                    RunJob(
                        variation=variation,
                        mode=mode,
                        config_path=entry[mode],
                        seed=seed,
                        run_dir=run_dir,
                        run_id=run_id,
                    )
                )
    return jobs


def prepare_job_config(job: RunJob) -> Path:
    """Materialize a run-specific config with seed and output directory.

    The file is named ``job_config.yaml``: the simulation itself owns the
    ``config_resolved.yaml`` and ``run_metadata.json`` names inside the run
    directory, so the runner's artifacts must not clobber them.

    Args:
        job: Run specification to resolve.

    Returns:
        Path to written ``job_config.yaml`` under this run directory.
    """
    cfg = load_yaml(job.config_path)
    cfg.setdefault("simulation", {})
    cfg["simulation"]["seed"] = job.seed
    cfg["simulation"]["output_dir"] = str(job.run_dir)

    cfg.setdefault("config", {})
    base_name = str(cfg["config"].get("name", f"{job.variation}_{job.mode}"))
    cfg["config"]["name"] = f"{base_name}_seed{job.seed:03d}"

    job.run_dir.mkdir(parents=True, exist_ok=True)
    out_path = job.run_dir / "job_config.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def run_one(job: RunJob, repo_root: Path) -> RunResult:
    """Execute one run and persist metadata/log files.

    Args:
        job: Run specification.
        repo_root: Repository root used as subprocess working directory.

    Returns:
        Structured run result including status, timing, and paths.
    """
    started = now_utc()
    started_iso = started.isoformat()
    log_path = job.run_dir / "run.log"
    job_config = prepare_job_config(job)
    command = [sys.executable, "-m", "swarm_perception", str(job_config)]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    returncode: int | None = None
    status = "failed"
    error = ""
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                command,
                cwd=repo_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        returncode = proc.returncode
        status = "success" if proc.returncode == 0 else "failed"
        if proc.returncode != 0:
            error = f"Process exited with return code {proc.returncode}."
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    # The simulation writes its artifacts directly into the run dir; the
    # event log is the ground truth that a run actually produced output.
    events_path = job.run_dir / "events.jsonl"
    if status == "success" and not events_path.exists():
        status = "failed"
        error = f"events.jsonl not found in run dir: {job.run_dir}"

    finished = now_utc()
    finished_iso = finished.isoformat()
    duration = (finished - started).total_seconds()

    metadata = {
        "run_id": job.run_id,
        "variation": job.variation,
        "mode": job.mode,
        "seed": job.seed,
        "status": status,
        "returncode": returncode,
        "duration_sec": duration,
        "config_path": str(job.config_path),
        "job_config_path": str(job_config),
        "command": command,
        "started_at_utc": started_iso,
        "finished_at_utc": finished_iso,
        "log_path": str(log_path),
        "events_path": str(events_path),
        "error": error,
    }
    # Runner-level metadata is batch_meta.json; run_metadata.json belongs to the sim.
    with (job.run_dir / "batch_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return RunResult(
        run_id=job.run_id,
        variation=job.variation,
        mode=job.mode,
        seed=job.seed,
        status=status,
        returncode=returncode,
        duration_sec=duration,
        run_dir=str(job.run_dir),
        config_path=str(job.config_path),
        command=" ".join(command),
        started_at=started_iso,
        finished_at=finished_iso,
        error=error,
    )


def write_batch_summary(batch_dir: Path, results: list[RunResult]) -> None:
    """Write JSON and CSV summaries for an experiment batch.

    Args:
        batch_dir: Batch output directory.
        results: All run results from this batch.
    """
    summary_json = batch_dir / "batch_summary.json"
    payload = {
        "batch_id": batch_dir.name,
        "total_runs": len(results),
        "success_runs": sum(r.status == "success" for r in results),
        "failed_runs": sum(r.status != "success" for r in results),
        "results": [r.__dict__ for r in results],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_csv = batch_dir / "batch_summary.csv"
    fieldnames = [
        "run_id",
        "variation",
        "mode",
        "seed",
        "status",
        "returncode",
        "duration_sec",
        "run_dir",
        "config_path",
        "command",
        "started_at",
        "finished_at",
        "error",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def main() -> int:
    """Run the full batch workflow from CLI args to summary files.

    Returns:
        Exit code ``0`` when all runs succeed, ``2`` if any run fails, and
        ``1`` for argument/discovery setup errors.
    """
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = (repo_root / args.config_dir).resolve()
    runs_root = (repo_root / args.runs_root).resolve()

    if not config_dir.exists():
        print(f"Config directory not found: {config_dir}")
        return 1

    variation_pairs = discover_variation_pairs(config_dir)
    if not variation_pairs:
        print(f"No *_comm.yaml / *_noncomm.yaml pairs found in: {config_dir}")
        return 1

    discovered = sorted(variation_pairs.keys())
    selected = sorted(args.variations) if args.variations else discovered
    unknown = [name for name in selected if name not in variation_pairs]
    if unknown:
        print(f"Unknown variation(s): {unknown}")
        print(f"Available variations: {discovered}")
        return 1

    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.repeats))
    if not seeds:
        print("No seeds provided/generated.")
        return 1

    batch_id = f"batch_{ts_for_name(now_utc())}"
    batch_dir = runs_root / batch_id
    jobs = build_jobs(variation_pairs, selected, seeds, batch_dir)

    print("Planned runs:")
    print(f"- batch_dir: {batch_dir}")
    print(f"- variations: {selected}")
    print(f"- seeds: {seeds}")
    print(f"- total jobs: {len(jobs)}")
    print(f"- max_workers: {args.max_workers}")
    if args.dry_run:
        for job in jobs:
            print(f"  DRY-RUN {job.run_id} -> {job.config_path}")
        return 0

    results: list[RunResult] = []
    if args.max_workers <= 1:
        for job in jobs:
            print(f"[RUN] {job.run_id}")
            result = run_one(job, repo_root)
            print(f"[{result.status.upper()}] {result.run_id} ({result.duration_sec:.1f}s)")
            results.append(result)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(run_one, job, repo_root): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                print(f"[{result.status.upper()}] {result.run_id} ({result.duration_sec:.1f}s)")
                results.append(result)

    results.sort(key=lambda r: (r.variation, r.mode, r.seed))
    write_batch_summary(batch_dir, results)
    success = sum(r.status == "success" for r in results)
    failed = len(results) - success
    print(f"\nBatch complete: {batch_dir}")
    print(f"Success: {success} | Failed: {failed} | Total: {len(results)}")
    print(f"Summary files: {batch_dir / 'batch_summary.json'} and {batch_dir / 'batch_summary.csv'}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
