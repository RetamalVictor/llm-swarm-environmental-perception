from __future__ import annotations
"""Evaluate swarm observation quality against ground truth using cosine metrics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer


def split_sentences(text: str) -> list[str]:
    """Split observation text into period-delimited sentences for scoring.

    Note:
        This intentionally uses simple period splitting for metric stability
        across generated logs, rather than the punctuation regex used in the
        runtime merge helper.
    """
    if not text:
        return []
    parts = text.replace("\n", " ").split(".")
    return [p.strip() + "." for p in parts if p.strip()]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between two embedding matrices."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a / a_norm) @ (b / b_norm).T


def cosine_gt_coverage(
    observation: str,
    gt_embeddings: np.ndarray,
    st_model: SentenceTransformer,
    threshold: float,
) -> float:
    """Legacy recall-only helper kept for compatibility scripts.

    Args:
        observation: Observation paragraph to evaluate.
        gt_embeddings: Ground-truth embedding matrix.
        st_model: SentenceTransformer model used for encoding.
        threshold: Cosine similarity threshold for a GT fact to count as covered.

    Returns:
        Fraction of ground-truth facts covered by at least one sentence.
    """
    sentences = split_sentences(observation)
    if not sentences:
        return 0.0
    obs_embeddings = st_model.encode(
        sentences, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    sims = cosine_matrix(obs_embeddings, gt_embeddings)
    best_fact_scores = (
        sims.max(axis=0)
        if sims.size
        else np.zeros((gt_embeddings.shape[0],), dtype=np.float32)
    )
    if len(best_fact_scores) == 0:
        return 0.0
    return float((best_fact_scores > threshold).sum() / len(best_fact_scores))


def score_observation_metrics(
    observation: str,
    gt_embeddings: np.ndarray,
    st_model: SentenceTransformer,
    threshold: float,
) -> dict[str, float]:
    """Compute recall, precision, and F1 for one observation paragraph.

    Args:
        observation: Observation text to score.
        gt_embeddings: Ground-truth embedding matrix.
        st_model: SentenceTransformer model used for encoding.
        threshold: Cosine threshold defining matched facts.

    Returns:
        Dictionary with ``recall``, ``precision``, and ``f1`` values in [0, 1].
    """
    sentences = split_sentences(observation)
    if not sentences:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    obs_embeddings = st_model.encode(
        sentences, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    sims = cosine_matrix(obs_embeddings, gt_embeddings)
    if sims.size == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    best_per_gt = sims.max(axis=0)
    best_per_obs = sims.max(axis=1)

    recall = (
        float((best_per_gt >= threshold).sum() / len(best_per_gt))
        if len(best_per_gt)
        else 0.0
    )
    precision = (
        float((best_per_obs >= threshold).sum() / len(best_per_obs))
        if len(best_per_obs)
        else 0.0
    )
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = float(2 * recall * precision / (recall + precision))
    return {"recall": recall, "precision": precision, "f1": f1}


def load_ground_truth(path: Path) -> list[str]:
    """Load ground-truth facts from list format or metadata object format."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Legacy format: top-level list of facts.
        facts = data
    elif isinstance(data, dict):
        # New format: metadata object with a `facts` field.
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            raise ValueError(f"'facts' must be a JSON list in: {path}")
    else:
        raise ValueError(
            f"Ground truth must be either a JSON list or object with 'facts': {path}"
        )
    return [str(x).strip() for x in facts if str(x).strip()]


def find_robots_json(run_dir: Path) -> Path | None:
    """Find the most recently modified ``robots.json`` under one run directory."""
    candidates = sorted(run_dir.rglob("robots.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_batch_rows(batch_csv: Path) -> list[dict[str, str]]:
    """Load rows from ``batch_summary.csv`` produced by ``run_experiments.py``."""
    with batch_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def infer_batch_id(batch_csv: Path) -> str:
    """Resolve experiment batch id from sibling JSON or parent directory name."""
    batch_dir = batch_csv.resolve().parent
    summary_json = batch_dir / "batch_summary.json"
    if summary_json.exists():
        with summary_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        batch_id = payload.get("batch_id")
        if batch_id:
            return str(batch_id)
    if batch_dir.name.startswith("batch_"):
        return batch_dir.name
    return f"{batch_dir.name}_{batch_csv.stem}"


def resolve_metrics_output_dir(
    batch_csv: Path,
    out_dir: Path,
    rows: list[dict[str, str]],
) -> Path:
    """Return ``out_dir / <batch_id>__<variation(s)>`` for metrics artifacts."""
    batch_id = infer_batch_id(batch_csv)
    variations = sorted({row["variation"] for row in rows if row.get("variation")})
    variation_slug = "_".join(variations) if variations else "unknown_variation"

    if len(variations) == 1:
        subdir = f"{batch_id}__{variations[0]}"
    else:
        subdir = f"{batch_id}__{variation_slug}"

    return out_dir.resolve() / subdir


def aggregate_progression(per_snapshot_rows: list[dict]) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Aggregate cosine-recall snapshots into mean/std curves per condition."""
    grouped: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_snapshot_rows:
        key = (row["variation"], row["mode"])
        grouped[key][int(row["snapshot_index"])].append(float(row["cosine_recall"]))

    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key, step_map in grouped.items():
        steps = sorted(step_map.keys())
        means = np.array([np.mean(step_map[s]) for s in steps], dtype=np.float32)
        stds = np.array([np.std(step_map[s]) for s in steps], dtype=np.float32)
        out[key] = {
            "steps": np.array(steps, dtype=np.int32),
            "means": means,
            "stds": stds,
        }
    return out


def make_progression_plot(
    per_snapshot_rows: list[dict],
    out_path: Path,
    title: str,
) -> None:
    """Render recall progression dashboard for comm vs noncomm runs."""
    curves = aggregate_progression(per_snapshot_rows)
    if not curves:
        raise ValueError("No snapshot curves available for plotting.")

    variations = sorted({k[0] for k in curves.keys()})
    n = len(variations)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False, sharey=True)
    color_map = {"comm": "#1f77b4", "noncomm": "#ff7f0e"}
    label_map = {"comm": "Communication", "noncomm": "Non-Communication"}

    for i, variation in enumerate(variations):
        ax = axes[0, i]
        for mode in ("comm", "noncomm"):
            key = (variation, mode)
            if key not in curves:
                continue
            c = curves[key]
            steps = c["steps"]
            means = c["means"]
            stds = c["stds"]
            color = color_map.get(mode, "#888888")
            label = label_map.get(mode, mode)

            ax.plot(steps, means, linewidth=2.4, color=color, label=label)
            ax.fill_between(
                steps,
                np.clip(means - stds, 0, 1),
                np.clip(means + stds, 0, 1),
                color=color,
                alpha=0.18,
            )

        ax.set_title(f"{variation}: Knowledge Progression")
        ax.set_xlabel("Snapshot Index")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend()

    axes[0, 0].set_ylabel("Cosine Recall vs Ground Truth")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def aggregate_metric_curves(
    per_snapshot_rows: list[dict],
) -> dict[tuple[str, str], dict[str, dict[str, np.ndarray]]]:
    """Aggregate recall/precision/F1 curves into mean/std arrays per condition."""
    metric_names = ("recall", "precision", "f1")
    grouped: dict[tuple[str, str], dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: {m: defaultdict(list) for m in metric_names}
    )

    for row in per_snapshot_rows:
        key = (row["variation"], row["mode"])
        snapshot_idx = int(row["snapshot_index"])
        for metric in metric_names:
            grouped[key][metric][snapshot_idx].append(float(row[metric]))

    out: dict[tuple[str, str], dict[str, dict[str, np.ndarray]]] = {}
    for key, metric_map in grouped.items():
        out[key] = {}
        for metric, snapshot_map in metric_map.items():
            snapshots = sorted(snapshot_map.keys())
            means = np.array([np.mean(snapshot_map[s]) for s in snapshots], dtype=np.float32)
            stds = np.array([np.std(snapshot_map[s]) for s in snapshots], dtype=np.float32)
            out[key][metric] = {
                "snapshots": np.array(snapshots, dtype=np.int32),
                "means": means,
                "stds": stds,
            }
    return out


def make_metrics_grid_plot(
    per_snapshot_rows: list[dict],
    out_path: Path,
    title: str,
) -> None:
    """Render a grid dashboard of recall, precision, and F1 trajectories."""
    curves = aggregate_metric_curves(per_snapshot_rows)
    if not curves:
        raise ValueError("No metric curves available for plotting.")

    variations = sorted({k[0] for k in curves.keys()})
    metrics = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1 Score"),
    ]
    n_rows = len(metrics)
    n_cols = len(variations)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 4.2 * n_rows),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    color_map = {"comm": "#1f77b4", "noncomm": "#ff7f0e"}
    label_map = {"comm": "Communication", "noncomm": "Non-Communication"}

    for row_idx, (metric_key, metric_label) in enumerate(metrics):
        for col_idx, variation in enumerate(variations):
            ax = axes[row_idx, col_idx]
            for mode in ("comm", "noncomm"):
                key = (variation, mode)
                if key not in curves:
                    continue
                curve = curves[key][metric_key]
                snapshots = curve["snapshots"]
                means = curve["means"]
                stds = curve["stds"]
                color = color_map.get(mode, "#888888")
                label = label_map.get(mode, mode)

                ax.plot(snapshots, means, linewidth=2.2, color=color, label=label)
                ax.fill_between(
                    snapshots,
                    np.clip(means - stds, 0, 1),
                    np.clip(means + stds, 0, 1),
                    color=color,
                    alpha=0.16,
                )

            if row_idx == 0:
                ax.set_title(f"Variation: {variation}")
            if col_idx == 0:
                ax.set_ylabel(f"{metric_label} (0-1)")
            ax.set_xlabel("Snapshot Index")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for metrics aggregation and plotting."""
    parser = argparse.ArgumentParser(
        description="Aggregate repeated experiment runs and plot cosine coverage averages."
    )
    parser.add_argument(
        "--batch-summary-csv",
        type=Path,
        required=True,
        help="Path to experiments batch_summary.csv generated by run_experiments.py",
    )
    parser.add_argument(
        "--ground-truth-json",
        type=Path,
        default=Path("experiments/metrics/ground_truth_148.json"),
        help="Path to ground truth facts JSON list.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Cosine threshold used to mark GT claim as covered.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for cosine embeddings.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/metrics/outputs"),
        help="Root output directory. Artifacts are saved in a batch-scoped subfolder.",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation pipeline from batch CSV to metrics CSVs and plots."""
    args = parse_args()
    gt_facts = load_ground_truth(args.ground_truth_json.resolve())
    rows = load_batch_rows(args.batch_summary_csv.resolve())
    ok_rows = [r for r in rows if r.get("status") == "success"]
    if not ok_rows:
        raise RuntimeError("No successful runs found in batch summary.")

    print(f"Loading embedding model: {args.model}")
    st_model = SentenceTransformer(args.model)
    gt_embeddings = st_model.encode(
        gt_facts, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)

    per_snapshot_rows: list[dict] = []
    per_run_final_summary: list[dict] = []

    for row in ok_rows:
        run_dir = Path(row["run_dir"])
        robots_json = find_robots_json(run_dir)
        if robots_json is None:
            print(f"[WARN] No robots.json found under {run_dir}")
            continue

        with robots_json.open("r", encoding="utf-8") as f:
            robots_data = json.load(f)
        if not isinstance(robots_data, dict):
            continue

        run_final_scores: list[dict[str, float]] = []
        for robot_id, observations in robots_data.items():
            if not isinstance(observations, list) or not observations:
                continue
            for snapshot_idx, obs in enumerate(observations):
                obs_text = str(obs).strip()
                metric_scores = score_observation_metrics(
                    obs_text, gt_embeddings, st_model, args.threshold
                )
                per_snapshot_rows.append(
                    {
                        "run_id": row["run_id"],
                        "variation": row["variation"],
                        "mode": row["mode"],
                        "seed": int(row["seed"]),
                        "robot_id": robot_id,
                        "snapshot_index": snapshot_idx,
                        "cosine_recall": metric_scores["recall"],
                        "recall": metric_scores["recall"],
                        "precision": metric_scores["precision"],
                        "f1": metric_scores["f1"],
                        "robots_json_path": str(robots_json),
                    }
                )

            final_obs = str(observations[-1]).strip()
            run_final_scores.append(
                score_observation_metrics(
                    final_obs, gt_embeddings, st_model, args.threshold
                )
            )

        if run_final_scores:
            per_run_final_summary.append(
                {
                    "run_id": row["run_id"],
                    "variation": row["variation"],
                    "mode": row["mode"],
                    "seed": int(row["seed"]),
                    "n_robots": len(run_final_scores),
                    "final_recall_mean": float(np.mean([s["recall"] for s in run_final_scores])),
                    "final_recall_std": float(np.std([s["recall"] for s in run_final_scores])),
                    "final_precision_mean": float(np.mean([s["precision"] for s in run_final_scores])),
                    "final_precision_std": float(np.std([s["precision"] for s in run_final_scores])),
                    "final_f1_mean": float(np.mean([s["f1"] for s in run_final_scores])),
                    "final_f1_std": float(np.std([s["f1"] for s in run_final_scores])),
                    "robots_json_path": str(robots_json),
                }
            )

    if not per_snapshot_rows:
        raise RuntimeError("No usable snapshot scores found. Check run output folders.")

    metrics_dir = resolve_metrics_output_dir(
        args.batch_summary_csv.resolve(),
        args.out_dir,
        ok_rows,
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_snapshot_csv = metrics_dir / "per_robot_per_snapshot_cosine_metrics.csv"
    per_run_csv = metrics_dir / "per_run_final_summary_cosine_metrics.csv"

    with per_snapshot_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "variation",
            "mode",
            "seed",
            "robot_id",
            "snapshot_index",
            "cosine_recall",
            "recall",
            "precision",
            "f1",
            "robots_json_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_snapshot_rows)

    with per_run_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "run_id",
            "variation",
            "mode",
            "seed",
            "n_robots",
            "final_recall_mean",
            "final_recall_std",
            "final_precision_mean",
            "final_precision_std",
            "final_f1_mean",
            "final_f1_std",
            "robots_json_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_run_final_summary)

    dashboard_path = metrics_dir / "cosine_knowledge_progression_dashboard.png"
    make_progression_plot(
        per_snapshot_rows=per_snapshot_rows,
        out_path=dashboard_path,
        title="Repeated Runs: Communication vs Non-Communication Recall Progression",
    )
    metrics_grid_path = metrics_dir / "cosine_metrics_progression_grid.png"
    make_metrics_grid_plot(
        per_snapshot_rows=per_snapshot_rows,
        out_path=metrics_grid_path,
        title="Repeated Runs: Recall, Precision, and F1 by Variation and Snapshot",
    )

    print(f"Output directory: {metrics_dir}")
    print(f"Saved: {per_snapshot_csv}")
    print(f"Saved: {per_run_csv}")
    print(f"Saved: {dashboard_path}")
    print(f"Saved: {metrics_grid_path}")

    grouped = defaultdict(list)
    for row in per_run_final_summary:
        grouped[(row["variation"], row["mode"])].append(
            (
                row["final_recall_mean"],
                row["final_precision_mean"],
                row["final_f1_mean"],
            )
        )
    print("\nFinal-snapshot averages over repeated runs:")
    for (variation, mode), vals in sorted(grouped.items()):
        recall_vals = [v[0] for v in vals]
        precision_vals = [v[1] for v in vals]
        f1_vals = [v[2] for v in vals]
        print(
            f"  {variation}/{mode}: "
            f"recall={np.mean(recall_vals):.4f} +/- {np.std(recall_vals):.4f}, "
            f"precision={np.mean(precision_vals):.4f} +/- {np.std(precision_vals):.4f}, "
            f"f1={np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}, n={len(vals)}"
        )


if __name__ == "__main__":
    main()
