import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready PDF plots from experiment metric CSVs."
    )
    parser.add_argument(
        "--per-snapshot-csv",
        type=Path,
        default=Path("experiments/metrics/outputs/per_robot_per_snapshot_cosine_metrics.csv"),
        help="Path to per_robot_per_snapshot_cosine_metrics.csv.",
    )
    parser.add_argument(
        "--per-run-summary-csv",
        type=Path,
        default=Path("experiments/metrics/outputs/per_run_final_summary_cosine_metrics.csv"),
        help="Path to per_run_final_summary_cosine_metrics.csv (for validation/context).",
    )
    parser.add_argument(
        "--variation",
        type=str,
        default="bg2500-big",
        help="Variation name to plot (for example: bg2500-big).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory where PDF plots will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="bg2500_big",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def _plot_metric_panel(
    ax: plt.Axes,
    epochs: np.ndarray,
    comm_mean: np.ndarray,
    comm_std: np.ndarray,
    noncomm_mean: np.ndarray,
    noncomm_std: np.ndarray,
    y_label: str,
    show_legend: bool = False,
) -> None:
    ax.plot(epochs, comm_mean, color="#1f77b4", linewidth=2.4, label="Communication")
    ax.fill_between(
        epochs,
        np.clip(comm_mean - comm_std, 0.0, 1.0),
        np.clip(comm_mean + comm_std, 0.0, 1.0),
        color="#1f77b4",
        alpha=0.16,
    )

    ax.plot(
        epochs,
        noncomm_mean,
        color="#ff7f0e",
        linewidth=2.4,
        label="Non-Communication",
    )
    ax.fill_between(
        epochs,
        np.clip(noncomm_mean - noncomm_std, 0.0, 1.0),
        np.clip(noncomm_mean + noncomm_std, 0.0, 1.0),
        color="#ff7f0e",
        alpha=0.16,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
    if show_legend:
        ax.legend(frameon=True)


def _metric_arrays(eval_data: dict, metric: str, n_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    per_epoch = eval_data["per_epoch"]
    means = np.array(per_epoch[f"{metric}_mean"][:n_epochs], dtype=np.float32)
    stds = np.array(per_epoch[f"{metric}_std"][:n_epochs], dtype=np.float32)
    return means, stds


def _load_per_snapshot_rows(path: Path, variation: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("variation") == variation]


def _assert_variation_present_in_summary(path: Path, variation: str) -> None:
    # This confirms the variation exists in the requested summary file the user pointed to.
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        present = any(row.get("variation") == variation for row in reader)
    if not present:
        raise ValueError(f"Variation '{variation}' not found in {path}")


def _build_eval_like_from_rows(rows: list[dict[str, str]], mode: str) -> dict:
    metric_names = ("recall", "precision", "f1")
    # Two-stage aggregation:
    # 1) average robots within each (seed, snapshot)
    # 2) average seeds within each snapshot (this is what the paper plot should show)
    per_seed_snapshot_metric: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in metric_names}
    )
    grouped: dict[str, dict[int, list[float]]] = {metric: defaultdict(list) for metric in metric_names}
    snapshot_ids: set[int] = set()
    seed_ids: set[int] = set()
    robot_ids: set[str] = set()

    for row in rows:
        if row.get("mode") != mode:
            continue
        seed = int(row["seed"])
        snapshot_idx = int(row["snapshot_index"])
        robot_ids.add(str(row["robot_id"]))
        snapshot_ids.add(snapshot_idx)
        seed_ids.add(seed)
        for metric in metric_names:
            per_seed_snapshot_metric[(seed, snapshot_idx)][metric].append(float(row[metric]))

    if not snapshot_ids:
        raise ValueError(f"No rows found for mode='{mode}'.")

    # Collapse robot-level values -> one value per (seed, snapshot, metric).
    for (seed, snapshot_idx), metric_lists in per_seed_snapshot_metric.items():
        _ = seed  # seed is part of key for clarity; no direct use required below
        for metric in metric_names:
            vals = metric_lists[metric]
            if vals:
                grouped[metric][snapshot_idx].append(float(np.mean(vals)))

    sorted_snapshots = sorted(snapshot_ids)
    per_epoch: dict[str, list[float]] = {}
    for metric in metric_names:
        per_epoch[f"{metric}_mean"] = [
            float(np.mean(grouped[metric][idx])) for idx in sorted_snapshots
        ]
        per_epoch[f"{metric}_std"] = [
            float(np.std(grouped[metric][idx])) for idx in sorted_snapshots
        ]

    return {
        "num_robots": len(robot_ids),
        "num_seeds": len(seed_ids),
        "num_epochs": len(sorted_snapshots),
        "per_epoch": per_epoch,
    }


def main() -> None:
    args = parse_args()
    _assert_variation_present_in_summary(args.per_run_summary_csv, args.variation)
    rows = _load_per_snapshot_rows(args.per_snapshot_csv, args.variation)
    if not rows:
        raise ValueError(
            f"No per-snapshot rows found for variation '{args.variation}' in {args.per_snapshot_csv}"
        )

    comm_eval = _build_eval_like_from_rows(rows, mode="comm")
    noncomm_eval = _build_eval_like_from_rows(rows, mode="noncomm")
    n_epochs = min(int(comm_eval["num_epochs"]), int(noncomm_eval["num_epochs"]))
    epochs = np.arange(1, n_epochs + 1)

    metric_specs = [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1 Score"),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Combined 3-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, (metric, y_label) in enumerate(metric_specs):
        comm_mean, comm_std = _metric_arrays(comm_eval, metric, n_epochs)
        noncomm_mean, noncomm_std = _metric_arrays(noncomm_eval, metric, n_epochs)
        _plot_metric_panel(
            ax=axes[idx],
            epochs=epochs,
            comm_mean=comm_mean,
            comm_std=comm_std,
            noncomm_mean=noncomm_mean,
            noncomm_std=noncomm_std,
            y_label=y_label,
            show_legend=(idx == 0),
        )

    plt.tight_layout()
    combined_path = args.out_dir / f"{args.prefix}_metrics_combined.pdf"
    plt.savefig(combined_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Separate single-metric PDFs.
    for metric, y_label in metric_specs:
        comm_mean, comm_std = _metric_arrays(comm_eval, metric, n_epochs)
        noncomm_mean, noncomm_std = _metric_arrays(noncomm_eval, metric, n_epochs)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        _plot_metric_panel(
            ax=ax,
            epochs=epochs,
            comm_mean=comm_mean,
            comm_std=comm_std,
            noncomm_mean=noncomm_mean,
            noncomm_std=noncomm_std,
            y_label=y_label,
            show_legend=True,
        )
        plt.tight_layout()
        single_path = args.out_dir / f"{args.prefix}_{metric}.pdf"
        plt.savefig(single_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {single_path}")

    metadata_path = args.out_dir / f"{args.prefix}_metrics_plot_metadata.json"
    metadata = {
        "variation": args.variation,
        "per_snapshot_csv": str(args.per_snapshot_csv.resolve()),
        "per_run_summary_csv": str(args.per_run_summary_csv.resolve()),
        "combined_pdf": str(combined_path.resolve()),
        "metric_pdfs": {
            "recall": str((args.out_dir / f"{args.prefix}_recall.pdf").resolve()),
            "precision": str((args.out_dir / f"{args.prefix}_precision.pdf").resolve()),
            "f1": str((args.out_dir / f"{args.prefix}_f1.pdf").resolve()),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {combined_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
