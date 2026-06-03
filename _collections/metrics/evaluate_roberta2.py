import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from bert_score import score as bert_score_fn
except ImportError as e:
    raise ImportError(
        "bert-score is required. Install with: pip install bert-score"
    ) from e


# Keep defaults aligned with evaluate_roberta.py (override with CLI args as needed).
GROUND_TRUTH_JSON = Path(
    "/scratch/ay2395/capstone/metrics/ground_truth/ground_truth_facts.json"
)
COMM_RUN_JSON = Path(
    "/scratch/ay2395/capstone/metrics/output/robots_progress_comm.json"
)
NONCOMM_RUN_JSON = Path(
    "/scratch/ay2395/capstone/metrics/output/robots_progress_noncomm.json"
)
OUTPUT_DIR = Path("/scratch/ay2395/capstone/metrics/analysis_outputs")

ROBERTA_MODEL_TYPE = "roberta-large"
BATCH_SIZE = 16

# Match threshold on pairwise BERTScore F1 for "fact is observed".
MATCH_THRESHOLD = 0.86


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_progress(path: Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): v for k, v in data.items()}


def load_ground_truth(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    facts = data.get("facts", [])
    return [fact.strip() for fact in facts if isinstance(fact, str) and fact.strip()]


def bertscore_pairwise_f1_matrix(
    observation_facts: list[str],
    gt_facts: list[str],
    model_type: str,
    batch_size: int,
) -> np.ndarray:
    """
    Returns matrix shape [num_obs_facts, num_gt_facts] with pairwise BERTScore F1.
    """
    if not observation_facts or not gt_facts:
        return np.zeros((len(observation_facts), len(gt_facts)), dtype=np.float32)

    cands: list[str] = []
    refs: list[str] = []
    for obs_fact in observation_facts:
        for gt_fact in gt_facts:
            cands.append(obs_fact)
            refs.append(gt_fact)

    _, _, f1 = bert_score_fn(
        cands,
        refs,
        model_type=model_type,
        verbose=False,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    values = np.array([float(x) for x in f1], dtype=np.float32)
    return values.reshape(len(observation_facts), len(gt_facts))


def score_snapshot(
    observation: str,
    gt_facts: list[str],
    covered_gt_indices: set[int],
    model_type: str,
    batch_size: int,
    threshold: float,
) -> dict[str, float | int]:
    """
    Scores one robot snapshot using fact-level matching:
    - Recall: cumulative GT coverage up to this epoch.
    - Precision: fraction of observation facts that map to at least one GT fact.
    - F1: harmonic mean of cumulative recall and per-snapshot precision.
    """
    obs_facts = split_sentences(observation)
    if not obs_facts:
        recall = len(covered_gt_indices) / len(gt_facts) if gt_facts else 0.0
        return {
            "recall": float(recall),
            "precision": 0.0,
            "f1": 0.0,
            "new_matches": 0,
            "total_covered": len(covered_gt_indices),
            "obs_fact_count": 0,
        }

    sim = bertscore_pairwise_f1_matrix(obs_facts, gt_facts, model_type, batch_size)
    if sim.size == 0:
        recall = len(covered_gt_indices) / len(gt_facts) if gt_facts else 0.0
        return {
            "recall": float(recall),
            "precision": 0.0,
            "f1": 0.0,
            "new_matches": 0,
            "total_covered": len(covered_gt_indices),
            "obs_fact_count": len(obs_facts),
        }

    best_per_gt = sim.max(axis=0)  # shape [num_gt_facts]
    matched_gt = set(np.where(best_per_gt >= threshold)[0].tolist())
    new_matches = matched_gt - covered_gt_indices
    covered_gt_indices.update(new_matches)

    best_per_obs = sim.max(axis=1)  # shape [num_obs_facts]
    precision = float((best_per_obs >= threshold).sum() / len(best_per_obs))
    recall = float(len(covered_gt_indices) / len(gt_facts)) if gt_facts else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "new_matches": len(new_matches),
        "total_covered": len(covered_gt_indices),
        "obs_fact_count": len(obs_facts),
    }


def evaluate_run(
    run_data: dict[str, list[str]],
    gt_facts: list[str],
    model_type: str,
    batch_size: int,
    threshold: float,
) -> dict:
    robot_ids = sorted(run_data.keys(), key=lambda x: int(x))
    num_epochs = min(len(run_data[rid]) for rid in robot_ids)

    # Epoch 0 baseline (before any observation is processed).
    per_epoch = {
        "recall_mean": [0.0],
        "recall_std": [0.0],
        "precision_mean": [0.0],
        "precision_std": [0.0],
        "f1_mean": [0.0],
        "f1_std": [0.0],
        "new_matches_mean": [0.0],
        "new_matches_std": [0.0],
        "coverage_count_mean": [0.0],
        "coverage_count_std": [0.0],
    }
    final_per_robot: dict[str, dict[str, float | int]] = {}
    covered_by_robot = {rid: set() for rid in robot_ids}

    for epoch in range(num_epochs):
        epoch_recalls: list[float] = []
        epoch_precisions: list[float] = []
        epoch_f1s: list[float] = []
        epoch_new_matches: list[int] = []
        epoch_coverage_counts: list[int] = []

        for rid in robot_ids:
            obs = run_data[rid][epoch]
            scores = score_snapshot(
                obs,
                gt_facts,
                covered_by_robot[rid],
                model_type,
                batch_size,
                threshold,
            )
            epoch_recalls.append(float(scores["recall"]))
            epoch_precisions.append(float(scores["precision"]))
            epoch_f1s.append(float(scores["f1"]))
            epoch_new_matches.append(int(scores["new_matches"]))
            epoch_coverage_counts.append(int(scores["total_covered"]))
            if epoch == num_epochs - 1:
                final_per_robot[rid] = scores

        per_epoch["recall_mean"].append(float(np.mean(epoch_recalls)))
        per_epoch["recall_std"].append(float(np.std(epoch_recalls)))
        per_epoch["precision_mean"].append(float(np.mean(epoch_precisions)))
        per_epoch["precision_std"].append(float(np.std(epoch_precisions)))
        per_epoch["f1_mean"].append(float(np.mean(epoch_f1s)))
        per_epoch["f1_std"].append(float(np.std(epoch_f1s)))
        per_epoch["new_matches_mean"].append(float(np.mean(epoch_new_matches)))
        per_epoch["new_matches_std"].append(float(np.std(epoch_new_matches)))
        per_epoch["coverage_count_mean"].append(float(np.mean(epoch_coverage_counts)))
        per_epoch["coverage_count_std"].append(float(np.std(epoch_coverage_counts)))

    return {
        "num_robots": len(robot_ids),
        "num_epochs": num_epochs,
        "num_epochs_with_baseline": num_epochs + 1,
        "gt_fact_count": len(gt_facts),
        "per_epoch": per_epoch,
        "final_per_robot": final_per_robot,
    }


def plot_progress(comm_eval: dict, noncomm_eval: dict, out_path: Path) -> None:
    n = min(
        comm_eval["num_epochs_with_baseline"], noncomm_eval["num_epochs_with_baseline"]
    )
    epochs = np.arange(0, n)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    metrics = [
        ("recall", "Cumulative GT Recall"),
        ("precision", "Snapshot GT Precision"),
        ("f1", "Cumulative/Snapshot F1"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        cm = np.array(comm_eval["per_epoch"][f"{metric}_mean"][:n])
        cs = np.array(comm_eval["per_epoch"][f"{metric}_std"][:n])
        nm = np.array(noncomm_eval["per_epoch"][f"{metric}_mean"][:n])
        ns = np.array(noncomm_eval["per_epoch"][f"{metric}_std"][:n])

        ax.plot(epochs, cm, color="#1f77b4", linewidth=2.4, label="Communication")
        ax.fill_between(
            epochs,
            np.clip(cm - cs, 0, 1),
            np.clip(cm + cs, 0, 1),
            color="#1f77b4",
            alpha=0.16,
        )
        ax.plot(epochs, nm, color="#ff7f0e", linewidth=2.4, label="Non-Communication")
        ax.fill_between(
            epochs,
            np.clip(nm - ns, 0, 1),
            np.clip(nm + ns, 0, 1),
            color="#ff7f0e",
            alpha=0.16,
        )
        ax.set_xlabel("Epoch (0 = pre-observation baseline)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.set_title(f"{ylabel} Progress")

    axes[0].legend(frameon=True)
    fig.suptitle(
        "Communication vs Non-Communication Ground-Truth Alignment (RoBERTa, Fact Coverage)",
        fontsize=12.5,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def final_means(eval_data: dict) -> dict[str, float]:
    vals = list(eval_data["final_per_robot"].values())
    if not vals:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    return {
        "recall": float(np.mean([float(v["recall"]) for v in vals])),
        "precision": float(np.mean([float(v["precision"]) for v in vals])),
        "f1": float(np.mean([float(v["f1"]) for v in vals])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RoBERTa fact-coverage evaluator for communication vs non-communication. "
            "Includes epoch-0 zero baseline."
        )
    )
    parser.add_argument(
        "--ground-truth-json",
        type=Path,
        default=GROUND_TRUTH_JSON,
        help="Path to ground_truth_facts.json",
    )
    parser.add_argument(
        "--comm-run-json",
        type=Path,
        default=COMM_RUN_JSON,
        help="Path to communication robots_progress.json",
    )
    parser.add_argument(
        "--noncomm-run-json",
        type=Path,
        default=NONCOMM_RUN_JSON,
        help="Path to non-communication robots_progress.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for output plot and summary JSON",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=ROBERTA_MODEL_TYPE,
        help="RoBERTa model for BERTScore (e.g., roberta-large, roberta-base)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for BERTScore",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help="BERTScore-F1 threshold to count a GT fact as matched",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_facts = load_ground_truth(args.ground_truth_json.resolve())
    comm_data = load_progress(args.comm_run_json.resolve())
    noncomm_data = load_progress(args.noncomm_run_json.resolve())

    print(f"Loaded ground truth facts: {len(gt_facts)}")
    print(f"Using RoBERTa model: {args.model_type}")
    print(f"Match threshold (pairwise BERTScore-F1): {args.match_threshold}")

    comm_eval = evaluate_run(
        comm_data,
        gt_facts,
        args.model_type,
        args.batch_size,
        args.match_threshold,
    )
    noncomm_eval = evaluate_run(
        noncomm_data,
        gt_facts,
        args.model_type,
        args.batch_size,
        args.match_threshold,
    )

    plot_path = output_dir / "comm_vs_noncomm_ground_truth_roberta2_progress.png"
    plot_progress(comm_eval, noncomm_eval, plot_path)

    comm_final = final_means(comm_eval)
    noncomm_final = final_means(noncomm_eval)
    delta = {
        "recall": comm_final["recall"] - noncomm_final["recall"],
        "precision": comm_final["precision"] - noncomm_final["precision"],
        "f1": comm_final["f1"] - noncomm_final["f1"],
    }

    summary = {
        "ground_truth_json": str(args.ground_truth_json.resolve()),
        "ground_truth_fact_count": len(gt_facts),
        "comm_run_json": str(args.comm_run_json.resolve()),
        "noncomm_run_json": str(args.noncomm_run_json.resolve()),
        "method": "bert_score_fact_coverage",
        "roberta_model_type": args.model_type,
        "batch_size": args.batch_size,
        "match_threshold": args.match_threshold,
        "baseline_epoch_included": True,
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "final_means": {
            "communication": comm_final,
            "non_communication": noncomm_final,
            "delta_comm_minus_noncomm": delta,
        },
        "plot": str(plot_path),
    }
    summary_path = output_dir / "comm_vs_noncomm_ground_truth_roberta2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved plot: {plot_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "Final mean scores | "
        f"Comm: R={comm_final['recall']:.3f}, P={comm_final['precision']:.3f}, F1={comm_final['f1']:.3f} | "
        f"Non-Comm: R={noncomm_final['recall']:.3f}, P={noncomm_final['precision']:.3f}, F1={noncomm_final['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
