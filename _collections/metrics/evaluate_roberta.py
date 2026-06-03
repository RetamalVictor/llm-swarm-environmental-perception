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


# Defaults mirror evaluate_cosine.py and can be overridden via CLI args.
GROUND_TRUTH_JSON = Path(
    "/scratch/ay2395/capstone/metrics/ground_truth/ground_truth_facts.json"
)
COMM_RUN_JSON = Path(
    "/scratch/ay2395/capstone/metrics/output/bg2500_comm_2026-04-21_13-14-55/robots_progress.json"
)
NONCOMM_RUN_JSON = Path(
    "/scratch/ay2395/capstone/metrics/output/bg2500_noncomm_2026-04-21_13-23-03/robots_progress.json"
)
OUTPUT_DIR = Path("/scratch/ay2395/capstone/metrics/analysis_outputs")

# High-accuracy RoBERTa backbone (slower than roberta-base, but better quality).
ROBERTA_MODEL_TYPE = "roberta-large"
BATCH_SIZE = 8


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


def score_observation(
    observation: str,
    gt_reference_text: str,
    model_type: str,
    batch_size: int,
) -> dict[str, float]:
    # Match cosine behavior: score factual units, not the raw full paragraph.
    facts = split_sentences(observation)
    if not facts:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    refs = [gt_reference_text] * len(facts)
    precision, recall, f1 = bert_score_fn(
        facts,
        refs,
        model_type=model_type,
        verbose=False,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )

    return {
        "recall": float(recall.mean().item()),
        "precision": float(precision.mean().item()),
        "f1": float(f1.mean().item()),
    }


def evaluate_run(
    run_data: dict[str, list[str]],
    gt_reference_text: str,
    model_type: str,
    batch_size: int,
) -> dict:
    robot_ids = sorted(run_data.keys(), key=lambda x: int(x))
    num_epochs = min(len(run_data[rid]) for rid in robot_ids)

    per_epoch = {
        "recall_mean": [],
        "recall_std": [],
        "precision_mean": [],
        "precision_std": [],
        "f1_mean": [],
        "f1_std": [],
    }
    final_per_robot = {}

    for epoch in range(num_epochs):
        epoch_recalls: list[float] = []
        epoch_precisions: list[float] = []
        epoch_f1s: list[float] = []

        for rid in robot_ids:
            obs = run_data[rid][epoch]
            scores = score_observation(obs, gt_reference_text, model_type, batch_size)
            epoch_recalls.append(scores["recall"])
            epoch_precisions.append(scores["precision"])
            epoch_f1s.append(scores["f1"])
            if epoch == num_epochs - 1:
                final_per_robot[rid] = scores

        per_epoch["recall_mean"].append(float(np.mean(epoch_recalls)))
        per_epoch["recall_std"].append(float(np.std(epoch_recalls)))
        per_epoch["precision_mean"].append(float(np.mean(epoch_precisions)))
        per_epoch["precision_std"].append(float(np.std(epoch_precisions)))
        per_epoch["f1_mean"].append(float(np.mean(epoch_f1s)))
        per_epoch["f1_std"].append(float(np.std(epoch_f1s)))

    return {
        "num_robots": len(robot_ids),
        "num_epochs": num_epochs,
        "per_epoch": per_epoch,
        "final_per_robot": final_per_robot,
    }


def plot_progress(comm_eval: dict, noncomm_eval: dict, out_path: Path) -> None:
    epochs = np.arange(1, min(comm_eval["num_epochs"], noncomm_eval["num_epochs"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metrics = [
        ("recall", "GT Recall"),
        ("precision", "GT Precision"),
        ("f1", "GT F1"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        cm = np.array(comm_eval["per_epoch"][f"{metric}_mean"][: len(epochs)])
        cs = np.array(comm_eval["per_epoch"][f"{metric}_std"][: len(epochs)])
        nm = np.array(noncomm_eval["per_epoch"][f"{metric}_mean"][: len(epochs)])
        ns = np.array(noncomm_eval["per_epoch"][f"{metric}_std"][: len(epochs)])

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
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.set_title(f"{ylabel} Progress")

    axes[0].legend(frameon=True)
    fig.suptitle(
        "Communication vs Non-Communication Ground-Truth Alignment (RoBERTa)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def final_means(eval_data: dict) -> dict[str, float]:
    vals = list(eval_data["final_per_robot"].values())
    return {
        "recall": float(np.mean([v["recall"] for v in vals])),
        "precision": float(np.mean([v["precision"] for v in vals])),
        "f1": float(np.mean([v["f1"] for v in vals])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate communication vs non-communication runs with RoBERTa-based BERTScore."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_facts = load_ground_truth(args.ground_truth_json.resolve())
    gt_reference_text = " ".join(gt_facts)
    comm_data = load_progress(args.comm_run_json.resolve())
    noncomm_data = load_progress(args.noncomm_run_json.resolve())

    print(f"Loaded ground truth facts: {len(gt_facts)}")
    print(f"Using RoBERTa model: {args.model_type}")

    comm_eval = evaluate_run(
        comm_data, gt_reference_text, args.model_type, args.batch_size
    )
    noncomm_eval = evaluate_run(
        noncomm_data, gt_reference_text, args.model_type, args.batch_size
    )

    plot_path = output_dir / "comm_vs_noncomm_ground_truth_roberta_progress.png"
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
        "method": "bert_score",
        "roberta_model_type": args.model_type,
        "batch_size": args.batch_size,
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "final_means": {
            "communication": comm_final,
            "non_communication": noncomm_final,
            "delta_comm_minus_noncomm": delta,
        },
        "plot": str(plot_path),
    }
    summary_path = output_dir / "comm_vs_noncomm_ground_truth_roberta_summary.json"
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
