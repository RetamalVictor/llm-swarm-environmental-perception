"""
Communication vs non-communication evaluation using BERTScore with a RoBERTa backbone.

This follows the spirit of scripts/tests.ipynb (bert_score), which uses contextual
embeddings rather than sentence-MiniLM cosine similarity.

BERTScore compares each robot observation (candidate) to a single frozen reference
string built from ground-truth facts. The **recall** component measures how much of
the reference is reflected in the candidate (useful for "knowledge of the map").
**F1** gives a balanced soft score.

Does not modify evaluate_comm_vs_noncomm.py.

Usage:
  python metrics/evaluate_comm_vs_noncomm_bertscore_roberta.py \\
    --output-dir output \\
    --analysis-dir analysis_outputs

  Or pass explicit paths:
  python metrics/evaluate_comm_vs_noncomm_bertscore_roberta.py \\
    --comm-json output/large_comm_1_.../robots.json \\
    --noncomm-json output/large_noncomm_1_.../robots.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from bert_score import score as bert_score_fn
except ImportError as e:
    raise ImportError(
        "bert-score is required. Install with: pip install bert-score"
    ) from e


# Same 21-fact ground truth as metrics/evaluate_comm_vs_noncomm.py (do not import that module).
GROUND_TRUTH_FACTS = [
    "A mountain with snowy peaks is present.",
    "A stone bridge is present.",
    "A blue river is present.",
    "A large green tree is present.",
    "A dark volcano with orange lava is present.",
    "A red castle is present.",
    "A blue moat surrounds the castle.",
    "A green hedge maze is present.",
    "A group of four people is present.",
    "A gray parking lot is present.",
    "Six cars are present in the parking lot.",
    "A football stadium is present.",
    "A windmill is present.",
    "A fenced graveyard with tombstones is present.",
    "A colorful hot air balloon is present.",
    "A dense patch of trees is present.",
    "A blue stone fountain is present.",
    "A fenced zoo enclosure is present.",
    "A giraffe is present in the enclosure.",
    "An elephant is present in the enclosure.",
    "A lion is present in the enclosure.",
]


def build_reference_text() -> str:
    return " ".join(GROUND_TRUTH_FACTS)


def load_run(path: Path) -> dict[str, list[str]]:
    with open(path, "r") as f:
        return json.load(f)


def discover_comm_noncomm_json(output_dir: Path) -> tuple[Path, Path]:
    comm_candidates: list[Path] = []
    noncomm_candidates: list[Path] = []
    for entry in sorted(output_dir.iterdir()):
        if not entry.is_dir():
            continue
        robots = entry / "robots.json"
        if not robots.exists():
            continue
        name = entry.name.lower()
        if "noncomm" in name:
            noncomm_candidates.append(robots)
        elif "comm" in name:
            comm_candidates.append(robots)
    if not comm_candidates or not noncomm_candidates:
        raise FileNotFoundError(
            f"Could not find both comm and noncomm runs under {output_dir}. "
            "Expected folder names containing 'comm' and 'noncomm' with robots.json."
        )
    # If multiple runs exist, use the newest robots.json by mtime.
    comm_path = max(comm_candidates, key=lambda p: p.stat().st_mtime)
    noncomm_path = max(noncomm_candidates, key=lambda p: p.stat().st_mtime)
    return comm_path, noncomm_path


def run_bertscore_batch(
    cands: list[str],
    ref: str,
    model_type: str,
    batch_size: int,
) -> tuple[list[float], list[float], list[float]]:
    """Returns per-candidate precision, recall, f1 as Python floats."""
    if not cands:
        return [], [], []
    refs = [ref] * len(cands)
    P, R, F1 = bert_score_fn(
        cands,
        refs,
        model_type=model_type,
        verbose=False,
        batch_size=batch_size,
        rescale_with_baseline=False,
    )
    return (
        [float(x) for x in P.cpu().tolist()],
        [float(x) for x in R.cpu().tolist()],
        [float(x) for x in F1.cpu().tolist()],
    )


def evaluate_run(
    data: dict[str, list[str]],
    ref_text: str,
    model_type: str,
    batch_size: int,
) -> dict:
    robot_ids = sorted(data.keys(), key=lambda x: int(x))
    n_snapshots = min(len(data[r]) for r in robot_ids)

    per_snapshot = {
        "bertscore_recall_mean": [],
        "bertscore_recall_std": [],
        "bertscore_f1_mean": [],
        "bertscore_f1_std": [],
        "bertscore_precision_mean": [],
        "bertscore_precision_std": [],
    }
    per_robot_final: dict[str, dict[str, float]] = {}

    for s in range(n_snapshots):
        cands = [data[rid][s] for rid in robot_ids]
        P, R, F1 = run_bertscore_batch(cands, ref_text, model_type, batch_size)

        per_snapshot["bertscore_recall_mean"].append(float(np.mean(R)))
        per_snapshot["bertscore_recall_std"].append(float(np.std(R)))
        per_snapshot["bertscore_f1_mean"].append(float(np.mean(F1)))
        per_snapshot["bertscore_f1_std"].append(float(np.std(F1)))
        per_snapshot["bertscore_precision_mean"].append(float(np.mean(P)))
        per_snapshot["bertscore_precision_std"].append(float(np.std(P)))

        if s == n_snapshots - 1:
            for i, rid in enumerate(robot_ids):
                per_robot_final[rid] = {
                    "bertscore_recall": R[i],
                    "bertscore_f1": F1[i],
                    "bertscore_precision": P[i],
                }

    return {
        "num_robots": len(robot_ids),
        "num_snapshots": n_snapshots,
        "per_snapshot": per_snapshot,
        "per_robot_final": per_robot_final,
    }


def mann_whitney(a: list[float], b: list[float]) -> dict:
    u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"u_stat": float(u_stat), "p_value": float(p)}


def compare_runs(comm_eval: dict, noncomm_eval: dict) -> dict:
    def extract(key: str, src: dict) -> list[float]:
        return [src["per_robot_final"][rid][key] for rid in sorted(src["per_robot_final"], key=lambda x: int(x))]

    comm_r = extract("bertscore_recall", comm_eval)
    non_r = extract("bertscore_recall", noncomm_eval)
    comm_f1 = extract("bertscore_f1", comm_eval)
    non_f1 = extract("bertscore_f1", noncomm_eval)

    return {
        "final_bertscore_recall_test": mann_whitney(comm_r, non_r),
        "final_bertscore_f1_test": mann_whitney(comm_f1, non_f1),
        "means": {
            "comm_final_bertscore_recall": float(np.mean(comm_r)),
            "noncomm_final_bertscore_recall": float(np.mean(non_r)),
            "comm_final_bertscore_f1": float(np.mean(comm_f1)),
            "noncomm_final_bertscore_f1": float(np.mean(non_f1)),
        },
    }


def plot_dashboard(
    comm_eval: dict,
    noncomm_eval: dict,
    out_file: Path,
    model_type: str,
) -> None:
    x = np.arange(1, min(comm_eval["num_snapshots"], noncomm_eval["num_snapshots"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"BERTScore (RoBERTa: {model_type}) — Communication vs Non-Communication",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    specs = [
        (
            axes[0, 0],
            "bertscore_recall",
            "BERTScore Recall",
            "Reference coverage in agent text (higher = closer to ground truth)",
        ),
        (
            axes[0, 1],
            "bertscore_f1",
            "BERTScore F1",
            "Balanced soft match vs full ground-truth paragraph",
        ),
    ]

    for ax, prefix, ylabel, subtitle in specs:
        cm = np.array(comm_eval["per_snapshot"][f"{prefix}_mean"][: len(x)])
        cs = np.array(comm_eval["per_snapshot"][f"{prefix}_std"][: len(x)])
        nm = np.array(noncomm_eval["per_snapshot"][f"{prefix}_mean"][: len(x)])
        ns = np.array(noncomm_eval["per_snapshot"][f"{prefix}_std"][: len(x)])

        ax.plot(x, cm, color="#1f77b4", linewidth=2.4, label="Communication")
        ax.fill_between(x, np.clip(cm - cs, 0, 1), np.clip(cm + cs, 0, 1), color="#1f77b4", alpha=0.16)
        ax.plot(x, nm, color="#ff7f0e", linewidth=2.4, label="Non-Communication")
        ax.fill_between(x, np.clip(nm - ns, 0, 1), np.clip(nm + ns, 0, 1), color="#ff7f0e", alpha=0.16)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xlabel("Observation snapshot index")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.legend(frameon=True)

    comm_r = [comm_eval["per_robot_final"][rid]["bertscore_recall"] for rid in sorted(comm_eval["per_robot_final"], key=lambda x: int(x))]
    non_r = [noncomm_eval["per_robot_final"][rid]["bertscore_recall"] for rid in sorted(noncomm_eval["per_robot_final"], key=lambda x: int(x))]
    comm_f1 = [comm_eval["per_robot_final"][rid]["bertscore_f1"] for rid in sorted(comm_eval["per_robot_final"], key=lambda x: int(x))]
    non_f1 = [noncomm_eval["per_robot_final"][rid]["bertscore_f1"] for rid in sorted(noncomm_eval["per_robot_final"], key=lambda x: int(x))]

    for ax, vals_a, vals_b, ylabel, title in [
        (axes[1, 0], comm_r, non_r, "BERTScore Recall", "Final snapshot — Recall"),
        (axes[1, 1], comm_f1, non_f1, "BERTScore F1", "Final snapshot — F1"),
    ]:
        bp = ax.boxplot(
            [vals_a, vals_b],
            labels=["Communication", "Non-Communication"],
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.8},
        )
        for patch, color in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_file, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERTScore (RoBERTa) comm vs non-comm evaluation on robots.json runs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Project output folder; used to auto-discover comm/noncomm robots.json",
    )
    parser.add_argument("--comm-json", type=str, default=None, help="Override path to communication robots.json")
    parser.add_argument("--noncomm-json", type=str, default=None, help="Override path to non-communication robots.json")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="analysis_outputs",
        help="Where to write plots and summary JSON",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="roberta-base",
        help="BERTScore backbone (e.g. roberta-base, roberta-large)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="BERTScore batch size")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_base = (project_root / args.output_dir).resolve()
    analysis_dir = (project_root / args.analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if args.comm_json and args.noncomm_json:
        comm_path = Path(args.comm_json).resolve()
        noncomm_path = Path(args.noncomm_json).resolve()
    else:
        comm_path, noncomm_path = discover_comm_noncomm_json(out_base)

    ref_text = build_reference_text()
    comm_data = load_run(comm_path)
    noncomm_data = load_run(noncomm_path)

    print(f"Using comm:   {comm_path}")
    print(f"Using noncomm:{noncomm_path}")
    print(f"BERTScore model_type={args.model_type} (RoBERTa family)")

    comm_eval = evaluate_run(comm_data, ref_text, args.model_type, args.batch_size)
    noncomm_eval = evaluate_run(noncomm_data, ref_text, args.model_type, args.batch_size)
    comparison = compare_runs(comm_eval, noncomm_eval)

    dashboard_path = analysis_dir / "comm_vs_noncomm_bertscore_roberta_dashboard.png"
    plot_dashboard(comm_eval, noncomm_eval, dashboard_path, args.model_type)

    summary = {
        "method": "bert_score",
        "model_type": args.model_type,
        "reference": "single string = space-joined GROUND_TRUTH_FACTS (21 facts)",
        "comm_json": str(comm_path),
        "noncomm_json": str(noncomm_path),
        "ground_truth_facts": GROUND_TRUTH_FACTS,
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "comparison": comparison,
        "plots": {"dashboard": str(dashboard_path)},
    }
    summary_path = analysis_dir / "comm_vs_noncomm_bertscore_roberta_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved dashboard: {dashboard_path}")
    print(f"Saved summary:   {summary_path}")
    print("Mean final metrics:")
    for k, v in comparison["means"].items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
