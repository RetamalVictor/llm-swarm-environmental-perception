import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

from evaluate_comm_vs_noncomm import GROUND_TRUTH_FACTS


ENTITY_KEYWORDS = {
    "mountain": ["mountain", "snowy peak", "snow-capped"],
    "bridge": ["bridge"],
    "river": ["river"],
    "tree": ["tree"],
    "volcano": ["volcano", "lava"],
    "castle": ["castle"],
    "moat": ["moat"],
    "maze": ["maze", "hedge"],
    "people_group": ["people", "person", "shirt"],
    "parking_lot": ["parking lot", "parked", "car"],
    "stadium": ["stadium", "field", "seats"],
    "windmill": ["windmill"],
    "graveyard": ["graveyard", "tombstone", "cross", "fence"],
    "hot_air_balloon": ["hot air balloon", "balloon"],
    "forest_patch": ["dense patch of trees", "forest", "pine"],
    "fountain": ["fountain"],
    "zoo_enclosure": ["zoo", "enclosure", "giraffe", "elephant", "lion"],
}


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a / a_norm) @ (b / b_norm).T


def gt_similarity_sum(observation: str, gt_embeddings: np.ndarray, model: SentenceTransformer) -> float:
    """
    Soft GT coverage:
    - For each GT fact, take its best cosine match across observation sentences.
    - Sum all best-match scores (no thresholding / binarization).
    """
    sentences = split_sentences(observation)
    if not sentences:
        return 0.0
    obs_embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    sims = cosine_matrix(obs_embeddings, gt_embeddings)
    best_fact_scores = sims.max(axis=0) if sims.size else np.zeros((gt_embeddings.shape[0],), dtype=np.float32)
    return float(best_fact_scores.sum())


def entity_coverage(observation: str) -> float:
    text = normalize_text(observation)
    found = 0
    total = len(ENTITY_KEYWORDS)
    for keywords in ENTITY_KEYWORDS.values():
        if any(keyword in text for keyword in keywords):
            found += 1
    return found / total if total else 0.0


def load_run(path: Path) -> dict[str, list[str]]:
    with open(path, "r") as f:
        return json.load(f)


def evaluate_run(data: dict[str, list[str]], model: SentenceTransformer, gt_embeddings: np.ndarray) -> dict:
    robot_ids = sorted(data.keys(), key=lambda x: int(x))
    snapshots = min(len(data[r]) for r in robot_ids)

    per_snapshot = {
        "similarity_sum_mean": [],
        "similarity_sum_std": [],
        "entity_coverage_mean": [],
        "entity_coverage_std": [],
    }
    per_robot_final = {}

    for s in range(snapshots):
        similarity_vals = []
        coverage_vals = []
        for rid in robot_ids:
            obs = data[rid][s]
            similarity_val = gt_similarity_sum(obs, gt_embeddings, model)
            coverage = entity_coverage(obs)
            similarity_vals.append(similarity_val)
            coverage_vals.append(coverage)

            if s == snapshots - 1:
                per_robot_final[rid] = {
                    "similarity_sum": similarity_val,
                    "entity_coverage": coverage,
                }

        per_snapshot["similarity_sum_mean"].append(float(np.mean(similarity_vals)))
        per_snapshot["similarity_sum_std"].append(float(np.std(similarity_vals)))
        per_snapshot["entity_coverage_mean"].append(float(np.mean(coverage_vals)))
        per_snapshot["entity_coverage_std"].append(float(np.std(coverage_vals)))

    return {
        "num_robots": len(robot_ids),
        "num_snapshots": snapshots,
        "per_snapshot": per_snapshot,
        "per_robot_final": per_robot_final,
    }


def mann_whitney(a: list[float], b: list[float]) -> dict:
    u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"u_stat": float(u_stat), "p_value": float(p)}


def compare_runs(comm_eval: dict, noncomm_eval: dict) -> dict:
    def extract(metric: str, src: dict) -> list[float]:
        return [src["per_robot_final"][rid][metric] for rid in sorted(src["per_robot_final"], key=lambda x: int(x))]

    comm_similarity = extract("similarity_sum", comm_eval)
    noncomm_similarity = extract("similarity_sum", noncomm_eval)
    comm_cov = extract("entity_coverage", comm_eval)
    noncomm_cov = extract("entity_coverage", noncomm_eval)

    return {
        "final_similarity_sum_test": mann_whitney(comm_similarity, noncomm_similarity),
        "final_entity_coverage_test": mann_whitney(comm_cov, noncomm_cov),
        "means": {
            "comm_final_similarity_sum": float(np.mean(comm_similarity)),
            "noncomm_final_similarity_sum": float(np.mean(noncomm_similarity)),
            "comm_final_entity_coverage": float(np.mean(comm_cov)),
            "noncomm_final_entity_coverage": float(np.mean(noncomm_cov)),
        },
    }


def plot_dashboard(
    comm_eval: dict,
    noncomm_eval: dict,
    out_file: Path,
    title_prefix: str,
    gt_count: int,
) -> None:
    snapshots = np.arange(1, min(comm_eval["num_snapshots"], noncomm_eval["num_snapshots"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{title_prefix}: Communication vs Non-Communication",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    cm = np.array(comm_eval["per_snapshot"]["similarity_sum_mean"][: len(snapshots)])
    cs = np.array(comm_eval["per_snapshot"]["similarity_sum_std"][: len(snapshots)])
    nm = np.array(noncomm_eval["per_snapshot"]["similarity_sum_mean"][: len(snapshots)])
    ns = np.array(noncomm_eval["per_snapshot"]["similarity_sum_std"][: len(snapshots)])

    axes[0, 0].plot(snapshots, cm, color="#1f77b4", linewidth=2.4, label="Communication")
    axes[0, 0].fill_between(snapshots, cm - cs, cm + cs, color="#1f77b4", alpha=0.16)
    axes[0, 0].plot(snapshots, nm, color="#ff7f0e", linewidth=2.4, label="Non-Communication")
    axes[0, 0].fill_between(snapshots, nm - ns, nm + ns, color="#ff7f0e", alpha=0.16)
    axes[0, 0].set_title("Soft GT Similarity Over Observation Snapshots", fontsize=11)
    axes[0, 0].set_xlabel("Observation Snapshot Index")
    axes[0, 0].set_ylabel(f"Similarity Sum (max possible ~{gt_count})")
    axes[0, 0].grid(alpha=0.28, linestyle="--", linewidth=0.8)
    axes[0, 0].legend(frameon=True)

    cm_cov = np.array(comm_eval["per_snapshot"]["entity_coverage_mean"][: len(snapshots)])
    cs_cov = np.array(comm_eval["per_snapshot"]["entity_coverage_std"][: len(snapshots)])
    nm_cov = np.array(noncomm_eval["per_snapshot"]["entity_coverage_mean"][: len(snapshots)])
    ns_cov = np.array(noncomm_eval["per_snapshot"]["entity_coverage_std"][: len(snapshots)])

    axes[0, 1].plot(snapshots, cm_cov, color="#1f77b4", linewidth=2.4, label="Communication")
    axes[0, 1].fill_between(
        snapshots,
        np.clip(cm_cov - cs_cov, 0, 1),
        np.clip(cm_cov + cs_cov, 0, 1),
        color="#1f77b4",
        alpha=0.16,
    )
    axes[0, 1].plot(snapshots, nm_cov, color="#ff7f0e", linewidth=2.4, label="Non-Communication")
    axes[0, 1].fill_between(
        snapshots,
        np.clip(nm_cov - ns_cov, 0, 1),
        np.clip(nm_cov + ns_cov, 0, 1),
        color="#ff7f0e",
        alpha=0.16,
    )
    axes[0, 1].set_title("Detected Landmark Breadth Over Observation Snapshots", fontsize=11)
    axes[0, 1].set_xlabel("Observation Snapshot Index")
    axes[0, 1].set_ylabel("Entity Coverage")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.28, linestyle="--", linewidth=0.8)
    axes[0, 1].legend(frameon=True)

    comm_final_similarity = [
        comm_eval["per_robot_final"][rid]["similarity_sum"] for rid in sorted(comm_eval["per_robot_final"], key=lambda x: int(x))
    ]
    noncomm_final_similarity = [
        noncomm_eval["per_robot_final"][rid]["similarity_sum"] for rid in sorted(noncomm_eval["per_robot_final"], key=lambda x: int(x))
    ]
    comm_final_cov = [
        comm_eval["per_robot_final"][rid]["entity_coverage"] for rid in sorted(comm_eval["per_robot_final"], key=lambda x: int(x))
    ]
    noncomm_final_cov = [
        noncomm_eval["per_robot_final"][rid]["entity_coverage"] for rid in sorted(noncomm_eval["per_robot_final"], key=lambda x: int(x))
    ]

    for ax, vals_a, vals_b, ylabel, subtitle in [
        (axes[1, 0], comm_final_similarity, noncomm_final_similarity, "Similarity Sum", "Final Snapshot Similarity Sum Distribution"),
        (axes[1, 1], comm_final_cov, noncomm_final_cov, "Entity Coverage", "Final Snapshot Entity Coverage Distribution"),
    ]:
        bp = ax.boxplot(
            [vals_a, vals_b],
            labels=["Communication", "Non-Communication"],
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.8},
        )
        colors = ["#1f77b4", "#ff7f0e"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        ax.set_title(subtitle, fontsize=11)
        ax.set_ylabel(ylabel)
        if ylabel == "Entity Coverage":
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_file, dpi=200)
    plt.close(fig)


def run_bundle(
    gt_name: str,
    gt_facts: list[str],
    comm_data: dict[str, list[str]],
    noncomm_data: dict[str, list[str]],
    model: SentenceTransformer,
    out_dir: Path,
) -> dict:
    gt_embeddings = model.encode(gt_facts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    comm_eval = evaluate_run(comm_data, model, gt_embeddings)
    noncomm_eval = evaluate_run(noncomm_data, model, gt_embeddings)
    comparison = compare_runs(comm_eval, noncomm_eval)

    dashboard_path = out_dir / f"{gt_name}_comm_vs_noncomm_similarity_sum_dashboard.png"
    plot_dashboard(comm_eval, noncomm_eval, dashboard_path, title_prefix=gt_name.upper(), gt_count=len(gt_facts))

    summary = {
        "ground_truth_name": gt_name,
        "ground_truth_count": len(gt_facts),
        "ground_truth_facts": gt_facts,
        "metric": "sum_of_per_gt_best_cosine_similarities",
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "comparison": comparison,
        "plots": {"dashboard": str(dashboard_path)},
    }
    summary_path = out_dir / f"{gt_name}_comm_vs_noncomm_similarity_sum_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{gt_name}] Saved summary: {summary_path}")
    print(f"[{gt_name}] Saved dashboard: {dashboard_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate communication vs non-communication observations using summed GT similarity and entity coverage."
    )
    parser.add_argument("--comm-json", required=True, type=str, help="Path to communication run robots.json")
    parser.add_argument("--noncomm-json", required=True, type=str, help="Path to non-communication run robots.json")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs", help="Directory for outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    comm_data = load_run(Path(args.comm_json).resolve())
    noncomm_data = load_run(Path(args.noncomm_json).resolve())

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    baseline = run_bundle(
        gt_name="gt_baseline",
        gt_facts=GROUND_TRUTH_FACTS,
        comm_data=comm_data,
        noncomm_data=noncomm_data,
        model=model,
        out_dir=out_dir,
    )

    combined = {
        "comm_json": args.comm_json,
        "noncomm_json": args.noncomm_json,
        "model": args.model,
        "evaluations": {
            "gt_baseline": baseline,
        },
    }
    combined_path = out_dir / "comm_vs_noncomm_similarity_sum_combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
