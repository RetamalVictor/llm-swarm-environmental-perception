import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

# One-time defaults (edit as needed).
GROUND_TRUTH_JSON = Path(
    # "/Users/absera/Capstone/swarm-robotics/pre_assets/ground_truth/ground_truth_facts_old.json"
    "/Users/absera/Capstone/swarm-robotics/pre_assets/ground_truth/ground_truth_facts_197.json"
)
COMM_RUN_JSON = Path(
    "/Users/absera/Capstone/swarm-robotics/output/bg2500-20_big_comm_2026-04-23_14-09-50/robots.json"
)
NONCOMM_RUN_JSON = Path(
    "/Users/absera/Capstone/swarm-robotics/output/bg2500-20_big_noncomm_2026-04-23_14-09-54/robots.json"
)
OUTPUT_DIR = Path("/Users/absera/Capstone/swarm-robotics/analysis_outputs")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COSINE_THRESHOLD = 0.62


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a / a_norm) @ (b / b_norm).T


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
    gt_embeddings: np.ndarray,
    model: SentenceTransformer,
    threshold: float,
) -> dict[str, float]:
    facts = split_sentences(observation)
    if not facts:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    obs_embeddings = model.encode(
        facts, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)
    sims = cosine_matrix(obs_embeddings, gt_embeddings)
    if sims.size == 0:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    best_per_gt = sims.max(axis=0)
    best_per_obs = sims.max(axis=1)

    recall = float((best_per_gt >= threshold).sum() / len(best_per_gt)) if len(best_per_gt) else 0.0
    precision = float((best_per_obs >= threshold).sum() / len(best_per_obs)) if len(best_per_obs) else 0.0
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = float(2 * recall * precision / (recall + precision))

    return {"recall": recall, "precision": precision, "f1": f1}


def evaluate_run(
    run_data: dict[str, list[str]],
    model: SentenceTransformer,
    gt_embeddings: np.ndarray,
    threshold: float,
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
            scores = score_observation(obs, gt_embeddings, model, threshold)
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
        ax.fill_between(epochs, np.clip(cm - cs, 0, 1), np.clip(cm + cs, 0, 1), color="#1f77b4", alpha=0.16)
        ax.plot(epochs, nm, color="#ff7f0e", linewidth=2.4, label="Non-Communication")
        ax.fill_between(epochs, np.clip(nm - ns, 0, 1), np.clip(nm + ns, 0, 1), color="#ff7f0e", alpha=0.16)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
        ax.set_title(f"{ylabel} Progress")

    axes[0].legend(frameon=True)
    fig.suptitle("Communication vs Non-Communication Ground-Truth Alignment", fontsize=13, fontweight="bold")
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gt_facts = load_ground_truth(GROUND_TRUTH_JSON)
    comm_data = load_progress(COMM_RUN_JSON)
    noncomm_data = load_progress(NONCOMM_RUN_JSON)

    print(f"Loaded ground truth facts: {len(gt_facts)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    gt_embeddings = model.encode(
        gt_facts, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)

    comm_eval = evaluate_run(comm_data, model, gt_embeddings, COSINE_THRESHOLD)
    noncomm_eval = evaluate_run(noncomm_data, model, gt_embeddings, COSINE_THRESHOLD)

    plot_path = OUTPUT_DIR / "comm_vs_noncomm_ground_truth_cosine_progress.png"
    plot_progress(comm_eval, noncomm_eval, plot_path)

    comm_final = final_means(comm_eval)
    noncomm_final = final_means(noncomm_eval)
    delta = {
        "recall": comm_final["recall"] - noncomm_final["recall"],
        "precision": comm_final["precision"] - noncomm_final["precision"],
        "f1": comm_final["f1"] - noncomm_final["f1"],
    }

    summary = {
        "ground_truth_json": str(GROUND_TRUTH_JSON),
        "ground_truth_fact_count": len(gt_facts),
        "comm_run_json": str(COMM_RUN_JSON),
        "noncomm_run_json": str(NONCOMM_RUN_JSON),
        "embedding_model": EMBEDDING_MODEL,
        "cosine_threshold": COSINE_THRESHOLD,
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "final_means": {
            "communication": comm_final,
            "non_communication": noncomm_final,
            "delta_comm_minus_noncomm": delta,
        },
        "plot": str(plot_path),
    }
    summary_path = OUTPUT_DIR / "comm_vs_noncomm_ground_truth_cosine_summary.json"
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
