from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# =========================
# Configure inputs/outputs
# =========================
GROUND_TRUTH_JSON = Path(
    "/Users/absera/Capstone/swarm-robotics/pre_assets/ground_truth/ground_truth_facts_148.json"
)
COMM_RUN_JSON = Path(
    "/Users/absera/Capstone/swarm-robotics/output/bg2500_big_comm_2026-04-22_21-44-57/robots_progress.json"
)
NONCOMM_RUN_JSON = Path(
    "/Users/absera/Capstone/swarm-robotics/output/bg2500_big_noncomm_2026-04-23_09-11-06/robots_progress.json"
)
OUTPUT_DIR = Path("/Users/absera/Capstone/swarm-robotics/analysis_outputs")

# =========================
# Evaluation hyperparameters
# =========================
RETRIEVER_MODEL = "all-MiniLM-L6-v2"
VERIFIER_MODEL = "cross-encoder/nli-roberta-base"
TOP_K = 4
RETRIEVER_BATCH_SIZE = 512
VERIFIER_BATCH_SIZE = 512


def split_sentences(text: str) -> list[str]:
    """Fast sentence splitter without NLTK download overhead."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_ground_truth(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        facts = data.get("facts", [])
    elif isinstance(data, list):
        facts = data
    else:
        raise ValueError(f"Unsupported ground-truth format: {path}")
    return [str(x).strip() for x in facts if str(x).strip()]


def load_swarm_progress(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"robots_progress JSON must be an object: {path}")
    cleaned: dict[str, list[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            cleaned[str(k)] = [str(x) for x in v]
    if not cleaned:
        raise ValueError(f"No robot trajectories found in: {path}")
    return cleaned


class SwarmEvaluator:
    """
    Coverage metric (paper-style):
    Cov(sigma, C) = 1/n * sum_j OR_i [l_{j,i} = 1]
    where:
      - j indexes ground-truth claims,
      - i indexes KB sentences from one agent snapshot,
      - l_{j,i} is entailment decision.
    """

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Device: {self.device}")

        self.retriever = SentenceTransformer(RETRIEVER_MODEL, device=self.device)
        self.verifier = CrossEncoder(VERIFIER_MODEL, device=self.device)
        id2label = getattr(self.verifier.model.config, "id2label", {})
        self.entailment_idx = None
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                self.entailment_idx = int(idx)
                break
        if self.entailment_idx is None:
            # Standard for cross-encoder/nli-roberta-base:
            # 0=contradiction, 1=entailment, 2=neutral.
            self.entailment_idx = 1
        print(
            f"[INFO] NLI entailment label index: {self.entailment_idx} "
            f"(id2label={id2label})"
        )

    def calculate_agent_snapshot_coverage(
        self,
        ground_truth: list[str],
        gt_embeddings: torch.Tensor,
        agent_kb_sentences: list[str],
        top_k: int = TOP_K,
    ) -> float:
        if not ground_truth or not agent_kb_sentences:
            return 0.0

        kb_embeddings = self.retriever.encode(
            agent_kb_sentences,
            convert_to_tensor=True,
            batch_size=RETRIEVER_BATCH_SIZE,
            show_progress_bar=False,
        )

        cosine_scores = util.cos_sim(gt_embeddings, kb_embeddings)
        num_claims = len(ground_truth)
        covered_count = 0

        for claim_idx, claim in enumerate(ground_truth):
            top_indices = torch.topk(
                cosine_scores[claim_idx],
                k=min(top_k, len(agent_kb_sentences)),
            ).indices.tolist()
            candidate_pairs = [(agent_kb_sentences[kb_idx], claim) for kb_idx in top_indices]
            logits = self.verifier.predict(
                candidate_pairs,
                batch_size=VERIFIER_BATCH_SIZE,
                show_progress_bar=False,
            )
            pred_labels = np.asarray(logits).argmax(axis=1)
            # Coverage OR condition: claim is covered if any candidate entails it.
            if (pred_labels == self.entailment_idx).any():
                covered_count += 1

        return covered_count / num_claims

    def evaluate_swarm_run(
        self,
        ground_truth: list[str],
        swarm_data: dict[str, list[str]],
        top_k: int = TOP_K,
    ) -> dict:
        robot_ids = sorted(swarm_data.keys(), key=lambda x: int(x))
        num_snapshots = min(len(swarm_data[rid]) for rid in robot_ids)
        if num_snapshots == 0:
            raise ValueError("At least one snapshot is required in swarm data.")

        gt_embeddings = self.retriever.encode(
            ground_truth,
            convert_to_tensor=True,
            batch_size=RETRIEVER_BATCH_SIZE,
            show_progress_bar=False,
        )

        per_snapshot_agent_scores: list[list[float]] = []
        snapshot_mean: list[float] = []
        snapshot_std: list[float] = []

        for snapshot_idx in range(num_snapshots):
            current_scores: list[float] = []
            for robot_id in robot_ids:
                kb_text = swarm_data[robot_id][snapshot_idx]
                kb_sentences = split_sentences(kb_text)
                score = self.calculate_agent_snapshot_coverage(
                    ground_truth=ground_truth,
                    gt_embeddings=gt_embeddings,
                    agent_kb_sentences=kb_sentences,
                    top_k=top_k,
                )
                current_scores.append(score)

            per_snapshot_agent_scores.append(current_scores)
            snapshot_mean.append(float(np.mean(current_scores)))
            snapshot_std.append(float(np.std(current_scores)))
            print(
                f"[snapshot {snapshot_idx:02d}] mean={snapshot_mean[-1]:.4f}, "
                f"std={snapshot_std[-1]:.4f}, robots={len(current_scores)}"
            )

        return {
            "num_robots": len(robot_ids),
            "num_snapshots": num_snapshots,
            "snapshot_mean": snapshot_mean,
            "snapshot_std": snapshot_std,
            "per_snapshot_agent_scores": per_snapshot_agent_scores,
        }


def plot_coverage_progress(
    comm_eval: dict,
    noncomm_eval: dict,
    out_path: Path,
) -> None:
    n = min(comm_eval["num_snapshots"], noncomm_eval["num_snapshots"])
    snapshots = np.arange(n)

    comm_mean = np.array(comm_eval["snapshot_mean"][:n], dtype=np.float32)
    comm_std = np.array(comm_eval["snapshot_std"][:n], dtype=np.float32)
    non_mean = np.array(noncomm_eval["snapshot_mean"][:n], dtype=np.float32)
    non_std = np.array(noncomm_eval["snapshot_std"][:n], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(snapshots, comm_mean, color="#1f77b4", linewidth=2.4, label="Individual + Social Learning (Communication)")
    ax.fill_between(
        snapshots,
        np.clip(comm_mean - comm_std, 0, 1),
        np.clip(comm_mean + comm_std, 0, 1),
        color="#1f77b4",
        alpha=0.16,
    )

    ax.plot(snapshots, non_mean, color="#ff7f0e", linewidth=2.4, label="Individual Learning Only (Non-Communication)")
    ax.fill_between(
        snapshots,
        np.clip(non_mean - non_std, 0, 1),
        np.clip(non_mean + non_std, 0, 1),
        color="#ff7f0e",
        alpha=0.16,
    )

    ax.set_title("Coverage Score Progression (NLI Entailment)")
    ax.set_xlabel("Snapshot Index")
    ax.set_ylabel("Coverage Score (0-1)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.28, linestyle="--", linewidth=0.8)
    ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(GROUND_TRUTH_JSON)
    comm_data = load_swarm_progress(COMM_RUN_JSON)
    noncomm_data = load_swarm_progress(NONCOMM_RUN_JSON)

    print(f"[INFO] Ground-truth claims: {len(ground_truth)}")
    print(f"[INFO] Communication run: {COMM_RUN_JSON}")
    print(f"[INFO] Non-communication run: {NONCOMM_RUN_JSON}")

    evaluator = SwarmEvaluator()
    comm_eval = evaluator.evaluate_swarm_run(ground_truth, comm_data, top_k=TOP_K)
    noncomm_eval = evaluator.evaluate_swarm_run(ground_truth, noncomm_data, top_k=TOP_K)

    plot_path = OUTPUT_DIR / "comm_vs_noncomm_nli_coverage_progress.png"
    plot_coverage_progress(comm_eval, noncomm_eval, plot_path)

    summary = {
        "ground_truth_json": str(GROUND_TRUTH_JSON),
        "comm_run_json": str(COMM_RUN_JSON),
        "noncomm_run_json": str(NONCOMM_RUN_JSON),
        "retriever_model": RETRIEVER_MODEL,
        "verifier_model": VERIFIER_MODEL,
        "top_k": TOP_K,
        "device": evaluator.device,
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "plot_path": str(plot_path),
    }
    summary_path = OUTPUT_DIR / "comm_vs_noncomm_nli_coverage_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved plot: {plot_path}")
    print(f"[DONE] Saved summary: {summary_path}")
    print(
        "[DONE] Final snapshot mean coverage | "
        f"Comm={comm_eval['snapshot_mean'][-1]:.4f}, "
        f"Non-Comm={noncomm_eval['snapshot_mean'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
