from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer


# Colab-friendly constants:
# - Place comm_robots.json and noncomm_robots.json in the same directory as this script/notebook.
# - Uses GPU automatically when available.
GROUND_TRUTH_FACTS = [
    "A mountain with snowy peaks is present.",
    "A stone bridge is present.",
    "A blue river is present.",
    "A large green tree is present.",
    "A dark volcano with orange lava is present.",
    "A red castle is present.",
    "A blue moat surrounds the castle.",
    "A green hedge maze is present.",
    # "A group of four people is present.",
    "A gray parking lot is present.",
    "Six cars are present in the parking lot.",
    "A football stadium is present.",
    "A windmill is present.",
    "A fenced graveyard with tombstones is present.",
    "A colorful hot air balloon is present.",
    "A dense patch of trees is present.",
    "A blue stone fountain is present.",
    # "A fenced zoo enclosure is present.",
    "A giraffe is present in the enclosure.",
    "An elephant is present in the enclosure.",
    "A lion is present in the enclosure.",
]

COSINE_MODEL_NAME = "all-MiniLM-L6-v2"
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
COSINE_THRESHOLD = 0.60

COMM_JSON = "comm_robots.json"
NONCOMM_JSON = "noncomm_robots.json"
PLOT_OUT = "sanity_final_obs_comm_vs_noncomm.png"
ENTAILMENT_INDEX = 1  # contradiction=0, entailment=1, neutral=2


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = text.replace("\n", " ").split(".")
    return [p.strip() + "." for p in parts if p.strip()]


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1e-12
    b_norm[b_norm == 0] = 1e-12
    return (a / a_norm) @ (b / b_norm).T


def extract_last_observations(run_data: dict) -> list[str]:
    last_obs: list[str] = []
    for robot_id, timeline in run_data.items():
        if not isinstance(timeline, list) or not timeline:
            continue
        obs = str(timeline[-1]).strip()
        if obs:
            last_obs.append(obs)
    return last_obs


def cosine_gt_coverage(
    observation: str,
    gt_embeddings: np.ndarray,
    st_model: SentenceTransformer,
    threshold: float,
) -> float:
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


def entailment_gt_coverage(
    observation: str, gt_facts: list[str], nli_model: CrossEncoder
) -> float:
    sentences = [s.strip() for s in split_sentences(observation) if s.strip()]
    if not sentences:
        return 0.0

    pairs = [(sent, claim) for claim in gt_facts for sent in sentences]
    logits = nli_model.predict(pairs, batch_size=64, show_progress_bar=False)
    preds = np.argmax(np.asarray(logits), axis=1)
    mat = preds.reshape(len(gt_facts), len(sentences))
    mask = (mat == ENTAILMENT_INDEX).any(axis=1).tolist()
    return float(sum(mask) / len(mask)) if mask else 0.0


def plot_combined(
    comm_cos: list[float],
    noncomm_cos: list[float],
    comm_ent: list[float],
    noncomm_ent: list[float],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data = [comm_cos, noncomm_cos, comm_ent, noncomm_ent]
    positions = [1, 2, 4, 5]
    labels = ["Comm", "Non-Comm", "Comm", "Non-Comm"]
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]

    bp = ax.boxplot(
        data,
        positions=positions,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.6},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Coverage (0-1)")
    ax.set_title("Final Observation Coverage: Cosine vs Entailment")
    ax.axvline(3, color="gray", linestyle="--", alpha=0.5)
    ax.text(1.5, 1.02, "Cosine Similarity", ha="center", va="bottom")
    ax.text(4.5, 1.02, "Entailment Coverage", ha="center", va="bottom")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    gt_facts = GROUND_TRUTH_FACTS
    if not gt_facts:
        raise RuntimeError("GROUND_TRUTH_FACTS is empty.")

    comm_path = Path(COMM_JSON).resolve()
    noncomm_path = Path(NONCOMM_JSON).resolve()
    if not comm_path.exists():
        raise FileNotFoundError(f"Missing file: {comm_path}")
    if not noncomm_path.exists():
        raise FileNotFoundError(f"Missing file: {noncomm_path}")

    print(f"Loading cosine model: {COSINE_MODEL_NAME}")
    st_model = SentenceTransformer(COSINE_MODEL_NAME, device=device)
    print(f"Loading entailment model: {NLI_MODEL_NAME}")
    nli_model = CrossEncoder(NLI_MODEL_NAME, device=device)
    gt_embeddings = st_model.encode(
        gt_facts, convert_to_numpy=True, show_progress_bar=False
    ).astype(np.float32)

    comm_cos: list[float] = []
    noncomm_cos: list[float] = []
    comm_ent: list[float] = []
    noncomm_ent: list[float] = []

    with comm_path.open("r", encoding="utf-8") as f:
        comm_data = json.load(f)
    with noncomm_path.open("r", encoding="utf-8") as f:
        noncomm_data = json.load(f)

    comm_final_observations = extract_last_observations(comm_data)
    noncomm_final_observations = extract_last_observations(noncomm_data)

    for obs in comm_final_observations:
        comm_cos.append(cosine_gt_coverage(obs, gt_embeddings, st_model, COSINE_THRESHOLD))
        comm_ent.append(entailment_gt_coverage(obs, gt_facts, nli_model))

    for obs in noncomm_final_observations:
        noncomm_cos.append(
            cosine_gt_coverage(obs, gt_embeddings, st_model, COSINE_THRESHOLD)
        )
        noncomm_ent.append(entailment_gt_coverage(obs, gt_facts, nli_model))

    if not (comm_cos and noncomm_cos and comm_ent and noncomm_ent):
        raise RuntimeError(
            "Insufficient data to plot. Check comm_robots.json and noncomm_robots.json."
        )

    plot_out = Path(PLOT_OUT).resolve()
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    plot_combined(comm_cos, noncomm_cos, comm_ent, noncomm_ent, plot_out)

    print(f"Saved plot: {plot_out}")
    print(
        "Counts | "
        f"comm={len(comm_cos)} observations, "
        f"noncomm={len(noncomm_cos)} observations"
    )
    print(
        "Means  | "
        f"cosine(comm={np.mean(comm_cos):.4f}, noncomm={np.mean(noncomm_cos):.4f}) "
        f"entailment(comm={np.mean(comm_ent):.4f}, noncomm={np.mean(noncomm_ent):.4f})"
    )


if __name__ == "__main__":
    main()

