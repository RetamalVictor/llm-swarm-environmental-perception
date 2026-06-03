from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize
from scipy import stats
from tqdm import tqdm

# cross-encoder/nli-roberta-base outputs labels in this order:
# 0 = contradiction, 1 = entailment, 2 = neutral
_ENTAILMENT_IDX = 1


def _load_nli_model(model_name: str):
    from sentence_transformers import CrossEncoder

    print(f"Loading NLI model: {model_name}")
    return CrossEncoder(model_name)


def load_ground_truth(path: Path) -> list[str]:
    """
    Ground-truth file can be:
    - plain text paragraph (split into sentences), or
    - JSON list[str], or
    - JSON object with one of: claims, facts, snippets (list[str]).
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"Ground truth file is empty: {path}")

    # Try JSON first.
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            claims = [str(x).strip() for x in payload if str(x).strip()]
            if claims:
                return claims
        if isinstance(payload, dict):
            for key in ("claims", "facts", "snippets"):
                value = payload.get(key)
                if isinstance(value, list):
                    claims = [str(x).strip() for x in value if str(x).strip()]
                    if claims:
                        return claims
    except json.JSONDecodeError:
        pass

    # Fallback: treat as paragraph text.
    claims = [s.strip() for s in sent_tokenize(raw) if s.strip()]
    if not claims:
        raise ValueError(f"Could not parse any claims from ground truth: {path}")
    return claims


def load_run(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Run JSON not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict[str, list[str]] in {path}")
    return data


def compute_coverage_score(summary: str, claims: list[str], nli_model) -> float:
    """
    Fraction of claims that are entailed by at least one summary sentence.
    """
    if not summary or not claims:
        return 0.0

    sentences = sent_tokenize(summary)
    if not sentences:
        return 0.0

    n_sent = len(sentences)
    n_claims = len(claims)

    # claim-major order
    pairs = [(sent, claim) for claim in claims for sent in sentences]
    probs = nli_model.predict(pairs, apply_softmax=True)

    best_label_per_pair = probs.argmax(axis=1).reshape(n_claims, n_sent)
    entailed_per_claim = (best_label_per_pair == _ENTAILMENT_IDX).any(axis=1)
    return float(entailed_per_claim.sum()) / n_claims


def evaluate_run(
    data: dict[str, list[str]],
    claims: list[str],
    nli_model,
    cache: dict[tuple[str, tuple[str, ...]], float],
) -> dict:
    robot_ids = sorted(data.keys(), key=lambda x: int(x))
    if not robot_ids:
        raise ValueError("Run JSON has no robots.")

    n_snapshots = min(len(data[rid]) for rid in robot_ids)
    if n_snapshots == 0:
        raise ValueError("Run JSON robots have no observations.")

    per_snapshot = {
        "coverage_mean": [],
        "coverage_std": [],
    }
    per_robot_final: dict[str, dict[str, float]] = {}

    claims_key = tuple(claims)
    for s in tqdm(range(n_snapshots), desc="Scoring snapshots", leave=False):
        scores = []
        for rid in robot_ids:
            obs = data[rid][s]
            key = (obs, claims_key)
            if key not in cache:
                cache[key] = compute_coverage_score(obs, claims, nli_model)
            score = cache[key]
            scores.append(score)

            if s == n_snapshots - 1:
                per_robot_final[rid] = {"coverage": score}

        per_snapshot["coverage_mean"].append(float(np.mean(scores)))
        per_snapshot["coverage_std"].append(float(np.std(scores)))

    return {
        "num_robots": len(robot_ids),
        "num_snapshots": n_snapshots,
        "claims_count": len(claims),
        "per_snapshot": per_snapshot,
        "per_robot_final": per_robot_final,
    }


def compare_runs(comm_eval: dict, noncomm_eval: dict) -> dict:
    def extract(src: dict) -> list[float]:
        return [
            src["per_robot_final"][rid]["coverage"]
            for rid in sorted(src["per_robot_final"], key=lambda x: int(x))
        ]

    comm_final = extract(comm_eval)
    noncomm_final = extract(noncomm_eval)
    u_stat, p_val = stats.mannwhitneyu(comm_final, noncomm_final, alternative="two-sided")

    return {
        "final_coverage_test": {
            "u_stat": float(u_stat),
            "p_value": float(p_val),
        },
        "means": {
            "comm_final_coverage": float(np.mean(comm_final)),
            "noncomm_final_coverage": float(np.mean(noncomm_final)),
        },
    }


def plot_dashboard(comm_eval: dict, noncomm_eval: dict, out_path: Path, title: str) -> None:
    x = np.arange(1, min(comm_eval["num_snapshots"], noncomm_eval["num_snapshots"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Time-series mean +/- std
    cm = np.array(comm_eval["per_snapshot"]["coverage_mean"][: len(x)])
    cs = np.array(comm_eval["per_snapshot"]["coverage_std"][: len(x)])
    nm = np.array(noncomm_eval["per_snapshot"]["coverage_mean"][: len(x)])
    ns = np.array(noncomm_eval["per_snapshot"]["coverage_std"][: len(x)])

    ax = axes[0]
    ax.plot(x, cm, color="#1f77b4", linewidth=2.2, label="Communication")
    ax.fill_between(x, np.clip(cm - cs, 0, 1), np.clip(cm + cs, 0, 1), color="#1f77b4", alpha=0.16)
    ax.plot(x, nm, color="#ff7f0e", linewidth=2.2, label="Non-Communication")
    ax.fill_between(x, np.clip(nm - ns, 0, 1), np.clip(nm + ns, 0, 1), color="#ff7f0e", alpha=0.16)
    ax.set_xlabel("Observation Snapshot Index")
    ax.set_ylabel("Coverage (entailed claims / total claims)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.legend(frameon=True)
    ax.set_title("Average coverage over time (across robots)")

    # Final snapshot boxplot
    comm_final = [
        comm_eval["per_robot_final"][rid]["coverage"]
        for rid in sorted(comm_eval["per_robot_final"], key=lambda x: int(x))
    ]
    noncomm_final = [
        noncomm_eval["per_robot_final"][rid]["coverage"]
        for rid in sorted(noncomm_eval["per_robot_final"], key=lambda x: int(x))
    ]
    ax = axes[1]
    bp = ax.boxplot(
        [comm_final, noncomm_final],
        labels=["Communication", "Non-Communication"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.8},
    )
    for patch, color in zip(bp["boxes"], ["#1f77b4", "#ff7f0e"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)
    ax.set_title("Final snapshot coverage distribution (per robot)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare communication vs non-communication robot observations using "
            "RoBERTa NLI entailment coverage against provided ground truth."
        )
    )
    parser.add_argument("--ground-truth", required=True, type=str, help="Path to ground truth file (text or JSON).")
    parser.add_argument("--comm-json", required=True, type=str, help="Path to communication robots.json")
    parser.add_argument("--noncomm-json", required=True, type=str, help="Path to non-communication robots.json")
    parser.add_argument(
        "--model",
        type=str,
        default="cross-encoder/nli-deberta-v3-large",
        help="CrossEncoder NLI model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_outputs",
        help="Directory to write dashboard and summary JSON",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth).resolve()
    comm_path = Path(args.comm_json).resolve()
    noncomm_path = Path(args.noncomm_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    claims = load_ground_truth(ground_truth_path)
    comm_data = load_run(comm_path)
    noncomm_data = load_run(noncomm_path)

    nli_model = _load_nli_model(args.model)

    shared_cache: dict[tuple[str, tuple[str, ...]], float] = {}
    print("Evaluating communication run...")
    comm_eval = evaluate_run(comm_data, claims, nli_model, shared_cache)
    print("Evaluating non-communication run...")
    noncomm_eval = evaluate_run(noncomm_data, claims, nli_model, shared_cache)
    comparison = compare_runs(comm_eval, noncomm_eval)

    dashboard_path = output_dir / "comm_vs_noncomm_roberta_two_dashboard.png"
    summary_path = output_dir / "comm_vs_noncomm_roberta_two_summary.json"

    plot_dashboard(
        comm_eval,
        noncomm_eval,
        dashboard_path,
        title="RoBERTa NLI Coverage: Communication vs Non-Communication",
    )

    result = {
        "method": "roberta_nli_coverage",
        "model": args.model,
        "ground_truth_file": str(ground_truth_path),
        "ground_truth_claims": claims,
        "comm_json": str(comm_path),
        "noncomm_json": str(noncomm_path),
        "comm_eval": comm_eval,
        "noncomm_eval": noncomm_eval,
        "comparison": comparison,
        "plots": {"dashboard": str(dashboard_path)},
    }

    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Claims loaded: {len(claims)}")
    print(f"Communication robots: {comm_eval['num_robots']}")
    print(f"Non-communication robots: {noncomm_eval['num_robots']}")
    print("Final mean coverage:")
    print(f"  comm:    {comparison['means']['comm_final_coverage']:.4f}")
    print(f"  noncomm: {comparison['means']['noncomm_final_coverage']:.4f}")
    print("Mann-Whitney U:")
    print(f"  U={comparison['final_coverage_test']['u_stat']:.4f}, p={comparison['final_coverage_test']['p_value']:.6f}")
    print(f"Saved dashboard: {dashboard_path}")
    print(f"Saved summary:   {summary_path}")


if __name__ == "__main__":
    main()
