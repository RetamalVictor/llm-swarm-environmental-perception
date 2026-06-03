from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NLI helpers
# ---------------------------------------------------------------------------

# Index layout returned by cross-encoder/nli-deberta-v3-large after softmax:
# 0 = contradiction, 1 = entailment, 2 = neutral
_ENTAILMENT_IDX = 1


def _load_nli_model():
    from sentence_transformers import CrossEncoder

    print("Loading NLI model (cross-encoder/nli-deberta-v3-large) …")
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")


def compute_coverage_score(
    summary: str,
    claims: list[str],
    nli_model,
) -> float:
    """
    Returns the fraction of *claims* that are entailed by the *summary*.

    For each claim the model checks every sentence in the summary; a claim
    is considered covered if at least one sentence entails it.

    Returns 0.0 for empty summaries or claim lists.
    """
    if not summary or not claims:
        return 0.0

    sentences = sent_tokenize(summary)
    if not sentences:
        return 0.0

    n_sent = len(sentences)
    n_claims = len(claims)

    # Build (premise=sentence, hypothesis=claim) pairs, claim-major order
    pairs = [
        (sent, claim)
        for claim in claims
        for sent in sentences
    ]

    probs = nli_model.predict(pairs, apply_softmax=True)

    # Reshape into [n_claims, n_sent] then take max over sentences per claim
    entailment_probs = probs[:, _ENTAILMENT_IDX].reshape(n_claims, n_sent)
    # A claim is entailed if the best sentence has entailment as the argmax
    best_label_per_pair = probs.argmax(axis=1).reshape(n_claims, n_sent)
    entailed_per_claim = (best_label_per_pair == _ENTAILMENT_IDX).any(axis=1)

    return float(entailed_per_claim.sum()) / n_claims


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_run(run_folder: Path) -> tuple[dict, dict]:
    """Return (experiment_data, metadata) for a run folder."""
    exp_path = run_folder / "experiment.json"
    meta_path = run_folder / "metadata.json"

    if not exp_path.exists():
        raise FileNotFoundError(f"experiment.json not found in {run_folder}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {run_folder}")

    exp_data = json.loads(exp_path.read_text())
    metadata = json.loads(meta_path.read_text())
    return exp_data, metadata


def get_claims(metadata: dict) -> list[str]:
    """
    Return the ground-truth snippets that act as claims for coverage scoring.

    Reads ``ground_truth_key`` from *metadata* (e.g. ``"career_fair_low"``),
    looks up the matching entry in ``GROUND_TRUTH_LIBRARY`` by its ``name``
    field, and returns that entry's ``snippets`` list.
    """
    import sys
    from pathlib import Path as _Path

    # Make sure constants.py is importable regardless of cwd
    _root = _Path(__file__).parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from constants import GROUND_TRUTH_LIBRARY

    gt_key = metadata.get("ground_truth_key", "")
    for entry in GROUND_TRUTH_LIBRARY.values():
        if entry.get("name") == gt_key:
            snippets = entry.get("snippets", [])
            if not snippets:
                raise ValueError(
                    f"Ground truth entry '{gt_key}' has no snippets."
                )
            return list(snippets)

    raise ValueError(
        f"No entry with name='{gt_key}' found in GROUND_TRUTH_LIBRARY. "
        f"Available names: {[v['name'] for v in GROUND_TRUTH_LIBRARY.values()]}"
    )


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def process_run(
    run_folder: Path,
    nli_model,
    cache: dict | None = None,
) -> tuple[list[int], np.ndarray, list[str], str]:
    """
    Compute per-agent entailment coverage scores for every snapshot.

    Parameters
    ----------
    run_folder : Path
        Path to a single experiment run directory.
    nli_model :
        Loaded CrossEncoder NLI model.
    cache : dict, optional
        Mutable dict used to avoid recomputing scores for identical summaries.
        Pass the same dict across multiple calls to share cache between runs.

    Returns
    -------
    timestamps : list[int]
        Snapshot timestamps in seconds.
    all_scores : np.ndarray, shape (n_agents, n_timestamps)
        Coverage score for every agent at every timestamp.
    agent_ids : list[str]
        Agent identifier strings, in the same row order as *all_scores*.
    label : str
        Human-readable label for the run (used in plot titles/legends).
    """
    if cache is None:
        cache = {}

    exp_data, metadata = load_run(run_folder)
    claims = get_claims(metadata)
    claims_key = tuple(claims)

    timestamps = [int(t) for t in exp_data["timestamps"]]
    ts_to_idx = {t: i for i, t in enumerate(timestamps)}

    agents = exp_data["agents"]
    n_agents = len(agents)
    all_scores = np.zeros((n_agents, len(timestamps)))
    agent_ids: list[str] = []

    for agent_idx, (agent_id, agent_data) in enumerate(
        tqdm(agents.items(), desc=f"{run_folder.name}", leave=True)
    ):
        agent_ids.append(agent_id)
        summaries: dict[str, str] = agent_data.get("summaries", {})

        for ts_str, summary in summaries.items():
            ts = int(ts_str)
            if ts not in ts_to_idx:
                continue

            cache_key = (summary, claims_key)
            if cache_key not in cache:
                cache[cache_key] = compute_coverage_score(summary, claims, nli_model)

            all_scores[agent_idx, ts_to_idx[ts]] = cache[cache_key]

    label = f"{run_folder.parent.name}/{run_folder.name}"
    return timestamps, all_scores, agent_ids, label


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------



def plot_runs(
    run_results: list[tuple[list[int], np.ndarray, list[str], str]],
    output_path: Path | str | None = None,
    title: str = "Entailment Coverage Score Over Time",
) -> plt.Figure:
    """
    Plot per-agent coverage trajectories and the swarm average.

    For a single run a single axes is produced.  For multiple runs each run
    gets its own subplot (one column per run) so agent lines from different
    runs don't overlap.

    Parameters
    ----------
    run_results :
        List of (timestamps, all_scores, agent_ids, label) tuples as returned
        by :func:`process_run`, one entry per run.
    output_path :
        If provided, save the figure here instead of showing it interactively.
    title :
        Overall figure title.

    Returns
    -------
    matplotlib Figure
    """
    n_runs = len(run_results)
    fig, axes = plt.subplots(
        1, n_runs,
        figsize=(max(11, 10 * n_runs), 5),
        sharey=True,
        squeeze=False,
    )

    color_cycle = plt.get_cmap("tab20").colors

    for col, (timestamps, all_scores, agent_ids, run_label) in enumerate(run_results):
        ax = axes[0, col]
        ts = np.array(timestamps)
        n_agents = all_scores.shape[0]

        avg_raw = all_scores.mean(axis=0)

        # Zoom x-axis to where the average plateaus, keep a short tail
        plateau_idx = np.argmax(avg_raw >= avg_raw[-1] * 0.995)
        tail = max(5, int(0.10 * (len(ts) - plateau_idx)))
        x_max = float(ts[min(plateau_idx + tail, len(ts) - 1)])

        # One thin, semi-transparent line per agent, each with its own color
        for agent_idx in range(n_agents):
            color = color_cycle[agent_idx % len(color_cycle)]
            ax.plot(
                ts, all_scores[agent_idx],
                linewidth=1.3, alpha=0.55, color=color,
                label=f"agent {agent_ids[agent_idx]}" if n_agents <= 15 else None,
            )

        # Bold black average line on top
        ax.plot(
            ts, avg_raw,
            linewidth=2.8, color="black", label="average", zorder=5,
        )

        ax.set_xlim(ts[0], x_max)
        ax.set_ylim(-0.02, 1.08)
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_title(run_label, fontsize=11, pad=6)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.7)

    axes[0, 0].set_ylabel("Coverage score  (entailed claims / total claims)", fontsize=11)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {output_path}")
    else:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def main(
    run_folders: list[str | Path],
    output_path: str | Path | None = None,
    title: str | None = None,
    nli_model=None,
) -> list[tuple[list[int], np.ndarray, list[str], str]]:
    """
    Process one or more experiment run folders and plot coverage scores.

    Parameters
    ----------
    run_folders :
        List of paths to experiment run directories.
    output_path :
        Path to save the output plot. Defaults to
        ``<run_folder>/coverage_over_time.png`` when a single run is given.
    title :
        Custom plot title.
    nli_model :
        Pre-loaded CrossEncoder model. If None, the model is loaded here.

    Returns
    -------
    list of (timestamps, all_scores, agent_ids, label) – one entry per run.
    """
    run_folders = [Path(f) for f in run_folders]

    if nli_model is None:
        nli_model = _load_nli_model()

    run_results = []
    cache: dict = {}

    for folder in run_folders:
        print(f"\nProcessing {folder} …")
        result = process_run(folder, nli_model, cache)
        run_results.append(result)

    # Resolve output path
    out = output_path
    if out is None and len(run_folders) == 1:
        out = run_folders[0] / "coverage_over_time.png"

    plot_runs(
        run_results,
        output_path=out,
        title=title or "Entailment Coverage Score Over Time",
    )

    return run_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute NLI-based entailment coverage scores for experiment runs "
            "and plot the average score over time."
        )
    )
    parser.add_argument(
        "run_folders",
        nargs="+",
        metavar="RUN_FOLDER",
        help="One or more experiment run folder paths (e.g. experiments/baseline_social/run_0042).",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="PATH",
        default=None,
        help=(
            "Output path for the plot image (PNG). "
            "Defaults to <run_folder>/coverage_over_time.png for a single run."
        ),
    )
    parser.add_argument(
        "--title",
        "-t",
        metavar="TITLE",
        default=None,
        help="Custom title for the plot.",
    )
    args = parser.parse_args()
    main(args.run_folders, output_path=args.output, title=args.title)
