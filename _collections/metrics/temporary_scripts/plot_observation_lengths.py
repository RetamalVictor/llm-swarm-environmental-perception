import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def split_tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_run(path: Path) -> dict[str, list[str]]:
    with open(path, "r") as f:
        return json.load(f)


def per_snapshot_lengths(data: dict[str, list[str]]) -> dict:
    robot_ids = sorted(data.keys(), key=lambda x: int(x))
    snapshots = min(len(data[r]) for r in robot_ids)
    words_mean = []
    words_std = []
    tokens_mean = []
    tokens_std = []
    sentences_mean = []
    sentences_std = []

    for s in range(snapshots):
        words = []
        tokens = []
        sentences = []
        for rid in robot_ids:
            obs = data[rid][s]
            word_count = len(obs.split())
            token_count = len(split_tokens(obs))
            sentence_count = len(split_sentences(obs))
            words.append(word_count)
            tokens.append(token_count)
            sentences.append(sentence_count)

        words_mean.append(float(np.mean(words)))
        words_std.append(float(np.std(words)))
        tokens_mean.append(float(np.mean(tokens)))
        tokens_std.append(float(np.std(tokens)))
        sentences_mean.append(float(np.mean(sentences)))
        sentences_std.append(float(np.std(sentences)))

    return {
        "num_robots": len(robot_ids),
        "num_snapshots": snapshots,
        "words_mean": words_mean,
        "words_std": words_std,
        "tokens_mean": tokens_mean,
        "tokens_std": tokens_std,
        "sentences_mean": sentences_mean,
        "sentences_std": sentences_std,
    }


def plot_lengths(lengths: dict, output_png: Path, title: str) -> None:
    x = np.arange(1, lengths["num_snapshots"] + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    wm = np.array(lengths["words_mean"])
    ws = np.array(lengths["words_std"])
    tm = np.array(lengths["tokens_mean"])
    ts = np.array(lengths["tokens_std"])

    axes[0].plot(x, wm, color="#2ca02c", linewidth=2.4, label="Average Words")
    axes[0].fill_between(x, np.clip(wm - ws, 0, None), wm + ws, color="#2ca02c", alpha=0.18)
    axes[0].plot(x, tm, color="#9467bd", linewidth=2.2, label="Average Tokens")
    axes[0].fill_between(x, np.clip(tm - ts, 0, None), tm + ts, color="#9467bd", alpha=0.16)
    axes[0].set_xlabel("Observation Snapshot Index")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Observation Length Over Snapshots")
    axes[0].grid(alpha=0.3, linestyle="--")
    axes[0].legend()

    sm = np.array(lengths["sentences_mean"])
    ss = np.array(lengths["sentences_std"])
    axes[1].plot(x, sm, color="#1f77b4", linewidth=2.4, label="Average Sentences")
    axes[1].fill_between(x, np.clip(sm - ss, 0, None), sm + ss, color="#1f77b4", alpha=0.18)
    axes[1].set_xlabel("Observation Snapshot Index")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sentence Count Over Snapshots")
    axes[1].grid(alpha=0.3, linestyle="--")
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot average observation word/token length from a robots.json file.")
    parser.add_argument("--json", required=True, type=str, help="Path to robots.json")
    parser.add_argument("--output-dir", type=str, default="analysis_outputs", help="Directory for output files")
    parser.add_argument("--name", type=str, default="observation_lengths", help="Output file name prefix")
    args = parser.parse_args()

    json_path = Path(args.json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_run(json_path)
    lengths = per_snapshot_lengths(data)

    png_path = output_dir / f"{args.name}.png"
    summary_path = output_dir / f"{args.name}.json"
    plot_lengths(lengths, png_path, title=f"Observation Length Diagnostics: {json_path.parent.name}")

    payload = {
        "json_path": str(json_path),
        "lengths": lengths,
        "plot": str(png_path),
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved plot: {png_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
