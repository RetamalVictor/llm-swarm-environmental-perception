"""
Build a comprehensive ground-truth fact list from all PNGs in pre_assets/pngs.

This script uses the same Gemini model and generation settings defined in a run config.
It asks the model for short atomized facts in the same style as the swarm prompts:
    "[Entity] is present." or "[Entity] has [Attribute]."
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from swarm_perception.utils.config import SwarmConfig  # noqa: E402

# One-time script settings (edit these directly if needed).
CONFIG_PATH = PROJECT_ROOT / "configs" / "bg2500-big_comm.yaml"
PNG_DIR = PROJECT_ROOT / "pre_assets" / "pngs" / "10"
OUTPUT_PATH = PROJECT_ROOT / "pre_assets" / "ground_truth" / "ground_truth_facts_10.json"
PER_IMAGE_MAX_FACTS = 10
FINAL_MAX_FACTS = 100


def split_facts(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def normalize_fact(fact: str) -> str:
    lowered = fact.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def deduplicate_facts(facts: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for fact in facts:
        key = normalize_fact(fact)
        if key and key not in seen:
            seen.add(key)
            unique.append(fact)
    return unique


def build_per_image_prompt(max_facts: int) -> str:
    return f"""
You are building a high-recall, factual ground-truth map for a robot-swarm benchmark.
Analyze the provided image and output an extensive list of atomic facts.

STRICT OUTPUT FORMAT:
- Every sentence MUST follow one of:
  1) "[Entity] is present."
  2) "[Entity] has [Attribute]."
- Keep each fact short and independent.
- No conjunctions ("and", "or", "but"), no commas, no compound sentences.
- Prefer concrete observable entities, landmarks, colors, shapes, relative placement cues.
- Include small but useful scene details when visible.
- Do not invent hidden/unseen objects.
- Target up to {max_facts} facts for this image.

Output only one paragraph made of these short fact sentences.
""".strip()


def build_final_merge_prompt(max_facts: int) -> str:
    return f"""
You are consolidating many partial visual fact logs into one final ground-truth set.

TASK:
- Merge all facts below into one unique, de-duplicated, high-coverage fact list.
- Keep as many distinct correct details as possible.

STRICT OUTPUT FORMAT:
- Every sentence MUST follow one of:
  1) "[Entity] is present."
  2) "[Entity] has [Attribute]."
- Keep each fact short and independent.
- No conjunctions ("and", "or", "but"), no commas, no compound sentences.
- Remove duplicates and near-duplicates.
- Do not add facts not present in the input facts.
- Target up to {max_facts} facts.

Output only one paragraph of fact sentences.
""".strip()


def to_generation_config(config_obj: Any) -> genai.GenerationConfig:
    temperature = getattr(config_obj.llm, "temperature", 0.05)
    max_output_tokens = getattr(config_obj.llm, "max_output_tokens", 220)
    max_output_tokens = max(max_output_tokens, 1200)
    return genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def load_pngs(png_dir: Path) -> list[Path]:
    pngs = sorted(png_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG files found in {png_dir}")
    return pngs


def main() -> None:
    load_dotenv()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is missing. Add it to your environment/.env.")

    config = SwarmConfig(CONFIG_PATH).load_config()
    model_name = config.llm.model_name
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    generation_config = to_generation_config(config)

    png_dir = Path(PNG_DIR)
    output_path = Path(OUTPUT_PATH)
    png_paths = load_pngs(png_dir)

    per_image_prompt = build_per_image_prompt(PER_IMAGE_MAX_FACTS)
    per_image_facts: dict[str, list[str]] = {}
    all_facts: list[str] = []

    for image_path in png_paths:
        image = Image.open(image_path)
        response = model.generate_content(
            [per_image_prompt, image],
            generation_config=generation_config,
        )
        text = (response.text or "").strip()
        facts = deduplicate_facts(split_facts(text))
        per_image_facts[image_path.name] = facts
        all_facts.extend(facts)
        print(f"[ground-truth] {image_path.name}: {len(facts)} facts")

    seed_facts = deduplicate_facts(all_facts)
    final_merge_prompt = build_final_merge_prompt(FINAL_MAX_FACTS)
    seed_text = " ".join(seed_facts)
    final_response = model.generate_content(
        f"{final_merge_prompt}\n\n[INPUT FACTS]\n{seed_text}",
        generation_config=generation_config,
    )
    final_text = (final_response.text or "").strip()
    final_facts = deduplicate_facts(split_facts(final_text))

    payload = {
        "model_name": model_name,
        "source_png_dir": str(png_dir),
        "num_source_images": len(png_paths),
        "num_seed_facts": len(seed_facts),
        "num_final_facts": len(final_facts),
        "facts": final_facts,
        "facts_by_image": per_image_facts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[ground-truth] wrote {len(final_facts)} facts to {output_path}")


if __name__ == "__main__":
    main()
