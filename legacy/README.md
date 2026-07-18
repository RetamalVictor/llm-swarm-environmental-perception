# legacy/

Retired code from the original LLM-based pipeline, kept for reference only.
**Not maintained, not imported by the live package.**

- `build_ground_truth.py` — generated fuzzy text "ground truth" by asking Gemini
  to describe object PNGs. Replaced by exact geometric ground truth
  (`layout.json` persisted by the background generator) in milestone M2. Its
  hardcoded paths are stale; do not expect it to run as-is.
