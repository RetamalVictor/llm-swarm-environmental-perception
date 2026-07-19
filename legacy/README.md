# legacy/

Retired code from the original LLM-based pipeline, kept for reference only.
**Not maintained, not imported by the live package.**

- `build_ground_truth.py` — generated fuzzy text "ground truth" by asking Gemini
  to describe object PNGs. Replaced by exact geometric ground truth
  (`layout.json` persisted by the background generator) in milestone M2. Its
  hardcoded paths are stale; do not expect it to run as-is.
- `plot_cosine_experiment_averages.py` — scored old experiment runs against the
  LLM-era text ground truth with sentence-transformers cosine similarity. Its
  dependency stack (sentence-transformers, bert-score, nltk, transformers) was
  removed from the `eval` extra when the text metrics were retired; install
  sentence-transformers manually if you ever need to rerun it.
- `ground_truth_148.json` / `ground_truth_21.json` — LLM-era (Gemini) ground-truth
  fact lists produced by `build_ground_truth.py`, kept only for historical
  comparison against old experiment runs. Note: `ground_truth_21.json` carries
  metadata copy-pasted from the 148 file — its `num_final_facts` says 148 (and
  `facts_by_image` matches the 148 file) but its actual `facts` list differs
  (19 entries).
