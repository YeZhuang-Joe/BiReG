# English Main Experiment — EN1920

This directory contains 1,920 English prompts and 34,560 image scores for six pipelines (SDXL, RPG, Kolors, RPG+Kolors, RAGD and BiReG) and three generation seeds (2026, 3407, 5678). The six categories each contain 320 prompts: color, shape, texture, spatial, non_spatial and complex.

## Files

- `data/prompts.frozen.jsonl`: prompt IDs and English prompt text.
- `data/scores.jsonl`: one record per prompt, method and seed, with fields `prompt_id`, `method`, `seed`, `category` and `score`.
- `summarize.py`: checks complete prompt/method/seed coverage and reproduces category means and standard deviations.

## Run

```bash
python summarize.py
```

Python standard library only; no GPU or API required. The script creates `results/summary.json` (full precision) and `results/table.md` (readable table). Run it from any working directory. Keep the previously released prompt file at the path above.

## Scores and aggregation

For each pipeline and category, average the 320 image scores separately for each seed, then report the mean and sample standard deviation (`ddof=1`) of those three seed means. SD therefore describes generation-seed variation, not variation across individual prompts or repeated planning. Scores remain on their archived scales; they are not human 1–5 ratings.

Color, shape and texture use the retained BLIP-VQA scores, spatial uses the retained object-detection score, and non_spatial uses CLIP similarity. Complex uses the retained experiment adapter: the arithmetic mean of attribute/spatial, attribute/CLIP, or attribute/spatial/CLIP branches according to the archived complex subtype. It is not the unmodified official complex metric. This script aggregates stored scores; it does not rerun the automatic evaluators.

