# Chinese Main Experiment — ZH150

This directory contains 150 Chinese prompts with frozen English translations and 2,700 image-level evaluation records for six pipelines (SDXL, RPG, Kolors, RPG+Kolors, RAGD and BiReG) and three generation seeds (1234, 2468, 42). Each pipeline has 450 evaluated images.

## Files

- `data/prompts.frozen.jsonl`: prompt IDs, original Chinese text (`prompt`) and frozen English translations (`translation_en`).
- `data/scores.jsonl`: one record per prompt, method and seed, with fields `prompt_id`, `method`, `seed`, `testpoint` and `score`. The two arrays `testpoint` and `score` contain aligned checkpoint labels and binary scores (1 = satisfied, 0 = not satisfied).
- `summarize.py`: checks complete prompt/method/seed coverage and consistent checkpoint labels, then reproduces dimension means and standard deviations.

## Run

```bash
python summarize.py
```

Python standard library only; no GPU or API required. The script creates `results/summary.json` (full precision on the 0–1 scale, including checkpoint counts and subdimensions) and `results/table.md` (the ten main dimensions, expressed as percentages). Run it from any working directory. Keep the previously released prompt file at the path above.

## Scores and aggregation

The ten main dimensions are attribute, action, relation, entity layout, composite, style, world knowledge, grammar, logical reasoning and text rendering. Archived Chinese checkpoint labels are preserved. Labels with a hyphen also contribute to the main dimension before the hyphen. A prompt can contribute multiple checkpoints and multiple dimensions; the 150 prompts are not ten mutually exclusive groups of equal size.

For each pipeline, dimension and seed, divide the number of satisfied checkpoints by the total number of evaluated checkpoints in that dimension. Then report the mean and sample standard deviation (`ddof=1`) of the three seed-level accuracies. Higher accuracy is better. This is checkpoint-weighted aggregation, not an unweighted average of per-image accuracies. SD describes generation-seed variation with one archived evaluation per image; it does not measure repeated evaluator-call variability. These scores are not human 1–5 ratings. The script aggregates stored scores and does not rerun the automatic evaluator.

## Language paths and evaluation scope

Kolors, RPG+Kolors and BiReG use the original Chinese prompts; SDXL, RPG and RAGD use the frozen English translations. All six pipelines are evaluated against the same original Chinese prompts and checkpoint labels. Differences across backbones and language paths cannot be attributed solely to regional planning.

The six RPG images for `UGB-ZH-110` and `UGB-ZH-278` across the three seeds use base-generation fallback. They remain in the RPG scores and do not count as successful regional planning.

The archived 150-prompt set has a post-hoc selection history based on earlier scores. Results describe this evaluation set and should not be interpreted as an unbiased estimate over the full benchmark. This release covers prompts and archived automatic scores; the separate human-evaluation ratings are not used in these aggregates.
