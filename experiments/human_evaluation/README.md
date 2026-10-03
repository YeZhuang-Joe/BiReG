# Human Evaluation — EN60 / ZH29

This directory provides the submitted V4 human-rating data and statistical reproduction for 60 English and 29 Chinese prompts, six pipelines (SDXL, RPG, Kolors, RPG+Kolors, RAGD and BiReG), and 534 evaluated images. There are 21 anonymous evaluator codes and three ratings per image: 1,602 image-level assessments and 3,834 applicable dimension scores. Image files are not included.

## Files

- `data/prompts.jsonl`: task and prompt IDs, original prompt text, Chinese-to-English generation translations where applicable, generation seed and sampling stratum.
- `data/ratings.jsonl`: one record per evaluator and image, containing anonymous evaluator/group codes, task/image/prompt IDs, method, seed and the three dimension scores. Scores and their evaluator/image associations are preserved from the submitted V4 files. Timestamps, free-text notes, collection metadata and server paths are omitted from this public export.
- `reproduce.py` and `requirements.txt`: statistical reproduction using NumPy.
- `results/statistics.json`: full-precision results, paired differences, agreement coefficients and sample counts.
- `results/table.md`: readable tables, including inter-rater agreement.

## Run

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

Use Python 3.11 or later. No GPU, images or API calls are required. The script resolves files relative to its own directory and overwrites the two result files. It validates coverage, score ranges, applicability and evaluator assignments before aggregation.

## Rating dimensions and aggregation

The dimensions are attribute alignment (`attribute`), spatial alignment (`spatial`) and visual quality (`quality`). Scores range from 1 to 5, with higher values indicating better performance. A `null` attribute or spatial score means that the dimension is not applicable; it is excluded rather than treated as zero. Visual quality is applicable to every image.

For each image and dimension, average its three evaluator scores. Report the mean and sample standard deviation (`ddof=1`) of these image means separately for each language and method. The applicable prompt counts per method are 36/34/60 for English and 27/27/29 for Chinese (attribute/spatial/quality). This SD measures variation across evaluated images, not the three-generation-seed SD used in the automatic main-experiment tables.

The 95% percentile confidence intervals use 20,000 paired prompt-level bootstrap resamples within the archived sampling strata (seven English and two Chinese), with random seed 20260930. All six methods are resampled together. Paired differences are BiReG minus each baseline. Intervals are pointwise, without multiplicity adjustment, and condition on the observed prompts, fixed images and assigned evaluator panel.

Inter-rater agreement uses ordinal Krippendorff's alpha, retaining all 21 evaluator identities and treating unassigned evaluator/image cells as missing. The implementation uses the ordinal coincidence-matrix formula; its outputs were checked against the archived krippendorff 0.8.2 results. Agreement is reported separately for each language and dimension across all six methods. Higher alpha indicates stronger agreement; alpha is distinct from the 1–5 performance scores.

## Evaluation scope

Each prompt contributes one fixed image per method. English seeds 2026/3407/5678 each cover 20 prompts; Chinese seeds 1234/2468/42 cover 10/10/9 prompts. Fifteen evaluator codes cover 13 prompts each and six cover 12 each. Task HZ012 (`UGB-ZH-278`) was excluded across all six methods because of RPG base-generation fallback, without replacement.

The evaluation sample comes from previously post-hoc-selected pools; these results do not establish unbiased performance over the full benchmarks. The archive retains AI-assisted applicability annotations, without a recorded completed human confirmation of that annotation step.

This release reproduces the submitted V4 values. Earlier V3 and V4 revisions changed BiReG Chinese quality and attribute scores respectively; the basis for those changes and the discrepancy between recorded rating dates and the materials version are not documented in the supplied archive. Numerical reproduction and agreement coefficients do not authenticate independent rating collection.
