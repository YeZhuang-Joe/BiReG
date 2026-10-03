# Human Evaluation — EN60 / ZH29

## 1. Evaluation design, rating rules and results

### Evaluation organization

The submitted data cover 60 English and 29 Chinese prompts, six pipelines (SDXL, RPG, Kolors, RPG+Kolors, RAGD and BiReG), and 534 images. Each prompt contributes one fixed image per method. There are 21 anonymous evaluator codes arranged in seven groups of three, with three ratings per image: 1,602 image-level assessments and 3,834 applicable dimension scores. Fifteen evaluator codes cover 13 prompts each and six cover 12 each.

The supplied participant materials use individual offline rating pages, anonymous image codes and evaluator-specific presentation orders. Participants are instructed to read the original English or Chinese prompt, inspect the image at its original size, and score independently without discussing images or using AI to rate them. The instructions specify voluntary participation, an introduction to the scale and practice on examples outside the evaluation sample; these are documented instructions, not independent verification of their administration.

### Rating rules

Three dimensions are scored separately: attribute binding (`attribute`), spatial alignment (`spatial`) and visual quality (`quality`). Attribute binding concerns whether specified colors, sizes, shapes, materials and textures belong to the correct objects. Spatial alignment concerns explicitly requested relations and positions. Visual quality concerns clarity, structural plausibility, coherence and artifacts in the requested style.

| Score | Attribute binding | Spatial alignment | Visual quality |
|---|---|---|---|
| 5 | Specified attributes correct, with no evident binding errors or omissions | Specified relations correct, with no evident position errors | Clear, natural and coherent, with no evident artifacts |
| 4 | Mostly correct, with minor local deviations | Main relations correct, with minor position deviations | Good, with minor local flaws |
| 3 | Partly correct; at least one key attribute incorrect or missing | Partly correct; at least one key relation incorrect or ambiguous | Evident problems, but content remains recognizable |
| 2 | Multiple key attributes incorrectly bound or missing | Multiple key relations incorrect | Severe structural errors or extensive artifacts |
| 1 | Requirements largely unmet, or target object absent | Requirements largely unmet, or relevant objects absent | Severely fragmented, distorted or unrecognizable |

Higher scores indicate better performance. Ties between methods are allowed; no ranking or combined total score is required. Action and text-rendering requirements are not automatically treated as attribute binding. An illustration is not penalized for being non-photographic.

A `null` attribute or spatial score means the dimension is not applicable to the prompt and is excluded, not treated as zero. If the prompt requires an object but the generated image omits it, that is a generation error to score, not a reason to mark the dimension inapplicable. Visual quality applies to every image. Applicable prompt counts per method are 36/34/60 for English and 27/27/29 for Chinese (attribute/spatial/quality).

### Results

Each image score is the average of its three evaluator ratings. The tables below report the mean and sample SD (`ddof=1`) across these image averages. This SD describes image-to-image variation, unlike the three-generation-seed SD in the automatic main-experiment tables.

#### English

| Method | Attribute | Spatial | Quality |
|---|---:|---:|---:|
| SDXL | 3.954 ± 1.096 | 3.402 ± 1.219 | 3.944 ± 0.326 |
| RPG | 4.278 ± 0.882 | 3.520 ± 1.274 | 4.017 ± 0.256 |
| Kolors | 3.861 ± 1.040 | 3.402 ± 1.315 | 4.011 ± 0.086 |
| RPG+Kolors | 4.120 ± 1.096 | 3.431 ± 1.332 | 4.033 ± 0.243 |
| RAGD | 4.593 ± 0.722 | 4.324 ± 1.000 | 4.033 ± 0.227 |
| BiReG | 4.269 ± 1.026 | 3.676 ± 1.389 | 4.044 ± 0.264 |

#### Chinese

| Method | Attribute | Spatial | Quality |
|---|---:|---:|---:|
| SDXL | 3.469 ± 1.005 | 3.222 ± 1.202 | 4.000 ± 0.267 |
| RPG | 3.531 ± 1.099 | 3.358 ± 1.261 | 3.977 ± 0.333 |
| Kolors | 4.049 ± 0.714 | 4.235 ± 0.973 | 3.977 ± 0.198 |
| RPG+Kolors | 3.753 ± 1.065 | 4.037 ± 1.130 | 4.057 ± 0.346 |
| RAGD | 3.951 ± 0.866 | 4.222 ± 0.925 | 3.989 ± 0.227 |
| BiReG | 4.136 ± 0.622 | 4.259 ± 1.059 | 4.115 ± 0.312 |

#### Inter-rater agreement

Ordinal Krippendorff's alpha is computed separately for each language and dimension across all six methods, retaining all 21 evaluator identities and treating unassigned cells as missing. Higher alpha indicates stronger agreement; alpha is distinct from performance scores.

| Language | Attribute | Spatial | Quality |
|---|---:|---:|---:|
| en | 0.842011 | 0.869049 | 0.797289 |
| zh | 0.719559 | 0.878509 | 0.725812 |

Full 95% confidence intervals and paired differences (BiReG minus each baseline) are available in [results/table.md](results/table.md). They use 20,000 paired prompt-level bootstrap resamples within seven English and two Chinese sampling strata, with random seed 20260930. The same prompts are resampled across all methods. Intervals are pointwise, without multiplicity adjustment, and condition on the observed prompts, fixed images and assigned panel. The largest mean alone does not establish a reliable difference.

### Scope and data notes

Each prompt contributes one fixed image per method. English seeds 2026/3407/5678 each cover 20 prompts; Chinese seeds 1234/2468/42 cover 10/10/9 prompts. Task HZ012 (`UGB-ZH-278`) was excluded across all six methods because of RPG base-generation fallback, without replacement.

## 2. Reproduction materials and instructions

### Recompute the reported statistics

Download this directory and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

Use Python 3.11 or later. NumPy is the only dependency; no GPU, image download or API calls are needed. The script validates coverage, score ranges, applicability and evaluator assignments, then writes `results/statistics.json` and `results/table.md`. It resolves inputs relative to its own location and overwrites those result files. Its ordinal alpha calculation was checked against the archived krippendorff 0.8.2 results.

| File | Purpose |
|---|---|
| `data/prompts.jsonl` | Task/prompt IDs, original prompt, generation translation where applicable, seed and sampling stratum |
| `data/ratings.jsonl` | One evaluator–image assessment per row, with anonymous codes, image/task/prompt IDs, method, seed and dimension scores |
| `reproduce.py` | Coverage checks and statistical reproduction |
| `requirements.txt` | Python dependency |
| `results/statistics.json` | Full-precision estimates, intervals, agreement and sample counts |
| `results/table.md` | Readable results, agreement and paired comparisons |

The public export preserves submitted V4 scores and evaluator/image associations. It omits timestamps, free-text notes, collection metadata and server paths.

### Conduct a new human evaluation

Use the prompts, fixed images and rating rules in Part 1 to organize a new evaluator panel. New ratings may differ from the submitted results. Match images to the data using `image_code`, and preserve anonymous method presentation while collecting ratings.

**Image download: pending.** The intended archive is `Human_evaluation_images.zip`, containing 534 evaluated images. Its Baidu Netdisk link, extraction code, final file size and SHA-256 will be added after the complete archive has been verified and uploaded. Image bytes are not included in this GitHub directory.
