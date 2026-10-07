# Main-experiment image evaluation — v1 (server smoke validated)

Unified scoring of **new outputs produced by the main-experiment generation
runner**, for EN1920 and ZH150 and all six pipelines: SDXL, RPG, Kolors,
RPG+Kolors, RAGD and BiReG. See [中文操作说明](README_zh.md).

This directory belongs at `experiments/main_experiment/evaluation/`, beside
`generation/`, `en1920/` and `zh150/`. It does not replace the archived scores or
their existing `summarize.py` scripts. Older generation layouts require their
historical scoring wrappers; this entry point does not silently reinterpret them.

## What is retained

- EN: pinned T2I-CompBench evaluation revision
  `4aa404212eb5d06e5adbcd9cee696c750d0d25a5`; BLIP-VQA for color/shape/texture,
  UniDet for spatial, CLIP for non_spatial. Complex runs all three branches and
  uses `(a+s)/2`, `(a+c)/2` or `(a+s+c)/3` by frozen complex subtype. This is the
  retained experiment adapter, not the unmodified official complex metric.
- ZH: hash-checked historical `eval_common.py` and `vllm_request.py`, the original
  Chinese system-prompt construction and checkpoint descriptions, and response
  parsing. The collected pair is associated with UniGenBench revision
  `64e57abc52bf24bfe901cd9beb7de5f00fc0a286`.
- Declared Chinese judge:
  `CodeGoat24/UniGenBench-EvalModel-qwen3vl-32b-v1@f07df2dd73ee2e0979211d018bc5a23a7fd619ae`.
  Model configuration files and the historical judge declaration are preserved
  as evidence. This is not a complete weight-hash attestation.
- Only 1,920 English prompts and 150 Chinese prompts/annotations are included.
  Regional planning comparisons and human ratings are not part of this package.

`vendor/historical_en.py` is the supplied RAGD English scoring wrapper retained
byte-for-byte. The new orchestrator reuses its evaluator provenance inspection,
branch invocation, score parsing and branch receipts. New code provides shared
input binding and the same aggregation definitions across all six methods.
The historical wrapper's original standalone CLI is not the public entry point.

Chinese evaluation imports the **external, hash-checked historical source pair**;
it does not vendor or rewrite the evaluation algorithm. Preserve the corresponding
upstream repository and its license. Dataset attribution: UniGenBench, Tencent;
the supplied dataset license is retained in `references/UniGenBench_License.txt`.
The full upstream dataset is not republished here.

## Setup

Use Linux and the already working evaluation environments on the respective
servers. This package installs nothing and starts no service. English and Chinese
can be scored on separate servers. Keep the working T2I evaluator, its weights,
and the historical UniGenBench server environment. The generation environments
are not necessarily suitable for scoring. A clean-machine dependency lock is
not provided.

Copy `configs/local.example.json` to `configs/local.json`. Paths inside that
JSON are resolved relative to the JSON file's directory; absolute paths also
work. Set `generation_dir`, `image_root`, and the evaluator paths for the language
being used. For Chinese, also set `chinese_model_dir` and `api_url`.
`image_root` is the folder containing `en/` and/or `zh/`, not the individual
method folder. The default API URL is local and requests contain the image and
Chinese evaluation prompt. No planner or cloud evaluator is called automatically.

Typical existing Python interpreters:

- EN: `/root/autodl-tmp/envs/t2i-eval/bin/python`
- ZH: `/root/autodl-tmp/unigen_eval/venv/bin/python`

These are example server paths, not portable installation requirements.

## Checks without scoring

Run from this `evaluation/` directory. Every command may instead use the script's
absolute path from another working directory.

Before creating local configuration, validate all frozen input manifests:

```bash
python3 run_evaluation.py check --language en --generation-dir /path/to/generation
python3 run_evaluation.py check --language zh --generation-dir /path/to/generation
python3 -m unittest discover -s tests -v
```

With `image_root` configured, `check` also validates every selected PNG and JSON.
For the existing first-task smoke outputs, use `--limit 1`:

```bash
/root/autodl-tmp/envs/t2i-eval/bin/python run_evaluation.py check --language en --limit 1 --check-environment
/root/autodl-tmp/unigen_eval/venv/bin/python run_evaluation.py check --language zh --limit 1 --check-environment
```

`--check-environment` validates evaluator evidence (including EN local weight
hashes and ZH configuration/source matches); it does not contact the service or
run GPU inference. File hashes can take time. Pillow is needed for image checks.
The Chinese check does not prove which weights/configuration a running server
actually loaded.

## Explicit scoring

Run each language in its corresponding evaluator environment. For Chinese, the
historical judge must already be running under the `QwenVL` service alias; this
package never replaces an existing service or changes its sampling settings.

```bash
/root/autodl-tmp/envs/t2i-eval/bin/python run_evaluation.py run --language en --limit 1 --execute
/root/autodl-tmp/unigen_eval/venv/bin/python run_evaluation.py run --language zh --limit 1 --execute
```

`--limit 1` checks the first selected task per method, matching the supplied
12-image generation smoke set. The English first task is a color prompt; it
**does not exercise all English scoring branches**. Before a full run, use
repeated `--prompt-id` options and `--seed 2026` to select available images from
all categories and all complex subtypes, then inspect their outputs.

After those checks, a full run is an explicit separate action:

```bash
/root/autodl-tmp/envs/t2i-eval/bin/python run_evaluation.py run --language en --execute
/root/autodl-tmp/unigen_eval/venv/bin/python run_evaluation.py run --language zh --execute
```

`--method` selects one of the six methods (default `all`). `--prompt-id` may be
repeated; `--seed` must belong to the language's frozen seed set. `--limit` is a
task limit per method, applied after filtering, not a prompt count.

## Outputs and resume

Default output: `evaluation/outputs/<en|zh>/<method>/<full|subset_HASH>/`.
Use `--output /path/to/separate_evaluation_outputs` for a new location.
Each completed selection contains `scores.jsonl`, `summary.json`, evaluator
provenance and frozen input bindings. EN retains branch logs/attempts and raw
scores; ZH retains per-image final responses and historical client outputs.
Subset results are explicitly labeled and never called full-experiment results.

A successful identical run reuses completed results after input and score hash
checks. An incomplete/failed attempt stops; inspect its records and use
`--retry-failed` for an explicit retry. Old attempts remain. EN retry is at the
branch level. ZH retry is at the image level and retains the historical nested
retry policy within each attempt. A process lock prevents concurrent writes to
one output root. `aggregate` rebuilds summaries from completed records without
model inference or requests, while validating the inputs and evaluator evidence.

Checks refuse changed task/plan bindings, image hashes, existing frozen records,
missing scores, duplicate score IDs, nonfinite EN scores or nonbinary ZH scores.
EN reports mean and sample SD across seed means. ZH uses checkpoint-weighted
accuracies per seed and then mean/sample SD, preserving subdimensions. A subset
with fewer than three seeds reports `sample_sd: null`. Means use stored metric
scales; ZH is 0–1. These are not human 1–5 ratings.

## Historical Chinese request behavior

The collected client converts images to RGB JPEG at quality 95. It sends
`model=QwenVL`, `max_tokens=4096`, and `do_sample=False`; it does not explicitly
send temperature, top-p or top-k. The saved model generation configuration has
`do_sample=true`, temperature 0.7, top-p 0.8 and top-k 20. The 2026-10-03 smoke service log confirms that `do_sample` was ignored and the
model defaults temperature 0.7, top-p 0.8 and top-k 20 were loaded. This evidence
applies to the tested service, not every historical deployment. The package does
**not** claim deterministic judging or change the historical request.

There are up to three outer parsing/evaluation rounds (`max_retries=2`) and up
to ten HTTP attempts per inner client invocation: up to 30 HTTP attempts in the
worst case for one image evaluation. This is a code-derived bound, not an observed
request count. The historical client does not retain complete per-HTTP-attempt
response text. The new runner isolates its `results.json` under each image/attempt
instead of allowing different tasks to append to a shared working-directory file.

## Validation boundary

See `references/VALIDATION.json` and `references/scoring_smoke_20261003/`.
All 37,260 frozen input tasks and the supplied 12 real generation-smoke
image/sidecar pairs passed offline checks. Six regression tests passed.
Server scoring on 2026-10-03 completed 11 EN images and 6 ZH images:

- EN: one color image per method; five additional SDXL images exercise spatial,
  non_spatial and all three complex subtypes. Branch records, complex arithmetic
  and summaries were checked. Shape and texture were not separately smoke-tested.
- ZH: all six methods use UGB-ZH-004, seed 1234, with two checkpoints each.
  Original image/sidecar hashes, response hashes, labels, binary scores and
  per-method/combined summaries agree. This covers action and style, not all ten dimensions.
- The 12 original smoke PNGs were independently checked; the additional five
  English PNGs were not included in the scoring evidence archive. Their recorded
  hash bindings and branch results were checked, not their image bytes locally.

This is workflow validation on these tasks, not full reproduction of the archived
scores, model rankings, or all 37,260 tasks. Re-evaluated scores remain separate.
Complete environment locks, weight-content attestation and full historical-image
joins/downloads remain outside the validated scope.

## Release update and existing outputs

This update changes documentation/evidence only. Executable files and frozen
inputs match the tested manifest in `references/scoring_smoke_20261003/tested_CHECKSUMS.json`.
The package checksum changes when documentation changes. Because existing output
bindings include that checksum, keep the tested server directory and its results;
use a fresh `--output` directory for runs made with this updated package. Do not
edit old receipts to force reuse. No scoring rerun is required to retain the smoke evidence.
