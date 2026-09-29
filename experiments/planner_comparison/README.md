# BiReG: effect of the LLM planner (Section 4.6)

This package reproduces the English and Chinese planner-comparison tables from retained evaluation outputs. It contains the frozen 102 English prompts (17 per category), 50 Chinese prompts, 1,224 English per-image scores, and 600 Chinese per-image checkpoint-score records. All four planners use the same prompt/seed bindings within each language.

## Reproduce the tables

From this directory, run:

```bash
python reproduce.py
```

Python 3.8+ and its standard library are sufficient. No API key, GPU, model download, or third-party Python package is required. The script verifies data hashes, task completeness, shared prompt/testpoint bindings, and agreement with archived summaries. It writes full-precision statistics, CSV tables, readable Markdown tables and an audit report under `results/`.

## Experimental setup

| Setting | English | Chinese |
|---|---|---|
| Frozen prompts | 102; 17 per category | 50 |
| Generation seeds | 2026, 3407, 5678 | 1234, 2468, 42 |
| Images per planner | 306 | 150 |
| Resolution (width × height) | 1024 × 1024 | 1536 × 1024 |
| Steps / CFG / global fusion weight | 20 / 7 / 0.5 | 30 / 4.5 / 0.2 |
| Scheduler | DPMSolverMultistepScheduler, Karras | EulerDiscreteScheduler |

API identifiers: `deepseek-flash`, `deepseek-v4-pro`, `qwen3.8-flash`, and `gpt-5.5`. GPT was accessed through the HopeAI third-party service; its name is the requested/returned identifier, not an independent verification of underlying model weights. DeepSeek/Qwen used temperature 0, output limit 1200 and disabled thinking. GPT omitted optional temperature, reasoning-effort and output-limit parameters. This comparison describes these tested configurations.

The same language-specific template and generator settings were used across planners. Each executable plan was frozen and reused across three generation seeds. GPT's 148 originally strict-accepted plans were supplemented by two whitespace-normalized plans and two plans from separately recorded supplemental requests. These are not 152 first-attempt successes. Request/failure timing is documented separately in `../efficiency/`.

## Metrics and statistical definitions

English: compute the mean image score within each category and generation seed (17 images), then the mean and sample SD (`ddof=1`) of the three seed-level means. Color/shape/texture use the retained BLIP-VQA pipeline, spatial uses UniDet, and non-spatial uses CLIP similarity. Complex follows the retained BiReG main-experiment adapter, NOT the unmodified official complex score. No auxiliary macro average is presented as an official overall metric.

Chinese: compute checkpoint-weighted accuracy within each dimension and seed, then the mean and sample SD of three seed-level accuracies. The CSV/Markdown presentation uses percentages; full-precision JSON uses fractions. Subdimensions are retained in `results/summary.json`. A checkpoint count is not an image count. There are six text-generation checkpoints per seed; all four planners scored zero on these checkpoints, which does not mean all image content failed.

SD describes generation-seed variation conditional on frozen plans, not repeated planner sampling. No statistical-significance claim is made. English and Chinese scores use different evaluators and must not be averaged together or compared as a common scale.

## Provenance and scope

The input records were exported from completed scoring runs and are retained at full precision. Server-specific absolute paths were omitted; score fields and available English image/task/record hashes were retained. Chinese data include judge checkpoint decisions and explanations, not PNG hashes. `data/evaluator_provenance.json` records pinned evaluator identities and English source hashes. Reference summaries are used as checks, not as the source of recomputed table means.

The 152 prompts are a frozen planner-comparison subset; they do not replace the 1,920-English/150-Chinese main evaluation sets. Retained sampling identifiers/ranks describe this subset; this package alone does not reconstruct its parent-set selection. It makes no additional claim about how the parent benchmarks were selected.

This is a **score-to-table reproduction package**. It does not independently rerun planning, image generation, or model-based scoring. Model weights, generated PNGs, complete frozen plans, API credentials and full raw request archives are not bundled. Use the repository generation workflow and separately released image/planning resources when available for those stages. Recomputed statistics alone do not verify visual correctness of an automatic judge.

## Interpretation

Planner rankings vary by dimension. No tested planner is uniformly best. The experiment supports configuration-specific comparisons, not a universal ranking of LLM reasoning strength or causal attribution of failures to a particular component.
