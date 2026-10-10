# BiReG main-experiment generation 

Unified offline generation entry point for the six pipelines: SDXL, RPG, Kolors,
RPG+Kolors (`rpg_kolors`), RAGD and BiReG. Only EN1920 and ZH150 prompts and their
associated frozen plans are included. No planner API calls or scoring calls occur.

**Status:** CPU manifest checks and archived adapter input validation passed.
On 2026-10-02, all six pipelines completed one English and one Chinese GPU smoke
task on AutoDL (12 images total). Image decoding, hashes, task/plan identity and
configuration agreement were independently checked against the supplied outputs.
See `references/GPU_SMOKE_20261002.json`. Full execution, special Chinese RPG
branches and bitwise agreement with historical images have not been validated.


## Contents

- `run_generation.py`: single CLI; sequential subprocesses isolate methods and
  translated-ZH RPG regional/fallback modes.
- `configs/en.json`, `configs/zh.json`: archived generation settings.
- `configs/local.example.json`: user-local model paths and Python interpreters.
- `data/en`, `data/zh`: frozen main prompts, translations, plans and task manifests.
- `vendor`: retained historical renderers/adapters; small bridge dependency and
  orchestration code connect them without changing regional text or ratios.
- `references`: source hashes, available environment evidence and validation report.
- `CHECKSUMS.json`: package file integrity (exclude local config and outputs).

EN1920 uses 1,920 prompts × 3 seeds × 6 methods = 34,560 tasks. ZH150 uses
150 × 3 × 6 = 2,700. The EN set maps to 611 original900 and 1,309 extension1800
records; only those retained IDs are exported. This packaging does not establish
or change the original prompt-selection procedure.

EN RPG and RPG+Kolors share the same 1,920 plans. EN BiReG and RAGD each have
1,920 separate plans. ZH RPG, RPG+Kolors, BiReG and RAGD each have 150 plan or
explicit fallback records. ZH RPG UGB-ZH-110/278 are SDXL base-generation fallback,
not successful regional plans. Their six tasks remain under RPG. ZH UGB-ZH-387
retains the full-canvas single-region compatibility handling.

## Setup

Use the existing, working generation environments first. The historical RPG/Kolors
environment reports Python 3.9.21, torch 2.2.2+cu121, transformers 4.39.3,
accelerate 0.29.1, safetensors 0.4.2 and **diffusers 0.33.0.dev0**. That dev version
alone does not identify an exact installable commit. Do not substitute a guessed
pip release and claim the same environment. A clean-machine installation lock
for this environment is still incomplete; use the original environment for smoke
validation. xformers and the model-specific tokenizer dependencies are required.

RAGD reports Python 3.12.3, torch 2.5.1+cu124, diffusers 0.31.0,
transformers 4.44.2, accelerate 1.1.1, safetensors 0.4.5,
huggingface-hub 0.26.2 (see `references/ragd_environment.json`).
RAGD requires a BF16-capable CUDA GPU and uses model CPU offload.

Copy `configs/local.example.json` to `configs/local.json` and edit all three
model directories and both Python executable paths. `python.default` points to
the existing RPG/Kolors environment; `python.ragd` points to the RAGD environment.
Models must already be downloaded. No weights or API keys are included.

Archived absolute model paths in task metadata are documentary; runtime replaces
only the model location with the local configuration. Model revision identity is
not established by a path. Complete weight hashes are not available for all models.

## Validate and run

All commands may be run from any working directory using the script's path.

```bash
python3 run_generation.py --language en --check
python3 run_generation.py --language zh --check
python3 validate_adapters.py

# One image for each method (per-language initial GPU smoke check)
python3 run_generation.py --language en --method all --limit 1
python3 run_generation.py --language zh --method all --limit 1

# Explicitly exercise Chinese RPG fallback and single-region paths
python3 run_generation.py --language zh --method rpg --prompt-id UGB-ZH-110 --seed 1234
python3 run_generation.py --language zh --method rpg --prompt-id UGB-ZH-278 --seed 1234
python3 run_generation.py --language zh --method rpg --prompt-id UGB-ZH-387 --seed 1234

# Full runs, after smoke validation
bash run_en.sh --method all
bash run_zh.sh --method all
```

`--limit` limits each method before splitting execution modes. `--prompt-id` may
be repeated; `--seed` must be an archived seed. `--output /new/path` selects a
new output root. Use a new directory, not the archived experiment output folders.

Output: `outputs/{en|zh}/{method}/{prompt_id}/seed_{seed}.png` and matching JSON.
JSON records the exact task/plan, configuration, source release digest, environment,
image hash and execution metadata. Existing outputs are skipped only when their
fingerprint and image hash agree. Partial or mismatched outputs stop the run;
inspect and move them aside rather than overwrite them. Failures are recorded.
`--retry-failed` permits retry after inspection; no retry, seed change or new
fallback is automatic. Creating `STOP` inside the output root stops between images.

## Scope and limitations

Only generation inputs/code are released here. Evaluator code, scores, full images
and image download links belong to separate releases. Broader source directories,
planner comparisons, ablations and unused historical planning responses are excluded.
Some retained task provenance points back to historical batches; those links are
provenance, not additional released evaluation samples.

The original ZH three-method task records were checked against 1,350 archived
sidecars. EN five-method tasks are rebuilt from the frozen batch prompt/plan files,
reference tasks and documented task-construction code, filtered to EN1920. A full
join to all archived EN five-method generation sidecars has not been performed in
this package. RAGD plan/config files were collected from its actual output folder.
The 12 supplied smoke images were decoded and hash-verified. Historical full image
collections have not been byte-verified, and the complete task set has not been rerun.

The unified runner changes packaging, local paths, output naming and resume checks;
rendering functions and archived parameters are retained. The smoke evidence covers only the two named prompts and their selected seeds;
it does not establish full-plan executability or a clean-machine installation lock.

## 中文说明

本包只包含英文EN1920、中文ZH150主实验的六条生成管道、配置和冻结规划。
运行入口统一，底层保留历史实现；中英文参数分别配置。无需再次调用LLM。
已通过CPU完整性和任务输入校验，以及中英文各六管道、共12项GPU试跑。
12张图的哈希、解码、任务、规划和配置已核对；未全量重跑，也未核对历史图像逐字节一致性。
中文操作步骤与环境边界见 `README_zh.md`。
全部历史权重和开发版依赖的精确安装锁尚不完整，暂不声称可在任意新机器上一键复现。

## Release update and existing smoke outputs

The importlib scope fix is included. No rendering functions, frozen plans or
sampling settings were changed by this documentation update. Four English smoke
outputs used the pre-fix release; the other eight used the import-fix release.
Their exact release digests are preserved in the smoke report.

The runner fingerprints the whole CHECKSUMS.json, including documentation. This
updated package therefore has a new release digest. Keep old smoke outputs as
validation evidence; select a fresh `--output` directory when using this package.
Do not bypass mismatched-output checks. The archive does not include local.json,
model weights, generated images or credentials.
