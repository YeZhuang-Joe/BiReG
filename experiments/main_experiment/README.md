# Main Experiments — EN1920 and ZH150

This directory groups the main-experiment generation workflow, frozen inputs,
archived automatic scores and score-aggregation scripts for SDXL, RPG, Kolors,
RPG+Kolors (`rpg_kolors`), RAGD and BiReG.

| Evaluation set | Prompts | Generation seeds | Tasks per pipeline | Total tasks / archived score records |
|---|---:|---|---:|---:|
| EN1920 | 1,920 | 2026, 3407, 5678 | 5,760 | 34,560 |
| ZH150 | 150 | 1234, 2468, 42 | 450 | 2,700 |

## Directory guide

| Directory | Contents |
|---|---|
| [generation](generation/README.md) | Unified six-pipeline runner, archived generation settings, frozen main-experiment plans and task manifests |
| [evaluation](evaluation/README.md) | English and Chinese-source image-to-score evaluation runner, local configuration, and validation records |
| [en1920](en1920/README.md) | English prompts, archived automatic scores and category aggregation |
| [zh150](zh150/README.md) | Chinese prompts, frozen English translations, archived checkpoint scores and dimension aggregation |

The generation workflow here is specific to the retained main experiments.
It uses frozen planning records without new planner API calls. Method demos,
planner-comparison experiments, ablations and human-evaluation materials are
outside this directory's scope. Standard SDXL and Kolors generation do not
require regional plans.

## 1. Reproduce tables from archived scores

Run these commands from the **repository root**:

```bash
python3 experiments/main_experiment/en1920/summarize.py
python3 experiments/main_experiment/zh150/summarize.py
```

Only the Python standard library is required; no GPU or API is needed.
Each script checks prompt/method/seed coverage and writes `results/summary.json`
and `results/table.md` inside its own evaluation directory. These scripts
aggregate the released scores; they do not evaluate newly generated images.

EN1920 contains 320 prompts in each of six categories. For each category and
pipeline, the script averages image scores within each seed, then reports the
mean and sample standard deviation (`ddof=1`) across the three seed means.

ZH150 reports ten main dimensions. It divides satisfied checkpoint counts by
evaluated checkpoint counts for each pipeline, dimension and seed, then reports
the mean and sample standard deviation across seeds. This is checkpoint-weighted
aggregation. A prompt may contribute to several dimensions.

The two evaluations use different metrics and scales. EN complex scores use the
archived experiment adapter rather than the unmodified official complex metric.
Neither evaluation contains human 1–5 ratings. See the respective READMEs for
metric definitions.

## 2. Check frozen generation inputs

From the repository root:

```bash
python3 experiments/main_experiment/generation/run_generation.py --language en --check
python3 experiments/main_experiment/generation/run_generation.py --language zh --check
python3 experiments/main_experiment/generation/validate_adapters.py
```

These checks do not generate images or request new plans.

## 3. Generate images from frozen plans

Follow the environment requirements in the
[generation guide](generation/README.md), also available
[in Chinese](generation/README_zh.md). Model weights must be available locally.
The historical environments and model revision evidence have limitations
documented there; a complete clean-machine installation lock is not provided.

In `experiments/main_experiment/generation/`, copy
`configs/local.example.json` to `configs/local.json`. Set the three model
directories and both Python interpreter paths for your machine.

Then run an initial GPU check from the repository root:

```bash
python3 experiments/main_experiment/generation/run_generation.py --language en --method all --limit 1
python3 experiments/main_experiment/generation/run_generation.py --language zh --method all --limit 1
```

Each command generates one task per pipeline. After checking those outputs,
full generation can be started explicitly:

```bash
bash experiments/main_experiment/generation/run_en.sh --method all
bash experiments/main_experiment/generation/run_zh.sh --method all
```

The default output is `generation/outputs/{en|zh}/{method}/{prompt_id}/`,
with a PNG and matching JSON for each seed. Use `--output /path/to/new_outputs`
to select another output root. Keep historical images and earlier smoke outputs
separate. Existing results are reused only when the runner's fingerprint and
image-hash checks agree.

Running generation does not update the archived score files or run evaluators.

## Language paths and retained exceptions

For ZH150, Kolors, RPG+Kolors and BiReG use original Chinese prompts; SDXL, RPG
and RAGD use frozen English translations. All six methods' archived evaluations
refer to the same original Chinese prompts and checkpoint labels.

ZH RPG retains base-generation fallback for `UGB-ZH-110` and `UGB-ZH-278`
at all three seeds. Those six tasks remain in RPG results and are not counted as
successful regional planning. `UGB-ZH-387` retains its single-region compatibility
handling. See the generation guide for commands that exercise these paths.

## Validation and release scope

On 2026-10-02, all six pipelines completed an English smoke task
(`T2I-color-001`, seed 2026) and a Chinese smoke task
(`UGB-ZH-004`, seed 1234), for 12 images total. Image decoding, hashes,
task/plan identity and configuration agreement were checked. Evidence is recorded
in [the smoke report](generation/references/GPU_SMOKE_20261002.json).

This validates those tested tasks. It does not establish full-plan executability,
exercise every fallback/compatibility path, or prove byte-for-byte agreement with
historical images. The unified package has not been rerun over all 37,260 tasks.

The release provides frozen prompts and regional plans, generation code and configurations, an [image-to-score evaluation entry](evaluation/README.md), archived automatic scores, and score-aggregation scripts. The fixed main-experiment images and their manifests are available through the download links below.

### Download main-experiment images

The English and Chinese main-experiment image archives are available in the **BiReG_Main_Experiment_Images** folder:

**[Download from Baidu Netdisk](https://pan.baidu.com/s/1LLymiKuYsck1E4Mv-fmEUA?pwd=juui)**  
**Extraction code:** `juui`

| Archive | Prompts | Seeds | Methods | Images | Size |
|---|---:|---|---:|---:|---:|
| `Main_Experiment_Images_EN.zip` | 1,920 (320 per category) | 2026, 3407, 5678 | 6 | 34,560 | 46.20 GiB |
| `Main_Experiment_Images_ZH.zip` | 150 | 1234, 2468, 42 | 6 | 2,700 | 5.51 GiB |

Both archives cover SDXL, RPG, Kolors, RPG+Kolors, RAGD and BiReG. Each archive includes the images, the frozen prompt list, an image manifest containing source-path mappings and per-image SHA-256 hashes, and a README.

Images are organized as:
`<language>/<method>/<prompt_id>/seed_<seed>.png`

For Chinese Kolors, RPG+Kolors and BiReG, the archived images are byte-preserving copies of the evaluation inputs, verified against the hashes in the corresponding evaluation records. Their original filenames and paths are retained in the manifest.

**Archive SHA-256 checksums**

```text
63435bf5f5a14728c21f3bcf17af4006c995af16ed24c697bf75af8e61eae98f  Main_Experiment_Images_EN.zip
f7080061513b3e4db1cc0eccb63faff1c5ed190b4c73a4d230a2b4b6b98d812f  Main_Experiment_Images_ZH.zip
```

These archives contain the fixed images corresponding to the retained main-experiment prompt sets. Human-evaluation materials are distributed separately.
