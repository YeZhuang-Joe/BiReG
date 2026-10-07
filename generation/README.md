# BiReG — Prompt-to-Image Generation

This folder provides a standalone workflow:
**prompt → Chinese/English template routing → API-based region planning → Kolors image**.
Automatic scoring is not part of this entry point. [中文说明](README_zh.md).

## Quick start

Run from this `generation/` directory, using the existing validated Linux/CUDA `kolors` environment and local Kolors weights. See [environment notes](docs/ENVIRONMENT.md); the exact development Diffusers build is not yet a portable installation lock.

```bash
conda activate kolors
export OMP_NUM_THREADS=1
python tools/configure_api.py
python -m bireg --prompt "A red cup to the left of a blue bowl." --detect-only
python -m bireg --prompt "A red cup to the left of a blue bowl." --output outputs/demo_en --check
python -m bireg --prompt "A red cup to the left of a blue bowl." --output outputs/demo_en
```

`configure_api.py` prompts for endpoint, model identifier and a hidden API key. Credentials are written to `private/api_config.json` with owner-only permissions. The public template is `configs/api.example.json`. The endpoint must be a full HTTPS chat-completions endpoint; the provider must support a non-streaming `messages` request and `choices[0].message.content` response. Changing an endpoint alone does not guarantee protocol compatibility. Remove the optional `thinking` field for providers that do not accept it.

The default model path is `/root/autodl-tmp/weights/Kolors`. Override with `--model-path /absolute/path/to/Kolors`.

```bash
python -m bireg --prompt "木桌上，左边是一个红色陶瓷杯，右边是一个蓝色玻璃碗。" --output outputs/demo_zh
python -m bireg --prompt-file /absolute/path/prompt.txt --language zh --seed 1234 --output outputs/custom
```

Prompts are preserved, not translated. `--language auto` is the default; use `en` or `zh` to override. Ambiguous mixed-language prompts stop before any API call. The routing rule is documented in [workflow details](docs/WORKFLOW.md).

## Output and resumption

Each output directory contains:

| Path | Content |
|---|---|
| `image.png` | Final generated image |
| `output.json` | Image hash, task identifier, routing and generation record |
| `input.jsonl`, `config.json`, `workflow.json` | Input, public configuration and launch provenance |
| `run/planning/` | Requests, raw responses, attempts and validation results |
| `run/plans.frozen.json` | Accepted regional plan used for generation |
| `run/generation/` | Canonical image, task settings, runtime metadata and failures |

Repeating the same command verifies and reuses completed results. Changed prompts, settings or code require a new output directory. Interrupted or failed generation is retained; inspect its record before explicitly adding `--retry-failed`. Failed planning never silently changes to an equal grid or another model.

`--detect-only` requires neither credentials nor model files. `--check` checks local source hashes, checkpoint inventory and configuration without writing outputs or calling the API/GPU; it does not establish model execution compatibility. Run `python tools/check_environment.py` for an import/tokenizer probe.

## Generation profiles

Dimensions are width × height. Profiles are in `configs/generation.json`.

| Routed language | Resolution | Steps | CFG | Scheduler | Karras sigmas | Global weight λ | Default seed |
|---|---|---|---|---|---|---|---|
| English | 1024 × 1024 | 20 | 7.0 | DPM multistep | Yes | 0.5 | 2026 |
| Chinese | 1536 × 1024 | 30 | 4.5 | Euler discrete | No | 0.2 | 1234 |

Both profiles use float16, batch size 1, CPU offload and xformers. The weights are configured values, not claimed optima. Language comparisons using these profiles also differ in generation settings. The maximum regional count is 7 and the retained encoder limit is 256 tokens per prompt part; actual token counts and truncation are logged.

## Verification and provenance

```bash
python -m unittest discover -s tests -v
python tools/verify_package.py
```

The six archived backend files and two templates are unchanged; their hashes are checked at runtime. The previous scheduler `-inf` JSON issue is handled in the metadata writer without modifying sampling. This distribution reorganizes the working code; its offline tests mock HTTP and GPU execution. Separately, the reorganized entry completed one English and one Chinese live API/GPU smoke run on the existing AutoDL environment on 2026-09-27, as documented by the author-supplied console logs. Both plans were accepted on the first attempt and both runs reported image and JSON record saving. See the [smoke-run record](docs/SMOKE_TEST_20260927.md) and [validation status](docs/VALIDATION.json). These two functional checks are not the formal planner-comparison or efficiency experiments.

This is a reusable generation entry, not a replacement for frozen paper-experiment manifests or scores. Regenerating an API plan may change it; seeds alone do not guarantee identical pixels across software/hardware environments. Existing historical run directories must stay in their original installation.

Do not upload `private/` contents, outputs, model weights or local logs. Browser uploads do not provide a substitute for checking selected files.

Author-created BiReG software contributions are licensed under the [Apache License 2.0](../LICENSE). Third-party code retains its original license conditions and copyright notices; see [NOTICE](../NOTICE) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
