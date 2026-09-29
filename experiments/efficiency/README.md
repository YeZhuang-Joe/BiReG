# Section 4.7: efficiency records

This directory reproduces the efficiency table for four BiReG planner deployments. It contains only records needed for this experiment, not intermediate API recovery scripts. No API calls, GPU, model download, or additional packages are required for table reproduction.

## Reproduce

From the repository root:

```bash
python experiments/efficiency/reproduce.py
```

The standard-library script checks the frozen data hashes, matches all prompt/model/seed bindings, sums recorded request and parsing durations, averages generation durations over three seeds, and recomputes the table. It checks the result against the previously audited summary.

Outputs: `results/table47.csv`, `results/table47.tex`, and `results/audit_summary.json`.

## Scope

All four configurations use the same fixed 102 English and 50 Chinese prompts. Each prompt has one selected plan and three generated images: 456 images per planner, 1,824 images in total. There are 631 recorded requests, including unsuccessful attempts and supplemental requests. This directory does not establish or redefine the sampling history of the parent benchmarks.

| Interface model ID | Endpoint | Temperature | Token limit | Thinking configuration |
|---|---|---:|---:|---|
| deepseek-flash | https://api.deepseek.com/chat/completions | 0 | 1200 | disabled |
| deepseek-v4-pro | https://api.deepseek.com/chat/completions | 0 | 1200 | disabled |
| qwen3.8-flash | https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions | 0 | 1200 | enable_thinking=false |
| gpt-5.5 | https://api.hopeai.cc/v1/chat/completions | omitted | omitted | reasoning_effort omitted; provider defaults |

GPT is an identifier requested from and returned by a third-party service; underlying model identity was not independently verified. Omitted thinking parameters do not mean thinking was disabled. Some GPT responses contain reasoning-token usage. These measurements characterize observed deployments, including network/queue/proxy effects and differing request defaults, not intrinsic LLM inference speed. Runs were not randomized/interleaved repeated timing trials.

## Timing definitions

- **Planning P:** sum of all observed HTTP/SDK request durations and recorded parse durations for a prompt. Failed requests, format retries, and supplemental requests remain included.
- **Generation D:** mean `generation_stage_seconds` across the three generation seeds for that prompt; includes preparation, scheduler configuration, token-length checks, synthesis, image conversion/PNG saving and metadata construction. Excludes model loading and final result.json writing.
- Means and sample SDs are computed across prompts separately for English (n=102) and Chinese (n=50). The SD is not computed from just three generation seeds.
- Planning excludes explicit backoff, ordinary inter-request pauses, manual recovery intervals, offline whitespace normalization, and file writes. These are not continuous end-to-end latency measurements.
- The paper table keeps P and D separate. The retained per-prompt records also contain P+D (reconstructed single-image active time) and P/3+D (amortized cost for this three-image batch); neither is a continuous wall-clock observation.
- No warmup observations were removed. Image-scoring time is excluded.

## Generation configuration

| Setting | English | Chinese |
|---|---|---|
| Resolution, width × height | 1024 × 1024 | 1536 × 1024 |
| Steps | 20 | 30 |
| Scheduler | DPMSolverMultistepScheduler, Karras | EulerDiscreteScheduler |
| CFG | 7.0 | 4.5 |
| Global fusion weight | 0.5 | 0.2 |
| Seeds | 2026, 3407, 5678 | 1234, 2468, 42 |
| Dtype | float16 | float16 |

Same generation settings within each language across all planners. Recorded generator environment: NVIDIA GeForce RTX 4080 SUPER, Python 3.8.20, torch 2.2.1+cu118. The GPT planning client used OpenAI SDK 3.17.0. Chinese/English runtime differences cannot be attributed to language alone.

## Failures and recovery retained

| Planner | Original requests | Supplemental requests | First-attempt strict success | Final original strict success | Selected whitespace recovery | Final usable plans |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-flash | 153 | 0 | 151 | 152 | 0 | 152 |
| deepseek-v4-pro | 152 | 0 | 152 | 152 | 0 | 152 |
| qwen3.8-flash | 153 | 0 | 151 | 152 | 0 | 152 |
| gpt-5.5 | 171 | 2 | 133 | 148 | 2 | 152 |

GPT's original requests include 13 HTTP errors (500×1, 502×6, 524×6) and 10 strict parsing failures. Two English plans were recovered by stripping outer ASCII whitespace from saved responses; the raw attempt outcomes were retained. The two unresolved Chinese prompts UGB-ZH-237 and UGB-ZH-300 were completed in a separate supplemental phase. Supplemental success is not retroactively counted as original first-attempt success.

Twelve saved explicit-wait records sum to 660.090 seconds, excluded from P. This does not include all possible waits: ordinary pauses and manual recovery intervals were not fully measured.

## Additional timing details (seconds)

| Planner | EN planning median | ZH planning median | Smoke model load | Batch model load |
|---|---:|---:|---:|---:|
| deepseek-flash | 0.979 | 1.164 | 15.494 | 15.864 |
| deepseek-v4-pro | 1.492 | 1.633 | 15.853 | 16.279 |
| qwen3.8-flash | 1.410 | 1.725 | 17.504 | 16.454 |
| gpt-5.5 | 14.039 | 25.804 | 18.678 | 17.914 |

Model loading occurred twice per planner (two smoke images, then 454 batch images). A load is not charged anew to each image. No matched no-region timing baseline was measured, so these results do not establish negligible regional-control overhead.

## Files and provenance

- `data/per_request_timings.json`: 631 extracted request records, including unsuccessful statuses, original source paths, timestamps, token usage, and durations.
- `data/per_image_timings.json`: 1,824 extracted image timing records with prompt/model/seed, image SHA256, GPU and session fields.
- `data/per_prompt_timings.json`: 608 derived prompt/model records.
- `data/explicit_wait_records.json`: 12 recorded backoff events.
- `data/summary.json`: original audited summary, model-loading records, generation configuration and checkpoint-configuration hashes.
- `data/SOURCE_SHA256.txt`: identity of the private source export used for the audit.
- `data_sha256.json`: checksums of the released data files.

These are extracted measurement records, not complete raw HTTP response bodies or image files. The source archive contained 8,064 files verified against its export manifest; this release does not include that full archive or temporary recovery scripts. Reproduction here verifies the released timing data and arithmetic, not the authenticity of server-side model identity or checkpoint weight bytes. The hash of a private source export is a provenance reference, not a public download link.

No API keys, account credentials, or private API configuration files are included. Image-quality evaluation is reported separately in Section 4.6.
