# BiReG generation entry: English and Chinese smoke runs

The author ran the reorganized `python -m bireg` entry in the existing AutoDL `kolors` environment and supplied the console transcripts on **2026-09-27 (Asia/Shanghai)**. Both transcripts show automatic language routing, first-attempt planning acceptance, completed sampling and successful image/JSON record saving.

This record summarizes the supplied logs. The original PNG files and `output.json` files were not supplied for independent inspection in this documentation update. Image contents, output dimensions and image hashes are therefore not independently verified here. No API or GPU call was made to prepare this update.

| Observation | English | Chinese |
|---|---:|---:|
| Automatic language route | en | zh |
| Accepted planning attempt | 1 | 1 |
| HTTP request, seconds | 0.913 | 0.737 |
| Pipeline loading, seconds | 17.03 | 17.01 |
| Generation, seconds | 21.94 | 26.16 |
| Sampling steps shown | 20 | 30 |
| Seed in task identifier | 2026 | 1234 |
| Image and JSON saving reported | Yes | Yes |

Times reproduce rounded console values. HTTP time includes network/service waiting and full response receipt. Generation time covers text encoding, denoising and decoding with CUDA synchronization, excluding pipeline loading and PNG writing. The progress-bar duration covers a narrower sampling loop and is not the full generation time. The English and Chinese configured profiles differ; these two runs do not measure an isolated language effect.

## English invocation and completion lines

```bash
python -m bireg \
  --prompt "A red ceramic cup is on the left of a blue glass bowl on a wooden table." \
  --output outputs/release_en
```

Selected lines copied from the supplied English console transcript:

```text
LANGUAGE: en | automatic | H= 0 E= 17
user_f672782baae7220e3c15 repeat 1 attempt 1: accepted | HTTP 0.913s
Pipeline loaded: 17.03s
DONE user_f672782baae7220e3c15_seed_2026 | 21.94s
IMAGE: /root/autodl-tmp/BiReG_generation_v1/generation/outputs/release_en/image.png
RECORD: /root/autodl-tmp/BiReG_generation_v1/generation/outputs/release_en/output.json
```

## Chinese invocation and completion lines

```bash
python -m bireg \
  --prompt "木桌上，左边是一个红色陶瓷杯，右边是一个蓝色玻璃碗。" \
  --output outputs/release_zh
```

Selected lines copied from the supplied Chinese console transcript:

```text
LANGUAGE: zh | automatic | H= 23 E= 0
user_7aa557ac7585ff756f7e repeat 1 attempt 1: accepted | HTTP 0.737s
Pipeline loaded: 17.01s
DONE user_7aa557ac7585ff756f7e_seed_1234 | 26.16s
IMAGE: /root/autodl-tmp/BiReG_generation_v1/generation/outputs/release_zh/image.png
RECORD: /root/autodl-tmp/BiReG_generation_v1/generation/outputs/release_zh/output.json
```

## Scope / 验证范围

These are two functional smoke runs, one per language. They confirm the reported execution of the reorganized entry in the existing environment. They are not formal planner-comparison results (Section 4.6), efficiency estimates (Section 4.7), an assessment of image quality, or a clean-install reproduction study. Formal experiments remain separate.

中英文各一次真实流程测试已完成，依据为作者提供的终端日志。日志显示规划、生图及图片/JSON 保存成功。本记录保留展示精度，不将单次耗时当作正式效率实验结果，也不据此宣称视觉质量或跨环境像素一致。
