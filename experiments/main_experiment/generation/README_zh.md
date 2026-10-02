# 主实验六管道统一生成包

本包仅包含 EN1920、ZH150 主实验的提示词、对应冻结规划、生成配置和生成代码。
六条管道为 SDXL、RPG、Kolors、RPG+Kolors、RAGD 和 BiReG。运行时复用冻结规划，无需调用规划 API。

## 范围与语言路径

| 实验 | 提示词 | 种子 | 每管道任务数 | 六管道任务数 |
|---|---:|---|---:|---:|
| EN1920 | 1920 | 2026、3407、5678 | 5760 | 34560 |
| ZH150 | 150 | 1234、2468、42 | 450 | 2700 |

中文 Kolors、RPG+Kolors、BiReG 使用中文原文；SDXL、RPG、RAGD 使用冻结英文翻译。
英文 RPG 与 RPG+Kolors 共用 RPG 规划。SDXL、Kolors 不需要区域规划。
仅收录主实验保留提示词对应的规划；不收录规划器对比、消融或其余历史样本。
历史批次标签仅用于溯源，不表示本次发布样本数量，也不重新确立采样过程。

## 环境准备

优先使用服务器上已验证的两个环境。下面版本来自本次试跑记录：

| 项目 | SDXL/RPG/Kolors/RPG+Kolors/BiReG | RAGD |
|---|---|---|
| Python | 3.9.21 | 3.12.3 |
| torch（包元数据） | 2.2.2 | 2.5.1+cu124 |
| diffusers | 0.33.0.dev0 | 0.31.0 |
| transformers | 4.39.3 | 4.44.2 |
| accelerate | 0.29.1 | 1.1.1 |

开发版 diffusers 的确切安装提交尚未确定，所以此处不提供猜测的 pip 一键安装命令。
原环境中的 xformers、分词器及其他依赖仍需保留；完整新机器安装锁待补齐。
模型需预先下载，包内不含权重。RAGD 使用 BF16 与 CPU offload。

在解压目录执行：

```bash
cp configs/local.example.json configs/local.json
```

编辑 local.json 的三个模型路径（SDXL、Kolors、FLUX.1-dev）和两个 Python 路径。
`python.default` 指向 RPG/Kolors 环境，`python.ragd` 指向 RAGD 环境。
路径只表示本地位置，不等同于模型版本；尚未取得全部权重文件的完整哈希。

## 校验与运行

在解压目录执行以下命令。校验仅用标准库，不加载模型：

```bash
python3 run_generation.py --language en --check
python3 run_generation.py --language zh --check
python3 validate_adapters.py
```

小样本运行，每种语言每条管道一项任务：

```bash
python3 run_generation.py --language en --method all --limit 1 --output ./outputs_release_v1
python3 run_generation.py --language zh --method all --limit 1 --output ./outputs_release_v1
```

需要全量生成时可执行（本次发布整理不要求重跑）：

```bash
bash run_en.sh --method all --output ./outputs_release_v1
bash run_zh.sh --method all --output ./outputs_release_v1
```

`--method` 可选 sdxl、rpg、kolors、rpg_kolors、ragd、bireg 或 all。
`--prompt-id` 可重复传入；`--seed` 指定归档种子；`--limit` 限制每种方法的任务数，而非提示词数。
输出按语言/方法/提示词ID/种子组织，PNG 与同名 JSON 配对。
再次运行仅在任务指纹与图像哈希均吻合时复用输出；不一致会停止。
在输出根目录创建 STOP 文件可在图像之间停止；继续前移除该文件。
失败后保留记录，核对原因后使用 `--retry-failed`；不会自动换种子或重新规划。

## 参数与特殊分支

| 实验/管道 | 宽×高 | 步数 | CFG |
|---|---|---:|---:|
| 英文 RAGD | 1024×1024 | 20 | 3.5 |
| 英文其余五管道 | 1024×1024 | 20 | 7 |
| 中文 SDXL/RPG | 1536×1024 | 30 | 7 |
| 中文 Kolors/RPG+Kolors/BiReG | 1536×1024 | 30 | 4.5 |
| 中文 RAGD | 1536×1024 | 30 | 3.5 |

完整参数以 configs/en.json、configs/zh.json 及冻结任务为准。
中文 RPG 的 UGB-ZH-110、UGB-ZH-278 在三个种子下共六项任务沿用基础生图回退，
仍归于 RPG，不算区域规划成功。UGB-ZH-387 保留单区域全画布兼容逻辑。
本次12图试跑没有覆盖这三个特殊提示词；英文 README 提供定向执行命令。

## 已验证内容与边界

2026-10-02，英文 T2I-color-001/2026 与中文 UGB-ZH-004/1234 各六管道执行成功。
共12张图均能解码，图像哈希、任务指纹、提示词、适用的规划和配置均核对通过。
详见 references/GPU_SMOKE_20261002.json；报告保留输入证据包和各输出文件的哈希。
这不是全量运行验证，也未验证与历史主实验图像逐字节一致。

中文原三管道任务曾与1350份历史侧录核对。英文五管道任务由历史冻结批次资料和任务构造代码恢复，
尚未与全部历史生成侧录完整连接核对。历史路径和批次字段作为溯源信息保留。

## 本次更新与发布边界

已修复统一入口的 importlib 作用域问题。本轮整理只更新说明和验证记录，未改生成逻辑、参数或冻结规划。
由于运行指纹包含 CHECKSUMS.json，文档更新也会改变发布包指纹。
原有12张试跑图应保留作证据；使用本包时选择新的输出目录，避免与旧指纹混用。
不需要为文档更新重新跑实验。

本包不含自动评分实现、评分汇总代码、完整历史图像或下载链接；这些属于后续独立整理内容。
不能将本包单独称为完整端到端复现发布。
