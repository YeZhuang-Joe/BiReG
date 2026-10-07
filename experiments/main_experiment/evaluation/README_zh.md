# 主实验统一评测入口 v1（已通过服务器小样本评分验证）

本包用于评价统一主实验生成入口产出的图像，支持 EN1920、ZH150 和六条管道。
建议放在 `experiments/main_experiment/evaluation/`，与 `generation/`、`en1920/`、
`zh150/` 同级。原有评分与 `summarize.py` 保持不变。

## 放置与配置

也可独立解压到 AutoDL，使用绝对路径关联现有生成包。
将 `configs/local.example.json` 复制为 `configs/local.json` 后填写：

- `generation_dir`：之前的统一生成包目录。
- `image_root`：统一生成输出根目录，其下应有 `en/` 或 `zh/`。
- `english_repo`：已有 T2I-CompBench-eval 目录。
- `chinese_repo`：实际使用的 UniGenBench 仓库目录。
- `chinese_model_dir`：历史中文评测模型目录，用于核对两个配置文件。
- `api_url`：已有中文评测服务地址，默认本机 `http://127.0.0.1:8080`。

相对路径以配置 JSON 所在目录为基准。中英文可在不同服务器运行；只需配置
当前语言需要的评测器。不要复制示例路径后就假定模型和环境已经安装。
本包不安装依赖、不启动或替换服务、不调用规划器。

## 第一步：只检查，不评分

在本目录运行。未创建 local.json 时，可只检查冻结输入：

```bash
python3 run_evaluation.py check --language en --generation-dir /root/autodl-tmp/BiReG_Main_Generation_v1
python3 run_evaluation.py check --language zh --generation-dir /root/autodl-tmp/BiReG_Main_Generation_v1
```

配置好图像目录以后，检查现有的每管道一张试跑图。下面两个 Python 路径来自
已有服务器记录，请按本机实际情况使用：

```bash
/root/autodl-tmp/envs/t2i-eval/bin/python run_evaluation.py check --language en --limit 1 --check-environment
/root/autodl-tmp/unigen_eval/venv/bin/python run_evaluation.py check --language zh --limit 1 --check-environment
```

检查 PNG 解码、尺寸、图像哈希、同名 JSON、完整任务和冻结规划、生成配置。
环境检查核对英文评分源码/权重证据，以及中文历史源码与模型配置；不会向服务
发请求，不进行 GPU 推理。读取权重哈希可能需要一些时间。

仅有 12 张试跑图时必须保留 `--limit 1`，去掉它会要求所选方法的全部图像。
当前环境中的评测权重/服务身份仍需与实际部署对应，模型配置匹配不等于所有权重认证。

## 第二步：显式评分试跑

第一步通过后运行：

```bash
/root/autodl-tmp/envs/t2i-eval/bin/python run_evaluation.py run --language en --limit 1 --execute
/root/autodl-tmp/unigen_eval/venv/bin/python run_evaluation.py run --language zh --limit 1 --execute
```

中文需要事先部署历史 `QwenVL` 服务。英文第一张是颜色题，只覆盖 BLIP 分支；
全量运行前还应选取空间、非空间和各复杂子类的已有图片验证。
可重复指定 `--prompt-id`，配合 `--seed` 和 `--method` 精确选择。
`--limit` 限制每个方法的任务数，不是提示词数。

2026-10-03 已完成英文 11 张、中文 6 张真实评分测试，具体范围见下文。

## 结果与恢复

新评分写入本包 `outputs/<en|zh>/<method>/<full或subset_HASH>/`：

- `scores.jsonl`：逐图分数，字段兼容主实验评分记录；英文额外保留分支值。
- `summary.json`：按生成种子求均值和样本标准差；中文按考点加权并保留子维度。
- `run.frozen.json`、`evaluator.json`：输入绑定和评测器证据。
- 英文分支执行日志和原始分数；中文逐图最终响应、客户端输出和失败记录。

少于三个种子的子集标准差为 null。子集不冒充全量结果；原归档评分不自动覆盖。
可以通过 `--output` 指定独立目录。同一选择、输入、代码及环境一致时复用已完成结果。
遇到失败先看日志，明确要重试时添加 `--retry-failed`；此前尝试保留。
`aggregate` 只校验并汇总已完成记录，不发评分请求或进行推理。

全量评分须另行显式去掉 `--limit`，保留 `run --execute`，要求全部输入图像齐备。

## 保留的历史评分口径

英文：颜色/形状/纹理使用 BLIP-VQA，空间使用 UniDet，非空间使用 CLIP。
复杂题按冻结子类取属性与空间、属性与 CLIP、或三分支算术平均。
这是主实验保留的适配规则，不是未经修改的官方 complex 总分。

中文：所有方法都用中文原文、完整考点及考点描述评分，保留历史提示模板和解析器。
SDXL/RPG/RAGD 的英文翻译用于生成输入绑定，不替换中文评测文本。
RPG 两条基础生成回退以及单区域兼容图像照常纳入，不另行排除。

中文历史客户端有两层重试：外层最多 3 轮，每轮内部最多 10 次 HTTP 尝试。
单图最坏可到 30 次 HTTP 尝试；这是代码上限，不是实际观测次数。
客户端请求带 `do_sample=False`，但保存的模型配置为温度 0.7、top-p 0.8、top-k 20、
`do_sample=true`。本次服务日志确认忽略 `do_sample`，加载模型默认的温度 0.7、top-p 0.8、top-k 20。
这项证据仅对应本次服务，不能反推所有历史部署；不宣称确定性评分，不修改历史请求。图像沿用 RGB、JPEG quality=95 的历史请求编码。

保留的两个中文源码文件在外部历史仓库读取，并逐字节校验；不下载新版本替换。
历史源码不记录完整的每次 HTTP 原始响应，本包不虚构这类记录。
原英文包装脚本保留于 vendor，新入口复用其分支执行与来源检查。

## 验证范围

- 英文 34,560、中文 2,700 个任务的冻结输入覆盖核对通过。
- 中英文各六张真实生成图片，共 12 组 PNG/JSON 输入绑定通过。
- 六项离线回归测试通过。
- 英文：六条管道各一张颜色题；SDXL 另加空间、非空间及三个复杂子类共五张。
  分支记录、复杂组合公式和汇总核对通过；形状、纹理未单独试跑。
- 中文：六条管道均评测 UGB-ZH-004、seed 1234，每图两个考点。
  原图/侧录哈希、响应哈希、标签、二值分数及单独/合并汇总一致。
  本次只覆盖动作、风格，未覆盖所有十个维度。
- 新增五张英文图未随评分证据包上传，本地核对其记录绑定与评分，未核验图片字节。

证据见 `references/VALIDATION.json` 和 `references/scoring_smoke_20261003/`。
这些测试验证所列任务的流程，不是全量历史分数重现，也不能用于方法排名。
完整新机安装锁、全量权重认证及历史图像下载/完整关联仍未补齐。

## 更新包与已有结果

本次仅更新说明与验证记录，可执行代码和冻结输入与测试版本一致。
由于包校验和包含说明文件，新包的校验和会变化。请保留服务器原测试目录及结果；
若使用更新包再次评分，用 `--output` 指向新目录，避免已有冻结记录冲突。
不要修改旧记录绕过校验，也不需要为此次文档更新重新评分。

## 上传到 GitHub

将本包的整个 `evaluation` 文件夹放入本地仓库的
`experiments/main_experiment/`，与 `generation`、`en1920`、`zh150` 同级。
上传源码、示例配置、冻结输入和 references；不上传本机配置、权重或运行输出。
主实验总 README 可增加评测入口链接：`[Image evaluation](evaluation/README.md)`。
此前“评测工作流待集成”的说明应改为：已提供统一评测入口并通过指定小样本验证，
尚未使用该入口全量重评，完整历史图像下载链接和索引仍待集成。
