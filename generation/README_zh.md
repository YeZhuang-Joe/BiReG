# BiReG：输入提示词，生成图像

统一入口为 `python -m bireg`。流程包括语言路由、API 区域规划和 Kolors 生图，不接入自动评分。

## 在原有服务器验证

进入本目录并使用已经跑通的环境，不需要重新下载模型或重装依赖：

```bash
conda activate kolors
export OMP_NUM_THREADS=1
python tools/configure_api.py
python -m unittest discover -s tests -v
python tools/verify_package.py
python tools/check_environment.py
```

设置 API 时，网址和模型名直接回车即可沿用示例；密钥隐藏输入，保存到本地 `private/api_config.json`。如果文件已经存在，工具不会覆盖。更换接口时可在服务器本地编辑这个 JSON 文件；其他服务商不支持 `thinking` 参数时删除该字段。

先做英文检查，再生成一张图：

```bash
python -m bireg --prompt "A red ceramic cup is on the left of a blue glass bowl on a wooden table." --output outputs/release_en --check
python -m bireg --prompt "A red ceramic cup is on the left of a blue glass bowl on a wooden table." --output outputs/release_en
```

然后验证中文：

```bash
python -m bireg --prompt "木桌上，左边是一个红色陶瓷杯，右边是一个蓝色玻璃碗。" --output outputs/release_zh
```

图片分别位于 `outputs/release_en/image.png` 与 `outputs/release_zh/image.png`。
默认权重目录为 `/root/autodl-tmp/weights/Kolors`，可以通过 `--model-path` 修改。

## 常用选项

| 选项 | 用途 |
|---|---|
| `--prompt` | 直接输入提示词 |
| `--prompt-file` | 读取 UTF-8 提示词文件，与上一项二选一 |
| `--language auto/en/zh` | 自动路由或手动指定；默认 auto |
| `--detect-only` | 只检查语言，不需要密钥和模型 |
| `--check` | 本地配置、模型目录和源文件检查，不调用 API、不生图 |
| `--seed` | 指定正整数种子 |
| `--output` | 本次输出目录；更换提示词或参数时使用新目录 |
| `--api-config` | 指定本地私有 API 配置文件 |
| `--retry-failed` | 检查失败记录后，明确允许重试生图 |

自动路由只用于中英文模板选择，不翻译、不修改提示词。混合输入证据不明确时要求手动指定语言。
英文默认 1024×1024、20 步、CFG 7、λ=0.5；中文默认 1536×1024、30 步、CFG 4.5、λ=0.2。完整配置见 `configs/generation.json`。

## 记录与发布

`output.json` 保存图片哈希、参数和实际生成记录。`run/` 保存请求、回复、冻结规划及每次生图记录。相同命令再次执行会复用已核验的结果；原历史实验目录不要迁移到这个新入口下续跑。

本次整合的底层 6 个渲染文件及两份模板未改动。25 项离线检查已通过；此外，2026 年 9 月 27 日，作者在既有 AutoDL 环境中完成了整合版入口的英文和中文各一次真实 API/GPU 流程测试。作者提供的终端日志显示，两次规划均首次通过，图片及 JSON 记录保存成功。详见 [中英文运行记录](docs/SMOKE_TEST_20260927.md)。这两次运行用于功能验证，不作为 4.6、4.7 的正式比较实验结果；新环境仍需自行验证。

上传 GitHub 使用下载包中的干净 `generation/` 文件夹，不要使用服务器上填过密钥、产生过输出的整个工作目录。正式对外发布前，按 `THIRD_PARTY_NOTICES.md` 补齐沿用源码的许可与归属信息。
