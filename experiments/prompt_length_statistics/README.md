# BiReG prompt-length statistics

This package reports length statistics for the 50 Chinese–English development prompt pairs, 1,920 English main-experiment prompts, 150 original Chinese main-experiment prompts, and their 150 frozen English translations. It contains descriptive text statistics only.

## Counting rules

- **Chinese characters:** count Unicode code points after excluding whitespace and punctuation (Unicode General Category beginning with `P`). Any retained Latin letters, numbers, or symbols count individually.
- **English words:** split on Unicode whitespace and count tokens containing at least one letter or number. A hyphenated or apostrophized form without whitespace counts as one token; punctuation-only tokens do not count.
- Text is not normalized, rewritten, translated, or truncated. These are text-length measures rather than model-token counts.
- Statistics use one entry per prompt text; generation seeds do not multiply prompt counts. The two development rows describe the same 50 paired scenarios. The English translations describe the same 150 Chinese-source prompts, not additional independent prompts.

| Collection | N | Unit | Mean | Median | Min–max |
| --- | ---: | --- | ---: | ---: | --- |
| Development (zh) | 50 | characters | 56.68 | 47.5 | 7–184 |
| Development (en) | 50 | words | 46.06 | 37 | 5–153 |
| English main | 1,920 | words | 9.07 | 8 | 5–25 |
| Chinese main | 150 | characters | 43.85 | 43 | 28–61 |
| English translations | 150 | words | 27.59 | 27 | 15–43 |

Mean values are rounded to two decimal places in the table. `results/summary.json` retains the unrounded values.

## Reproduce

Python 3.8 or later; no third-party packages required. To reproduce from the bundled original-file snapshots:

```bash
python reproduce.py
```

Suggested repository location: `experiments/prompt_statistics/`. To read the existing frozen lists directly from a local BiReG checkout, run from the repository root:

```bash
python experiments/prompt_statistics/reproduce.py --repo-root . --output experiments/prompt_statistics/results
```

The script verifies record counts, unique prompt identifiers, nonempty texts, and, when using bundled snapshots, their recorded Git blob hashes. It produces summary JSON and CSV files, one length record per analyzed text, a Markdown table, and a LaTeX table. It does not call a language model or modify any source prompts.

## Source provenance

Sources were retrieved on 2026-10-07 from [YeZhuang-Joe/BiReG](https://github.com/YeZhuang-Joe/BiReG), commit `d7ef2546eb5b6a4c3a19582a388505922ca30951`:

1. `data/development_prompts/bireg_development_pairs_50_v1.jsonl`: fields `prompt_zh` and `prompt_en`.
2. `experiments/main_experiment/en1920/data/prompts.frozen.jsonl`: field `prompt`.
3. `experiments/main_experiment/zh150/data/prompts.frozen.jsonl`: fields `prompt` and `translation_en`.

`source_manifest.json` records repository paths and Git blob hashes. `results/summary.json` records source-file SHA-256 hashes and the Unicode database version used for counting. Bundled snapshots match the recorded Git blobs.

## Paper placement

Insert `results/table.tex` in Appendix A.4, **Text Overlap Checks and Collection Statistics**, and add:

```latex
Table~\ref{tab:prompt_length_statistics} summarizes prompt lengths for the development and main-experiment collections, including the frozen English translations of the Chinese-source prompts.
```

The table requires `booktabs`, already used in the manuscript. Its caption is a single source line.

## 中文说明

本包统计50对开发提示词、1,920条英文主实验提示词、150条中文主实验提示词及其冻结英文翻译的长度。中文按排除空白和标点后的Unicode字符数计数，保留的英文字母、数字和符号各计一个字符；英文按空白分词，仅计算含字母或数字的词项，无空白分隔的连字符词和带撇号词计为一个词。各组分别统计，不将种子数量计入提示词数量，也不将开发集语言版本或中文提示词的英文翻译视为新增独立场景。

表格中的平均值保留两位小数，JSON保留完整精度。可直接运行`python reproduce.py`复算；如已将文件夹上传到GitHub仓库的`experiments/prompt_statistics/`，也可通过上方`--repo-root`命令读取仓库内原有冻结名单。统计不改变提示词，不重新规划、生图或评分。
