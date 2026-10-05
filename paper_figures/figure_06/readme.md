# Chinese-Source Qualitative Comparisons

This directory contains the prompt pairs and original generated images for the four qualitative cases identified by manuscript figure label `fig:regional_control3`.

## Generation inputs

BiReG uses the original Chinese descriptions (`prompt_zh`). RPG+Kolors uses the corresponding English translations (`prompt_en`). In the manuscript figure, BiReG appears on the left and RPG+Kolors on the right for each case.

The complete input texts are stored in [prompts.jsonl](prompts.jsonl). They preserve the generation inputs as provided, including their original wording and punctuation.

## Image mapping

| Case | Description | BiReG — Chinese input | RPG+Kolors — English input |
| --- | --- | --- | --- |
| (a) | Scholar and squirrel | [fig06_a_bireg.png](fig06_a_bireg.png) | [fig06_a_rpg_kolors.png](fig06_a_rpg_kolors.png) |
| (b) | Tibetan plateau scene | [fig06_b_bireg.png](fig06_b_bireg.png) | [fig06_b_rpg_kolors.png](fig06_b_rpg_kolors.png) |
| (c) | Fisherman casting a net | [fig06_c_bireg.png](fig06_c_bireg.png) | [fig06_c_rpg_kolors.png](fig06_c_rpg_kolors.png) |
| (d) | Sichuan opera performance | [fig06_d_bireg.png](fig06_d_bireg.png) | [fig06_d_rpg_kolors.png](fig06_d_rpg_kolors.png) |

## Prompt file format

`prompts.jsonl` is UTF-8 encoded and contains four records, one JSON object per line.

| Field | Meaning |
| --- | --- |
| `case_id` | Panel identifier: `a`, `b`, `c`, or `d`. |
| `case_title` | English case title used in the figure. |
| `prompt_zh` | Original Chinese input used by BiReG. |
| `prompt_en` | English translation used by RPG+Kolors. |
| `images.bireg` | BiReG image filename, relative to this directory. |
| `images.rpg_kolors` | RPG+Kolors image filename, relative to this directory. |

## Figure annotations

The original PNG files retain their native pixel dimensions. Method labels, case labels, and yellow boxes are added when composing the manuscript figure.

The yellow boxes mark the following details for visual inspection:

- **(a):** Head regions of the scholar.
- **(b):** Handheld objects in both images and the distant stupa in the BiReG image.
- **(c):** The fisherman's hands and forearms.
- **(d):** The spatial relationship between the performer's face and the flames.

These files accompany the qualitative analysis in the [BiReG repository](https://github.com/YeZhuang-Joe/BiReG).
