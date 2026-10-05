# English Compositional Generation: Qualitative Comparisons

This directory contains the English prompts and original generated images for the four qualitative cases identified by manuscript figure label `fig:regional_control2`.

## Generation inputs and layout

SDXL, RPG, Kolors, RPG+Kolors, and BiReG use the English prompt recorded in `prompt_en` for each case. The complete texts are available in [prompts.jsonl](prompts.jsonl), preserving the wording and punctuation shown in the original figure.

In the manuscript figure, columns correspond to cases (a)--(d). Rows correspond to SDXL, RPG, Kolors, RPG+Kolors, and BiReG, in that order.

## Image mapping

Each filename below is relative to this directory. The four cases and five pipelines correspond to 20 images.

| Pipeline | (a) Desk objects | (b) Five rooftop pine trees | (c) Two girls talking | (d) Misty village scene |
| --- | --- | --- | --- | --- |
| SDXL | [fig05_a_sdxl.png](fig05_a_sdxl.png) | [fig05_b_sdxl.png](fig05_b_sdxl.png) | [fig05_c_sdxl.png](fig05_c_sdxl.png) | [fig05_d_sdxl.png](fig05_d_sdxl.png) |
| RPG | [fig05_a_rpg.png](fig05_a_rpg.png) | [fig05_b_rpg.png](fig05_b_rpg.png) | [fig05_c_rpg.png](fig05_c_rpg.png) | [fig05_d_rpg.png](fig05_d_rpg.png) |
| Kolors | [fig05_a_kolors.png](fig05_a_kolors.png) | [fig05_b_kolors.png](fig05_b_kolors.png) | [fig05_c_kolors.png](fig05_c_kolors.png) | [fig05_d_kolors.png](fig05_d_kolors.png) |
| RPG+Kolors | [fig05_a_rpg_kolors.png](fig05_a_rpg_kolors.png) | [fig05_b_rpg_kolors.png](fig05_b_rpg_kolors.png) | [fig05_c_rpg_kolors.png](fig05_c_rpg_kolors.png) | [fig05_d_rpg_kolors.png](fig05_d_rpg_kolors.png) |
| BiReG | [fig05_a_bireg.png](fig05_a_bireg.png) | [fig05_b_bireg.png](fig05_b_bireg.png) | [fig05_c_bireg.png](fig05_c_bireg.png) | [fig05_d_bireg.png](fig05_d_bireg.png) |

## Prompt file format

`prompts.jsonl` is UTF-8 encoded and contains four records, one JSON object per line.

| Field | Meaning |
| --- | --- |
| `case_id` | Panel identifier: `a`, `b`, `c`, or `d`. |
| `case_title` | English case title used in the figure. |
| `input_language` | `en`, indicating English input for all five pipelines. |
| `prompt_en` | English prompt corresponding to the case. |
| `images` | Image filenames keyed by `sdxl`, `rpg`, `kolors`, `rpg_kolors`, and `bireg`. |

## Visual inspection

The prompts specify the following content:

- **(a):** One document, two cups, three fountain pens, and two potted plants on a desk.
- **(b):** Five pine trees on a rooftop.
- **(c):** Two girls talking on a street corner.
- **(d):** A misty morning village, villagers washing clothes by a creek, smoke from distant chimneys, birds on rooftops, and a dog resting by a door.

The original PNG files retain their native pixel dimensions. Method labels and case labels are added when composing the manuscript figure.

These files accompany the qualitative analysis in the [BiReG repository](https://github.com/YeZhuang-Joe/BiReG).
