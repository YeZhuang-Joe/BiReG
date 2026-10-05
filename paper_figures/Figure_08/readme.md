# Figure 8 — Planning and Regional Control

This folder contains the 12 original images used in Figure 8 of the BiReG manuscript, together with the four bilingual descriptions, layout ratios, and image-file mappings in `prompts.jsonl`.

## Configurations

| Configuration | Description |
| --- | --- |
| Origin | Direct generation from the input prompt without regional planning. |
| Merged | Planned regional descriptions combined into a single global prompt, without spatially assigned regional conditioning. |
| BiReG | Planned regional descriptions coupled with spatially assigned regional conditioning. |

The figure arranges these configurations in three columns: Origin, Merged, and BiReG. Each case shows a layout diagram on the left and the Chinese description and English translation below the image row.

## Cases and original images

| Case | Layout | Origin | Merged | BiReG |
| --- | --- | --- | --- | --- |
| (a) | Left–Right | [fig08_a_origin.png](fig08_a_origin.png) | [fig08_a_merged.png](fig08_a_merged.png) | [fig08_a_bireg.png](fig08_a_bireg.png) |
| (b) | Top–Bottom | [fig08_b_origin.png](fig08_b_origin.png) | [fig08_b_merged.png](fig08_b_merged.png) | [fig08_b_bireg.png](fig08_b_bireg.png) |
| (c) | Three-Region | [fig08_c_origin.png](fig08_c_origin.png) | [fig08_c_merged.png](fig08_c_merged.png) | [fig08_c_bireg.png](fig08_c_bireg.png) |
| (d) | Four-Grid | [fig08_d_origin.png](fig08_d_origin.png) | [fig08_d_merged.png](fig08_d_merged.png) | [fig08_d_bireg.png](fig08_d_bireg.png) |

## Prompt records

`prompts.jsonl` is UTF-8 encoded and contains one JSON object per case.

| Field | Meaning |
| --- | --- |
| `case_id` | Panel identifier: `a`, `b`, `c`, or `d`. |
| `case_title` | Layout title displayed in the figure. |
| `prompt_zh` | Chinese description displayed below the image row. |
| `prompt_en` | Corresponding English translation displayed in the figure. |
| `layout_type` | Layout identifier. |
| `split_ratio` | Layout ratio string corresponding to the displayed diagram. |
| `images` | Relative image filenames for the three configurations. |

## Layout ratios

In `split_ratio`, semicolons separate rows from top to bottom. The first number in each row specifies its fraction of the image height; subsequent numbers specify column-width fractions within that row, from left to right.

| Case | `split_ratio` | Spatial arrangement |
| --- | --- | --- |
| (a) | `1,0.5,0.5` | One full-height row divided into equal left and right regions. |
| (b) | `0.45,1;0.55,1` | Full-width upper and lower regions occupying 45% and 55% of the image height. |
| (c) | `0.4,1;0.6,0.5,0.5` | A full-width upper region occupying 40% of the image height, with the remaining 60% divided into equal left and right regions. |
| (d) | `0.5,0.5,0.5;0.5,0.5,0.5` | Two equal-height rows, each divided into two equal-width columns. |

## Figure presentation

The PNG files retain the original generated images. Labels, shaded text bands, layout diagrams, and colored dashed outlines are added separately in LaTeX. The outlines indicate the regional conditioning partitions and correspond to the colored layout diagrams; they are not object segmentation boundaries. Open or download the linked PNG files to inspect the images at their original resolution.

Project repository: [BiReG](https://github.com/YeZhuang-Joe/BiReG).
