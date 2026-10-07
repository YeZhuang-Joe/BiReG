# Figure 9 — Region–Global Fusion Weight

This folder contains the 12 original images used in Figure 9 of the BiReG manuscript. The accompanying `prompts.jsonl` records the two bilingual descriptions, layout ratios, fusion weights, and corresponding image filenames.

## Comparison

The figure compares six region–global fusion weights: λ = 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0. Here λ denotes the global-conditioning weight in the method's fusion equation. Smaller values emphasize regional conditioning; larger values increase the relative contribution of global conditioning.

- **Case (a): Three-region layout.** Sky and distant mountains above, a traveler at the lower left, and a cat at the lower right.
- **Case (b): Four-grid layout.** Sun at the upper left, clouds at the upper right, a child at the lower left, and a dog at the lower right.

These descriptions and layout ratios correspond to cases (c) and (d) in the Figure 8 materials. The bilingual text is retained in full in this folder.

## Original images

| Fusion weight λ | (a) Three-region layout | (b) Four-grid layout |
| --- | --- | --- |
| 0.0 | [fig09_a_lambda_00.png](fig09_a_lambda_00.png) | [fig09_b_lambda_00.png](fig09_b_lambda_00.png) |
| 0.2 | [fig09_a_lambda_02.png](fig09_a_lambda_02.png) | [fig09_b_lambda_02.png](fig09_b_lambda_02.png) |
| 0.4 | [fig09_a_lambda_04.png](fig09_a_lambda_04.png) | [fig09_b_lambda_04.png](fig09_b_lambda_04.png) |
| 0.6 | [fig09_a_lambda_06.png](fig09_a_lambda_06.png) | [fig09_b_lambda_06.png](fig09_b_lambda_06.png) |
| 0.8 | [fig09_a_lambda_08.png](fig09_a_lambda_08.png) | [fig09_b_lambda_08.png](fig09_b_lambda_08.png) |
| 1.0 | [fig09_a_lambda_10.png](fig09_a_lambda_10.png) | [fig09_b_lambda_10.png](fig09_b_lambda_10.png) |

The figure presents each case in two rows of three images. The upper row uses λ = 0.0, 0.2, and 0.4; the lower row uses λ = 0.6, 0.8, and 1.0. The two-digit filename suffix encodes the displayed weight, for example `02` for 0.2 and `10` for 1.0.

## Prompt records

`prompts.jsonl` is UTF-8 encoded and contains one JSON object per case. Each object maps the description and layout to all six weight-specific images.

| Field | Meaning |
| --- | --- |
| `case_id` | Figure panel identifier: `a` or `b`. |
| `case_title` | Layout title used in the figure. |
| `prompt_zh` | Chinese description of the scene and spatial arrangement. |
| `prompt_en` | Corresponding English translation. |
| `layout_type` | Layout identifier. |
| `split_ratio` | Layout ratio string. |
| `images` | Six records pairing a numerical fusion weight (`lambda`) with a relative image filename (`image_path`). |

## Layout ratios

In `split_ratio`, semicolons separate rows from top to bottom. The first number in each row specifies its fraction of the image height; subsequent numbers specify column-width fractions within that row, from left to right.

| Case | `split_ratio` | Spatial arrangement |
| --- | --- | --- |
| (a) | `0.4,1;0.6,0.5,0.5` | A full-width upper region occupying 40% of the image height; the remaining 60% is divided into equal left and right regions. |
| (b) | `0.5,0.5,0.5;0.5,0.5,0.5` | Two equal-height rows, each divided into two equal-width columns. |

## Figure presentation

The PNG files retain the original generated images. Case headings and weight labels are added separately in LaTeX. Open or download the linked PNG files to inspect the images at their original resolution.

This qualitative comparison illustrates changes in entity scale, position, and count as the fusion weight varies.

Project repository: [BiReG](https://github.com/YeZhuang-Joe/BiReG).
