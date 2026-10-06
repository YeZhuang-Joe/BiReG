# Figure 4 — Regional Control Examples

This folder contains the four source JPEG images used in Figure 4, together with their regional descriptions and intended layouts.

## Image mapping

| Case | Layout | File | Regional assignment |
| --- | --- | --- | --- |
| (a) Heaven and Hell | Left–Right | `fig04_a_left_right.jpg` | Left: Hell; right: Heaven |
| (a) Heaven and Hell | Top–Bottom | `fig04_a_top_bottom.jpg` | Top: Heaven; bottom: Hell |
| (b) Forest and Desert | Left–Right | `fig04_b_left_right.jpg` | Left: Forest; right: Desert |
| (b) Forest and Desert | Top–Bottom | `fig04_b_top_bottom.jpg` | Top: Forest; bottom: Desert |

All four examples use equal regional allocation ratios of 0.5:0.5.

## Prompt records

`prompts.jsonl` contains four UTF-8 JSON records, one per image. Each record identifies the case, image filename, layout, regional ratios, and regional descriptions.

The Chinese descriptions are transcribed from the original manuscript figure. The English texts are documentation translations for readers; they are not recorded English generation inputs. “Desert” is the concise figure label for the Chinese barren-wasteland description.

## Figure presentation

The manuscript uses a single-column, two-row, two-column layout. Region names and dashed division lines are added as LaTeX overlays. Dashed lines identify the intended equal-area layout divisions.

The four JPEG files should be uploaded alongside this README and `prompts.jsonl`. The manuscript accesses them under `paper_figures/figure_04/`. This folder documents the displayed images and regional descriptions; generation seeds and request settings are not specified in these prompt records.

