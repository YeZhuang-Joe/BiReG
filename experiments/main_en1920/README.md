# English Main Experiment — EN1920

This directory provides the 1,920 English prompts used in the BiReG main experiment, covering six categories with 320 prompts each: color, shape, texture, spatial relationships, non-spatial relationships, and complex compositions.

## Prompt File

`data/prompts.frozen.jsonl`

Each line is a JSON record containing:

- `prompt_id`: the experiment identifier linking the prompt to its generation and evaluation records.
- `prompt`: the English prompt text.

The same prompt set is used to compare six pipelines: SDXL, RPG, Kolors, RPG+Kolors, RAGD, and BiReG.
