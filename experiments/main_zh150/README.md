# Chinese Main Experiment — ZH150

This directory provides the 150 Chinese prompts and their corresponding fixed English translations used in the BiReG main experiment.

## Prompt File

`data/prompts.frozen.jsonl`

Each line is a JSON record containing:

- `prompt_id`: the experiment identifier linking the prompt to its generation and evaluation records.
- `prompt`: the original Chinese prompt text.
- `translation_en`: the corresponding fixed English translation.

Six pipelines are compared: SDXL, RPG, Kolors, RPG+Kolors, RAGD, and BiReG. Kolors, RPG+Kolors, and BiReG use the original Chinese prompts, while SDXL, RPG, and RAGD use the same fixed English translations.
