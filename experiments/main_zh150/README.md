# Chinese Main Experiment — ZH150

## Frozen prompt manifest

`data/prompts.frozen.jsonl` contains the 150 retained main-evaluation prompts. This release exports the complete parent-pool snapshot archived with the human-sampling materials; it is not the 89-prompt human-evaluation subset. Prompt strings and identifiers are preserved exactly. No prompts were newly selected, rewritten or dropped during this export.

Run from this directory:

```bash
python verify_prompts.py
```

Python standard library only; no API or GPU required. The verifier checks the frozen file hash, record count, unique IDs/texts and required fields.

## Field definitions and provenance

- `prompt_id`: retained experiment identifier, used to join planning, image and score records.
- `prompt_original`: exact archived original-language prompt.
- `source_dataset`: benchmark attribution of the archived pool.
- `source_record_id`: null; an official native record ID has not been independently established.
- `source_index_archived`: original archived index, preserved without asserting that it is an official ID or official row number.
- `source_manifest_file` / `source_manifest_line_1based`: filename and 1-based line in the archived source snapshot.
- `data/prompts.provenance.json`: source/archive checksums, transformation notes and the released manifest checksum. These identify archived files, not an official dataset version.

These retained evaluation pools have a post-hoc selection history. This export does not change that history or establish random sampling. Full upstream selection decisions and benchmark-native identifiers are not reconstructed here.

`annotations` preserves the benchmark annotations and checkpoints. `translation_en` is the archived FINAL English translation (`final_translation_en`, verified equal to `prompt_en`), not the earlier draft alias. There are 138 retained and 12 corrected translations. Review was AI-assisted; status/type and whether human review was recorded are retained explicitly. These flags must not be described as completed human verification.

In the reported Chinese comparison, Kolors, RPG+Kolors and BiReG use original Chinese, while SDXL, RPG and RAGD use the frozen English translations. This manifest does not by itself verify every historical generation record's input.

## Cross-checks and release scope

Exact prompt text was cross-checked against all 89 retained human-evaluation prompts and all 152 planner-comparison prompts; the Chinese human subset also matched the frozen final translations. These subset checks are not a full join against all six pipelines' main-generation and score records.

This update releases prompts only. Main-experiment frozen plans, generation configurations/scripts, archived scores, statistical reproduction and complete image download links remain to be integrated. Do not interpret this directory as a complete end-to-end reproduction release yet.
