# Workflow details

## Language routing

A normalized copy is used for detection only. URLs and paired quoted text are removed from the detection copy; quoted content is restored if the remaining text contains no supported letters. Count Han characters H and ASCII English words E, with apostrophes allowed. H/(H+E) ≥ 0.7 selects Chinese; ≤ 0.3 selects English. Intermediate or empty evidence requires `--language`. This is a deterministic two-language routing heuristic, not a calibrated language probability or a general multilingual detector. The original prompt remains unchanged. Routing decisions, overrides, counts, thresholds and Unicode database version are logged.

## API and planning

The public example requests `deepseek-flash`, temperature 0, max_tokens 1200 and thinking disabled. Availability and schema support depend on the configured provider. The returned model field is recorded as returned, not independently authenticated. The key is read from a local file, temporarily supplied to the planning module through a process environment variable, then restored before rendering. Authorization headers are not serialized.

At most two attempts are configured by default. A structurally rejected response may receive one retry with the previous response and validation error. HTTP errors and interrupted/uncertain transport attempts stop rather than silently repeating an unknown request. The first executable plan is used. No output-quality selection, automatic baseline fallback, or post-hoc numerical repair is performed by this daily generation entry.

The native row/column representation normalizes positive weights. Validation checks region-description count, reserved tokens, finite positive geometry, maximum region count and nonempty discrete crops at supported feature resolutions. Acceptance is an execution check, not proof of semantic correctness. Prompt parts exceeding the backend encoder limit may be truncated by the retained encoder and are recorded accordingly.

## Rendering and records

The regional backend and language templates are preserved byte-for-byte. The wrapper handles a single-cell layout and restores the historical seed helper's reassignment of `torch.use_deterministic_algorithms`. Scheduler nonfinite metadata is encoded using explicit `__nonfinite_float__` markers; sampling values are not changed. Other JSON fields remain strict.

Planning time covers HTTP client wall time through full response-body receipt, including network/service queuing, excluding parsing, file writes and GPU work. Generation time uses CUDA synchronization around text encoding, denoising and decoding; it excludes model loading and PNG saving. These per-run diagnostics do not by themselves constitute a formal efficiency experiment or a multi-planner comparison.

Each output is bound to its prompt, configuration, source hashes and selected plan. Completed image hashes are checked on resume. The key itself and its hash are omitted from public configuration snapshots. Local responses and logs should still be reviewed before any publication.

## Release boundaries

This entry is for fresh prompt-to-image runs. It does not reconstruct old paper images solely from their seeds and does not alter archived experiments. Paper tables, frozen manifests, scoring and human-evaluation materials belong in the corresponding experiment directories. One English and one Chinese full inference run on the existing target server have now completed, according to the author-supplied console logs dated 2026-09-27; see [the smoke-run record](SMOKE_TEST_20260927.md). These functional checks are separate from the included offline tests and formal experiments. Complete third-party notices remain a separate release task.
