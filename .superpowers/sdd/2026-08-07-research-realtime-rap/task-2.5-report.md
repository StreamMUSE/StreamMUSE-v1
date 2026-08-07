# Task 2.5 Report: Representative Local MCFlow Sample Catalog

## Files Changed

- `src/streammuse/infrastructure/rap/sample_catalog.py`
- `scripts/build_mcflow_sample_catalog.py`
- `src/streammuse/infrastructure/rap/mcflow.py`
- `tests/unit/infrastructure/rap/test_sample_catalog.py`
- `tests/unit/infrastructure/rap/test_mcflow.py`
- `tests/unit/scripts/test_build_mcflow_sample_catalog.py`
- `docs/superpowers/plans/2026-08-07-research-realtime-rap.md` (pre-existing Task 2.5 amendment preserved)
- This report.

## TDD Evidence

Initial RED command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_sample_catalog.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q
```

Expected output: collection failed with `ModuleNotFoundError: No module named 'streammuse.infrastructure.rap.sample_catalog'`.

Additional RED commands:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_records_invalid_break_values_without_stopping_other_measures -q
```

Expected output: failed at `_parse_break` with `ValueError: invalid break value`.

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_sample_catalog.py::test_selection_sorts_mixed_rhyme_topologies_deterministically -q
```

Expected output: failed with `TypeError: '<' not supported between instances of 'NoneType' and 'int'` while sorting structural signatures.

Final focused GREEN command before broader regressions:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_sample_catalog.py tests/unit/infrastructure/rap/test_mcflow.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q --tb=no
```

Output: `41 passed in 0.07s`.

Final Task 2 and Task 2.5 command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py tests/unit/infrastructure/rap/test_sample_catalog.py tests/unit/scripts/test_extract_mcflow_templates.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q --tb=no
```

Output: `43 passed in 0.07s`.

Focused rap regression command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap tests/unit/scripts/test_extract_mcflow_templates.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q --tb=no
```

Output: `94 passed in 0.65s`.

## Catalog Contract

`structural_flow_signature` consists only of ordered slot tuples:
`(onset_tick, duration_ticks, target_stress, boundary_strength, canonical_rhyme_index)`.
All template metadata is ignored. Rhyme labels are replaced by integer indices assigned in first-occurrence order; absent rhymes remain null. Thus structurally equivalent arbitrary labels deduplicate, while a different rhyme topology does not.

Supported density ranges are sparse `4..7`, medium `8..11`, and dense `12..16` slots. Selection validates anonymous 4x4 extracted templates, keeps the lexicographically smallest anonymous template ID within each duplicate structure, sorts unique structures by a fully comparable normalized signature, and selects at most the requested limit at evenly spaced stable positions. The first and last stable entries are included whenever the limit is greater than one. The selected output concatenates sparse, medium, then dense templates.

The sidecar schema is `streammuse.mcflow_sample.v1`. It records exact ranges, requested limit, aggregate totals, and available/selected/underfilled counts per band. It intentionally excludes source hashes, IDs, names, filenames, paths, lyrics, IPA, artists, and titles. The sampled catalog itself remains a standard Task 2 extraction document and preserves parsed-file and rejection metadata.

Unsupported break annotations now cause an anonymous measure-level `invalid_break_value` rejection instead of aborting the entire local directory extraction. No value is retained in the rejection detail.

## CLI Behavior

`scripts/build_mcflow_sample_catalog.py` accepts `--mcflow-dir`, `--catalog-output`, `--report-output`, `--per-bucket`, and `--max-quantization-error-ticks`. It creates output parents, performs extraction and deterministic sampling, and prints aggregate counts plus the two requested output paths. Invalid input or arguments return `2`. It returns `1` for an empty selection or any empty density band, after writing the inspectable catalog and report.

## Five-File Local Experiment

The API and CLI both ran with `per_bucket=10` and a maximum quantization error of `0.25` ticks. Both selected catalogs reloaded through `load_extracted_templates` and `TemplateCatalog`; their aggregate sidecars matched exactly. Catalog and report JSON schema-whitelist audits passed, with the report confirmed aggregate-only.

| Field | Aggregate result |
| --- | ---: |
| Parsed files | 5 |
| Accepted source measures | 159 |
| Rejected source measures | 73 |
| Structurally unique templates | 143 |
| Duplicates removed | 16 |
| Duplicate rate | 10.06% |
| Out-of-range unique structures | 6 |
| Selected templates | 30 |
| Maximum accepted quantization error | 0.0 ticks |

| Density band | Available | Selected | Underfilled |
| --- | ---: | ---: | ---: |
| Sparse (4..7) | 19 | 10 | 0 |
| Medium (8..11) | 50 | 10 | 0 |
| Dense (12..16) | 68 | 10 | 0 |

Selected slot-count distribution: `4: 2`, `5: 3`, `6: 3`, `7: 2`, `9: 2`, `10: 5`, `11: 3`, `12: 3`, `13: 3`, `14: 3`, `15: 1`.

Rejection categories: `quantization_error: 26`, `incomplete_measure: 17`, `overfull_measure: 17`, `invalid_break_value: 8`, `unrepresentable_phrase_break: 5`.

Representative normalized structural timelines use `(onset, duration, stress, boundary, canonical_rhyme_index)` and contain no source identity.

```text
sparse: [(0,1,1.0,0,-), (1,1,0.0,0,-), (2,1,1.0,0,-), (3,1,0.0,0,-), (4,12,0.0,0,-)]
medium: [(0,1,0.0,0,-), (1,1,0.0,0,-), (2,1,1.0,0,-), (3,1,0.0,0,-), (4,2,1.0,0,0), (6,2,1.0,0,1), (8,2,1.0,2,-), (10,2,1.0,0,-), (12,2,1.0,3,2), (14,1,1.0,0,-), (15,1,0.0,0,-)]
dense: [(0,1,0.0,0,-), (1,1,0.0,0,-), (2,1,1.0,0,0), (3,1,0.0,0,-), (4,1,0.0,0,-), (5,1,1.0,0,1), (6,1,0.0,1,-), (7,1,0.0,0,-), (8,1,1.0,0,1), (9,1,0.0,0,-), (10,2,1.0,0,1), (12,2,1.0,0,0), (14,2,0.0,0,-)]
```

## Anonymity and Tracking Audit

All real-derived outputs were written only below `/tmp/streammuse-mcflow-sample-output`. This report contains aggregate counts and normalized structural timelines only. No corpus file, source filename, path, source hash, artist/title metadata, lyric, or IPA text is in the report. The commit audit must confirm no `.rap` file or temporary catalog/report is tracked.

## Self-Review and Concerns

The report contract and sampled-catalog write contract both validate count consistency. Unit coverage exercises duplicate representatives, metadata independence, rhyme canonicalization, structural distinctions, range handling, order independence, invalid inputs, standard-schema reload, sidecar anonymity, CLI underfill behavior, invalid break recovery, and mixed optional-rhyme sort order.

Concern: a source measure with an unsupported break annotation is deliberately rejected rather than approximated, which preserves conservative structural semantics but can reduce available templates in corpora with such annotations. The observed experiment retained sufficient structures for all three density bands.

## Fix Round 1

### Changed Files

- `src/streammuse/infrastructure/rap/mcflow.py`
- `src/streammuse/infrastructure/rap/sample_catalog.py`
- `scripts/build_mcflow_sample_catalog.py`
- `tests/unit/infrastructure/rap/test_mcflow.py`
- `tests/unit/infrastructure/rap/test_sample_catalog.py`
- `tests/unit/scripts/test_build_mcflow_sample_catalog.py`
- This report.

### RED Evidence

The following focused command was run before implementation:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_records_reciprocal_and_stress_parse_failures_per_measure tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_equal_resolved_destinations_before_writing tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_mutated_anonymous_source_count tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_fabricated_band_counts tests/unit/scripts/test_build_mcflow_sample_catalog.py::test_cli_returns_two_for_input_read_failure_without_echoing_source_path tests/unit/scripts/test_build_mcflow_sample_catalog.py::test_cli_rejects_equal_resolved_destinations_without_writing -q
```

Output: `6 failed`. The reciprocal case raised `ValueError: unsupported reciprocal duration`; the destination and report-correspondence cases did not raise; the mocked input read raised `OSError`; and the CLI alias case returned `1` after overwriting the catalog.

Focused GREEN command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_records_reciprocal_and_stress_parse_failures_per_measure tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_equal_resolved_destinations_before_writing tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_mutated_anonymous_source_count tests/unit/infrastructure/rap/test_sample_catalog.py::test_write_sample_catalog_rejects_fabricated_band_counts tests/unit/scripts/test_build_mcflow_sample_catalog.py::test_cli_returns_two_for_input_read_failure_without_echoing_source_path tests/unit/scripts/test_build_mcflow_sample_catalog.py::test_cli_rejects_equal_resolved_destinations_without_writing -q --tb=no
```

Output: `6 passed in 0.04s`.

Task 2, Task 2.5, and both catalog/extraction CLI test files:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py tests/unit/infrastructure/rap/test_sample_catalog.py tests/unit/scripts/test_extract_mcflow_templates.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q --tb=no
```

Output: `49 passed in 0.08s`.

Focused rap regressions:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap tests/unit/scripts/test_extract_mcflow_templates.py tests/unit/scripts/test_build_mcflow_sample_catalog.py -q --tb=no
```

Output: `100 passed in 0.21s`.

### Fixes

- Unsupported reciprocal-duration and stress annotations are now captured as anonymous `invalid_reciprocal_duration` and `invalid_stress_value` measure rejections. The parser retains file-structural failures as file-level errors, does not preserve malformed tokens, and continues to later measures.
- The local catalog CLI catches `OSError` from input extraction, emits `error: unable to read MCFlow input`, returns `2`, and does not expose the source path.
- Catalog/report paths are resolved and compared before validation, directory creation, or file writing. Equal destinations, including lexical aliases, are rejected; the CLI returns `2` and neither artifact is created.
- The public write boundary now recomputes deterministic selection from the supplied extraction and requested bucket limit. Both selected templates and the complete aggregate report must match that recomputation, preventing fabricated source-file or per-band values.

### Updated Five-File Experiment

API and CLI executions with `per_bucket=10` and maximum quantization error `0.25` completed successfully. Both outputs reloaded through `load_extracted_templates` and `TemplateCatalog`, passed catalog schema-whitelist and aggregate-only report audits, and had matching aggregate reports.

| Field | Aggregate result |
| --- | ---: |
| Parsed files | 5 |
| Accepted source measures | 159 |
| Rejected source measures | 73 |
| Structurally unique templates | 143 |
| Duplicates removed | 16 |
| Duplicate rate | 10.06% |
| Out-of-range unique structures | 6 |
| Selected templates | 30 |
| Maximum accepted quantization error | 0.0 ticks |

| Density band | Available | Selected | Underfilled |
| --- | ---: | ---: | ---: |
| Sparse (4..7) | 19 | 10 | 0 |
| Medium (8..11) | 50 | 10 | 0 |
| Dense (12..16) | 68 | 10 | 0 |

Selected slot-count distribution: `4: 2`, `5: 3`, `6: 3`, `7: 2`, `9: 2`, `10: 5`, `11: 3`, `12: 3`, `13: 3`, `14: 3`, `15: 1`.

Rejection categories: `quantization_error: 26`, `incomplete_measure: 17`, `overfull_measure: 17`, `invalid_break_value: 8`, `unrepresentable_phrase_break: 5`. No malformed reciprocal or stress annotation occurred in this transient five-file sample.

The API and CLI produced the same aggregate normalized slot timelines previously recorded for sparse, medium, and dense representatives; their equality was checked without reading or reporting corpus identity or text.

### Anonymity, Tracking, and Self-Review

All Fix Round 1 real-derived outputs remain only below `/tmp/streammuse-mcflow-sample-output`. The commit audit verifies that no `.rap` corpus content or temporary catalog/report is tracked. The report contains only aggregate counts, rejection categories, and normalized structural output.

Self-review confirmed measure-local failure handling does not weaken missing/repeated spine or record-width failures, destination comparison occurs before parent creation, and recomputation binds all report fields to the supplied extraction and deterministic selection. Remaining concern: destination identity is based on `Path.resolve()` at write time; filesystem races after preflight remain outside the scope of this local CLI.
