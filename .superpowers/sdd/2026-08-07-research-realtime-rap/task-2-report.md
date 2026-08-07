# Task 2 Implementation Report

## Files Changed

- `src/streammuse/infrastructure/rap/mcflow.py`
- `scripts/extract_mcflow_templates.py`
- `tests/unit/infrastructure/rap/test_mcflow.py`
- `tests/unit/scripts/test_extract_mcflow_templates.py`
- `tests/fixtures/rap/` (invented structural fixture only)
- `docs/superpowers/plans/2026-08-07-research-realtime-rap.md` (pre-existing intentional execution-plan update, preserved for this commit)
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-2-report.md`

## TDD Evidence

RED command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py tests/unit/scripts/test_extract_mcflow_templates.py -q --tb=no
```

RED output:

```text
ERROR tests/unit/infrastructure/rap/test_mcflow.py
Interrupted: 1 error during collection
1 error in 0.09s
```

The expected initial failure was the missing extraction module.

GREEN command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py tests/unit/scripts/test_extract_mcflow_templates.py -q --tb=no
```

GREEN output:

```text
11 passed in 0.04s
```

Focused regression command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/domain/rap tests/unit/infrastructure/rap tests/unit/application/rap tests/unit/presentation/rap -q --tb=no
```

Focused regression output:

```text
60 passed in 0.16s
```

## Supported Constructs

- Exclusive spine discovery in any column order. Required spines are reciprocal duration, stress, phrase break, rhyme, and lyrics; tone, IPA, and hype are ignored structurally.
- Full-line and local comments, tandem interpretations, measure records, and termination records with strict record-width validation.
- Exact `Fraction` reciprocal durations: ordinary positive reciprocals, augmentation dots, and untied rational forms such as `n%d`. Reciprocal rest markers and lyric rest markers advance time without a slot.
- Numeric stress values `0`, `1`, and experimental `2`, mapped to `0.0`, `1.0`, and `1.0`.
- Numeric break strengths `0` through `5`; a phrase start attaches its strength to the preceding lyric-bearing slot, including over an accepted measure boundary within one source file.
- 4/4 extraction into the Task 1 clock, with exact quantization checks for onsets and durations and a configurable maximum tick error.
- Deterministic recursive directory traversal, anonymous catalog JSON, validated reload, and an opt-in CLI that never downloads data.

## Rejected Constructs

- Missing or repeated required spines, missing exclusive interpretations, malformed record widths, malformed meters, unsupported reciprocal notation, dotted rational reciprocal notation, and invalid stress or break values.
- Measures outside 4/4, incomplete measures, overfull measures, empty lyric-bearing measures, quantization errors above tolerance, nonpositive quantized durations, slots outside a bar, and duplicate quantized onsets.
- Catalog documents with an unknown schema, unknown fields, or malformed template, slot, provenance, rejection, or aggregate fields.

## Anonymity Checks

- Public parsed data contains only exact timings, mapped stress, phrase-break strength, rhyme group, measure ordinal, and a SHA-256 content hash. It does not retain lyric or IPA fields.
- Template IDs are derived from content hash and measure ordinal, not a supplied file name or path. Template names and provenance source strings are fixed anonymous values.
- JSON serialization uses a field whitelist. The transient extraction check verified that known invented lyric, IPA, fixture-name, and fixture-path tokens were absent from the emitted catalog.
- CLI output reports only aggregate counts and the requested output location. It never prints input file names.

## Synthetic Extraction Check

The invented fixture produced two accepted templates with slot counts of four and one, zero quantization error, and no rejection categories. This verified the varied-slot, rest, dotted-duration, rational-duration, stress, rhyme, phrase-break, and anonymity paths.

## Self-Review

- Confirmed phrase breaks cannot cross separate input-file boundaries.
- Confirmed catalog reload rebuilds `FlowTemplate` values accepted by the existing catalog contract.
- Confirmed deterministic source hashing and IDs for identical bytes at different locations.
- Ran `git diff --check`; it completed without output.
- An independent reviewer tool was not available in this environment, so this report records a manual specification and implementation review.

## Concerns

The original lack of a transient public input was resolved by the smoke test below. No corpus content was copied into the repository or emitted in this report. No remaining Task 2 concern is known.

## Transient Real-Corpus Smoke Test

Commands were run through both the Python API and the opt-in directory CLI using only temporary input and output locations. The API command extracted, serialized, reloaded, and inserted the resulting templates into `TemplateCatalog`. The CLI command extracted the same transient input with the configured 0.25-tick tolerance; its JSON output was independently reloaded and inserted into `TemplateCatalog`.

Aggregate anonymous output from both paths:

```text
accepted_templates=26
rejected_measures=8
accepted_slot_counts=[4, 11, 12, 13, 14, 15, 16]
max_quantization_error_ticks=0.0
rejection_codes={incomplete_measure: 4, overfull_measure: 4}
anonymous_catalog=validated
```

The output-field audit passed: only whitelisted anonymous structural fields were inspected, no textual-content or source-identifying fields were present, and both serialized catalogs reloaded into usable `FlowTemplate` instances accepted by `TemplateCatalog`.

The smoke test initially exposed two valid Humdrum constructs missing from the parser: non-meter tandem interpretations beginning with `*M`, and null stress continuations including one initial null. Focused invented regressions now preserve non-meter tandem records, carry later null stress values forward, and map an initial null stress to the neutral unaccented value. The repeated API and CLI smoke test then completed with the aggregate result above.

## Fix Round 1

### Changed Files

- `src/streammuse/infrastructure/rap/mcflow.py`
- `tests/unit/infrastructure/rap/test_mcflow.py`
- `.superpowers/sdd/2026-08-07-research-realtime-rap/task-2-report.md`

### RED Evidence

Empty-measure ordinal regression:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_records_empty_measure_without_reusing_its_ordinal -q --tb=no
1 failed in 0.04s
```

Phrase-break regressions:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_shifts_a_phrase_break_annotated_on_a_rest tests/unit/infrastructure/rap/test_mcflow.py::test_extract_rejects_phrase_break_that_would_cross_a_rejected_measure -q --tb=no
2 failed in 0.02s
```

Malformed catalog metadata regression:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_load_extracted_templates_rejects_malformed_rejection_and_aggregate_scalars -q --tb=no
4 failed in 0.03s
```

Anonymous serialize/reload boundary regressions:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py::test_flow_template_to_dict_rejects_nonanonymous_extracted_metadata tests/unit/infrastructure/rap/test_mcflow.py::test_load_extracted_templates_rejects_nonanonymous_metadata -q --tb=no
2 failed in 0.02s
```

### Implementation

- Measure starts now advance independently of data records. Empty measures retain their source ordinal and become `empty_measure` rejections.
- Phrase starts are retained as anonymous timing events even when annotated on rests. A shift may target only the immediately preceding lyric-bearing slot in the same accepted measure or its immediately preceding accepted measure. Rejected measures reset that continuity; affected structures receive `unrepresentable_phrase_break` rejections.
- Catalog loading now validates rejection scalar types, nonempty error details, SHA-256 format, positive measure ordinals, nonnegative aggregate counts, and aggregate/template/rejection count consistency.
- Serialization and loading enforce fixed anonymous extracted-template name, identifier, provenance kind, provenance source, source-hash, and finite nonnegative quantization-error invariants.

### Covering Tests

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/infrastructure/rap/test_mcflow.py -q --tb=no
21 passed in 0.06s

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/scripts/test_extract_mcflow_templates.py -q --tb=no
2 passed in 0.02s

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/unit/domain/rap tests/unit/infrastructure/rap tests/unit/application/rap tests/unit/presentation/rap -q --tb=no
72 passed in 0.20s
```

### Transient Real-Corpus Smoke Test

The Python API and opt-in CLI were run against the supplied transient input using only temporary locations. Their serialized anonymous catalogs were separately field-audited, reloaded, and inserted into `TemplateCatalog`.

```text
api accepted_templates=25 rejected_measures=16
cli accepted_templates=25 rejected_measures=16
accepted_slot_counts=[4, 11, 12, 13, 14, 15, 16]
max_quantization_error_ticks=0.0
rejection_codes={incomplete_measure: 4, overfull_measure: 4, unrepresentable_phrase_break: 8}
anonymous_catalog=validated
```

### Self-Review

- Empty and rejected measures both create chronological barriers for phrase-boundary shifting.
- Rest annotations are retained independently from lyric slot creation.
- Rejection and aggregate validation rejects malformed scalars before templates are returned.
- Anonymous invariants are enforced on both public serialization and loading paths.
- No corpus content was copied into the repository or included in this report.
