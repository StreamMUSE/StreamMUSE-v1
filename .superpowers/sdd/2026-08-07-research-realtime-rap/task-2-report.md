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

No user-supplied public corpus file was available outside the invented test assets. Per the no-download constraint, no corpus data was fetched, copied, or committed; consequently the required transient real-corpus extraction observations could not be recorded. A later run should supply a local corpus checkout and record only aggregate counts, slot-count variation, quantization errors, and anonymous rejection categories.
