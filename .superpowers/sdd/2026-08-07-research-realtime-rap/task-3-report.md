# Task 3 Implementation Report: Pronunciation-Backed Prosody

## Changed Files

- `pyproject.toml`
- `uv.lock`
- `src/streammuse/application/rap/service.py`
- `src/streammuse/domain/rap/__init__.py`
- `src/streammuse/domain/rap/models.py`
- `src/streammuse/domain/rap/prosody.py`
- `src/streammuse/infrastructure/rap/prosody.py`
- `tests/unit/domain/rap/__init__.py`
- `tests/unit/domain/rap/test_prosody.py`
- `tests/unit/infrastructure/rap/__init__.py`
- `tests/unit/infrastructure/rap/test_prosody.py`

The two test-package markers keep the contract-mandated domain and infrastructure files, which share the basename `test_prosody.py`, from colliding during pytest collection.

## Implementation

- `Syllable` is still frozen, preserves `label`, and now stores integer `stress`. Its `stressed` property remains the historical boolean API (`stress > 0`).
- `ProsodyAnalysis` is a frozen data object containing normalized text, syllables, rhyme tail, OOV/fallback diagnostics, and zero-based punctuation boundaries.
- `ProsodyAnalyzer` is declared in `application/rap/service.py` as the replaceable application boundary.
- `HeuristicProsodyAnalyzer` wraps the existing `analyse_syllables(text)` algorithm. It preserves the public heuristic baseline while labeling its output `vowel_group_heuristic`.
- `CmuProsodyAnalyzer` takes `pronouncing.phones_for_word(word)[0]`, splits syllables at digit-bearing ARPABET vowel phones, retains phone chunks and integer stress, and computes the tail from the final primary/secondary stressed vowel.
- Per-word dictionary misses use the heuristic analyzer, retain the word in scheduling output, and populate both `oov_words` and `heuristic_words`.

## RED/GREEN Evidence

Initial focused RED command:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap/test_prosody.py \
  tests/unit/infrastructure/rap/test_prosody.py -v
```

It failed at collection because `ProsodyAnalysis` and `streammuse.infrastructure.rap.prosody` did not exist.

Additional RED cases caught two mapper/provenance defects:

- OOV syllables initially retained the legacy `heuristic` source instead of `vowel_group_heuristic`.
- `moving, moving!` initially returned `(3,)` rather than `(1, 3)` because repeated word strings were counted across subsequent occurrences.
- `night—day` initially produced no boundary because the em dash was not a punctuation token.

The final focused GREEN command passed all 13 tests:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap/test_prosody.py \
  tests/unit/infrastructure/rap/test_prosody.py -v
```

## Dependency and Lock Update

- Added `pronouncing>=0.2.0` to project dependencies.
- `uv sync` first encountered sandbox DNS failure. An approved unrestricted `uv lock` then resolved 194 packages and wrote the genuine lockfile update.
- The lock selects `pronouncing 0.3.0` and `cmudict 1.1.3`.
- `UV_CACHE_DIR=/tmp/streammuse-uv-cache uv lock --check` passed.
- No neural phonemizer or model is used.

## Tokenization and Punctuation

- Words are normalized to lower-case ASCII-English tokens, preserving apostrophes inside words such as `don't`.
- Punctuation runs containing comma, semicolon, colon, ASCII hyphen, en dash, em dash, question mark, exclamation mark, or period map to the preceding syllable's zero-based index.
- `"don't stop, now!"` yields boundaries `(1, 2)`; the internal apostrophe is not a boundary.
- `"moving, moving!"` yields `(1, 3)`.
- Empty or non-word input produces a valid analysis with empty tuples and normalized text `""`.

## CMU Stress and Rhyme Examples

For `"moving night"`, the selected first CMU pronunciations produce:

```text
moving: (M UW1 V), (IH0 NG)
night:  (N AY1 T)
stress: [1, 0, 1]
end rhyme tail: (AY1, T)
```

The selected first pronunciation is deterministic and stored as `cmudict_first_pronunciation` provenance.

## OOV Diagnostics

For `"xyzzy beat"`, `xyzzy` remains in `syllables`, has `vowel_group_heuristic` provenance, and appears in both `oov_words` and `heuristic_words`. `beat` still supplies the final CMU rhyme tail. If the final word is OOV, the rhyme tail is empty because no dictionary phones are available.

## Compatibility Notes

- `analyse_syllables(text)` remains public and preserves its vowel-group counts and boolean `.stressed` behavior.
- `Syllable.label` remains unchanged.
- Existing alignment and CLI code continue to consume `.stressed` without modification.

## Tests Run

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap/test_prosody.py \
  tests/unit/infrastructure/rap/test_prosody.py -v
# 13 passed

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/test_cli_config_parser.py \
  tests/integration/test_cli_entry_point.py -q
# 123 passed, 1 existing pretty_midi/pkg_resources deprecation warning

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest tests/ -q --tb=short
# passed; rap modules also passed compileall
```

The exact unscoped command `UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest` also ran, but collection entered the vendored `transformers` tree and failed because Keras 3 requires unavailable `tf_keras`. This is outside `tests/` and unrelated to Task 3.

## Limitations and Concerns

- Only the first CMU pronunciation is modeled. Alternate pronunciations, regional dialect, elision, pronunciation bending, and non-English words are not modeled.
- The OOV fallback estimates stress and syllables from vowel groups and cannot infer a phonetic rhyme tail.
- The vendored Transformers test collection remains blocked by the environment's missing `tf_keras`; the repository's normal `tests/` suite and all requested Task 3/rap/CLI scopes passed.

## Self-Review

- Confirmed frozen value objects, tuple-based analysis fields, and no neural dependency.
- Confirmed `git diff --check` has no whitespace errors.
- Direct analyzer probe confirmed CMU phone chunks/stress, `('AY1', 'T')` rhyme tail, OOV diagnostics, and apostrophe-safe punctuation boundaries.

## Fix Round 1

- Added direct CMU regression assertions for the selected
  `cmudict_first_pronunciation` provenance, the exact `moving` and `night`
  ARPABET syllable phone chunks, and exact rhyme tail `('AY1', 'T')`.
- Added `beat xyzzy` coverage proving that an OOV final word yields an empty
  dictionary rhyme tail while retaining `xyzzy` in `oov_words`.
- Rebuilt `uv.lock` using the official `uv 0.8.13` macOS ARM64 binary after
  restoring the prior genuine revision-3 lock as resolver input. The resulting
  lock remains `revision = 3`; compared with the pre-Task-3 lock, it adds only
  `pronouncing`, `cmudict`, `importlib-metadata`, `importlib-resources`,
  `zipp`, and the two StreamMUSE dependency declarations (57 inserted lines).
- `UV_CACHE_DIR=/tmp/streammuse-uv-cache .venv/bin/uv-aarch64-apple-darwin/uv lock --check`
  passed with the same resolver.
- Fix-round verification:

```text
UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap/test_prosody.py \
  tests/unit/infrastructure/rap/test_prosody.py -v
# 14 passed

UV_CACHE_DIR=/tmp/streammuse-uv-cache uv run python -m pytest \
  tests/unit/domain/rap tests/unit/application/rap \
  tests/unit/infrastructure/rap tests/unit/presentation/rap \
  tests/unit/presentation/test_cli_config_parser.py \
  tests/integration/test_cli_entry_point.py -q
# 124 passed, 1 existing pretty_midi/pkg_resources deprecation warning

git diff --check
# passed
```
