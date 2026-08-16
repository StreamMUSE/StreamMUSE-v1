# Task 1 Report: Corpus And Backend-Neutral Contracts

Date: 2026-08-16

## Scope

Implemented Task 1 in the existing `real_rap` worktree only:

- `src/streammuse/experiments/rap_audio_protocols/__init__.py`
- `src/streammuse/experiments/rap_audio_protocols/contracts.py`
- `src/streammuse/experiments/rap_audio_protocols/corpus.py`
- `tests/unit/experiments/rap_audio_protocols/test_corpus.py`
- `tests/fixtures/rap_audio_protocols/two_bar_records.jsonl`

I did not alter or stage the pre-existing dirty files already present in the worktree.

## Implementation Summary

Added a new `streammuse.experiments.rap_audio_protocols` package that establishes the immutable Task 1 contracts and strict corpus loader:

- `ProtocolId` string enum with the four required backend identifiers.
- `SyllableTarget` immutable request item carrying word identity, within-word index, ARPAbet phones, lexical stress, target stress, boundary strength, absolute tick, tick in chunk, and target seconds.
- `TwoBarRenderRequest` immutable two-bar request with canonical JSON bytes and stable SHA-256.
- `ChunkRenderRecord` immutable render-record placeholder contract with canonical JSON helpers.
- `SongCorpus` immutable container enforcing the fixed 90 BPM, 4/4, four-ticks-per-beat campaign contract.
- `load_song_corpus()` strict JSONL loader that:
  - requires contiguous zero-based bars,
  - enforces nine syllables per bar,
  - reconstructs analyzed word order from the stored schedule,
  - validates schedule slot/tick/stress data against the declared built-in MCFlow template,
  - groups bars into backend-neutral two-bar requests.

## TDD Log

### Red 1

Ran:

```bash
.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_corpus.py -q
```

Observed expected import failure because `streammuse.experiments.rap_audio_protocols` did not exist.

### Green 1

Implemented the new package and loader. First pass exposed a real bug:

- `NameError: tempo is not defined` in syllable target construction.

Fixed by threading `Tempo` through request assembly.

### Red 2

During self-review I noticed a contract hole: repeated surface words in the same bar would produce wrong `index_in_word` values. I added a regression test that rewrites one fixture bar to contain two separate `rocket` words and ran:

```bash
.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_corpus.py::test_load_song_corpus_resets_within_word_index_for_repeated_surface_words -q
```

It failed with `rocket_indices == [0, 1, 2, 3]`.

### Green 2

Fixed `index_in_word` to count only the current contiguous word run, then reran the targeted regression test and the full Task 1 file.

## Verification

Fresh verification commands run after implementation:

```bash
.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_corpus.py -q
git diff --check -- src/streammuse/experiments/rap_audio_protocols tests/unit/experiments/rap_audio_protocols tests/fixtures/rap_audio_protocols
```

Results:

- `6 passed in 0.35s`
- `git diff --check` returned clean output

## Self-Review

Checked the final Task 1 implementation against the brief:

- Required files exist.
- Required exported interfaces exist.
- Loader accepts the two-bar smoke fixture with the exact requested values.
- Negative coverage exists for:
  - noncontiguous bars,
  - schedule/word-order mismatch,
  - non-nine-syllable bars,
  - non-90-BPM corpus construction,
  - repeated-word within-word indexing.
- Canonical JSON and SHA-256 are exposed on the request contract.

## Residual Concerns

- `ChunkRenderRecord` is intentionally minimal because Task 1 only defines the immutable interface boundary; later tasks may extend how it is populated, but the immutable JSON/sha contract is already in place.
