# Task 1 Report: Remote Chunk Contracts And Binary Package Codec

## Status

Completed and committed from the requested worktree. The commit contains only
Task 1 source and test files; all pre-existing worktree changes remain
unstaged and untouched.

## Implementation

- Added immutable remote two-bar contracts in `remote_chunk.py`.
  - `REMOTE_CHUNK_SCHEMA_VERSION` is exactly `streammuse.rap_chunk.v1`.
  - Requests require exactly two consecutive 4/4 bars, a finite positive BPM,
    a positive remaining budget, and the fixed 24 kHz transport sample rate.
  - The expected frame count is deterministically derived from two 4/4 bars.
    At 90 BPM it is exactly 128,000 frames.
  - Request IDs are SHA-256 hashes of canonical JSON identity payloads that
    exclude `remaining_budget_ms`; the canonical transport payload still
    contains that budget for every attempt.
  - Payload readers require exact keys and reconstruct flow templates and
    scheduled syllables. Diagnostics are recursively frozen through mapping
    proxies and tuples.
  - Selected bars retain bar and flow-template identity. Prepared chunks
    require exactly two consecutive prepared bars.
- Added deterministic, in-memory ZIP package support in `chunk_package.py`.
  - The media type is exactly `application/vnd.streammuse.rap-chunk+zip`.
  - Packages contain only `manifest.json` and `vocals.wav`, with fixed ZIP
    member metadata and canonical manifest bytes.
  - Decode rejects packages over 4 MiB, oversized combined or individual
    members, duplicate members, traversal names, unexpected members, encrypted
    members, malformed JSON, identity mismatches, and SHA-256 mismatches.
  - WAV validation uses `wave`: uncompressed mono PCM16, 24 kHz, exact manifest
    frame count, and non-silent samples are required before package acceptance.
- Exported the new domain contract public names from `streammuse.domain.rap`.

## Files

- `src/streammuse/domain/rap/remote_chunk.py` (new)
- `src/streammuse/infrastructure/rap/chunk_package.py` (new)
- `src/streammuse/domain/rap/__init__.py` (modified)
- `tests/unit/domain/rap/test_remote_chunk.py` (new)
- `tests/unit/infrastructure/rap/test_chunk_package.py` (new)

## TDD Evidence

### RED

Command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py -q
```

Outcome: collection failed as intended before implementation with
`ImportError: cannot import name 'RemoteCandidatePolicy' from
'streammuse.domain.rap'`. The new remote contract API did not exist.

### GREEN

Required focused command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q
```

Outcome: `27 passed in 0.59s` after the commit.

Relevant regression command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py tests/unit/domain/rap/test_audio.py tests/unit/domain/rap/test_flow.py -q --tb=short
```

Outcome: `47 passed in 0.64s`.

Also run: `git diff --check`, with no whitespace errors.

## Test Coverage

- Two consecutive bars, 4/4 compatibility, finite positive BPM and budget.
- Canonical request ID stability across budget-only retries and changes when
  lyric/flow identity changes.
- 24 kHz two-bar frame count, selected-bar identity, frozen diagnostics, and
  exact prepared-chunk bar count.
- Deterministic ZIP members and bytes, canonical manifest bytes, hash and
  request identity verification.
- Duplicate/traversal/unexpected archive members, 4 MiB guard, malformed JSON,
  PCM16/mono/sample-rate/frame-count/silence validation, and float/non-finite
  WAV rejection.

## Self-Review

- Confirmed the idempotency identity excludes only `remaining_budget_ms` while
  every retry payload retains its original budget value.
- Confirmed the codec never extracts an archive to disk and verifies members,
  manifest, WAV structure, and the exact raw WAV hash before returning audio.
- Added a combined uncompressed-member limit in addition to archive and
  per-member limits to bound ZIP expansion.
- Confirmed the commit includes exactly the five owned Task 1 files.

## Commit

`085a1838 add remote rap chunk contracts`

## Concerns

- A valid PCM16 stream cannot contain non-finite samples by representation; the
  decoder retains a defensive finite-sample check and rejects non-PCM16 float
  WAV input before it can enter the transport contract.
- The full repository suite was not run because this worktree contains many
  unrelated uncommitted changes. The focused Task 1 suite and adjacent rap
  domain regressions are green.

## Review-Fix Loop (2026-08-21)

### Findings Addressed

- Added `RemoteRapChunkTransportAttempt`, derived from a request and retaining
  the original canonical request bytes. `retry_body()` returns those immutable
  bytes exactly; it does not rebuild a request or alter the original budget.
  `request_id` remains derived without `remaining_budget_ms`.
- Replaced the unconstrained manifest diagnostic mapping with the mandatory
  `RemoteRapChunkDiagnostics` envelope and `RemoteCandidateStats`. The
  envelope requires the accepted request budget, resolved policy, candidate
  counts plus bounded candidate/rejection summaries, every stage timing,
  alignment and audio diagnostics, model/tool versions, and warnings.
  Selected-bar diagnostics now require non-empty finite `component_scores`.
- Decoder WAV validation now rejects a short PCM16 data payload even if its
  WAV header declares the expected frame count, and normalizes `EOFError` and
  `wave.Error` parsing failures to `ValueError`.
- Added decoder coverage for a small compressed archive with an oversized
  uncompressed member and for oversized combined uncompressed contents.

### Changed Paths

- `src/streammuse/domain/rap/remote_chunk.py`
- `src/streammuse/domain/rap/__init__.py`
- `src/streammuse/infrastructure/rap/chunk_package.py`
- `tests/unit/domain/rap/test_remote_chunk.py`
- `tests/unit/infrastructure/rap/test_chunk_package.py`

### TDD Evidence

RED command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: collection failed with missing `RemoteCandidateStats` exports in both
Task 1 test modules before the new transport-attempt and diagnostic contracts
were implemented.

GREEN focused command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `34 passed in 0.66s`.

Relevant regression command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py tests/unit/domain/rap/test_audio.py tests/unit/domain/rap/test_flow.py -q --tb=short
```

Outcome: `54 passed in 0.64s`.

### Follow-Up Commit

`harden remote rap chunk contracts` (follow-up commit)
