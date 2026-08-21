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

## Second Review-Fix Loop (2026-08-21)

### Implementation

- Rejected JSON booleans in all numeric request, policy, flow, selected-bar,
  manifest, candidate-summary, timing, alignment, audio, and fallback-count
  fields. Wire diagnostics now recursively accept only finite JSON values and
  immutable mappings/sequences.
- Made successful selected bars structurally valid: nonblank text, nonempty
  ordered unique slots, consistent bar/template identity, bounded slot fields,
  and validated syllable word/index/count/stress/phoneme/source primitives.
  The contract intentionally does not compare the entire schedule to an
  originating request; that remains Task 5 behavior.
- Replaced arbitrary candidate/rejection diagnostic mappings with strict
  required-key records. Summaries have finite component scores, nonblank IDs
  and top text, nonempty rejection reasons, immutable JSON-safe values, and
  count-derived plus maximum-eight bounds.
- Enforced the re-review diagnostic invariants: nonempty equal anchor arrays,
  positive finite warp ratios, duration within one sample of frames/rate, peak
  in `[0, 1]`, nonempty `moss`/`aligner`/`rubberband` versions, and total timing no
  smaller than any component timing.
- Restricted package members to `ZIP_STORED` and `ZIP_DEFLATED`; unsupported
  compression and ZIP/JSON bounded-parser failures now normalize to
  `ValueError`.
- Preserved the existing request identity and immutable transport attempt:
  `remaining_budget_ms` remains excluded from `request_id`, while retry bytes
  retain the original canonical request body and budget exactly.

### Changed Paths

- `src/streammuse/domain/rap/remote_chunk.py`
- `src/streammuse/infrastructure/rap/chunk_package.py`
- `tests/unit/domain/rap/test_remote_chunk.py`
- `tests/unit/infrastructure/rap/test_chunk_package.py`
- `.superpowers/sdd/2026-08-21-realtime-remote-moss-renderer/task-1-report.md`

### TDD Evidence

RED command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py -q --tb=short
```

Outcome: `19 failed, 23 passed`; the new numeric-wire rejection cases exposed
boolean acceptance across the transport contract.

Complete RED command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `45 failed, 42 passed`; additionally exposed empty/malformed selected
bars, unconstrained diagnostic summaries, unsupported ZIP compression escaping
the boundary, and nested JSON `RecursionError` leakage.

GREEN focused command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `89 passed in 0.68s`.

Adjacent regression command:

```bash
uv run pytest tests/unit/domain/rap tests/unit/infrastructure/rap -q --tb=short
```

Outcome: `374 passed in 1.38s`.

Quality command:

```bash
git diff --check
```

Outcome: no whitespace errors.

### Self-Review

- Verified all response mappings that reach canonical JSON are frozen and
  recursively JSON-safe with finite numbers; `json.dumps(..., allow_nan=False)`
  remains used for canonical request and manifest serialization.
- Verified ZIP validation occurs before member reads, so unsupported methods do
  not reach decompression; malformed ZIP/JSON decoding now has a stable
  `ValueError` boundary.
- Verified no request-ID field was added or removed. The original transport
  attempt behavior and byte-for-byte retry test remain intact.
- Verified only Task 1 production/tests/report paths will be staged. In
  particular, coordinator-owned `progress.md` and `task-2-brief.md` are not
  part of this follow-up commit.

### Concerns

- Full-repository tests were not run because the worktree contains extensive
  unrelated uncommitted work. Focused and adjacent Task 1 rap modules are
  green.

### Follow-Up Commit

`finalize remote rap chunk contract bounds` (follow-up commit)

## Timing-Key Correction (2026-08-21)

- Replaced the sole mandatory stage timing key `mfa` with generic `aligner` in
  the Task 1 diagnostic contract and all Task 1 fixtures. The exact required
  set is now `generation`, `evaluation`, `moss`, `aligner`, `warp`,
  `packaging`, and `total`; no other timing key changed.
- Retained modular version diagnostics: `moss`, `aligner`, and `rubberband`
  are mandatory, while implementation-specific keys such as `mfa` remain
  allowed there.

RED command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py -q --tb=short
```

Outcome: `1 failed, 77 passed`; a diagnostic carrying `aligner` timing and no
`mfa` was rejected by the old required-stage-key set.

GREEN focused command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `99 passed in 0.76s`.

Adjacent regression command:

```bash
uv run pytest tests/unit/domain/rap tests/unit/infrastructure/rap -q --tb=short
```

Outcome: `384 passed in 1.39s`.

`git diff --check` reported no whitespace errors.

### Follow-Up Commit

`tighten remote rap chunk wire contracts` (follow-up commit)

## Final Review-Fix Loop (2026-08-21)

### Implementation

- Required every selected-bar schedule to carry `slot_index` values exactly
  `0..len(schedule)-1` in schedule order, in addition to the existing ordered
  unique tick validation. This remains local structural validation; full
  requested-flow comparison is still deferred to Task 5.
- Added a 32-container-depth limit for recursively frozen wire diagnostics.
  Constructors now reject excessive nesting as `ValueError`; manifest decoding
  and retained transport-attempt JSON decoding also normalize `RecursionError`
  to their existing `ValueError` boundary.
- Added `REMOTE_CHUNK_PACKAGE_MAX_BYTES` and the derived
  `MAX_REMOTE_CHUNK_PCM16_FRAMES` domain limit. Frame derivation rejects
  arithmetic failures, non-finite/zero rounded counts, and counts beyond the
  4 MiB package-ceiling-derived mono PCM16 transport bound.
- Applied the modular-aligner amendment: diagnostic versions now require
  nonblank `moss`, `aligner`, and `rubberband` keys. Extra version keys remain
  accepted; stage timing keys remain unchanged.

### Changed Paths

- `src/streammuse/domain/rap/remote_chunk.py`
- `src/streammuse/infrastructure/rap/chunk_package.py`
- `tests/unit/domain/rap/test_remote_chunk.py`
- `tests/unit/infrastructure/rap/test_chunk_package.py`
- `.superpowers/sdd/2026-08-21-realtime-remote-moss-renderer/task-1-report.md`

### TDD Evidence

RED command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `9 failed, 89 passed`. The failures demonstrated accepted duplicate
and noncontiguous slot indices, unbounded diagnostics, leaked retained-body
`RecursionError`, unrepresentable/zero/oversized frame counts, rejected
modular aligner versions, and accepted post-parse nested manifest diagnostics.

GREEN focused command:

```bash
uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q --tb=short
```

Outcome: `98 passed in 0.69s`.

Adjacent regression command:

```bash
uv run pytest tests/unit/domain/rap tests/unit/infrastructure/rap -q --tb=short
```

Outcome: `383 passed in 1.38s`.

Quality command:

```bash
git diff --check
```

Outcome: no whitespace errors.

### Self-Review

- Confirmed the frame limit is named and derived from the shared 4 MiB package
  ceiling, not from an arbitrary BPM range; normal 90 BPM requests remain at
  128,000 frames.
- Confirmed diagnostics are limited before Python recursion depth is reached,
  while codec and retained-body decoder boundaries still normalize parser or
  construction recursion independently.
- Confirmed request identity still excludes `remaining_budget_ms`, and the
  immutable transport attempt still returns its original canonical bytes.
- Confirmed no coordinator-owned `progress.md`, `task-2-brief.md`,
  `mms-fa-probe.py`, or review artifact will be staged.

### Concerns

- Full-repository tests were not run because the worktree contains extensive
  unrelated uncommitted work. Focused and adjacent Task 1 rap modules are
  green.
