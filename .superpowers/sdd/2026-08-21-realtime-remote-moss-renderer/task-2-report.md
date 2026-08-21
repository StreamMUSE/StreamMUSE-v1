# Task 2 Implementation Report

## Scope Delivered

- Added a budget-aware two-bar candidate planner that runs both initial waves
  before bar-specific rescue waves, clamps every wave to the remaining per-bar
  allowance, and checks the monotonic generation cutoff before each new wave.
- Reused `rank_candidates()` and its exact-syllable hard gate without copying
  validity or score logic. Every completed wave is analyzed and the accumulated,
  normalized, per-bar pool is reranked immediately.
- Added deterministic pair selection ordered by mean existing total score, then
  `lexical_continuity(right, history=(left,), topic=right_topic)`, then
  `rhyme_quality(right_tail, left_tail)`, then the two source-order indices.
- Built exact `TwoBarRenderRequest` values from the selected schedules. At four
  ticks per beat, `target_seconds = tick_in_chunk * 60 / (BPM * 4)`.
- Added immutable complete candidate ledgers, aggregate wire stats, eight-item
  aggregate summaries, and up to eight top/rejected summaries inside each
  selected bar's diagnostics. This satisfies per-bar inspection while retaining
  Task 1's eight-item aggregate transport bound.
- Added `PhraseVocalRenderer`, `PhraseRenderResult`,
  `RemoteChunkRenderArtifact`, typed failures, and `RapChunkOrchestrator`.
  Orchestration validates identity, manifest diagnostics, exact mono PCM16 WAV
  duration, non-silence, and byte hash through the shared package codec.
- Added API-level independent OpenAI choices to `LocalChatModelClient` using the
  existing persistent async client, cancellation, timeout, retry, and close
  lifecycle. Legacy `generate()` retains its existing payload and behavior.
- Added `IndependentChoiceCandidateGenerator`. Each choice is prompted for one
  line with the full flow, syllable target, pronunciation rule, history, and
  visible variation seed. It defaults to temperature `1.0` and 32 tokens per
  choice; H200 composition supplies `top_p=0.95` through client configuration.

## Conservative Decisions

- A hard-valid candidate stops rescue for its bar even when it is below the
  minimum score, because the binding brief defines rescue eligibility using
  `minimum_valid_candidates`. Final selection still enforces `minimum_score`.
- The planner counts requested API choices, not only returned lines, against the
  per-bar maximum. Extra returned choices are retained in the ledger as
  `over_returned` but are not analyzed or counted as parseable.
- Package encoding is invoked only as a validation boundary in Task 2.
  `packaging=0.0` and the warning `packaging timing is provisional` make clear
  that Task 4 owns measured transport packaging time.

## Changed Paths

- `src/streammuse/application/rap/chunk_orchestration.py`
- `src/streammuse/application/rap/__init__.py`
- `src/streammuse/domain/rap/generation.py`
- `src/streammuse/infrastructure/inference/local_chat_client.py`
- `src/streammuse/infrastructure/rap/generators.py`
- `src/streammuse/infrastructure/rap/__init__.py`
- `tests/unit/application/rap/test_chunk_orchestration.py`
- `tests/unit/infrastructure/inference/test_local_chat_model_client.py`
- `tests/unit/infrastructure/rap/test_generators.py`
- `.superpowers/sdd/2026-08-21-realtime-remote-moss-renderer/task-2-report.md`

## TDD Evidence

RED was observed before each production surface:

- Client tests failed at collection because `LocalChatChoicesResponse` did not
  exist.
- Generator tests failed at collection because
  `IndependentChoiceCandidateGenerator` did not exist.
- Orchestration tests failed at collection because `chunk_orchestration` did
  not exist.
- The later explicit-error-batch regression failed with zero ledger error rows
  before that behavior was implemented.

GREEN evidence:

- Required focused command: `86 passed in 2.12s` after formatting.
- Adjacent scoring/alignment/service/remote-contract/package suite:
  `128 passed in 1.02s`.
- Ruff behavior lint: `All checks passed`.
- The final broad suite excluding two unrelated, concurrently added Task 5
  tests passed: `1541 passed, 4 skipped, 1 pre-existing warning in 40.21s`.
  An earlier run had one intermittent web-runtime failure; that test passed in
  isolation and the complete rerun passed.

## H200 Evidence

Isolated deployment:

`/data/home/Andrew.Yang/StreamMUSE/deploy/real_rap_audio_task2_precommit_20260821`

- The H200 focused suite passed: `86 passed in 2.22s`.
- The first H200 attempt exposed an environment issue: the shared Python
  environment's editable install pointed at the old dirty checkout. Setting
  `PYTHONPATH` to the isolated deployment fixed source selection.
- A second mismatch came from the copied Task 1 snapshot predating the generic
  `aligner` key. Refreshing only the finalized Task 1 contracts/package codec
  fixed that deployment mismatch.
- Real vLLM adapter smoke against loopback `127.0.0.1:8010`, model
  `Qwen/Qwen3.6-27B`, `n=4`, temperature `1.0`, top-p `0.95`: four independent
  non-empty candidates in `292.93 ms`, with no warning or error.
- Real two-bar planner smoke used syncopated and staggered nine-syllable flows:
  `8` requested, `7` parseable, `5` valid/selectable; candidate generation
  `589.08 ms`, evaluation `1.19 ms`, planner total `857.26 ms`. Selected lines:
  `neon signs hum while the shadows creep` and
  `cold neon hums while the city wakes`.

## Review Round 1 Fixes

- Added explicit first-wave state so a cutoff reached after request validation
  but before any wave starts raises `RenderBudgetExpired`. Once a wave has
  started, an unusable two-bar result retains `NoValidCandidates` semantics.
- Moved the evaluation clock boundary ahead of prosody analysis and kept it
  open through accumulated reranking. Planner timings now expose generation,
  evaluation, total, and the non-negative residual `overhead` explicitly;
  manifest total timing includes that planning overhead without changing the
  Task 1 wire-stage keys.
- Added the backward-compatible optional
  `CandidateBatch.provider_choice_indices`. The independent-choice adapter
  preserves the provider index for every accepted choice, including a provider
  over-return, while planner `source_order` remains a separate deterministic
  local sequence.
- Added immutable `generation_warning` ledger rows and propagated those
  bounded, sanitized warnings into successful manifests ahead of renderer and
  packaging warnings. Over-returned choices remain in the complete ledger but
  are excluded from requested-candidate rejection summaries and counts.
- Wrapped workspace preparation `OSError` failures as `PhraseRenderFailed`
  with bounded, sanitized diagnostics. The boundary catches only expected
  filesystem failures there; `SystemExit` and other `BaseException` control
  flow still propagate.
- Added the named branch tests for bar 1-only rescue, both-bar rescue,
  mean-score priority over continuity/rhyme, over-returned choices, and the
  exact deadline race, plus timing-attribution and end-to-end warning/index
  retention tests.

The review's maintainability concern is deferred by controller ruling: the
planner/orchestrator module remains unsplit in this fix round to avoid changing
the public renderer boundary while Tasks 3/4 build the vertical slice.

## Review Round 1 TDD Evidence

RED evidence:

- Initial review test run: `7 failed, 59 passed in 0.94s`. Failures covered the
  deadline classification, omitted analysis time, missing provider metadata,
  missing warning propagation, and untyped workspace failure.
- After the first GREEN implementation, the over-return branch exposed its
  previously untested stats invariant: `1 failed, 65 passed in 0.84s`.
- A separate adapter-level RED proved the fifth provider choice was truncated:
  `1 failed in 0.55s`.

GREEN evidence:

- Review-specific planner/generator suite: `66 passed in 0.72s`.
- Adapter over-return regression: `1 passed in 0.49s`.
- Required three-module focused command after all fixes: `97 passed in 2.04s`.
- Adjacent alignment/scoring/service/realtime/domain/package regressions:
  `190 passed in 1.47s`.
- Ruff on all modified Python paths: `All checks passed!`.
