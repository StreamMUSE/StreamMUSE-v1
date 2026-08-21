# Task 3 Fix Report: Persistent Aligned MOSS Phrase Renderer

Date: 2026-08-21

## Status

The rejected Task 3 renderer has been repaired locally. The concrete renderer now
crosses `RapChunkOrchestrator` with the exact remote diagnostics schemas, handles
the production tick-zero flow as one continuous R3 warp, uses the remote contract's
authoritative frame calculation, retains canonical complete research artifacts,
reseeds MOSS per request, and cleans stale/cancelled publication state.

No H200 deployment or smoke was performed. That remains the coordinator's gate.

## Owned Files

- `src/streammuse/infrastructure/rap/moss_tts.py`: persistent lazy MOSS adapter,
  offline-equivalent per-request seeding, warmup isolation, source validation,
  immutable provenance, and `BaseException` cleanup.
- `src/streammuse/infrastructure/rap/mms_forced_alignment.py`: resident lazy MMS_FA
  adapter, post-resample inference timebase, retained original-source bounds,
  complete positional spans, and deterministic onset mapping.
- `src/streammuse/infrastructure/rap/moss_aligned_phrase.py`: exact wire diagnostics,
  tick-zero crop/boundary policy, shared target frame count, canonical artifacts,
  one continuous R3 warp, serialized model access, and publish-last ownership.
- `tests/unit/infrastructure/rap/test_moss_tts.py`: deterministic seed order,
  warmup isolation, reuse, invalid audio, stale output, and cancellation coverage.
- `tests/unit/infrastructure/rap/test_mms_forced_alignment.py`: resident reuse,
  normalization/coverage, all mapping methods, post-resample timing, failures,
  confidence warnings, and repeated-word positional evidence.
- `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py`: exact wire boundary,
  canonical artifacts, tick-zero endpoint controls, half-frame BPM, real
  orchestrator integration, ownership/cancellation, PCM16, and concurrency.
- This report.

No offline backend script, backend test, Task 1/2 contract/planner, Task 4 server,
Task 6 file, CLI, controller, UI, plan, or specification was edited or staged.

## RED/GREEN Evidence

The pre-fix Task 3 baseline was:

```text
uv run pytest tests/unit/infrastructure/rap/test_moss_tts.py tests/unit/infrastructure/rap/test_mms_forced_alignment.py tests/unit/infrastructure/rap/test_moss_aligned_phrase.py -q
23 passed in 0.61s
```

Every production change below followed an observed focused RED. Commands used the
exact node shown as `uv run pytest <node> -q`, followed by the same command for
GREEN.

| Behavior / exact test node | RED evidence | GREEN evidence |
|---|---|---|
| `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py::test_renders_one_continuous_r3_phrase_with_exact_pcm16_and_diagnostics` | `1 failed`; private/misnamed alignment keys violated the exact wire set | `1 passed`; `RemoteRapChunkDiagnostics` accepts the result |
| `...::test_preserves_complete_canonical_source_and_mms_alignment_artifacts` | `1 failed`; canonical `source.wav` absent | `1 passed`; complete `source.wav` and `mms_alignment.json` retained |
| `...::test_tick_zero_uses_cropped_acoustic_onset_as_single_boundary_endpoint` | `1 failed`; continuous warp rejected target sample zero | `1 passed`; one `(0,0)` endpoint and no duplicate target control |
| `...::test_uses_remote_contract_frame_count_at_non_90_bpm_half_frame_edge` | `1 failed`; renderer produced 57,601 versus wire 57,600 | `1 passed`; renderer delegates to `RemoteRapChunkRequest.frame_count_for()` |
| `tests/unit/infrastructure/rap/test_moss_tts.py::test_reseeds_each_request_with_offline_attempt_policy_after_warmup` | `1 failed`; warmup generation ran with no seed | `1 passed`; Torch/CUDA seed calls precede every generation |
| `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py::test_preflight_failure_removes_all_stale_renderer_owned_artifacts` | `1 failed`; stale success artifacts survived preflight | `1 passed`; owned paths are cleared before validation |
| `...::test_success_publishes_vocal_last_and_removes_stale_failure` | `1 failed`; `vocal.wav` existed during alignment publication | `1 passed`; successful vocal replacement is the final state transition |
| `...::test_base_exception_cancellation_removes_unpublished_vocal_artifacts` | `1 failed`; cancellation left `vocal.wav` | `1 passed`; `BaseException` removes final/partial vocal and re-raises |
| `...::test_failed_silent_warp_cleans_stale_final_but_retains_research_artifacts` | `1 failed`; late warp failure had no MMS artifact | `1 passed`; complete alignment/mapping evidence survives separately |
| `tests/unit/infrastructure/rap/test_mms_forced_alignment.py::test_ctc_seconds_use_actual_post_resample_waveform_and_keep_source_bounds` | `1 failed`; 0.0500020833 s used source duration instead of expected 0.0500031250 s | `1 passed`; 16,001-sample inference timebase is authoritative |
| `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py::test_preserves_complete_canonical_source_and_mms_alignment_artifacts` | `1 failed`; persisted source/inference/emission timebases absent | `1 passed`; all three timebases retained |
| `tests/unit/infrastructure/rap/test_moss_tts.py::test_base_exception_during_generation_removes_partial_and_stale_source` | `1 failed`; `.source.partial.wav` survived | `1 passed`; cancellation cleans both source paths |
| `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py::test_tick_zero_uses_cropped_acoustic_onset_as_single_boundary_endpoint` | `1 failed`; crop hash provenance fields absent | `1 passed`; WAV and float32le warp-input hashes are explicit |
| `...::test_rejects_wrong_request_type_with_typed_preflight_failure` | `1 failed`; failure JSON masked the typed error with `AttributeError` | `1 passed`; null hash is recorded and typed failure preserved |
| `...::test_atomic_alignment_write_removes_partial_on_base_exception` | `1 failed`; JSON partial survived interrupted replace | `1 passed`; atomic writer cleans in `finally` |
| `...::test_tick_zero_accepts_acoustic_onset_already_at_source_boundary` | `1 failed`; valid source sample zero was rejected | `1 passed`; zero uses the boundary policy without cropping frames |

The required real integration test is
`test_real_renderer_result_crosses_orchestrator_with_builtin_tick_zero_non_90_bpm`.
It uses `baseline_syncopated_9`, `baseline_staggered_9`, real
`ChunkCandidatePlanner`, real `RapChunkOrchestrator`, and real
`MossAlignedPhraseRenderer`; only MOSS, MMS, and Rubber Band are faked. It passes
with target zero, exact diagnostics, and 57,600 frames at BPM
`199.99826390395916`.

## Wire And Artifact Contracts

`PhraseRenderResult.alignment_diagnostics` has exactly:

```text
fallback_counts, source_anchors, target_anchors, local_warp_ratios
```

`PhraseRenderResult.audio_diagnostics` has exactly:

```text
sample_rate_hz, frame_count, duration_seconds, peak
```

Rich evidence is not placed on the wire. The caller workspace instead retains:

- `source.wav`: the complete, unmodified validated MOSS WAV;
- `mms_alignment.json`: unbounded normalized transcript, all character spans and
  scores, all positional word spans and nested characters, aligner identity,
  version, timing, confidence, warnings, original/inference/emission timebases,
  source/model/reference/settings hashes, every mapping anchor/method/score,
  requested and warp-domain coordinates, endpoint/crop evidence, effective R3
  anchors, local ratios, and output hash/metrics;
- `vocal.wav`: installed only after result construction and canonical evidence;
- `render_failure.json`: a separate bounded failure record, never a substitute
  for complete MMS evidence.

The canonical alignment artifact deliberately has no render `success=true` flag.
On a late failure it retains truthful alignment-ready evidence with warp status
`pending`; on success it records completed warp/output evidence before the final
vocal publication.

## Tick-Zero Endpoint Policy

Production built-in templates start at target tick/sample zero. The complete MMS
map still contains one source onset for every planned syllable. For a first target
at exact zero:

1. The renderer takes the first mapped MMS acoustic onset as `crop_start`.
2. `source.wav` remains complete; only the in-memory warp input is sliced from
   `crop_start` and all warp-domain source coordinates are normalized by that
   exact sample count.
3. The first mapped syllable becomes source/target `(0,0)` and is represented by
   `continuous_pitch_preserving_warp()`'s existing boundary endpoint, so it is
   omitted from the interior-anchor argument.
4. Every later source onset remains strictly increasing and every requested
   target time/sample is unchanged. No target shift or regularization occurs.
5. The exact original WAV hash, float32le warp-input hash, crop sample/time,
   original/cropped frame counts, complete original anchors, normalized anchors,
   and endpoint role are persisted.

If the first acoustic onset is already source sample zero, the same policy applies
with zero removed frames. A target that merely rounds to zero but is not exactly
zero fails typed. A missing interior anchor, terminal crop, duplicate/non-monotonic
map, or other impossible geometry also fails typed.

## Mapping And MMS Timebase

Transcript normalization is NFKD ASCII, lowercase ASCII word extraction, and
internal apostrophe deletion. Digits and an empty normalized transcript fail.
MMS token IDs, character count, word count, and positional word order must cover
the exact normalized request; repeated words remain distinct by `word_index`.

Per word, onset methods remain ordered as required:

1. exact orthographic vowel-group starts, conservatively dropping only a trailing
   silent `e` when that makes the count exact;
2. deterministic character boundaries weighted by planned CMU phoneme counts;
3. word-duration proportional interior points.

Every planned syllable receives one `VowelAnchor` with
`anchor_kind="syllable_onset"`. Target times are copied exactly. Low score and
fallback methods warn and continue when coverage is complete. Missing, reordered,
duplicate, incomplete, non-finite, out-of-bounds, or non-monotonic evidence raises
Task 2 `PhraseRenderFailed`.

The inference waveform object now retains both timebases. CTC frame seconds use:

```text
(post_resample_inference_frame_count / MMS_sample_rate) / emission_frame_count
```

Original source rate/frame/duration are retained separately and used for span and
mapped-sample bounds. The 24,001 -> 16,001 frame regression proves timestamps
follow the actual tensor given to MMS, not the pre-resample WAV.

## MOSS Settings And Seed Provenance

Heavy MOSS code is imported only inside `PersistentMossSynthesizer.load()`. The
adapter delegates runtime construction to the dirty offline backend's public
`create_runtime()` and connected-phrase generation to its existing
`_generate_chunk()`; the offline files were not changed.

Settings are read directly from that backend:

- model `OpenMOSS-Team/MOSS-TTS-v1.5`;
- language `English`;
- instruction `clear, rhythmically spoken rap with restrained pitch`;
- mode `generation`;
- `max_new_tokens=256`;
- `audio_temperature=1.7`, `audio_top_p=0.8`, `audio_top_k=25`;
- `audio_repetition_penalty=1.0`;
- token target from the shared `moss_token_target(request)` policy;
- the same reference WAV path, offline processor prompt, normalization, save,
  dtype, attention, and device behavior.

Each realtime call is one offline-equivalent attempt. Immediately before
`_generate_chunk()`, the adapter delegates to the offline functions:

```text
seed = DEFAULT_BASE_SEED + request.chunk_index * 1000 + (attempt - 1)
DEFAULT_BASE_SEED = 20260816
attempt = 1
```

The backend's `_seed_torch_best_effort()` applies `torch.manual_seed()` and, on
CUDA, `torch.cuda.manual_seed_all()`. Base seed, attempt, and resolved seed are
recorded in immutable generation settings. Warmup may consume RNG state, but both
the warmup phrase and every later request reseed immediately before generation;
the test proves identical request source hashes after warmup. A runtime without
`torch.manual_seed` is rejected instead of falsely claiming seeding.

## Lazy Dependencies And Concurrency

Normal package import does not import Torch, torchaudio, Transformers, or MOSS.
MOSS dependencies load through the offline runtime only when explicitly composed.
Torch/torchaudio load inside the default MMS runtime loader only. Unit tests inject
fakes and never load/download MOSS or MMS or execute Rubber Band.

MOSS runtime/reference conditioning and MMS model/tokenizer/aligner are each loaded
once. MMS inference has its own lock and inference-mode scope. The concrete renderer
holds one lock across MOSS, MMS, mapping, canonical evidence, and the single R3 warp,
protecting all resident non-thread-safe state.

## Failure And Warning Policy

- Low CTC confidence, phoneme-weighted/proportional mapping, and local ratios
  outside 0.5-2.0 are warnings and do not reject complete mappings.
- Wire warnings are deduplicated, limited to 32, and limited to 512 characters.
  Canonical research warnings and anchors are untruncated.
- Invalid source/output WAVs, non-finite audio, float or quantized silence, wrong
  rate/format/frame count, invalid coverage/geometry, and failed warp are typed
  hard failures. Successful silence is impossible.
- Renderer-owned stale final/partial files are removed before preflight.
- A complete source is retained after synthesis; complete alignment evidence is
  retained before warp. Bounded failure diagnostics remain separate.
- `BaseException` paths clean source/vocal/JSON partials as applicable and re-raise
  cancellation. `vocal.wav` is atomically installed last.

## Verification

```text
Focused command from brief: 75 passed in 1.49s
Adjacent rap infrastructure/experiments + Task 2 orchestration: 462 passed in 1.97s
Owned-file Ruff check: all checks passed
Owned-file Ruff format check: clean after formatting
```

Repository-wide regression run:

```text
1697 passed, 4 skipped, 2 failed in 41.21s
```

Both failures are outside Task 3 ownership. The deterministic failure is
`tests/unit/domain/rap/test_audio.py::test_rap_audio_event_names_are_canonical`:
concurrent Task 6 commit `0af9f57c` added five `RapEventType` values, while that
test still asserts the previous last-14 ordering. The second failure was
`tests/unit/presentation/rap_demo/test_server.py::test_control_endpoints_drive_the_concrete_restartable_audio_runtime`;
it passed immediately in isolation (`1 passed in 0.69s`) and is a suite-order
interaction in concurrently edited presentation state. Per ownership constraints,
no Task 6/domain or presentation test/source file was edited.

## H200 Assumptions And Remaining Concerns

- Real MOSS must emit non-silent 24 kHz mono audio. The renderer fails typed rather
  than silently resampling a different MOSS source rate.
- Torchaudio 2.8.x `MMS_FA` APIs/model cache remain available. The known 2.9 removal
  risk and CC-BY-NC 4.0 research-only licensing constraint remain.
- Rubber Band R3 is available and the coordinator supplies/probes the exact binary
  version; local tests inject a stretcher and execute no subprocess.
- Best-effort Torch seeding matches the accepted offline path, but deterministic
  CUDA kernels and real MOSS repeat hashes still require the H200 smoke.
- The concrete non-90 BPM flow currently relies on the shared worktree's
  `TwoBarRenderRequest.tempo_bpm` contract change, which is owned outside Task 3
  and intentionally unstaged here.
- GPU coexistence, real latency, perceptual alignment quality, and one MMS-versus-
  MFA comparison remain unmeasured locally. The coordinator should run the exact
  accepted revision in an isolated H200 directory and retain both canonical files,
  final WAV/package, request, logs, timings, GPU identity, and hash comparison.
