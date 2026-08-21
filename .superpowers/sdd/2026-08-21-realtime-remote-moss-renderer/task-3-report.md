# Task 3 Implementation Report

Date: 2026-08-21

## Status

Implemented the persistent connected-phrase MOSS synthesizer, resident
torchaudio MMS_FA aligner and deterministic syllable-onset mapper, and the
serialized full-phrase Rubber Band R3 renderer. The real H200 smoke was not run
locally; it remains the coordinator's post-review gate.

## Files

- `src/streammuse/infrastructure/rap/moss_tts.py`: lazy adapter over the
  accepted offline MOSS runtime, real warmup, atomic isolated synthesis,
  waveform validation, hashes, and immutable reproduction metadata.
- `src/streammuse/infrastructure/rap/mms_forced_alignment.py`: lazy resident
  MMS_FA runtime, inference-only resampling, character/word spans, exact token
  coverage, real warmup, and source-syllable onset mapping.
- `src/streammuse/infrastructure/rap/moss_aligned_phrase.py`: Task 2
  `PhraseVocalRenderer` implementation, serialized MOSS/MMS/R3 pipeline, exact
  PCM16 packaging, bounded diagnostics, retained artifacts, and typed failure
  cleanup.
- `tests/unit/infrastructure/rap/test_moss_tts.py`: persistent runtime,
  settings, isolation, stale/silent/invalid output, WAV suffix, and warmup.
- `tests/unit/infrastructure/rap/test_mms_forced_alignment.py`: resident model,
  normalization, resampling, inference mode, token/word coverage, all mapping
  methods, warnings, and typed invalid-map failures.
- `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py`: one full-chunk R3
  call, exact WAV, diagnostics, preflight, cleanup, determinism, concurrency,
  and post-quantization silence rejection.
- This report.

No offline script or offline script test was edited. Their substantial dirty
experiment work remains untouched and unstaged.

## RED/GREEN Evidence

All focused cycles used:

```bash
uv run pytest tests/unit/infrastructure/rap/<focused-test-file>.py -q
```

Recorded cycles:

| Behavior | RED | GREEN |
|---|---|---|
| MOSS module/API | missing-module collection error, then `NotImplementedError` (`1 failed`) | runtime reuse/settings test `1 passed` |
| MOSS stale, silent, NaN, Inf policy | `4 failed, 1 passed` | `5 passed` |
| Torchaudio-compatible temporary WAV suffix | `1 failed, 4 passed`; adapter supplied `.source.wav.partial` | `5 passed`; adapter supplies `.source.partial.wav` |
| Real MOSS warmup | `1 failed, 5 passed`; no generation occurred | `6 passed` |
| MMS module/API and resident reuse | missing-module collection error, then `NotImplementedError` (`1 failed`) | `1 passed` |
| Exact/weighted/proportional mapper | missing-symbol collection error, then `NotImplementedError` (`1 failed, 1 passed`) | `2 passed` |
| Low-confidence complete map | `1 failed, 2 passed`; warning absent | `3 passed` |
| MMS inference/no-grad scope | `1 failed, 8 passed`; CTC aligner ran outside inference mode | `9 passed` |
| Real MMS warmup | `1 failed, 8 passed`; old signature only reported load state | `9 passed` |
| Exact CTC token coverage | `1 failed, 9 passed`; wrong same-length token accepted | `10 passed` |
| Whole-phrase renderer | missing-module collection error, then `NotImplementedError` (`1 failed`) | initial connected R3 behavior `1 passed` |
| Encoded PCM16 silence | `1 failed, 5 passed`; sub-LSB float audio encoded to successful zeros | `6 passed` |

The mapper's behavior-level RED preceded implementation of complete mapping,
finite/bounds checks, and strict source/target monotonicity. The final mapper
matrix separately proves missing/reordered words, impossible duplicate-sample
maps, NaN, Inf, out-of-bounds spans, and phrase-level non-monotonic spans all
raise Task 2 `PhraseRenderFailed`.

## MOSS Settings Provenance

The production module lazily imports
`scripts.rap_audio_backends.moss_backend` only from
`PersistentMossSynthesizer.load()`. It delegates model construction to the
script's `create_runtime()` and connected-phrase generation to its existing
single-request `_generate_chunk()` implementation. This avoids copying a
second MOSS invocation while preserving the dirty script byte-for-byte.

The resolved settings are read directly from that module on every result:

- model: `OpenMOSS-Team/MOSS-TTS-v1.5`;
- language: `English`;
- instruction: `clear, rhythmically spoken rap with restrained pitch`;
- mode: `generation`;
- `max_new_tokens=256`;
- `audio_temperature=1.7`;
- `audio_top_p=0.8`;
- `audio_top_k=25`;
- `audio_repetition_penalty=1.0`;
- token target: `round(request.duration_seconds * 12.5)`, which is 67 for the
  90 BPM two-bar request;
- reference conditioning: the exact caller reference path, with SHA-256
  recorded once at load;
- normalization/save behavior: the offline `_normalise_audio_for_save()` and
  torchaudio save path are reused unchanged;
- runtime device/dtype/attention policy: the offline runtime's CUDA bfloat16
  and Flash Attention 2/SDPA selection (or CPU float32/eager fallback).

Model and processor construction happen once in `load()`. Later synthesis and
warmup calls reuse that runtime and reference conditioning. Model revision is
read from the loaded config/model `_commit_hash` when available, otherwise
reported as `unknown`.

## Lazy Dependency Behavior

Neither new module imports `torch`, `torchaudio`, `transformers`, or MOSS code
at package import time. MOSS dependencies load only through the offline
`create_runtime()` call. Torch and torchaudio load only inside the default MMS
runtime loader. The complete Mac suite imported these modules without a MOSS or
torchaudio runtime being loaded or downloaded.

NumPy/SciPy remain ordinary local dependencies for WAV validation and the
16 kHz inference copy. Unit tests inject fake runtime/model/tokenizer/aligner
and stretcher boundaries; they do not execute MOSS, MMS, or Rubber Band.

## Alignment And Mapping Policy

Transcript normalization is NFKD to ASCII, lowercase, ASCII word extraction,
and deletion of internal apostrophes. Punctuation and case are normalized;
digits and an empty normalized transcript fail typed. Returned MMS token IDs,
character counts, word counts, word order, and normalized transcript must all
cover the exact request.

The source WAV is read at its native rate. A mono float32 copy is resampled to
the MMS bundle rate (expected 16 kHz) only for model inference. CTC frame spans
are converted to seconds using source duration divided by emission-frame count,
so mapped source samples are later computed at the original MOSS WAV rate.

Planned syllables are grouped by contiguous `index_in_word` values and mapped
per word in this order:

1. `orthographic_vowel_groups`: regex vowel-group starts, with syllable zero at
   the word/first-character onset and later syllables at later group starts. A
   lone trailing final `e` is removed only when doing so makes the group count
   exactly match the planned count.
2. `phoneme_weighted_character`: cumulative boundaries proportional to each
   planned syllable's CMU phoneme count, clamped only by requiring distinct
   available character indices.
3. `word_duration_proportional`: word onset plus `index / syllable_count` of
   the aligned word duration when character allocation is impossible.

Every target produces exactly one `VowelAnchor` with
`anchor_kind="syllable_onset"`. Target seconds are copied exactly from the
`SyllableTarget`; source samples use the original MOSS rate. Source and target
samples must be finite, in bounds, unique, and strictly increasing.

## Renderer And Policy

The renderer validates transcript/syllable agreement before MOSS work and
serializes the complete render call with one lock. It synthesizes once, aligns
once, maps once, and invokes existing `continuous_pitch_preserving_warp()` once
with a stretcher constructed as
`RubberBandTimeMapStretcher(engine="r3", smoothing=False)`. It does not import
or call stress augmentation, target regularization, piecewise warping, or MFA.

The accepted output is exactly 24 kHz, mono, uncompressed PCM16, with
`round(request.duration_seconds * 24_000)` frames (128,000 at 90 BPM). Float
NaN/Inf, float silence, post-quantization PCM16 silence, wrong source metadata,
wrong rate/frames/channels/encoding, and warp failures raise
`PhraseRenderFailed`.

Low CTC confidence below 0.5, phoneme-weighted mapping, proportional mapping,
and local stretch ratios outside 0.5-2.0 warn and continue when the map is
otherwise complete. Missing/reordered/token-mismatched coverage, incomplete or
duplicate targets, impossible spacing, and non-finite/out-of-bounds/
non-monotonic geometry fail typed.

Warnings are deduplicated, capped at 32, and individually capped at 512
characters. Anchor diagnostics are capped at 128 entries with explicit total
and truncation fields. The caller workspace retains `moss-source.wav`,
`alignment.json`, and successful `vocal.wav`. Temporary partials and stale
final vocals are removed on failure. A later-stage failure preserves the raw
source and writes bounded failure JSON.

Version keys are exactly `moss`, `aligner`, and `rubberband`. The aligner value
explicitly names `torchaudio.pipelines.MMS_FA`; no result claims MFA.

## Verification

```text
Focused binding command: 61 passed in 1.58s
Adjacent rap infrastructure/experiments + Task 2 orchestration: 448 passed in 1.94s
Complete repository suite: 1649 passed, 4 skipped in 42.91s
Owned-file Ruff check: all checks passed
Owned-file Ruff format check: clean after formatting
```

The full-suite warning is pre-existing: `pretty_midi` imports deprecated
`pkg_resources`.

## H200 Assumptions And Concerns

- The accepted MOSS model emits 24 kHz mono audio. The realtime renderer fails
  typed rather than silently resampling a different MOSS output rate.
- Deployment runs from the exact repository revision where the lazily imported
  offline backend module is available; no wheel-only packaging gate was run.
- Torchaudio 2.8.x MMS_FA APIs and the cached model are available. The known
  forced-align API deprecation/removal risk in later torchaudio remains.
- The MMS_FA CC-BY-NC 4.0 restriction remains a research-prototype constraint.
- The Rubber Band binary/library supports R3 and the coordinator supplies the
  exact installed version string (expected 3.3.0) when constructing the
  renderer; the class default identifies R3 but cannot probe a binary version
  without executing a subprocess.
- `PersistentMossSynthesizer.warmup()` now performs one real disposable
  connected-phrase generation. `MmsForcedAligner.warmup(source_wav, transcript)`
  performs one real resident acoustic/CTC pass.
- The renderer serializes MOSS, MMS, and R3 access. GPU coexistence and latency
  under real MOSS/vLLM load remain unmeasured locally.
- The coordinator must still select an unused physical H200, render the known
  request twice in one process, verify no MFA subprocess, retain artifacts, and
  compare one MMS map against the retained MFA reference.
