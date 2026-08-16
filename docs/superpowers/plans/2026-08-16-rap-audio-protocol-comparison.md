# Rap Audio Protocol Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce twelve directly comparable 50-bar rap mixes by rendering the same three lyric/MCFlow schedules with four auditable audio protocols.

**Architecture:** A dependency-light experiment package loads and validates the fixed corpus, converts it into backend-neutral two-bar requests, and assembles exact-length stems, drums, metrics, and listening artifacts. Four isolated backend scripts run in separate H200 environments so MOSS-TTS, TED-TTS/IndexTTS2, NeMo, and alignment dependencies cannot destabilize the main StreamMUSE environment.

**Tech Stack:** Python 3.10+, NumPy/SciPy, MOSS-TTS-v1.5, TED-TTS with IndexTTS2, NVIDIA NeMo FastPitch/HiFi-GAN, Montreal Forced Aligner, Rubber Band, faster-whisper, pytest, shell/SSH.

**Spec:** `docs/superpowers/specs/2026-08-16-rap-audio-protocol-comparison-design.md`

## Global Constraints

- Use exactly songs 01, 02, and 03 from `output/rap_album_10x50_90bpm_20260816_v4/`.
- Use all 50 bars at 90 BPM, 4/4, four ticks per beat, and 48 kHz output.
- Synthesize 25 two-bar chunks per song and account for every chunk in logs.
- Keep lyrics, MCFlow schedules, and drum audio identical across protocols.
- Protocols 1 through 3 may not use post-generation forced alignment or waveform retiming.
- Protocol 4 must reuse Protocol 1's raw MOSS chunk and record its SHA-256.
- Heavy external model dependencies must remain outside the StreamMUSE project environment.
- A failed chunk becomes logged silence after bounded retries; it is never replaced by another protocol.
- Do not modify the existing realtime rap renderer, playback path, or website.
- Store generated model checkpoints and WAV artifacts outside Git; commit only code, tests, specifications, plans, and concise reports.

---

### Task 1: Corpus And Backend-Neutral Contracts

**Files:**
- Create: `src/streammuse/experiments/rap_audio_protocols/__init__.py`
- Create: `src/streammuse/experiments/rap_audio_protocols/contracts.py`
- Create: `src/streammuse/experiments/rap_audio_protocols/corpus.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_corpus.py`
- Create: `tests/fixtures/rap_audio_protocols/two_bar_records.jsonl`

**Interfaces:**
- Produces: `ProtocolId`, `SyllableTarget`, `TwoBarRenderRequest`, `ChunkRenderRecord`, `SongCorpus`, `load_song_corpus()`, and JSON serialization helpers.
- Consumes: existing `chosen_lyrics.jsonl` schema and `Tempo` timing behavior.

- [ ] **Step 1: Write fixture and failing corpus tests**

Create a two-record fixture copied from the public fields of the existing song-01 records. Tests must assert:

```python
corpus = load_song_corpus(path, song_id="01_space_exploration", expected_bars=2)
request = corpus.two_bar_requests()[0]
assert request.chunk_index == 0
assert request.start_bar == 0
assert request.end_bar == 2
assert request.duration_seconds == pytest.approx(16 / 3)
assert len(request.syllables) == 18
assert [item.absolute_tick for item in request.syllables] == sorted(
    item.absolute_tick for item in request.syllables
)
```

Also test rejection of noncontiguous bars, a schedule whose words do not reconstruct the lyric's analyzed word order, non-nine-syllable bars, and any tempo other than 90 BPM in the campaign constructor.

- [ ] **Step 2: Run the corpus tests and verify failure**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_corpus.py -q`

Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement immutable contracts and strict loading**

Use string-valued protocol identifiers:

```python
class ProtocolId(str, Enum):
    MOSS_GLOBAL = "moss_global"
    TED_LOCAL = "ted_local"
    FASTPITCH_PHONEME = "fastpitch_phoneme"
    MOSS_ALIGNED = "moss_aligned"
```

`SyllableTarget` must retain word, within-word index, ARPAbet phones, lexical stress, target stress, boundary strength, absolute tick, tick in chunk, and target seconds. `TwoBarRenderRequest` must contain exactly two complete bars and expose canonical JSON whose SHA-256 is stable. `load_song_corpus()` must validate 50 contiguous bars in production and permit an explicit smaller `expected_bars` only for tests/smoke fixtures.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_corpus.py -q`

Expected: all pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/streammuse/experiments/rap_audio_protocols tests/unit/experiments/rap_audio_protocols tests/fixtures/rap_audio_protocols
git commit -m "feat(rap): define offline audio comparison corpus"
```

### Task 2: Native Timing Plans

**Files:**
- Create: `src/streammuse/experiments/rap_audio_protocols/timing.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_timing.py`

**Interfaces:**
- Consumes: `TwoBarRenderRequest` from Task 1.
- Produces: `build_ted_segments(request) -> tuple[TimedTextSegment, ...]`, `build_fastpitch_phone_plan(request, tokenizer_labels) -> FastPitchPhonePlan`, and `moss_token_target(request) -> int`.

- [ ] **Step 1: Write failing duration-plan tests**

Tests must cover:

```python
assert moss_token_target(request) == 67  # round((16 / 3) * 12.5)
segments = build_ted_segments(request)
assert "".join(segment.text_with_spacing for segment in segments).split() == request.text.split()
assert sum(segment.target_seconds for segment in segments) == pytest.approx(16 / 3)
assert all(segment.target_seconds >= 0.02 for segment in segments)
```

For FastPitch, use a synthetic label sequence containing spaces/blanks and ARPAbet symbols. Assert one duration per tokenizer position, every duration is a nonnegative integer mel frame, the sum matches `round((16 / 3) * 22050 / 256)`, vowel positions contain the residual timing budget, and vowel-center positions are within one mel frame of the target MCFlow anchors whenever physically possible.

- [ ] **Step 2: Run timing tests and verify failure**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_timing.py -q`

Expected: import failure for `timing`.

- [ ] **Step 3: Implement MOSS and TED timing conversion**

MOSS uses exactly `round(request.duration_seconds * 12.5)`. TED segmentation must end only at word boundaries. Prefer a boundary after a word whose final syllable has `boundary_strength > 0`; otherwise preserve one segment per bar. Segment durations are cumulative differences between first-syllable target anchors, with leading space assigned to the first segment and the final segment ending at the two-bar boundary.

- [ ] **Step 4: Implement FastPitch phone-frame allocation**

Map labels to the existing ARPAbet sequence after removing tokenizer blanks and space padding. Fail closed on an unmatched lexical phone. Assign at least one mel frame to spoken phones, zero frames to boundary blanks unless required by the checkpoint, place onset consonants before the vowel target, codas after it, and allocate remaining interval frames to the vowel. Return diagnostics for anchor error and any compressed consonant region.

- [ ] **Step 5: Run timing tests**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_timing.py -q`

Expected: all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/streammuse/experiments/rap_audio_protocols/timing.py tests/unit/experiments/rap_audio_protocols/test_timing.py
git commit -m "feat(rap): derive native speech timing controls"
```

### Task 3: Common Audio Assembly And Artifact Integrity

**Files:**
- Create: `src/streammuse/experiments/rap_audio_protocols/audio.py`
- Create: `src/streammuse/experiments/rap_audio_protocols/artifacts.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_audio.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_artifacts.py`

**Interfaces:**
- Consumes: corpus requests and per-chunk WAV/results from backend scripts.
- Produces: `render_common_drums()`, `assemble_vocal_stem()`, `mix_stems()`, WAV metadata validation, SHA-256 manifests, and resumable JSONL record writing.

- [ ] **Step 1: Write failing exact-length and identity tests**

Generate tiny in-memory 24 kHz and 22.05 kHz mono fixtures. Assert resampling produces 48 kHz float32, a 50-bar stem contains exactly `6_400_000` frames, only silence is added to a short chunk, an overlong chunk is truncated with an explicit diagnostic, common drums hash identically for all protocol manifests, and duplicate/conflicting chunk records are rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_audio.py tests/unit/experiments/rap_audio_protocols/test_artifacts.py -q`

- [ ] **Step 3: Implement WAV conversion and assembly**

Use `scipy.io.wavfile` and `scipy.signal.resample_poly`; do not add a production dependency on librosa. Convert integer PCM to float32, downmix before resampling, pad or truncate only at two-bar boundaries, and write PCM-16 listening WAVs plus float32 intermediate arrays only when explicitly requested.

- [ ] **Step 4: Implement deterministic common drums and mixing**

Render every bar with `ProceduralBoomBapRenderer(seed=20260816 + song_index * 10_000)`, concatenate once per song, and reuse that exact `drums.wav`. Mix vocals at `0.80` and drums at `0.45`, applying one shared peak gain per full song only when the absolute peak exceeds `0.98`.

- [ ] **Step 5: Implement manifests and resumable records**

Use canonical JSON and SHA-256 for corpus requests, source chunks, stems, and mixes. A chunk is complete only if both its WAV and matching successful JSONL record exist and hashes agree.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_audio.py tests/unit/experiments/rap_audio_protocols/test_artifacts.py -q`

```bash
git add src/streammuse/experiments/rap_audio_protocols/audio.py src/streammuse/experiments/rap_audio_protocols/artifacts.py tests/unit/experiments/rap_audio_protocols
git commit -m "feat(rap): assemble comparable offline audio artifacts"
```

### Task 4: MOSS-TTS Global-Duration Backend

**Files:**
- Create: `scripts/rap_audio_backends/moss_backend.py`
- Create: `tests/unit/scripts/test_rap_audio_moss_backend.py`

**Interfaces:**
- Consumes: request JSONL, `OpenMOSS-Team/MOSS-TTS-v1.5`, and a reference WAV.
- Produces: 24 kHz mono raw chunk WAVs and one `ChunkRenderRecord` per request.

- [ ] **Step 1: Write failing dependency-free adapter tests**

Inject fake processor/model/torchaudio objects. Assert the adapter calls:

```python
processor.build_user_message(
    text=request.text,
    language="English",
    reference=[str(reference_wav)],
    tokens=67,
    instruction="clear, rhythmically spoken rap with restrained pitch",
)
```

and uses `mode="generation"`, `audio_temperature=1.7`, `audio_top_p=0.8`, `audio_top_k=25`, and `audio_repetition_penalty=1.0`. The shared processor accepts `instruction`, but MOSS-TTS-v1.5 does not document style-following as a guaranteed model capability; record that caveat in every campaign manifest. Assert retries preserve text/request identity, seed PyTorch on each attempt as best effort, and log that official deterministic seeding is unsupported.

- [ ] **Step 2: Run the test and verify failure**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_moss_backend.py -q`

- [ ] **Step 3: Implement lazy imports and batch inference**

Load `AutoProcessor` and `AutoModel` only inside backend startup, move the audio tokenizer and model to the selected GPU, use BF16, disable cuDNN SDPA, prefer Flash Attention 2 when installed, and otherwise use SDPA. Load the model once, process all pending requests sequentially, and save decoded audio using `processor.model_config.sampling_rate`.

- [ ] **Step 4: Run local fake tests and commit**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_moss_backend.py -q`

```bash
git add scripts/rap_audio_backends/moss_backend.py tests/unit/scripts/test_rap_audio_moss_backend.py
git commit -m "feat(rap): add MOSS duration-controlled backend"
```

### Task 5: TED-TTS Local-Duration Backend

**Files:**
- Create: `scripts/rap_audio_backends/ted_backend.py`
- Create: `tests/unit/scripts/test_rap_audio_ted_backend.py`

**Interfaces:**
- Consumes: request JSONL, precomputed TED segments, TED-TTS checkout, IndexTTS2 checkpoints, and a reference WAV.
- Produces: 22.05 kHz chunk WAVs and render records.

- [ ] **Step 1: Write failing adapter tests with a fake IndexTTS2**

Assert one inference call per two-bar request with literal pipe-separated segment text, equal-length pipe-separated emotion descriptions, and:

```python
target_duration_tokens=[int(seconds / 0.02) for seconds in target_seconds]
duration_mode="both"
use_random=False
do_sample=True
top_p=0.8
top_k=30
temperature=0.8
num_beams=3
repetition_penalty=10.0
method="hmm"
```

Assert segment/count mismatch fails before model invocation.

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_ted_backend.py -q`

- [ ] **Step 3: Implement the TED-specific Python adapter**

Import `IndexTTS2` from the configured TED checkout, construct it with `is_fp16=True`, and call the modified `infer()` API. Use the fixed description `clear, confident, rhythmic spoken rap with restrained melody` for every segment so duration, not changing prose, is the experimental variable. Record that `use_random=False` does not guarantee determinism because TED still samples.

- [ ] **Step 4: Run fake tests and commit**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_ted_backend.py -q`

```bash
git add scripts/rap_audio_backends/ted_backend.py tests/unit/scripts/test_rap_audio_ted_backend.py
git commit -m "feat(rap): add TED local-duration backend"
```

### Task 6: NeMo FastPitch Explicit-Phoneme Backend

**Files:**
- Create: `scripts/rap_audio_backends/fastpitch_backend.py`
- Create: `tests/unit/scripts/test_rap_audio_fastpitch_backend.py`

**Interfaces:**
- Consumes: request JSONL and official `tts_en_fastpitch`/`tts_en_hifigan` checkpoints.
- Produces: 22.05 kHz chunk WAVs, tokenizer-label diagnostics, planned duration frames, and anchor-error records.

- [ ] **Step 1: Write failing adapter tests**

Use fake FastPitch and HiFi-GAN objects. Assert the backend obtains `tokens = fp.parse(text)`, obtains labels using `fp.vocab.ids_to_tokens(...)`, creates a duration tensor with shape equal to `tokens.shape`, and invokes the direct model call:

```python
spect, *_ = fp(
    text=tokens,
    durs=durs,
    pitch=pitch,
    energy=energy,
    speaker=None,
    pace=1.0,
)
audio = hifigan.convert_spectrogram_to_audio(spec=spect)
```

Assert `generate_spectrogram()` is never used because it discards explicit durations.

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_fastpitch_backend.py -q`

- [ ] **Step 3: Implement checkpoint loading and controlled inference**

Load each checkpoint once on the requested GPU. Build explicit durations through Task 2. First run a duration-only dry call. Enable pitch/energy accents only when the installed NeMo forward signature accepts their generated shapes; otherwise record `prosody_controls="duration_only_version_guard"` rather than guessing incompatible tensors.

- [ ] **Step 4: Run fake tests and commit**

Run: `.venv/bin/pytest tests/unit/scripts/test_rap_audio_fastpitch_backend.py -q`

```bash
git add scripts/rap_audio_backends/fastpitch_backend.py tests/unit/scripts/test_rap_audio_fastpitch_backend.py
git commit -m "feat(rap): add explicit-duration FastPitch backend"
```

### Task 7: Forced-Alignment And Piecewise-Warp Baseline

**Files:**
- Create: `scripts/rap_audio_backends/aligned_moss_backend.py`
- Create: `src/streammuse/experiments/rap_audio_protocols/warp.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_warp.py`
- Create: `tests/unit/scripts/test_rap_audio_aligned_moss_backend.py`

**Interfaces:**
- Consumes: Protocol 1 raw MOSS chunks, known lyrics, MFA TextGrids, and target syllable slots.
- Produces: warped chunks, source hashes, anchor maps, local stretch diagnostics, and explicit failures.

- [ ] **Step 1: Write failing monotonic-warp tests**

Test vowel detection for stressed/unstressed ARPAbet phones, TextGrid phone extraction, source/target anchor count mismatch, non-monotonic anchors, impossible regions below 10 ms, exact target frame count, and source SHA propagation. Use impulse fixtures to assert mapped anchors land within one output sample before crossfade smoothing.

- [ ] **Step 2: Run and verify failure**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_warp.py tests/unit/scripts/test_rap_audio_aligned_moss_backend.py -q`

- [ ] **Step 3: Implement MFA batch staging and parsing**

Write one `.wav` and `.lab` pair per pending source chunk, invoke MFA with the English US ARPA dictionary/acoustic model, and parse phone intervals from TextGrid. Match aligned vowels to planned syllables in lexical order. Source p-center is vowel onset plus `min(0.030, 0.25 * vowel_duration)`.

- [ ] **Step 4: Implement simple pitch-preserving piecewise warp**

Define source and target region boundaries halfway between adjacent p-centers. Stretch each region to its target boundary interval with Rubber Band, then apply a 5 ms equal-power crossfade without moving the target region boundary. Fail rather than clamp when target mappings are non-monotonic or an aligned vowel is missing. Record every local stretch ratio.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols/test_warp.py tests/unit/scripts/test_rap_audio_aligned_moss_backend.py -q`

```bash
git add scripts/rap_audio_backends/aligned_moss_backend.py src/streammuse/experiments/rap_audio_protocols/warp.py tests/unit/experiments/rap_audio_protocols tests/unit/scripts/test_rap_audio_aligned_moss_backend.py
git commit -m "feat(rap): add forced-alignment warp baseline"
```

### Task 8: Campaign CLI, Evaluation, Listening Page, And H200 Setup

**Files:**
- Create: `src/streammuse/experiments/rap_audio_protocols/evaluation.py`
- Create: `src/streammuse/experiments/rap_audio_protocols/listening.py`
- Create: `scripts/run_rap_audio_protocol_comparison.py`
- Create: `scripts/setup_rap_audio_protocols_h200.sh`
- Create: `tests/unit/experiments/rap_audio_protocols/test_evaluation.py`
- Create: `tests/unit/experiments/rap_audio_protocols/test_listening.py`
- Create: `tests/unit/scripts/test_run_rap_audio_protocol_comparison.py`
- Create: `docs/developer-guide/rap-audio-protocol-comparison.md`

**Interfaces:**
- Consumes: Tasks 1 through 7.
- Produces: resumable `prepare`, `assemble`, `evaluate`, and `package` stages; H200 environment setup; `COMPARISON.md`; metrics JSON; blinded `listening.html` and `blind_map.json`.

- [ ] **Step 1: Write failing CLI and evaluator tests**

Assert `prepare` selects only songs 1-3, writes exactly 75 canonical requests, creates one common drum stem per song, and refuses a mismatched existing manifest. Assert `assemble` requires 25 records per protocol/song and produces 6,400,000-frame stems. Test exact Levenshtein WER, duration statistics, clipping counts, RMS/stress correlation, and failed-chunk counts.

- [ ] **Step 2: Write failing listening-package tests**

Assert the page contains three song sections and four audio controls per song, paths are relative, displayed methods are neutral `A` through `D`, the deterministic blind map is complete, and the page does not expose protocol names before unblinding.

- [ ] **Step 3: Run tests and verify failure**

Run: `.venv/bin/pytest tests/unit/experiments/rap_audio_protocols tests/unit/scripts/test_run_rap_audio_protocol_comparison.py -q`

- [ ] **Step 4: Implement resumable campaign stages**

The CLI must accept explicit `--source-album`, `--output-dir`, `--stage`, `--song`, and `--protocol` arguments. It must never assume a home-directory path. Print one dense progress line per chunk and write errors immediately to JSONL.

- [ ] **Step 5: Implement independent evaluation**

Use faster-whisper for transcription and word timestamps. Compute WER against normalized known lyrics. Estimate within-word syllable anchors proportionally only when the independent aligner returns a valid word interval; label this metric `estimated_syllable_timing_error_ms` and never call it phone-level ground truth. Compute stress/RMS correlation at target slots and basic signal integrity locally.

- [ ] **Step 6: Implement idempotent H200 setup**

Create separate environments below `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/` named `moss`, `ted`, `nemo`, and `align`. Pin external repository revisions into `environment_manifest.json`. Install model checkpoints under the shared Hugging Face cache, install MFA English US ARPA models in the align environment, and install Rubber Band without requiring administrator privileges. Never alter the existing StreamMUSE environment.

- [ ] **Step 7: Write operator documentation and run tests**

Document setup, smoke, resume, full render, rsync download, and local listening commands. Run:

```bash
.venv/bin/pytest tests/unit/experiments/rap_audio_protocols \
  tests/unit/scripts/test_rap_audio_moss_backend.py \
  tests/unit/scripts/test_rap_audio_ted_backend.py \
  tests/unit/scripts/test_rap_audio_fastpitch_backend.py \
  tests/unit/scripts/test_rap_audio_aligned_moss_backend.py \
  tests/unit/scripts/test_run_rap_audio_protocol_comparison.py -q
```

- [ ] **Step 8: Commit Task 8**

```bash
git add \
  src/streammuse/experiments/rap_audio_protocols/evaluation.py \
  src/streammuse/experiments/rap_audio_protocols/listening.py \
  scripts/run_rap_audio_protocol_comparison.py \
  scripts/setup_rap_audio_protocols_h200.sh \
  tests/unit/experiments/rap_audio_protocols/test_evaluation.py \
  tests/unit/experiments/rap_audio_protocols/test_listening.py \
  tests/unit/scripts/test_run_rap_audio_protocol_comparison.py \
  docs/developer-guide/rap-audio-protocol-comparison.md
git commit -m "feat(rap): orchestrate audio protocol comparison"
```

### Task 9: H200 Smoke Tests And Full Twelve-Song Render

**Files:**
- Create outside Git: `output/rap_audio_protocol_comparison_20260816/`
- Create: `docs/developer-guide/rap-audio-protocol-comparison-results-2026-08-16.md`

**Interfaces:**
- Consumes: all implementation tasks and unused H200 GPUs.
- Produces: twelve mixes, twelve vocal stems, three shared drum stems, logs, metrics, manifests, listening page, and concise results report downloaded to the local worktree.

- [ ] **Step 1: Synchronize code and fixed corpus to H200**

Use `rsync` with explicit source/destination paths. Do not synchronize model caches, existing ten-song WAVs, `.git`, or unrelated output directories. Verify the three source `chosen_lyrics.jsonl` SHA-256 values on both hosts.

- [ ] **Step 2: Provision isolated environments on unused GPUs**

Run the setup script, record `nvidia-smi` before model launch, and assign only GPUs reporting zero compute processes. Store GPU/model/environment identities in the campaign manifest.

- [ ] **Step 3: Run one shared two-bar smoke fixture through all four protocols**

Acceptance for each protocol: process exit zero, nonempty WAV, expected source sample rate, no missing/repeated transcript according to smoke ASR, complete render record, and Protocol 4 source hash equal to Protocol 1. Listen to all four smoke WAVs before launching the campaign; reject only catastrophic output such as silence, truncation before the final word, repeated loops, or unintelligible noise.

- [ ] **Step 4: Render Protocols 1 through 3 and monitor logs**

Run MOSS, TED, and FastPitch on separate unused GPUs when memory permits. Resume from verified chunk records after any interruption. Do not rerun successful chunks merely to improve a subjective sample.

- [ ] **Step 5: Render Protocol 4 from completed MOSS chunks**

Run MFA and warp only after all MOSS source chunks and hashes exist. Verify all 75 source hashes against Protocol 1 records.

- [ ] **Step 6: Assemble, evaluate, and package**

Generate stems/mixes, run independent evaluation, write blinded listening artifacts, and validate all expected frame counts and hashes. Run a final artifact audit requiring 12 mixes, 12 vocal stems, 3 shared drum stems, 300 successful-or-explicitly-failed chunk records, and no unaccounted chunk.

- [ ] **Step 7: Download artifacts to the local worktree**

Use `rsync --partial --progress` into `output/rap_audio_protocol_comparison_20260816/`. Compare the remote and local aggregate manifest SHA-256.

- [ ] **Step 8: Write and commit the concise results report**

Report protocol versions, failures, objective metrics, generation times, and artifact locations without declaring a perceptual winner before the user listens.

```bash
git add docs/developer-guide/rap-audio-protocol-comparison-results-2026-08-16.md
git commit -m "docs(rap): report audio protocol comparison"
```

- [ ] **Step 9: Run final repository verification**

Run:

```bash
.venv/bin/pytest tests/unit/experiments/rap_audio_protocols \
  tests/unit/scripts/test_rap_audio_moss_backend.py \
  tests/unit/scripts/test_rap_audio_ted_backend.py \
  tests/unit/scripts/test_rap_audio_fastpitch_backend.py \
  tests/unit/scripts/test_rap_audio_aligned_moss_backend.py \
  tests/unit/scripts/test_run_rap_audio_protocol_comparison.py -q
git status --short
```

Expected: all comparison tests pass; only known pre-existing worktree changes and ignored output artifacts remain.
