# Voice Length-Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing voice microbenchmark with reproducible STT/TTS word-length sweeps, raw trial data, and latency/duration line plots for Mac and H200.

**Architecture:** `scripts/voice_microbench.py` remains the single command surface. A new `--length-sweep` mode builds a deterministic phrase corpus, invokes the existing persistent batch backends with warmups and repeated trials, writes normalized trial rows, and renders plots from those rows. The legacy default mode remains unchanged.

**Tech Stack:** Python 3.10+, standard library (`csv`, `json`, `random`, `wave`), faster-whisper, Piper, espeak-ng, Matplotlib, pytest.

## Global Constraints

- Benchmark is isolated from microphone capture, VAD, networking, speaker playback, LLM inference, and StreamMUSE runtime orchestration.
- Test word counts are exactly `1, 2, 4, 8, 16, 32, 64`, with five deterministic phrases per bucket and ten warmed trials per phrase.
- STT records word count, input WAV duration, transcription latency, and transcript.
- TTS records word count, generation time, generated WAV duration, and real-time factor; duration is not reported as latency.
- Mac uses faster-whisper CPU int8 and espeak-ng; H200 uses faster-whisper GPU float16 and persistent Piper.
- Each run writes raw JSON/CSV, a manifest, summary JSON/Markdown, and PNG line plots with individual trials plus p50/p95 curves.

---

### Task 1: Add deterministic sweep data contracts and corpus generation

**Files:**
- Modify: `scripts/voice_microbench.py`
- Modify: `tests/unit/scripts/test_voice_microbench.py`

**Interfaces:**
- Produces `SweepPhrase(phrase_id: str, text: str, word_count: int)`.
- Produces `build_length_sweep_phrases(word_counts: tuple[int, ...], variants_per_length: int) -> tuple[SweepPhrase, ...]`.
- Produces `audio_duration_s(path: Path) -> float` using WAV metadata.

- [ ] **Step 1: Write failing tests for corpus determinism and word counts**

```python
def test_build_length_sweep_phrases_has_five_variants_for_each_requested_length() -> None:
    phrases = build_length_sweep_phrases((1, 2, 4), variants_per_length=5)
    assert len(phrases) == 15
    assert {item.word_count for item in phrases} == {1, 2, 4}
    assert all(len(item.text.split()) == item.word_count for item in phrases)
    assert len({item.phrase_id for item in phrases}) == len(phrases)
```

- [ ] **Step 2: Run the focused test to verify failure**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py::test_build_length_sweep_phrases_has_five_variants_for_each_requested_length -q`

Expected: FAIL because `build_length_sweep_phrases` is not defined.

- [ ] **Step 3: Implement the immutable phrase contract and corpus builder**

```python
@dataclass(frozen=True)
class SweepPhrase:
    phrase_id: str
    text: str
    word_count: int

def build_length_sweep_phrases(
    word_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    variants_per_length: int = 5,
) -> tuple[SweepPhrase, ...]:
    streams = (
        "cat watches soft rain fall across the quiet garden beside a warm window "
        "while evening lights glow over the small street after sunset today",
        "dog follows bright leaves along the calm river near the old bridge "
        "while gentle wind moves through the tall trees before dinner tonight",
        "bird carries a small twig toward the hidden nest above green branches "
        "as morning sun reaches the peaceful park beside our familiar school",
        "child finds red shells beside the clear water on a sandy beach "
        "then walks slowly home with a cheerful friend before lunch today",
        "friend brings fresh bread from the local market through the busy square "
        "and shares a warm meal with family after a long afternoon outside",
    )
    return tuple(
        SweepPhrase(f"w{count}-v{variant}", " ".join(stream.split()[:count]), count)
        for count in word_counts for variant, stream in enumerate(streams[:variants_per_length], 1)
    )
```

Implement `audio_duration_s()` with `wave.open()` and explicit validation for
zero sample rate. Reuse it from the existing sample-information path.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py -q`

Expected: PASS.

### Task 2: Collect persistent-backend sweep trial rows

**Files:**
- Modify: `scripts/voice_microbench.py`
- Modify: `tests/unit/scripts/test_voice_microbench.py`

**Interfaces:**
- Produces `SweepTrial` rows with `kind`, `phrase_id`, `text`, `word_count`, `repeat_index`, `latency_ms`, `audio_duration_s`, `real_time_factor`, `output`, and `error`.
- Adds `run_length_sweep(...) -> SweepRunResult`.
- Consumes the existing `build_piper_tts_batch_code()` and faster-whisper batch mechanism, extended to accept ordered warmup and measurement requests.

- [ ] **Step 1: Write failing summary and repeat-accounting tests**

```python
def test_summarize_sweep_trials_reports_p50_p95_and_separates_tts_duration() -> None:
    rows = (
        SweepTrial(kind="tts", phrase_id="one-a", text="cat", word_count=1,
                   repeat_index=1, latency_ms=20.0, audio_duration_s=0.5,
                   output="out.wav"),
        SweepTrial(kind="tts", phrase_id="one-a", text="cat", word_count=1,
                   repeat_index=2, latency_ms=40.0, audio_duration_s=0.5,
                   output="out.wav"),
    )
    summary = summarize_sweep_trials(rows)
    assert summary["by_word_count"]["1"]["generation_p50_ms"] == 30.0
    assert summary["by_word_count"]["1"]["audio_duration_mean_s"] == 0.5
    assert summary["by_word_count"]["1"]["rtf_p50"] == 0.06
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py::test_summarize_sweep_trials_reports_p50_p95_and_separates_tts_duration -q`

Expected: FAIL because `SweepTrial` and `summarize_sweep_trials` are not defined.

- [ ] **Step 3: Implement the sweep execution path**

Add CLI arguments:

```text
--length-sweep
--repetitions 10
--variants-per-length 5
--word-counts 1,2,4,8,16,32,64
--seed 20260717
```

In the batch subprocess code, load the model once; execute three unmeasured
warmup requests; process a deterministic shuffled schedule; return one JSON
row per measured request. Piper rows must include the duration of the created
WAV. Faster-whisper rows must retain each input WAV duration. Use `random.Random(seed)`
instead of global random state.

- [ ] **Step 4: Implement percentile summaries without NumPy**

```python
def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]
```

Emit `p50_ms`, `p95_ms`, mean, maximum, and count for all trial groups. For
TTS, emit separate `generation_*_ms`, `audio_duration_*_s`, and `rtf_*` fields.

- [ ] **Step 5: Run focused tests**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py -q`

Expected: PASS.

### Task 3: Write sweep artifacts and render plots

**Files:**
- Modify: `scripts/voice_microbench.py`
- Modify: `tests/unit/scripts/test_voice_microbench.py`

**Interfaces:**
- Produces `length_sweep_trials.json`, `length_sweep_trials.csv`, `length_sweep_summary.json`, `length_sweep_report.md`, and `plots/*.png`.
- Produces `render_length_sweep_plots(trials, plots_dir, title_prefix) -> tuple[Path, ...]`.

- [ ] **Step 1: Write failing artifact and plot tests**

```python
def test_write_length_sweep_outputs_creates_raw_data_summary_and_expected_plots(tmp_path: Path) -> None:
    paths = write_length_sweep_outputs(tmp_path, fixture_sweep_result())
    assert (tmp_path / "length_sweep_trials.json").exists()
    assert (tmp_path / "length_sweep_trials.csv").exists()
    assert (tmp_path / "length_sweep_summary.json").exists()
    assert {path.name for path in paths} == {
        "stt_latency_by_words.png", "stt_latency_by_audio_duration.png",
        "tts_generation_by_words.png", "tts_audio_duration_by_words.png",
        "tts_rtf_by_words.png",
    }
```

- [ ] **Step 2: Run focused test to verify failure**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py::test_write_length_sweep_outputs_creates_raw_data_summary_and_expected_plots -q`

Expected: FAIL because `write_length_sweep_outputs` is not defined.

- [ ] **Step 3: Implement raw artifact writing and plots**

Use the non-interactive Matplotlib backend (`Agg`). Plot individual trial
markers with low alpha, p50 as a solid line, and p95 as a dashed line. Never
put generation time and audio duration on the same axis. Ensure only the plots
for requested modalities are rendered.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py -q`

Expected: PASS.

### Task 4: Validate, run Mac and H200 matrices, and report findings

**Files:**
- Modify: `docs/developer-guide/voice-microbenchmark-results.md`
- Create: `voice_bench_runs/voice_length_sweep_YYYYMMDD-HHMMSS/` (ignored raw artifacts)
- Create on H200: `/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-v1/voice_bench_runs/voice_length_sweep_YYYYMMDD-HHMMSS/`

**Interfaces:**
- Mac command uses `FASTER_WHISPER_DEVICE=cpu`, `FASTER_WHISPER_COMPUTE_TYPE=int8`, `--stt-backends faster_whisper`, and `--tts-backends espeak_ng`.
- H200 command uses an idle GPU selected from `nvidia-smi`, then `FASTER_WHISPER_DEVICE=cuda`, `FASTER_WHISPER_COMPUTE_TYPE=float16`, required CUDA library path, `PIPER_MODEL`, `--stt-backends faster_whisper`, and `--tts-backends piper`.

- [ ] **Step 1: Run unit tests and a reduced Mac smoke sweep**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py -q`

Then run a one-word/one-variant/two-repeat sweep on the available Mac backend
and inspect the manifest, row count, duration fields, and PNG files.

- [ ] **Step 2: Run the full Mac matrix**

Run both selected backends with the approved seven buckets, five variants, and
ten repetitions. Preserve the exact command in the Markdown report.

- [ ] **Step 3: Inspect H200 availability and run the full H200 matrix**

Use `ssh H200` and `nvidia-smi` to choose an idle GPU. Do not terminate or
interfere with unrelated processes. Confirm the existing benchmark environment,
models, and CUDA libraries before running the same sweep there.

- [ ] **Step 4: Update the results report**

Add per-device setup/first-request/warmed summaries, raw-artifact locations,
and embedded/linkable plots. State that results exclude endpointing and
playback. Explain the maximum experimentally supported word length under one
second for each path.

- [ ] **Step 5: Run final regression tests**

Run: `uv run pytest tests/unit/scripts/test_voice_microbench.py -q`

Expected: PASS.
