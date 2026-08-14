# Realtime Rap Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sample-accurate robotic syllable performance and a stable drum reference to the realtime rap demo while keeping its current text-only terminal and web workflows working unchanged by default.

**Architecture:** The H200 remains a text-only vLLM service. A Mac Python process adds replaceable speech, drum, bar-rendering, playback, and recording components around the existing rolling planner; an audio candidate replaces a fallback only after its complete PCM bar is ready. Audio mode uses a PortAudio/CoreAudio sample clock, while the existing monotonic tick loop remains the default when audio is disabled.

**Tech Stack:** Python 3.10+, eSpeak NG 1.52+, NumPy/SciPy, sounddevice/PortAudio, FastAPI, WebSocket, vanilla HTML/CSS/JavaScript, pytest, vLLM OpenAI-compatible HTTP API.

**Spec:** `docs/superpowers/specs/2026-08-14-realtime-rap-audio-design.md`

## Global Constraints

- Preserve `streammuse-rap-demo` text-only behavior when `--audio-output none`; this remains the parser default.
- Preserve existing defaults of scenario tempo 92 BPM, candidate count 8, and lookahead 2 for text-only invocations.
- Use the explicit audio-demo configuration `--tempo 60 --candidate-count 12 --lookahead-bars 3` in documentation and acceptance runs.
- Keep the H200 text-only: no TTS model, PCM payload, or audio endpoint is added to the server.
- Keep all synthesis, mixing, playback, recording, and website hosting on the Mac.
- Use 48,000 Hz stereo float32 PCM and derive every onset from an absolute bar/tick sample calculation.
- Use eSpeak phoneme rendering first and best-effort pronunciation warnings; pronunciation never changes candidate validity or score.
- Never delay a scheduled syllable onset because an earlier waveform is long.
- Keep browser controls limited to Start, Stop, and Reset.
- Treat Start, Stop, and Reset as the only runtime controls; preserve existing monitor-only tools such as Follow live and candidate-table sorting.
- Stop completes the active bar before entering `STOPPED`.
- Do not perform HTTP, synthesis, rendering, disk I/O, formatting, or event dispatch in the audio callback.
- Keep all new audio dependencies lazy at runtime so importing or running text-only mode does not require a working audio device or eSpeak executable.
- After every task, run `uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q`; do not continue past a new failure.
- Commit each task separately and do not combine unrelated cleanup with an audio task.

---

## Planned File Structure

### New Files

- `src/streammuse/domain/rap/audio.py`: immutable PCM, warning, render, prepared-bar, playback-state, and playback-notice value objects.
- `src/streammuse/application/rap/audio_service.py`: speech, drum, bar renderer, and audio sink protocols.
- `src/streammuse/application/rap/audio_rendering.py`: absolute sample arithmetic, silence trimming, duration fitting, mixing, and limiting.
- `src/streammuse/application/rap/bar_renderer.py`: build one immutable vocal-and-drum PCM bar from `PlannedRapBar`.
- `src/streammuse/application/rap/audio_coordination.py`: asynchronous fallback/primary rendering and atomic source commitment.
- `src/streammuse/application/rap/playback.py`: audio state machine, sample-position observer, and bar-quantized stop behavior.
- `src/streammuse/infrastructure/rap/speech.py`: ARPAbet/eSpeak conversion, eSpeak subprocess adapter, cache, and degraded pronunciation fallback.
- `src/streammuse/infrastructure/rap/drums.py`: deterministic procedural boom-bap renderer.
- `src/streammuse/infrastructure/rap/audio_output.py`: sounddevice, WAV, composite, and null audio sinks.
- `tests/unit/domain/rap/test_audio.py`: value-object validation and event contract tests.
- `tests/unit/application/rap/test_audio_rendering.py`: sample math and syllable fitting tests.
- `tests/unit/application/rap/test_bar_renderer.py`: exact bar construction and diagnostics tests.
- `tests/unit/application/rap/test_audio_coordination.py`: fallback-first and atomic-commit tests.
- `tests/unit/application/rap/test_playback.py`: state machine, observer, stop, and reset tests.
- `tests/unit/infrastructure/rap/test_speech.py`: phoneme mapping, command construction, fallback, and cache tests.
- `tests/unit/infrastructure/rap/test_drums.py`: deterministic drum placement tests.
- `tests/unit/infrastructure/rap/test_audio_output.py`: callback, queue, WAV, composite, and null sink tests.
- `tests/integration/test_realtime_rap_audio.py`: complete fake-clock rolling audio tests.

### Modified Files

- `src/streammuse/domain/rap/events.py`: canonical audio event types.
- `src/streammuse/domain/rap/__init__.py`: public audio-domain exports.
- `src/streammuse/application/rap/realtime.py`: optional audio-ready proposal and commit path while retaining the current text path.
- `src/streammuse/application/rap/runtime.py`: selectable text clock or audio playback runtime and restart-safe controls.
- `src/streammuse/application/rap/monitoring.py`: audio state, warning, queue, and latency projections.
- `src/streammuse/application/rap/__init__.py`: public application exports.
- `src/streammuse/infrastructure/rap/__init__.py`: public infrastructure exports.
- `src/streammuse/infrastructure/rap/recorder.py`: audio fields in session summaries.
- `src/streammuse/presentation/rap_demo/cli.py`: additive audio and tempo-override configuration.
- `src/streammuse/presentation/rap_demo/server.py`: Start/Stop/Reset endpoints and controllable audio-mode lifecycle.
- `src/streammuse/presentation/rap_demo/terminal_stream.py`: dense audio lifecycle and warning records.
- `src/streammuse/presentation/rap_demo/terminal_state.py`: projected audio fields used by stream and split renderers.
- `src/streammuse/presentation/rap_demo/terminal_dashboard.py`: audio readiness, buffer, and warning panels.
- `src/streammuse/presentation/rap_demo/static/index.html`: exactly three runtime controls and audio telemetry.
- `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`: compact controls and warning treatment matching the existing monitor.
- `src/streammuse/presentation/rap_demo/static/js/rap-demo.js`: control requests and audio-state rendering.
- `tests/unit/application/rap/test_realtime.py`: optional audio path and unchanged text path.
- `tests/unit/application/rap/test_runtime.py`: selectable clocks and control lifecycle.
- `tests/unit/application/rap/test_monitoring.py`: audio projection and cumulative metrics.
- `tests/unit/presentation/rap_demo/test_cli.py`: parser compatibility and audio assembly.
- `tests/unit/presentation/rap_demo/test_server.py`: control API and repeated runtime starts.
- `tests/unit/presentation/rap_demo/test_terminal_stream.py`: structured audio records.
- `tests/unit/presentation/rap_demo/test_terminal_dashboard.py`: split-layout audio rows.
- `tests/unit/presentation/rap_demo/test_terminal_state.py`: audio view state.
- `pyproject.toml`: `sounddevice` dependency.
- `uv.lock`: resolved sounddevice package.
- `docs/developer-guide/rap-demo-quickstart.md`: split Mac/H200 audio commands and model-port SSH tunnel.
- `docs/developer-guide/rap-acceptance-report-2026-08-09.md`: link to the new audio acceptance evidence without rewriting the prior report.

---

### Task 1: Freeze The Existing Behavior And Add Audio Domain Contracts

**Files:**
- Create: `src/streammuse/domain/rap/audio.py`
- Create: `tests/unit/domain/rap/test_audio.py`
- Modify: `src/streammuse/domain/rap/events.py`
- Modify: `src/streammuse/domain/rap/__init__.py`

**Interfaces:**
- Consumes: existing `ScheduledSyllable` from `streammuse.domain.rap.models`.
- Produces: `AudioFormat`, `PcmAudio`, `AudioWarningCode`, `AudioWarningSeverity`, `AudioWarning`, `SyllableRenderRequest`, `RenderedSyllable`, `SyllablePlacementDiagnostic`, `PreparedRapBar`, `PlaybackState`, `AudioPlaybackNoticeKind`, `AudioPlaybackNotice`, and `AudioPlaybackSnapshot`.

- [ ] **Step 1: Run the existing rap regression baseline**

Run:

```bash
uv run pytest \
  tests/unit/domain/rap \
  tests/unit/application/rap \
  tests/unit/infrastructure/rap \
  tests/unit/presentation/rap_demo -q
```

Expected: all existing tests pass before any implementation file changes.

- [ ] **Step 2: Write failing validation and event-contract tests**

Add tests that pin float32 frame sizing, immutable prepared-bar metadata, playback states, and the complete event names:

```python
def test_pcm_audio_requires_exact_float32_frame_bytes() -> None:
    audio_format = AudioFormat(sample_rate_hz=48_000, channels=2)

    with pytest.raises(ValueError, match="frame byte length"):
        PcmAudio(format=audio_format, frame_count=2, data=b"short")


def test_pcm_audio_accepts_exact_stereo_float32_data() -> None:
    audio_format = AudioFormat(sample_rate_hz=48_000, channels=2)
    audio = PcmAudio(format=audio_format, frame_count=2, data=bytes(2 * 2 * 4))

    assert audio.duration_seconds == pytest.approx(2 / 48_000)


def test_rap_audio_event_names_are_canonical() -> None:
    assert RapEventType.AUDIO_RENDER_STARTED.value == "audio_render_started"
    assert RapEventType.BAR_AUDIO_COMMITTED.value == "bar_audio_committed"
    assert RapEventType.AUDIO_UNDERRUN.value == "audio_underrun"
    assert RapEventType.SESSION_RESET.value == "session_reset"
```

- [ ] **Step 3: Run the domain tests and verify the new imports fail**

Run:

```bash
uv run pytest tests/unit/domain/rap/test_audio.py -v
```

Expected: collection fails because the new audio types and event members do not exist.

- [ ] **Step 4: Implement immutable audio value objects**

Define the exact public shapes in `audio.py`:

```python
class AudioWarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class AudioWarningCode(str, Enum):
    PRONUNCIATION_FALLBACK = "pronunciation_fallback"
    TIMING_PRESSURE = "timing_pressure"
    FORCED_BAR_FIT = "forced_bar_fit"
    SYNTHESIS_FAILED = "synthesis_failed"
    AUDIO_DEADLINE_MISS = "audio_deadline_miss"
    AUDIO_UNDERRUN = "audio_underrun"
    AUDIO_DEVICE_FAILED = "audio_device_failed"


class PlaybackState(str, Enum):
    STOPPED = "stopped"
    PRIMING = "priming"
    RUNNING = "running"
    STOP_REQUESTED = "stop_requested"
    CLOSED = "closed"


@dataclass(frozen=True)
class AudioFormat:
    sample_rate_hz: int = 48_000
    channels: int = 2
    sample_width_bytes: int = 4


@dataclass(frozen=True)
class PcmAudio:
    format: AudioFormat
    frame_count: int
    data: bytes

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.format.sample_rate_hz
```

Define the remaining dataclasses with the field names below so later tasks use one contract:

```python
@dataclass(frozen=True)
class AudioWarning:
    code: AudioWarningCode
    severity: AudioWarningSeverity
    message: str
    bar: int | None = None
    slot_index: int | None = None
    word: str | None = None
    available_ms: float | None = None
    rendered_ms: float | None = None
    compression_ratio: float | None = None
    overlap_ms: float | None = None
    action: str | None = None


@dataclass(frozen=True)
class SyllableRenderRequest:
    bar: int
    slot_index: int
    word: str
    index_in_word: int
    syllable_count: int
    phonemes: tuple[str, ...]
    stress: int
    analysis_source: str
    voice: str
    speed_wpm: int
    pitch: int


@dataclass(frozen=True)
class RenderedSyllable:
    request: SyllableRenderRequest
    audio: PcmAudio
    renderer_phonemes: tuple[str, ...]
    pronunciation_source: str
    synthesis_latency_ms: float
    warnings: tuple[AudioWarning, ...] = ()


@dataclass(frozen=True)
class SyllablePlacementDiagnostic:
    bar: int
    slot_index: int
    word: str
    target_sample: int
    source_frames: int
    fitted_frames: int
    available_frames: int
    compression_ratio: float
    overlap_frames: int
    pronunciation_source: str
    software_error_samples: int = 0


@dataclass(frozen=True)
class PreparedRapBar:
    bar: int
    text: str
    source: str
    fallback_reason: str | None
    scheduled: tuple[ScheduledSyllable, ...]
    audio: PcmAudio
    diagnostics: tuple[SyllablePlacementDiagnostic, ...]
    warnings: tuple[AudioWarning, ...]
    render_latency_ms: float
```

Add `AudioPlaybackNoticeKind` members `BAR_STARTED`, `BAR_COMPLETED`, `STOPPED`, `UNDERRUN`, and `DEVICE_FAILED`, plus an `AudioPlaybackNotice` carrying `kind`, `bar`, `absolute_frame`, `queue_depth`, and `message`.

Define the sink snapshot before the application protocol consumes it:

```python
@dataclass(frozen=True)
class AudioPlaybackSnapshot:
    state: PlaybackState
    current_bar: int | None
    frame_in_bar: int
    absolute_frame: int
    queue_depth: int
    underrun_count: int
```

- [ ] **Step 5: Extend and export canonical event types**

Add these enum values to `RapEventType`:

```python
AUDIO_RENDER_STARTED = "audio_render_started"
AUDIO_RENDER_COMPLETED = "audio_render_completed"
PRONUNCIATION_FALLBACK = "pronunciation_fallback"
TIMING_PRESSURE = "timing_pressure"
BAR_AUDIO_READY = "bar_audio_ready"
BAR_AUDIO_COMMITTED = "bar_audio_committed"
BAR_PLAYBACK_STARTED = "bar_playback_started"
BAR_PLAYBACK_COMPLETED = "bar_playback_completed"
STOP_REQUESTED = "stop_requested"
SESSION_RESET = "session_reset"
AUDIO_UNDERRUN = "audio_underrun"
AUDIO_DEVICE_FAILED = "audio_device_failed"
```

Export every new domain symbol from `streammuse.domain.rap`.

- [ ] **Step 6: Run the new tests and the regression gate**

Run:

```bash
uv run pytest tests/unit/domain/rap/test_audio.py tests/unit/domain/rap -q
uv run pytest tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: all tests pass and no existing canonical event behavior changes.

- [ ] **Step 7: Commit the domain contract**

```bash
git add src/streammuse/domain/rap tests/unit/domain/rap
git commit -m "feat: define realtime rap audio contracts"
```

---

### Task 2: Implement Absolute Sample Timing And Syllable Fitting

**Files:**
- Create: `src/streammuse/application/rap/audio_rendering.py`
- Create: `tests/unit/application/rap/test_audio_rendering.py`
- Modify: `src/streammuse/application/rap/__init__.py`

**Interfaces:**
- Consumes: `Tempo`, `AudioFormat`, `PcmAudio`, and `AudioWarning` from Task 1.
- Produces: `bar_start_frame(bar, tempo, audio_format) -> int`, `bar_frame_count(bar, tempo, audio_format) -> int`, `tick_frame_in_bar(bar, tick_in_bar, tempo, audio_format) -> int`, `trim_silence(audio, threshold_dbfs, padding_ms) -> PcmAudio`, `fit_syllable(audio, available_frames, final_in_bar, context) -> FittedSyllable`, `mix_at(destination, source, onset_frame, gain)`, and `limit_peak(samples, peak=0.95)`.

- [ ] **Step 1: Write failing exact-timing tests**

```python
def test_sixty_bpm_bar_and_tick_frames_are_exact() -> None:
    tempo = Tempo(60.0, 4, 4)
    audio_format = AudioFormat(48_000, 2)

    assert bar_start_frame(0, tempo, audio_format) == 0
    assert bar_frame_count(0, tempo, audio_format) == 192_000
    assert tick_frame_in_bar(0, 7, tempo, audio_format) == 84_000


def test_fractional_bar_lengths_do_not_accumulate_drift_at_92_bpm() -> None:
    tempo = Tempo(92.0, 4, 4)
    audio_format = AudioFormat(48_000, 2)
    lengths = [bar_frame_count(bar, tempo, audio_format) for bar in range(100)]

    assert sum(lengths) == bar_start_frame(100, tempo, audio_format)
    assert set(lengths).issubset({125_217, 125_218})
```

- [ ] **Step 2: Write failing fitting and mixing tests**

Cover unchanged audio, 1.5x compression, 2.0x capped overlap, final-bar emergency fit, -45 dBFS trimming with 5 ms padding, exact onset mixing, and peak limiting:

```python
def test_long_nonfinal_syllable_caps_compression_and_reports_overlap() -> None:
    source = mono_pcm(frames=1_000, value=0.25)
    context = FitContext(bar=2, slot_index=4, word="timing")

    fitted = fit_syllable(source, available_frames=300, final_in_bar=False, context=context)

    assert fitted.audio.frame_count == 500
    assert fitted.compression_ratio == pytest.approx(2.0)
    assert fitted.overlap_frames == 200
    assert fitted.warnings[0].code == AudioWarningCode.TIMING_PRESSURE


def test_final_syllable_is_forced_to_bar_boundary() -> None:
    source = mono_pcm(frames=1_000, value=0.25)
    context = FitContext(bar=2, slot_index=8, word="ending")

    fitted = fit_syllable(source, available_frames=300, final_in_bar=True, context=context)

    assert fitted.audio.frame_count == 300
    assert fitted.warnings[-1].code == AudioWarningCode.FORCED_BAR_FIT
```

- [ ] **Step 3: Run the tests and verify missing functions fail**

```bash
uv run pytest tests/unit/application/rap/test_audio_rendering.py -v
```

Expected: collection fails on the new function imports.

- [ ] **Step 4: Implement absolute frame arithmetic**

Use rounding only from absolute musical time:

```python
def bar_start_frame(bar: int, tempo: Tempo, audio_format: AudioFormat) -> int:
    absolute_tick = bar * tempo.ticks_per_bar
    return round(tempo.tick_to_seconds(absolute_tick) * audio_format.sample_rate_hz)


def bar_frame_count(bar: int, tempo: Tempo, audio_format: AudioFormat) -> int:
    return bar_start_frame(bar + 1, tempo, audio_format) - bar_start_frame(bar, tempo, audio_format)


def tick_frame_in_bar(bar: int, tick_in_bar: int, tempo: Tempo, audio_format: AudioFormat) -> int:
    absolute_tick = bar * tempo.ticks_per_bar + tick_in_bar
    absolute_frame = round(tempo.tick_to_seconds(absolute_tick) * audio_format.sample_rate_hz)
    return absolute_frame - bar_start_frame(bar, tempo, audio_format)
```

Validate nonnegative bars/ticks and tick bounds.

- [ ] **Step 5: Implement deterministic trim, resample-to-length, overlap, mixing, and limiting**

Use NumPy float32 arrays internally and return immutable PCM bytes across the public boundary. Compute threshold amplitude as `10 ** (threshold_dbfs / 20)`. Use `scipy.signal.resample` for exact target lengths; pitch preservation is intentionally outside this prototype.

Define:

```python
@dataclass(frozen=True)
class FitContext:
    bar: int
    slot_index: int
    word: str


@dataclass(frozen=True)
class FittedSyllable:
    audio: PcmAudio
    compression_ratio: float
    overlap_frames: int
    warnings: tuple[AudioWarning, ...]
```

For nonfinal units, use `target_frames = max(1, ceil(source_frames / min(required_ratio, 2.0)))`; the excess over `available_frames` is overlap. For final units, resample directly to `available_frames` when the 2.0x result still exceeds the bar.

- [ ] **Step 6: Run focused tests and the regression gate**

```bash
uv run pytest tests/unit/application/rap/test_audio_rendering.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: all tests pass, including the 100-bar drift assertion.

- [ ] **Step 7: Commit sample timing and fitting**

```bash
git add src/streammuse/application/rap/audio_rendering.py src/streammuse/application/rap/__init__.py tests/unit/application/rap/test_audio_rendering.py
git commit -m "feat: add sample-accurate rap audio fitting"
```

---

### Task 3: Add The Replaceable eSpeak Phoneme Synthesizer

**Files:**
- Create: `src/streammuse/application/rap/audio_service.py`
- Create: `src/streammuse/infrastructure/rap/speech.py`
- Create: `tests/unit/infrastructure/rap/test_speech.py`
- Modify: `src/streammuse/application/rap/__init__.py`
- Modify: `src/streammuse/infrastructure/rap/__init__.py`

**Interfaces:**
- Consumes: `SyllableRenderRequest`, `RenderedSyllable`, `PcmAudio`, and warnings from Task 1.
- Produces: `SpeechSynthesizer.synthesize(request) -> RenderedSyllable`, `arpabet_syllable_to_espeak(phonemes) -> tuple[str, ...]`, `EspeakPhonemeSynthesizer`, and injectable `CommandRunner`.

- [ ] **Step 1: Write failing ARPAbet mapping tests**

```python
@pytest.mark.parametrize(
    ("arpabet", "expected"),
    [
        (("M", "UW1", "V"), ("'m", "u:", "v")),
        (("IH0", "NG"), ("I", "N")),
        (("S", "IH1", "T"), ("'s", "I", "t")),
        (("TH", "R", "UW2"), (",T", "r", "u:")),
    ],
)
def test_arpabet_syllable_maps_to_espeak_with_syllable_stress(arpabet, expected) -> None:
    assert arpabet_syllable_to_espeak(arpabet) == expected
```

The stress marker prefixes the first output token of the independently rendered syllable. Implement the complete CMU consonant and vowel inventory, not only these examples.

- [ ] **Step 2: Write failing synthesis, fallback, and cache tests**

Use a fake command runner returning an in-memory mono WAV. Assert the primary command contains deterministic mode, no terminal pause, explicit voice/speed/pitch, `--stdout`, and `[[...]]`. Add tests for:

```python
def test_espeak_synthesizer_uses_explicit_phonemes() -> None:
    runner = FakeEspeakRunner(wav_bytes=wav_bytes(frames=240))
    synthesizer = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner)

    result = synthesizer.synthesize(cmu_request(("M", "UW1", "V")))

    assert "[['mu:v]]" in " ".join(runner.commands[0])
    assert result.pronunciation_source == "cmudict_arpabet"
    assert result.audio.frame_count == 240


def test_missing_cmu_phonemes_use_espeak_g2p_and_warn() -> None:
    runner = FakeEspeakRunner(
        phoneme_stdout="s_t_r_'i:_m_m_j_'u:_z",
        wav_bytes=wav_bytes(frames=240),
    )

    result = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner).synthesize(oov_request("StreamMUSE"))

    assert result.pronunciation_source == "espeak_g2p"
    assert result.warnings[0].code == AudioWarningCode.PRONUNCIATION_FALLBACK


def test_identical_phoneme_requests_use_cached_pcm() -> None:
    runner = FakeEspeakRunner(wav_bytes=wav_bytes(frames=240))
    synthesizer = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner)
    request = cmu_request(("M", "UW1", "V"))

    synthesizer.synthesize(request)
    synthesizer.synthesize(request)

    assert runner.wav_command_count == 1
```

- [ ] **Step 3: Run the tests and verify the adapter is missing**

```bash
uv run pytest tests/unit/infrastructure/rap/test_speech.py -v
```

Expected: collection fails because `audio_service` and `speech` do not exist.

- [ ] **Step 4: Define replaceable protocols**

In `audio_service.py`, define:

```python
class SpeechSynthesizer(Protocol):
    def synthesize(self, request: SyllableRenderRequest) -> RenderedSyllable: ...


class DrumRenderer(Protocol):
    def render(self, template: FlowTemplate, tempo: Tempo, audio_format: AudioFormat, bar: int) -> PcmAudio: ...


class RapBarRenderer(Protocol):
    def render(self, plan: PlannedRapBar) -> PreparedRapBar: ...


class RapAudioSink(Protocol):
    def start(self) -> None: ...
    def enqueue(self, bar: PreparedRapBar) -> None: ...
    def request_stop_after_bar(self) -> None: ...
    def reset(self) -> None: ...
    def snapshot(self) -> AudioPlaybackSnapshot: ...
    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]: ...
    def close(self) -> None: ...
```

Use `TYPE_CHECKING` for `PlannedRapBar` to avoid an application import cycle.

- [ ] **Step 5: Implement eSpeak mapping, G2P fallback, rendering, and caching**

Use these ARPAbet bases:

```python
_ARPABET_TO_ESPEAK = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:",
    "AW": "aU", "AY": "aI", "EH": "E", "ER": "3:",
    "EY": "eI", "IH": "I", "IY": "i:", "OW": "oU",
    "OY": "OI", "UH": "U", "UW": "u:",
    "B": "b", "CH": "tS", "D": "d", "DH": "D", "F": "f",
    "G": "g", "HH": "h", "JH": "dZ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "N", "P": "p", "R": "r",
    "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "Z",
}
```

Render primary phonemes with:

```text
espeak-ng -D -z -v en-us -s 175 -p 50 --stdout "[[<phoneme-string>]]"
```

For OOV words, request tokenized eSpeak phonemes with `-q -x --sep=_`, split tokens around eSpeak vowel nuclei to the requested syllable count, and render the selected syllable token group. If the number of vowel nuclei does not match, split the normalized spelling deterministically around vowel groups and render that fragment as text. If every command fails, return correctly sized empty PCM plus `SYNTHESIS_FAILED` and `PRONUNCIATION_FALLBACK` warnings.

Cache decoded mono PCM by `(voice, speed_wpm, pitch, render_mode, render_text)` and keep request-specific warning metadata outside the cache.

- [ ] **Step 6: Run unit tests and a local eSpeak smoke command**

```bash
uv run pytest tests/unit/infrastructure/rap/test_speech.py -q
espeak-ng -D -z -v en-us -s 175 -p 50 -w /tmp/streammuse-moving.wav "[[m'u:v]][[IN]]"
test -s /tmp/streammuse-moving.wav
```

Expected: unit tests pass and the WAV file is nonempty. The smoke command is a Mac acceptance check, not a CI requirement.

- [ ] **Step 7: Run the regression gate and commit**

```bash
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
git add src/streammuse/application/rap/audio_service.py src/streammuse/application/rap/__init__.py src/streammuse/infrastructure/rap/speech.py src/streammuse/infrastructure/rap/__init__.py tests/unit/infrastructure/rap/test_speech.py
git commit -m "feat: render rap syllables with espeak phonemes"
```

---

### Task 4: Render Stable Drums And Complete Immutable Audio Bars

**Files:**
- Create: `src/streammuse/infrastructure/rap/drums.py`
- Create: `src/streammuse/application/rap/bar_renderer.py`
- Create: `tests/unit/infrastructure/rap/test_drums.py`
- Create: `tests/unit/application/rap/test_bar_renderer.py`
- Modify: `src/streammuse/application/rap/__init__.py`
- Modify: `src/streammuse/infrastructure/rap/__init__.py`

**Interfaces:**
- Consumes: `SpeechSynthesizer`, `DrumRenderer`, Task 2 fitting functions, existing `PlannedRapBar`, and existing MCFlow `FlowTemplate` slots.
- Produces: `ProceduralBoomBapRenderer.render(...) -> PcmAudio` and `DeterministicRapBarRenderer.render(plan) -> PreparedRapBar`.

- [ ] **Step 1: Write failing deterministic drum-placement tests**

```python
def test_boom_bap_has_stable_meter_hits_and_sixteenth_hats() -> None:
    renderer = ProceduralBoomBapRenderer(seed=20260814)
    audio = renderer.render(template(), Tempo(60.0, 4, 4), AudioFormat(48_000, 2), bar=0)
    samples = stereo_array(audio)

    assert has_energy(samples, frame=0, window=2_000)                 # kick + hat
    assert has_energy(samples, frame=48_000, window=2_000)            # snare + hat
    assert has_energy(samples, frame=96_000, window=2_000)            # kick + hat
    assert has_energy(samples, frame=144_000, window=2_000)           # snare + hat
    assert all(has_energy(samples, frame=tick * 12_000, window=1_500) for tick in range(16))


def test_boom_bap_is_reproducible_for_same_seed_and_bar() -> None:
    first = ProceduralBoomBapRenderer(seed=7).render(template(), tempo(), audio_format(), bar=3)
    second = ProceduralBoomBapRenderer(seed=7).render(template(), tempo(), audio_format(), bar=3)

    assert first.data == second.data
```

- [ ] **Step 2: Write failing complete-bar tests with a fake synthesizer**

```python
def test_bar_renderer_places_every_syllable_at_exact_target_sample() -> None:
    plan = planned_bar_with_slots(bar=0, ticks=(0, 2, 7, 12, 15))
    renderer = DeterministicRapBarRenderer(
        tempo=Tempo(60.0, 4, 4),
        audio_format=AudioFormat(48_000, 2),
        synthesizer=ImpulseSpeechSynthesizer(frames=1_000),
        drums=SilentDrumRenderer(),
    )

    prepared = renderer.render(plan)

    assert prepared.audio.frame_count == 192_000
    assert [item.target_sample for item in prepared.diagnostics] == [0, 24_000, 84_000, 144_000, 180_000]
    assert all(item.software_error_samples == 0 for item in prepared.diagnostics)


def test_bar_renderer_preserves_pronunciation_and_timing_warnings() -> None:
    prepared = warning_renderer().render(planned_warning_bar())

    assert {warning.code for warning in prepared.warnings} == {
        AudioWarningCode.PRONUNCIATION_FALLBACK,
        AudioWarningCode.TIMING_PRESSURE,
    }
```

- [ ] **Step 3: Run tests and verify missing renderers fail**

```bash
uv run pytest tests/unit/infrastructure/rap/test_drums.py tests/unit/application/rap/test_bar_renderer.py -v
```

Expected: collection fails on `ProceduralBoomBapRenderer` and `DeterministicRapBarRenderer`.

- [ ] **Step 4: Implement deterministic procedural drums**

Generate reusable mono hits once per renderer instance:

- Kick: 120 ms exponentially decaying sine sweep from 85 Hz to 45 Hz.
- Snare: 110 ms deterministic white noise plus a quiet 180 Hz body tone.
- Hat: 35 ms deterministic differentiated noise with exponential decay.

Place hats on all 16 ticks, kicks on ticks 0 and 8, and snares on ticks 4 and 12. Apply a small hat gain increase on beat boundaries and occupied flow slots with high target stress. Seed bar-local noise with `seed + bar` and normalize the drum bar below 0.65 peak.

- [ ] **Step 5: Implement the complete bar renderer**

For each `ScheduledSyllable` in slot order:

```python
request = SyllableRenderRequest(
    bar=plan.bar,
    slot_index=item.slot.slot_index,
    word=item.syllable.word,
    index_in_word=item.syllable.index_in_word,
    syllable_count=item.syllable.syllable_count,
    phonemes=item.syllable.phonemes,
    stress=item.syllable.stress,
    analysis_source=item.syllable.analysis_source,
    voice=self.voice,
    speed_wpm=self.speed_wpm,
    pitch=self.pitch,
)
```

Render, trim, fit to the next onset or bar end, mix at `tick_frame_in_bar`, append diagnostics, then mix the drum bar. Use vocal gain 0.80, drum gain 0.55, and final peak limit 0.95. Emit stereo float32 bytes with exactly `bar_frame_count(plan.bar, ...)` frames.

- [ ] **Step 6: Run focused tests and the regression gate**

```bash
uv run pytest tests/unit/infrastructure/rap/test_drums.py tests/unit/application/rap/test_bar_renderer.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: exact placement, deterministic drum, and all existing tests pass.

- [ ] **Step 7: Commit bar rendering**

```bash
git add src/streammuse/application/rap/bar_renderer.py src/streammuse/application/rap/__init__.py src/streammuse/infrastructure/rap/drums.py src/streammuse/infrastructure/rap/__init__.py tests/unit/application/rap/test_bar_renderer.py tests/unit/infrastructure/rap/test_drums.py
git commit -m "feat: mix rap syllables with deterministic drums"
```

---

### Task 5: Add Live, WAV, Composite, And Null Audio Sinks

**Files:**
- Create: `src/streammuse/infrastructure/rap/audio_output.py`
- Create: `tests/unit/infrastructure/rap/test_audio_output.py`
- Modify: `src/streammuse/infrastructure/rap/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `RapAudioSink`, `PreparedRapBar`, `PlaybackState`, `AudioPlaybackNotice`, and `AudioPlaybackSnapshot`.
- Produces: `SoundDeviceAudioSink`, `Float32WavAudioSink`, `CompositeAudioSink`, and `NullAudioSink`.

- [ ] **Step 1: Write failing callback and stop tests using an injected fake stream**

```python
def test_sounddevice_callback_copies_bar_bytes_across_arbitrary_blocks() -> None:
    stream_factory = FakeOutputStreamFactory(block_frames=7)
    sink = SoundDeviceAudioSink(audio_format=stereo_format(), stream_factory=stream_factory)
    sink.enqueue(prepared_bar(bar=0, frames=10, value=0.25))
    sink.enqueue(prepared_bar(bar=1, frames=10, value=0.50))

    sink.start()
    rendered = stream_factory.render_frames(20)

    assert rendered[:10] == pytest.approx([0.25] * 20)
    assert rendered[10:] == pytest.approx([0.50] * 20)
    assert [notice.kind for notice in sink.drain_notices()] == [
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
        AudioPlaybackNoticeKind.BAR_STARTED,
        AudioPlaybackNoticeKind.BAR_COMPLETED,
    ]


def test_stop_request_finishes_current_bar_without_dequeuing_next() -> None:
    sink = fake_live_sink(bar_frames=16)
    sink.enqueue(prepared_bar(bar=0, frames=16))
    sink.enqueue(prepared_bar(bar=1, frames=16))
    sink.start()
    sink.request_stop_after_bar()

    sink.render_frames(32)

    assert sink.snapshot().state == PlaybackState.STOPPED
    assert sink.snapshot().absolute_frame == 16
```

- [ ] **Step 2: Write failing WAV and composite tests**

```python
def test_float32_wav_sink_writes_ieee_float_header_and_exact_pcm(tmp_path: Path) -> None:
    path = tmp_path / "session.wav"
    sink = Float32WavAudioSink(path, stereo_format())
    bar = prepared_bar(bar=0, frames=8, value=0.25)

    sink.enqueue(bar)
    sink.mark_completed(bar)
    sink.close()

    header, payload = read_float_wav(path)
    assert header.audio_format_code == 3
    assert header.sample_rate == 48_000
    assert payload == bar.audio.data


def test_composite_truncates_unplayed_queued_bars_on_stop(tmp_path: Path) -> None:
    composite = composite_with_fake_live(tmp_path)
    composite.enqueue(prepared_bar(bar=0, frames=8, value=0.25))
    composite.enqueue(prepared_bar(bar=1, frames=8, value=0.50))
    composite.start()
    composite.request_stop_after_bar()
    composite.render_frames(16)
    composite.close()

    assert wav_frame_count(tmp_path / "session.wav") == 8
```

- [ ] **Step 3: Run tests and verify sink imports fail**

```bash
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -v
```

Expected: collection fails because the sink module does not exist.

- [ ] **Step 4: Add and lock sounddevice**

Run:

```bash
uv add "sounddevice>=0.5.2"
```

Expected: `pyproject.toml` and `uv.lock` contain sounddevice. Do not import sounddevice at module import time; import it inside the default stream factory so text-only mode remains device-independent.

- [ ] **Step 5: Implement the callback sink with injected stream construction**

Use a deque of immutable prepared bars, a short lock only when acquiring the next bar, a frame offset into the active bar, and a `SimpleQueue` for notices. Fill callback remainders with zero. Catch status flags as `UNDERRUN` notices. On a stop request, complete the current bar, leave future bars queued, stop the stream, and publish `STOPPED`.

- [ ] **Step 6: Implement streaming float32 WAV and composite behavior**

Write a 44-byte little-endian RIFF/WAVE header with format code 3, stereo channels, 48 kHz, 32 bits per sample, and placeholder sizes. Append committed bar bytes in order. Track completed live frames and patch/truncate the file to completed data on Stop/close so queued but unplayed bars are excluded.

`CompositeAudioSink` forwards lifecycle and enqueue operations to the live sink and recorder, drains live notices, and calls `mark_completed` for each `BAR_COMPLETED`. `NullAudioSink` records bars and permits tests without sleeping or opening devices.

- [ ] **Step 7: Run sink tests and the regression gate**

```bash
uv run pytest tests/unit/infrastructure/rap/test_audio_output.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: all tests pass without opening a physical audio device.

- [ ] **Step 8: Commit audio output adapters**

```bash
git add pyproject.toml uv.lock src/streammuse/infrastructure/rap/audio_output.py src/streammuse/infrastructure/rap/__init__.py tests/unit/infrastructure/rap/test_audio_output.py
git commit -m "feat: add realtime rap audio outputs"
```

---

### Task 6: Add The Python Playback State Machine And Sample Observer

**Files:**
- Create: `src/streammuse/application/rap/playback.py`
- Create: `tests/unit/application/rap/test_playback.py`
- Modify: `src/streammuse/application/rap/__init__.py`

**Interfaces:**
- Consumes: `RapAudioSink`, `RapEventPublisher`, `Tempo`, prepared bars, sink notices, and sink snapshots.
- Produces: `RapPlaybackService.prime(bar)`, `start()`, `enqueue(bar)`, `request_stop()`, `reset()`, `wait()`, and `close()`; the service invokes `on_tick(tick)` from observed sample progress.

- [ ] **Step 1: Write failing state-transition tests**

```python
def test_playback_moves_through_priming_running_and_bar_quantized_stop() -> None:
    sink = FakeRapAudioSink()
    service = RapPlaybackService(tempo(), sink, publisher(), on_tick=lambda _: None)

    service.prime(prepared_bar(bar=0))
    assert service.state == PlaybackState.PRIMING
    service.start()
    assert service.state == PlaybackState.RUNNING
    service.request_stop()
    assert service.state == PlaybackState.STOP_REQUESTED
    sink.complete_bar(0)
    service.poll()
    assert service.state == PlaybackState.STOPPED


def test_reset_requires_stopped_and_clears_audio_state() -> None:
    service, sink = stopped_service_after_one_bar()

    service.reset()

    assert service.state == PlaybackState.STOPPED
    assert sink.reset_calls == 1
    assert service.current_tick is None
```

- [ ] **Step 2: Write failing sample-observer tests**

```python
def test_observer_emits_each_tick_once_when_polling_skips_frames() -> None:
    ticks: list[int] = []
    service, sink = running_service(on_tick=ticks.append, tempo=Tempo(60.0, 4, 4))

    sink.set_absolute_frame(36_100)
    service.poll()

    assert ticks == [0, 1, 2, 3]


def test_syllable_event_reports_exact_mixed_sample_not_poll_delay() -> None:
    publisher = RecordingPublisher()
    service, sink = running_service(publisher=publisher, prepared=bar_with_syllable_at_tick(2))
    sink.set_absolute_frame(30_000)

    service.poll()

    event = publisher.only(RapEventType.SYLLABLE_EMITTED)
    assert event.payload["scheduled_sample"] == 24_000
    assert event.payload["software_error_samples"] == 0
```

- [ ] **Step 3: Run tests and verify the service is missing**

```bash
uv run pytest tests/unit/application/rap/test_playback.py -v
```

Expected: collection fails on `RapPlaybackService`.

- [ ] **Step 4: Implement playback lifecycle and event publication**

The service owns state but delegates bytes to the sink. `prime` requires bar 0 and enqueues it. `enqueue` delegates to `prime` when it receives the first bar in `STOPPED`, allowing the rolling controller to use one `on_audio_committed=playback.enqueue` callback for every bar. `start` requires at least one queued bar, starts a daemon observer thread, and emits session/audio state. `request_stop` is idempotent and emits `STOP_REQUESTED`. `reset` is accepted only in `STOPPED`; it resets counters and sink state, then emits `SESSION_RESET`. `close` is idempotent and joins the observer before closing the sink.

Convert sink notices to canonical events:

```python
AudioPlaybackNoticeKind.BAR_STARTED   -> RapEventType.BAR_PLAYBACK_STARTED
AudioPlaybackNoticeKind.BAR_COMPLETED -> RapEventType.BAR_PLAYBACK_COMPLETED
AudioPlaybackNoticeKind.UNDERRUN      -> RapEventType.AUDIO_UNDERRUN
AudioPlaybackNoticeKind.DEVICE_FAILED -> RapEventType.AUDIO_DEVICE_FAILED
```

- [ ] **Step 5: Implement non-authoritative tick and syllable observation**

Poll the sink snapshot at 5 ms intervals. For every absolute tick sample crossed since the previous poll, call `on_tick(tick)` once; the existing rolling controller remains the sole publisher of `TICK`. Emit `SYLLABLE_EMITTED` when the current prepared bar crosses a diagnostic target sample. Include `scheduled_sample`, `software_error_samples=0`, `observation_delay_ms`, word, label, stress, beat, and subdivision. Do not call the publisher or controller from the audio callback.

- [ ] **Step 6: Run focused tests and the regression gate**

```bash
uv run pytest tests/unit/application/rap/test_playback.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: state, stop, reset, skipped-frame tick emission, and all old tests pass.

- [ ] **Step 7: Commit playback control**

```bash
git add src/streammuse/application/rap/playback.py src/streammuse/application/rap/__init__.py tests/unit/application/rap/test_playback.py
git commit -m "feat: drive rap playback from the audio sample clock"
```

---

### Task 7: Add Fallback-First Rendering And Atomic Audio Commitment

**Files:**
- Create: `src/streammuse/application/rap/audio_coordination.py`
- Create: `tests/unit/application/rap/test_audio_coordination.py`
- Modify: `src/streammuse/application/rap/realtime.py`
- Modify: `tests/unit/application/rap/test_realtime.py`
- Modify: `src/streammuse/application/rap/__init__.py`

**Interfaces:**
- Consumes: `RapBarRenderer`, `PreparedRapBar`, `PlannedRapBar`, `RapEventPublisher`, and existing rolling planner timing.
- Produces: `BarAudioCoordinator.reserve_fallback(plan)`, `submit_primary(plan)`, `poll_primary(bar)`, `commit(bar)`, `reset()`, and `close()`; `RollingRapController` gains optional `audio_coordinator` and `on_audio_committed` dependencies.

- [ ] **Step 1: Write failing coordinator tests**

```python
def test_fallback_is_rendered_immediately_and_primary_is_provisional() -> None:
    renderer = ControlledBarRenderer()
    coordinator = BarAudioCoordinator(renderer, publisher=RecordingPublisher())
    fallback = planned_bar(bar=1, source="prevalidated_fallback")
    primary = planned_bar(bar=1, source="local_chat")

    coordinator.reserve_fallback(fallback)
    coordinator.submit_primary(primary)

    assert coordinator.commit(bar=1).source == "prevalidated_fallback"
    renderer.complete(primary)
    assert coordinator.poll_primary(bar=1).source == "local_chat"


def test_committed_bar_cannot_be_replaced_by_late_primary_audio() -> None:
    coordinator, renderer = coordinator_with_ready_fallback(bar=2)
    committed = coordinator.commit(2)
    renderer.complete(planned_bar(bar=2, source="local_chat"))

    assert coordinator.commit(2) is committed
    assert coordinator.poll_primary(2) is None
```

- [ ] **Step 2: Write failing rolling-controller audio-path tests**

Pin both paths:

```python
def test_text_controller_still_replaces_candidate_without_audio_coordinator() -> None:
    controller = existing_text_controller()
    controller.start()
    complete_primary_generation(controller, bar=1)
    controller.on_tick(1)

    assert controller.bar_for(1).source == "local_chat"


def test_audio_controller_replaces_text_only_after_primary_audio_is_ready() -> None:
    controller, audio = audio_controller()
    controller.start()
    complete_primary_generation(controller, bar=1)
    controller.on_tick(1)
    assert controller.bar_for(1).source == "prevalidated_fallback"

    audio.complete_primary(bar=1)
    controller.on_tick(2)
    assert controller.bar_for(1).source == "local_chat"


def test_audio_controller_commits_next_bar_one_tick_before_boundary() -> None:
    controller, committed = audio_controller_with_recorder()
    controller.start()
    controller.on_tick(15)

    assert committed.only().bar == 1
```

- [ ] **Step 3: Run tests and verify the coordinator is missing**

```bash
uv run pytest tests/unit/application/rap/test_audio_coordination.py tests/unit/application/rap/test_realtime.py -v
```

Expected: new tests fail while all previously existing `test_realtime.py` cases remain green or fail only on the not-yet-added constructor arguments.

- [ ] **Step 4: Implement asynchronous fallback and primary rendering**

Use a dedicated `ThreadPoolExecutor` with two workers so fallback preparation is not queued behind a slow primary render. Maintain maps for fallback futures/results, primary futures/results, and committed bars under one `RLock`.

`reserve_fallback` is idempotent. `submit_primary` replaces only a still-uncommitted primary proposal. `poll_primary` returns a result only once. `commit` chooses ready primary first, otherwise waits only for the already-reserved fallback; it then records an immutable committed result and rejects all later mutation.

Publish `AUDIO_RENDER_STARTED`, `AUDIO_RENDER_COMPLETED`, `BAR_AUDIO_READY`, pronunciation warnings, and timing warnings outside the render worker's critical section.

- [ ] **Step 5: Integrate the coordinator as an optional planner dependency**

Extend the constructor without changing defaults:

```python
def __init__(
    ...,
    audio_coordinator: BarAudioCoordinator | None = None,
    on_audio_committed: Callable[[PreparedRapBar], None] | None = None,
) -> None:
```

When no audio coordinator exists, retain the current `_drain_primary_result`, immediate `BAR_REPLACED`, `_freeze`, and `_emit` behavior.

When audio is enabled:

1. Reserve fallback audio whenever `_reserve_through` creates a bar.
2. Submit the selected primary as a copied `PlannedRapBar` without mutating the active fallback.
3. Poll completed primary audio on later ticks; only then mutate the active bar and emit `BAR_REPLACED`.
4. Commit bar 0 during `start()` before playback begins.
5. At tick `target_bar * ticks_per_bar - 1`, commit the target bar and call `on_audio_committed`.
6. Freeze at the normal bar-start tick and assert that the audio bar is already committed.
7. Emit `BAR_AUDIO_COMMITTED` with source, warnings, render latency, frame count, and one-tick deadline slack.

Do not emit `SYLLABLE_EMITTED` from `RollingRapController` in audio mode; Task 6's observer owns it.

- [ ] **Step 6: Add reset-safe planner state without changing close semantics**

Add `reset()` for audio-controlled sessions. It may run only after playback is stopped. Cancel uncommitted futures, clear bars/history/anchors/timing origins, reset `_next_primary_bar`, and ask the audio coordinator to reset. Do not close the LLM client or event publisher. Existing `close()` still permanently closes resources.

- [ ] **Step 7: Run focused tests and the full rap regression gate**

```bash
uv run pytest tests/unit/application/rap/test_audio_coordination.py tests/unit/application/rap/test_realtime.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: all old text-controller tests and all new audio-controller tests pass.

- [ ] **Step 8: Commit atomic audio commitment**

```bash
git add src/streammuse/application/rap/audio_coordination.py src/streammuse/application/rap/realtime.py src/streammuse/application/rap/__init__.py tests/unit/application/rap/test_audio_coordination.py tests/unit/application/rap/test_realtime.py
git commit -m "feat: commit rap lyrics only after audio is ready"
```

---

### Task 8: Assemble Audio Mode In The Runtime And CLI Without Changing Defaults

**Files:**
- Modify: `src/streammuse/application/rap/runtime.py`
- Modify: `src/streammuse/presentation/rap_demo/cli.py`
- Modify: `src/streammuse/infrastructure/rap/recorder.py`
- Modify: `tests/unit/application/rap/test_runtime.py`
- Modify: `tests/unit/presentation/rap_demo/test_cli.py`
- Create: `tests/integration/test_realtime_rap_audio.py`

**Interfaces:**
- Consumes: Tasks 3-7 components and existing `RapDemoDependencies`.
- Produces: additive CLI flags, `RapAudioDemoDependencies`, and a complete fake-clock audio integration workflow.

- [ ] **Step 1: Write parser compatibility and new-option tests**

```python
def test_existing_parser_defaults_remain_text_only() -> None:
    args = build_parser().parse_args([])

    assert args.audio_output == "none"
    assert args.tempo is None
    assert args.candidate_count == 8
    assert args.lookahead_bars == 2


def test_audio_parser_accepts_explicit_research_configuration() -> None:
    args = build_parser().parse_args([
        "--audio-output", "composite",
        "--tempo", "60",
        "--candidate-count", "12",
        "--lookahead-bars", "3",
        "--audio-device", "MacBook Pro Speakers",
        "--voice", "en-us",
    ])

    assert args.audio_output == "composite"
    assert args.tempo == 60.0
    assert args.sample_rate == 48_000
```

- [ ] **Step 2: Write failing assembly and lifecycle tests with injected factories**

```python
def test_text_build_uses_existing_tick_loop_and_no_audio_dependencies(base_args) -> None:
    demo = build_demo(base_args, audio_factories=FailIfCalledAudioFactories())

    assert type(demo) is RapDemoDependencies


def test_audio_build_uses_mac_synthesis_renderer_playback_and_recording(audio_args, fake_audio_factories) -> None:
    demo = build_demo(audio_args, audio_factories=fake_audio_factories)

    assert isinstance(demo, RapAudioDemoDependencies)
    assert demo.tempo.bpm == 60.0
    assert demo.autostart is False
```

Add lifecycle tests proving `start -> stop-after-bar -> start -> stop -> reset` does not close the publisher, recorder, generator, or audio device until `close()`.

- [ ] **Step 3: Write the failing end-to-end fake-clock integration test**

```python
def test_rolling_audio_runs_without_gap_when_generator_is_late(tmp_path: Path) -> None:
    harness = RealtimeRapAudioHarness(
        tempo=Tempo(60.0, 4, 4),
        generator=DelayedGenerator(delay_bars=4),
        max_bars=100,
        output_path=tmp_path / "session.wav",
    )

    result = harness.run()

    assert result.completed_bars == 100
    assert result.total_frames == bar_start_frame(100, harness.tempo, harness.audio_format)
    assert result.underruns == 0
    assert result.fallback_bars > 0
    assert result.software_timing_errors == {0}
```

- [ ] **Step 4: Run tests and verify audio assembly is absent**

```bash
uv run pytest tests/unit/presentation/rap_demo/test_cli.py tests/unit/application/rap/test_runtime.py tests/integration/test_realtime_rap_audio.py -v
```

Expected: new audio cases fail; existing CLI/runtime cases still pass.

- [ ] **Step 5: Add strictly additive CLI flags and validation**

Add:

```text
--audio-output {none,live,wav,composite}  default none
--tempo FLOAT                             default None, overrides scenario playback tempo
--audio-device TEXT                       default None
--sample-rate INT                         default 48000
--voice TEXT                              default en-us
--voice-speed INT                         default 175
--voice-pitch INT                         default 50
--max-compression FLOAT                   default 2.0
--audio-file PATH                         default <session-dir>/mixed.wav
```

Validate positive tempo/sample rate, speed 80-450, pitch 0-99, and compression 1.0-4.0. Require eSpeak and sounddevice only for `live` or `composite`; WAV mode still requires eSpeak but no physical device.

- [ ] **Step 6: Implement an audio-specific controllable dependency owner**

Add `RapAudioDemoDependencies` rather than changing the blocking/close behavior of `RapDemoDependencies`. It owns controller, coordinator, playback, publisher, dispatcher, recorder, and session directory. Its public contract is:

```python
class RapAudioDemoDependencies:
    autostart = False

    def start(self) -> None: ...              # blocks until stop/max-bars
    def request_stop(self) -> None: ...       # bar-quantized
    def reset(self) -> None: ...              # only while stopped
    def close(self) -> None: ...              # permanent and idempotent
    @property
    def control_state(self) -> PlaybackState: ...
```

Do not call `close()` at the end of a user-requested Stop. The FastAPI lifespan or terminal `finally` owns permanent close.

- [ ] **Step 7: Assemble the audio dependency graph only when requested**

For audio mode, build eSpeak, procedural drums, deterministic bar renderer, audio coordinator, selected sink, playback service, and rolling controller. Pass `playback.enqueue` as `on_audio_committed`, and pass the controller's planning-only tick handler to playback. Keep all current text-only assembly code and `RapTickLoop` intact.

Add audio configuration and artifact paths to `session.json`. Extend recorder summaries with render latency, commit slack, warning counts, underruns, and completed audio bars.

- [ ] **Step 8: Implement the fake-clock integration harness and run tests**

The harness uses fake speech and a test-only `ManualAudioSink` implementing `RapAudioSink` plus `advance(frames: int)`. It advances blocks without sleeping and exercises the real controller, coordinator, renderer, playback observer, publisher, projector, and recorder. `NullAudioSink` remains an inert collection sink for narrower unit tests.

Run:

```bash
uv run pytest tests/unit/application/rap/test_runtime.py tests/unit/presentation/rap_demo/test_cli.py tests/integration/test_realtime_rap_audio.py -q
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
```

Expected: 100-bar exact frame count and all old CLI tests pass.

- [ ] **Step 9: Commit additive runtime assembly**

```bash
git add src/streammuse/application/rap/runtime.py src/streammuse/presentation/rap_demo/cli.py src/streammuse/infrastructure/rap/recorder.py tests/unit/application/rap/test_runtime.py tests/unit/presentation/rap_demo/test_cli.py tests/integration/test_realtime_rap_audio.py
git commit -m "feat: assemble optional realtime rap audio mode"
```

---

### Task 9: Expose Start, Stop, Reset, Audio Telemetry, And Warnings

**Files:**
- Modify: `src/streammuse/application/rap/monitoring.py`
- Modify: `src/streammuse/presentation/rap_demo/server.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_state.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_stream.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_dashboard.py`
- Modify: `src/streammuse/presentation/rap_demo/static/index.html`
- Modify: `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`
- Modify: `src/streammuse/presentation/rap_demo/static/js/rap-demo.js`
- Modify: `tests/unit/application/rap/test_monitoring.py`
- Modify: `tests/unit/presentation/rap_demo/test_server.py`
- Modify: `tests/unit/presentation/rap_demo/test_terminal_state.py`
- Modify: `tests/unit/presentation/rap_demo/test_terminal_stream.py`
- Modify: `tests/unit/presentation/rap_demo/test_terminal_dashboard.py`

**Interfaces:**
- Consumes: canonical audio events and `RapAudioDemoDependencies` controls.
- Produces: projector fields `audio`, `audio_warnings`, and audio latency distributions; HTTP endpoints `/api/control/start`, `/stop`, `/reset`; exactly three visible control buttons.

- [ ] **Step 1: Write failing projector tests**

```python
def test_projector_tracks_audio_queue_warning_and_playback_state() -> None:
    projector = RapStateProjector()
    projector.apply(event(RapEventType.BAR_AUDIO_COMMITTED, bar=2, payload={
        "queue_depth": 3,
        "buffered_seconds": 12.0,
        "render_latency_ms": 48.0,
    }))
    projector.apply(event(RapEventType.PRONUNCIATION_FALLBACK, bar=2, payload={
        "word": "StreamMUSE",
        "action": "espeak_g2p",
    }))
    projector.apply(event(RapEventType.BAR_PLAYBACK_STARTED, bar=2, payload={"absolute_frame": 384_000}))

    state = projector.snapshot()
    assert state["audio"]["state"] == "running"
    assert state["audio"]["queue_depth"] == 3
    assert state["audio_warnings"][-1]["word"] == "StreamMUSE"
```

Add cumulative tests for synthesis latency, bar render latency, commit slack, pronunciation warnings, timing pressure, forced fits, and underruns.

- [ ] **Step 2: Write failing control API tests**

```python
def test_audio_runtime_does_not_autostart_and_start_endpoint_runs_it() -> None:
    runtime = ControllableFakeRuntime(autostart=False)
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        assert runtime.start_calls == 0
        response = client.post("/api/control/start")
        assert response.status_code == 202
        wait_until(lambda: runtime.start_calls == 1)


def test_stop_and_reset_endpoints_delegate_with_state_validation() -> None:
    runtime = ControllableFakeRuntime(autostart=False)
    app = create_app(runtime=runtime, projector=FakeProjector(), websocket_queue=Queue())

    with TestClient(app) as client:
        assert client.post("/api/control/stop").status_code == 409
        client.post("/api/control/start")
        assert client.post("/api/control/stop").status_code == 202
        runtime.finish_bar()
        assert client.post("/api/control/reset").status_code == 200
```

Retain a test proving `RapDemoDependencies.autostart` defaults to true and current text-only website startup remains automatic.

- [ ] **Step 3: Write failing terminal rendering tests**

Pin dense lines such as:

```text
[BAR 03][AUDIO] ready source=local_chat frames=192000 render_ms=48 warnings=1
[BAR 03][WARN] pronunciation word='StreamMUSE' source=espeak_g2p action=best_effort_rendered
[BAR 03][WARN] timing slot=7 word='trans' available_ms=163 rendered_ms=241 compression=1.45 overlap_ms=12
[BAR 03][PLAY] started queue=3 buffered_s=12 absolute_frame=384000
```

Assert summary detail shows lifecycle and warnings, while full detail includes renderer phonemes and exact sample fields.

- [ ] **Step 4: Run monitoring tests and verify new fields/endpoints are absent**

```bash
uv run pytest tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo/test_server.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py -v
```

Expected: only new audio expectations fail.

- [ ] **Step 5: Extend projector state without changing existing keys**

Add:

```python
"audio": {
    "state": "disabled",
    "current_bar": None,
    "queue_depth": 0,
    "buffered_seconds": 0.0,
    "underruns": 0,
    "device": None,
    "recording_path": None,
},
"audio_warnings": [],
```

Keep at most 128 warning records. Add latency aggregates named `synthesis_latency_ms`, `bar_render_latency_ms`, and `audio_commit_slack_ms`. Do not rename existing state, bars, candidates, latencies, or research metrics.

- [ ] **Step 6: Implement restart-safe server controls**

The FastAPI lifespan starts the WebSocket broadcaster for every runtime. It starts the runtime thread automatically only when `runtime.autostart` is true. Add POST endpoints:

```text
/api/control/start -> 202 {"state": "priming"}
/api/control/stop  -> 202 {"state": "stop_requested"}
/api/control/reset -> 200 {"state": "stopped"}
```

Return HTTP 409 for invalid transitions and 404 when a runtime does not expose the requested control. Permit a new runtime thread after a prior bar-quantized Stop; keep permanent `close()` only in lifespan shutdown.

- [ ] **Step 7: Extend terminal stream and split dashboard**

Add an `AUDIO` phase and a `WARN` phase using the existing dense prefix format. Show current playback state, queue bars/seconds, output device, recording path, render latency, commit slack, warning counts, and underruns. Keep existing planning, prompt, candidate, score, flow, fallback, and syllable sections unchanged.

- [ ] **Step 8: Add exactly three website controls and audio evidence fields**

In the existing app header, add one compact command group:

```html
<div class="runtime-controls" aria-label="Runtime controls">
  <button id="start-runtime" type="button">Start</button>
  <button id="stop-runtime" type="button">Stop</button>
  <button id="reset-runtime" type="button">Reset</button>
</div>
```

Do not add tempo, topic, candidate, lookahead, voice, drum, or volume controls. Add audio state, queue depth, buffered seconds, output device, recording, render latency, commit slack, and underrun telemetry to existing bands. Add a compact warning table with bar/slot, type, word, source, duration pressure, and action.

In JavaScript, POST commands through one `sendControl(action)` function, disable impossible actions based on projected state, and show request errors in the existing last-error area. Runtime controls must not own or schedule browser audio.

- [ ] **Step 9: Run tests and perform responsive visual verification**

```bash
uv run pytest tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo -q
uv run streammuse-rap-demo --generator phrase_bank --audio-output none --max-bars 3 --port 8012
```

Open `http://127.0.0.1:8012/` and capture desktop 1440x900 and mobile 390x844 screenshots. Verify no text overlaps, all existing research data remains visible, the three controls fit, and no additional controls appear. Stop the server after verification.

- [ ] **Step 10: Run the regression gate and commit monitoring/UI**

```bash
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap_demo -q
git add src/streammuse/application/rap/monitoring.py src/streammuse/presentation/rap_demo tests/unit/application/rap/test_monitoring.py tests/unit/presentation/rap_demo
git commit -m "feat: monitor and control realtime rap audio"
```

---

### Task 10: Document, Benchmark, And Verify The Split Mac/H200 Demo

**Files:**
- Modify: `docs/developer-guide/rap-demo-quickstart.md`
- Modify: `docs/developer-guide/rap-acceptance-report-2026-08-09.md`
- Modify: `README.md`
- Test: complete repository test suite and real Mac/H200 run artifacts under `logs/rap/`

**Interfaces:**
- Consumes: completed audio runtime, UI, H200 local-chat generator, session recorder, and existing summary script.
- Produces: reproducible setup commands, candidate-count evidence, Mac WAV artifacts, and final regression evidence.

- [ ] **Step 1: Update the quick start with separate H200 and Mac commands**

Document H200 model startup:

```bash
CUDA_VISIBLE_DEVICES=<UNUSED_GPU_ID> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.25
```

Document the Mac model tunnel:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8001:127.0.0.1:8001 \
  Andrew.Yang@masdar
```

Document Mac prerequisites and run command:

```bash
brew install espeak-ng portaudio
uv sync
uv run streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --audio-output composite \
  --tempo 60 \
  --candidate-count 12 \
  --lookahead-bars 3 \
  --max-bars 0 \
  --terminal-layout split \
  --terminal-detail full \
  --port 8012
```

State explicitly that the website is Mac-local at `http://127.0.0.1:8012/`; only vLLM port 8001 is forwarded in the new split architecture. Preserve the old H200-hosted website forwarding section as a labeled legacy text-only workflow.

- [ ] **Step 2: Run the complete local automated test suite**

```bash
uv run pytest tests/ -q
```

Expected: all tests pass. Record the command, date, platform, and pass count in a new audio section of the acceptance report.

- [ ] **Step 3: Run a deterministic Mac phrase-bank audio acceptance session**

```bash
uv run streammuse-rap-demo \
  --generator phrase_bank \
  --audio-output composite \
  --tempo 60 \
  --candidate-count 12 \
  --lookahead-bars 3 \
  --max-bars 12 \
  --terminal-layout stream \
  --terminal-detail full \
  --port 8012
```

Expected: 12 completed bars, zero cumulative software timing error, no audio underrun, audible drums and syllables, working Start/Stop/Reset, and a nonempty float32 mixed WAV. Record the session directory.

- [ ] **Step 4: Run the H200 candidate-count sweep through the Mac tunnel**

Run three eight-bar sessions at each tempo/candidate pair:

```bash
for tempo in 60 92; do
  for candidates in 8 12 16; do
    uv run streammuse-rap-demo \
      --generator local_chat \
      --model-url http://127.0.0.1:8001/v1 \
      --model qwen-rap \
      --audio-output composite \
      --tempo "$tempo" \
      --candidate-count "$candidates" \
      --lookahead-bars 3 \
      --max-bars 8 \
      --terminal-layout stream \
      --terminal-detail summary \
      --no-web
  done
done
```

Use `scripts/summarize_rap_session.py` on each session directory. Record generation p50/p95, decision slack, audio-commit slack, fallback bars, pronunciation warnings, timing pressure, underruns, and completed audio bars. Keep 12 as the documented prototype value unless its p95 misses the commit deadline at 60 BPM.

- [ ] **Step 5: Verify failure and bar-quantized control behavior**

Run audio mode with `--generator scripted_failure` for eight bars and confirm all eight bars play from fallback audio without an underrun. During a local-chat run, request Stop mid-bar and confirm the current bar completes while the next does not start; press Start and confirm the next complete bar begins; Stop again, Reset, and confirm the next Start returns to bar zero with cleared warnings and metrics.

- [ ] **Step 6: Verify the exported WAV against the canonical event log**

For the accepted local-chat session, calculate expected bar starts and syllable sample offsets from `events.jsonl`. Read `mixed.wav`, assert its frame count equals the sum of completed bar frame counts, and assert every `software_error_samples` value is zero. Document that physical speaker latency was not measured.

- [ ] **Step 7: Run final compatibility checks**

```bash
uv run streammuse-rap-demo --generator phrase_bank --audio-output none --max-bars 3 --no-web
uv run streammuse-rap --topic "space travel" --tempo 92 --pattern boom_bap --bars 2
uv run pytest tests/ -q
git diff --check
```

Expected: the original text-only demo and static rap planner still work, the complete test suite passes, and the diff has no whitespace errors.

- [ ] **Step 8: Commit documentation and acceptance evidence**

```bash
git add README.md docs/developer-guide/rap-demo-quickstart.md docs/developer-guide/rap-acceptance-report-2026-08-09.md
git commit -m "docs: add realtime rap audio demo guide"
```

- [ ] **Step 9: Review the branch before publication**

```bash
git status --short
git log --oneline --decorate -12
git diff origin/feature/real_rap...HEAD --stat
```

Expected: only intentional audio, monitoring, tests, dependency, and documentation changes are present; the worktree is clean; each task has a separate commit.
