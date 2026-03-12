# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StreamMUSE is a real-time AI music generation system that creates accompaniment for user-played melodies. The system follows **Clean Architecture** (4 layers: Presentation → Application → Domain → Infrastructure) and uses a transformer-based model (RoFormer/Stanley engine). The real-time application runs as a single CLI process that communicates with an inference server over HTTP.

## Development Commands

### Environment Setup
```bash
# Install dependencies and activate venv
uv sync

# Install the adapted transformers package (required for RoFormer positional encoding)
cd transformers
pip install -e .
cd ..
```

### Running Tests
```bash
uv run pytest tests/
uv run pytest tests/ -q --tb=no   # concise
```

### Real-time Application

#### Inference Server
```bash
# Start fake server for development/testing (no model required)
uv run python scripts/fake_inference_server.py

# Start real inference server
CHECKPOINT_PATH=path/to/model.ckpt uvicorn src.streammuse.infrastructure.inference.server:app --host 0.0.0.0 --port 8000
# Optional env vars: MODEL_MAX_SEQ_LEN_FRAMES=96, GENERATION_LENGTH_FRAMES=20, MODEL_SIZE=0.12B
```

#### CLI Client
```bash
# Keyboard input + console output (default server: http://localhost:8000)
uv run streammuse-cli --input-mode keyboard

# MIDI device input
uv run streammuse-cli --input-mode midi

# MIDI file simulation
uv run streammuse-cli --input-mode midi_file --midi-file path/to/song.mid

# Audio output
uv run streammuse-cli --input-mode keyboard --output-type audio

# Session logging (console + MIDI + JSON logs)
uv run streammuse-cli --input-mode keyboard --output-type composite --log-dir logs

# With music injection (pre-populate model history)
uv run streammuse-cli --input-mode keyboard --injection-file prompts/C_major/pop909_216_mel.mid --injection-length 50
```

#### Output Types
| `--output-type` | Description |
|---|---|
| `console` | Print events/stats to terminal (default) |
| `audio` | Real-time MIDI audio playback |
| `midi_file` | Record to MIDI file |
| `websocket` | Push events via WebSocket |
| `json_log` | Write `events.jsonl` + `inferences.json` to session dir |
| `session` | MIDI file + JSON logs combined |
| `composite` | Console + session logging (use with `--log-dir`) |

### Benchmarking
```bash
uv run python app/benchmark.py --output_file results/benchmark_test.csv --num_requests 100
```

## Architecture Overview

### Clean Architecture Layers

```
src/streammuse/
├── presentation/cli/       # Entry point: cli.py, config_parser.py
├── application/
│   ├── config/             # ApplicationConfig, TempoConfig, InputConfig, OutputConfig, InferenceConfig
│   ├── factories/          # InputSourceFactory, OutputSinkFactory, InferenceEngineFactory
│   └── services/           # RealTimeMusicService (3-thread orchestrator)
├── domain/
│   ├── interfaces/         # Protocols: InputSource, OutputSink, InferenceEngine, TimingInfo
│   ├── musical/            # MusicalEvent, EventType, Note, converters
│   ├── timing/             # Tempo, PlaybackScheduler, MusicalTime
│   └── logging/            # SessionManager, MetricsCalculator, LogEvent, InferenceEvent
└── infrastructure/
    ├── input/              # MidiDeviceInput, KeyboardInput, MidiFileInput, ListInput
    ├── output/             # AudioOutputSink, ConsoleOutputSink, MidiFileOutputSink,
    │                       # WebSocketOutputSink, CompositeOutputSink,
    │                       # JsonLoggerOutputSink, SessionLoggerOutputSink
    └── inference/          # HttpInferenceClient, StanleyInferenceEngine, LegacyInferenceEngineStanley
```

### Core Data Model

```python
@dataclass(frozen=True)
class MusicalEvent:
    tick: int           # absolute musical time (ticks)
    pitch: int          # MIDI pitch 0-127, or -1 for non-note events
    event_type: EventType   # NOTE_ON or NOTE_OFF
    velocity: int       # 0-127
    channel: int        # MIDI channel
    program: int        # MIDI program (instrument)
    source: str         # "user" or "model"
    is_placeholder: bool
```

### Key Protocols (domain/interfaces/)

```python
class InputSource(Protocol):
    def read_events(self) -> Iterator[MusicalEvent]: ...
    def close(self) -> None: ...

class OutputSink(Protocol):
    def output_event(self, event: MusicalEvent, source: str) -> None: ...
    def output_tick(self, tick: int, bar: int, beat: int) -> None: ...
    def output_stats(self, round_trip_ms=None, server_process_ms=None, ...) -> None: ...
    def output_status(self, state: str, message: str = "") -> None: ...
    def output_config(self, config: dict) -> None: ...
    def close(self) -> None: ...

class InferenceEngine(Protocol):
    def generate_accompaniment(
        self, melody_events: List[MusicalEvent],
        generation_start_tick: int, generation_length_frames: int,
    ) -> tuple[List[MusicalEvent], TimingInfo]: ...
    def inject_history(...) -> None: ...
    def clear_history(self) -> None: ...
```

### RealTimeMusicService — 3 Threads

- **`_input_worker`**: reads from `InputSource`, puts events into queues
- **`_tick_loop`**: advances musical time, schedules playback, triggers inference every `generation_interval_ticks` (default 2) ticks
- **`_inference_worker`**: calls `InferenceEngine.generate_accompaniment()`, puts results back into playback queue, calls `output_stats()` and `log_inference()` on the output sink

### Timing

- 1 tick = 1/4 beat (default `ticks_per_beat=4`)
- `generation_interval_ticks=2` → inference triggered every half-beat
- `generation_length_frames=20` → each inference generates 20 frames = 10 ticks ahead

### Session Logging

When `--output-type composite --log-dir logs` is used, a `logs/session_YYYYMMDD-HHMMSS/` directory is created containing:
- `events.jsonl` — one JSON line per musical event
- `inferences.json` — all inference request/response pairs with latency
- `performance.json` — latency percentiles (p95/p99), event counts, music analysis
- `statistics.csv` — summary metrics
- `session_config.json` — session configuration
- `combined.mid` — recorded MIDI (user + model)

## Stanley Inference Engine

The Stanley engine uses a two-layer adapter pattern:
- **`StanleyInferenceEngine`** (infrastructure): implements `InferenceEngine` protocol, converts `MusicalEvent` ↔ duration-note dicts
- **`LegacyInferenceEngineStanley`** (legacy): the actual RoFormer model wrapper; operates on `dict[pitch, tick, duration]` and piano-roll tensors

Model defaults: 96-frame context window, 4 ticks/beat, max polyphony 4.

## File Structure

```
src/streammuse/         # Main package (Clean Architecture)
scripts/                # fake_inference_server.py, utilities
tests/
  unit/                 # Fast, no I/O: domain, infrastructure, application layers
  integration/          # CLI entry point tests
prompts/                # Sample MIDI files for injection (organized by key)
transformers/           # Modified HuggingFace transformers (RoFormer positional encoding)
pyproject.toml          # uv project config, entry point: streammuse-cli → cli:main
```