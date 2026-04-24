# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Protocol (MANDATORY)

**Canonical docs live in `docs/audit/`. CLAUDE.md stays at the project root.**

**At the start of every session:**
1. Read `docs/audit/progress.txt` — the session bridge (COMPLETED / IN PROGRESS / KNOWN BUGS / NEXT).
2. Read `docs/audit/lessons.md` — past mistakes and the rules they produced. Do not repeat them.
3. Read the relevant `docs/audit/*_SPEC.md` for whatever layer you are about to touch.

**At the end of every session (before stopping):**
1. Update `docs/audit/progress.txt` — move completed items out of IN PROGRESS; add new KNOWN BUGS you discovered; set NEXT for the following session.
2. If you hit a dead end or a fix that was non-obvious, append to `docs/audit/lessons.md` in the format: `Problem: [observation] -> Rule: [constraint]`.
3. If you changed an API contract, data schema, or module boundary, update the corresponding `docs/audit/*_SPEC.md`. The specs are the source of truth after the code itself.

If you cannot complete a task, leave `progress.txt` honest — do not pretend work is done.

## Canonical Document Set

The project reality is frozen in `docs/audit/`. Treat these as authoritative alongside the code:

| Doc | Purpose |
|---|---|
| `docs/audit/PRD.md` | Every currently working feature as a hard requirement. Do not regress. |
| `docs/audit/TECH_STACK.md` | Verified library versions actually in use. |
| `docs/audit/DOMAIN_SPEC.md` | `MusicalEvent`, `Note`, `Tempo`, converters, scheduler — exact schemas. |
| `docs/audit/APPLICATION_SPEC.md` | `RealTimeMusicService` 3-thread contract, config dataclasses, factories. |
| `docs/audit/INFRASTRUCTURE_SPEC.md` | Input/output sink implementations and their contracts. |
| `docs/audit/INFERENCE_SPEC.md` | HTTP API contract, Lekai pathway (primary), Stanley pathway (secondary). |
| `docs/audit/PROCESS_FLOW.md` | End-to-end data flow and state transitions. |
| `docs/audit/lessons.md` | Problem → Rule log. Read before writing code. |
| `docs/audit/progress.txt` | Session bridge. Read at start, update at end. |

## Project Rules (Extracted from Audited Patterns)

These rules were extracted by reading the code, not by asking. Follow them unless the user explicitly overrides.

### Architecture
- **Rule:** The codebase uses **Clean Architecture** with 4 layers (`presentation` → `application` → `domain` → `infrastructure`). Dependencies only point inward. Domain has zero external imports. Do not violate this direction.
- **Rule:** Cross-layer wiring goes through **Factories** in `src/streammuse/application/factories/`. Do not instantiate `infrastructure/` classes directly from `presentation/`.
- **Rule:** Protocols in `src/streammuse/domain/interfaces/` are the contract. Every new input source / output sink / inference engine must implement the matching Protocol exactly.

### Domain Model
- **Rule:** `MusicalEvent` and `Note` are **frozen dataclasses**. Never mutate them — construct new ones. Validation runs in `__post_init__`; respect the ranges (pitch 0–127 or -1 for placeholders, velocity 0–127, channel 0–15, program 0–127).
- **Rule:** Time is expressed in **ticks**, not seconds, inside the domain and inference layers. Convert at I/O boundaries only, via `Tempo.tick_to_seconds` / `Tempo.seconds_to_tick`.
- **Rule:** Event-to-note conversion uses the **close-at-horizon policy** (`domain/musical/converters.py`). Unpaired note_ons are closed at `horizon_tick`; re-triggered same-pitch notes close the previous at the new note_on's tick. Do not invent alternative policies.

### Threading & Real-Time
- **Rule:** The `RealTimeMusicService` runs **exactly 3 threads**: `_input_worker`, `_tick_loop`, `_inference_worker`, decoupled by queues. Do not add more threads without updating `APPLICATION_SPEC.md`.
- **Rule:** The inference worker implements **latest-only** semantics — it drains pending requests, keeps the newest `generation_start_tick`, and merges melody events. Do not change this to FIFO without explicit approval; it exists to prevent inference backlog under load.
- **Rule:** On new inference response, the tick loop **clears future model events** from the scheduler before scheduling the new ones (`PlaybackScheduler.clear_future_events(from_tick, source="model")`). This preserves the user's authoritative stream and replaces stale predictions.

### Inference
- **Rule:** **Lekai is the primary inference pathway.** Stanley is retained as a secondary / historical path. New inference work targets Lekai unless specified otherwise. (Note: `InferenceConfig.model_name` still defaults to `"stanley"` in code — see `progress.txt` KNOWN BUGS.)
- **Rule:** The HTTP contract is frozen in `INFERENCE_SPEC.md`. Any change to request/response JSON is a breaking change — update the fake server (`scripts/fake_inference_server.py`) in the same commit.
- **Rule:** The Stanley engine uses a **two-layer adapter**: `StanleyInferenceEngine` (infrastructure, event↔note converter) wraps `LegacyInferenceEngineStanley` (RoFormer model on duration-note dicts). Do not call the legacy engine directly from application code.

### Configuration
- **Rule:** All configuration flows through the frozen dataclasses in `src/streammuse/application/config/models.py` (`ApplicationConfig` / `TempoConfig` / `InputConfig` / `OutputConfig` / `InferenceConfig`). Environment-variable parsing is centralized in `presentation/cli/config_parser.py`. Do not read env vars or argparse in infrastructure code.

### Testing
- **Rule:** Tests live in `tests/unit/` (fast, no I/O — domain, infrastructure adapters, application) and `tests/integration/` (CLI, simulator, Lekai runtime). Domain tests must not import `infrastructure/`.
- **Rule:** Run `uv run pytest tests/` before committing anything non-trivial. There are currently **zero** `xfail`/`skip` markers — keep it that way; fix tests rather than skip them.

### Scope
- **Rule:** This repository contains the **inference service and real-time application only**. Training, preprocessing pipelines, and dataset prep are **out of scope** — they live elsewhere. Do not add training code here, even if a comment in `README.md` suggests otherwise (README has stale sections; the code is the truth).

### Git & Workflow
- **Rule:** Development follows a **plan → implement → report** pattern. Planning docs live in `developing-logs/YYYY-M-D/*-plan.md`, results in `*-report.md`. For non-trivial changes, write a plan before touching code.
- **Rule:** The checked-in branch is `new_system_stanley`. Untracked asset: `FluidR3Mono_GM.sf3` (23.7 MB SoundFont) — do not commit it; treat it as local-only.

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