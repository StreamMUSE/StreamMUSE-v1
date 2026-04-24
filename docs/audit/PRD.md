# PRD.md — Functional Baseline (Frozen Scope)

**Audited commit:** `05fc2fa` on branch `new_system_stanley`
**Audit date:** 2026-04-24

This document lists every feature currently working in the codebase as a **hard requirement**. Any change that removes or breaks one of these features is a regression. "Working" means: traceable from an entry point, covered by a test or referenced in `docs/`, and producing observable output.

---

## 1. Real-Time CLI Application

**R1.1** — Entry point `streammuse-cli` (defined in `pyproject.toml:45` → `streammuse.presentation.cli.cli:main`) starts a real-time music-generation session from the shell.

**R1.2** — CLI accepts arguments parsed by `src/streammuse/presentation/cli/config_parser.py` into an immutable `ApplicationConfig`. Environment-variable overrides are honored when present.

**R1.3** — On startup, the CLI constructs components via factories (`InputSourceFactory`, `OutputSinkFactory`, `InferenceEngineFactory`), registers SIGINT/SIGTERM/atexit cleanup, and hands them to `RealTimeMusicService`.

**R1.4** — On `Ctrl-C` or SIGTERM, the service stops cleanly: `inference_engine.clear_history()`, `output_sink.close()`, `session_manager.save_summary()`.

---

## 2. Input Sources (`--input-mode`)

**R2.1 — `midi_device`** — Reads MIDI from a hardware device via `mido` (`MidiDeviceInput`). Device selectable by name.

**R2.2 — `keyboard`** — Reads keystrokes via `pynput` and maps to MIDI pitches using `DEFAULT_KEY_TO_PITCH` (`z`=C4, `x`=D4, …). Non-blocking iteration.

**R2.3 — `midi_file`** — Plays back a MIDI file as live input, timed against tempo. Supports `--midi-file-delay-ticks` for leading silence.

**R2.4 — `list`** — In-memory list of events, used by tests and programmatic scenarios.

**R2.5** — All four implementations satisfy the `InputSource` Protocol (`read_events() -> Iterator[MusicalEvent]`, `close()`).

---

## 3. Output Sinks (`--output-type`)

**R3.1 — `console`** — Prints events, ticks, and stats to stdout (`ConsoleOutputSink`).

**R3.2 — `audio`** — Streams MIDI events to a `mido` output port for real-time audible playback (`AudioOutputSink`).

**R3.3 — `midi_file`** — Records events to a `.mid` file via `pretty-midi` (`MidiFileOutputSink`).

**R3.4 — `websocket`** — Pushes events over a WebSocket connection (`WebSocketOutputSink`).

**R3.5 — `json_log`** — Writes `events.jsonl` and `inferences.json` per session (`JsonLoggerOutputSink`).

**R3.6 — `session`** — Combined MIDI + JSON logging into a session directory (`SessionLoggerOutputSink`).

**R3.7 — `composite`** — Fans out to multiple sinks simultaneously (`CompositeOutputSink`). Used with `--log-dir` to pair console with session logging.

**R3.8** — All seven implementations satisfy the `OutputSink` Protocol (`output_event`, `output_tick`, `output_stats`, `output_status`, `output_config`, `close`).

---

## 4. Inference Pathways

**R4.1 — Lekai (primary)** — HTTP-based inference against a Lekai model server. Served by `src/streammuse/infrastructure/inference/server_lekai.py` and the `lekai_model/` package. Reached via `HttpInferenceClient` with `model_name="lekai"`.

**R4.2 — Stanley (secondary)** — Two-layer RoFormer pathway. In-process via `StanleyInferenceEngine` → `LegacyInferenceEngineStanley`, or over HTTP with `model_name="stanley"`. Retained for parity with legacy data and comparison runs.

**R4.3 — Fake server** — `scripts/fake_inference_server.py` echoes the melody as accompaniment, for local development without a model or GPU.

**R4.4** — All pathways satisfy the `InferenceEngine` Protocol: `generate_accompaniment`, `inject_history`, `set_injection_offset`, `clear_history`.

---

## 5. Music Injection

**R5.1** — `--injection-file <path>` loads a melody MIDI file before session start; model history is pre-populated so generation begins with context.

**R5.2** — `--injection-length <ticks>` controls how many leading ticks of the injection are used.

**R5.3** — If a matching accompaniment file exists (e.g. `..._mel.mid` → `..._acc.mid`), it is also injected.

**R5.4** — Injection data is saved to the session directory as `melody_history.json` and `accompaniment_history.json`.

**R5.5** — Injection is only valid with `--input-mode midi_file` (validated at startup).

---

## 6. Timing Model

**R6.1** — Musical time is measured in **ticks**. Default: `ticks_per_beat = 4`, `beats_per_bar = 4`, `bpm = 120.0`.

**R6.2** — Inference triggers every `generation_interval_ticks` ticks (default: 2 → every half-beat).

**R6.3** — Each inference generates `generation_length_frames` frames ahead (default: 20 frames = 10 ticks ahead at `ticks_per_beat=4`).

**R6.4** — `MusicalTime.from_tick(tick, tempo)` converts absolute ticks to `(bar, beat, tick_in_beat)` for display.

---

## 7. Scheduling & Playback

**R7.1** — `PlaybackScheduler` is thread-safe; `schedule(event, tick)` and `get_events_at_tick(tick)` are the canonical add/drain operations.

**R7.2** — On new inference response, `clear_future_events(from_tick, source="model")` removes stale model predictions before the new ones are scheduled. User events are never cleared by this path.

**R7.3** — Late events (model events whose scheduled tick is already in the past) are rescheduled at `current_tick + 1` and have their `backup_level` field annotated with the delay.

---

## 8. Session Logging

**R8.1** — When any session-logging sink is active, a directory is created at `logs/YYYY-MM-DD/session_HHMMSS/` (with legacy-path fallback to `logs/session_HHMMSS/`).

**R8.2** — The session directory contains:
- `session_config.json` — the `ApplicationConfig` that was used
- `events.jsonl` — one JSON line per musical event
- `inferences.json` — all request/response pairs with latency
- `performance.json` — p95/p99 latency, event counts, music analysis
- `statistics.csv` — summary metrics
- `combined.mid` — recorded MIDI (user + model) when session logging is composed with MIDI file output
- `session_summary.txt` — human-readable summary written on shutdown
- `melody_history.json`, `accompaniment_history.json` — when injection is used

**R8.3** — `inference_log_detail` can be `"summary"` or `"full"`, controlling whether full request/response payloads are logged.

---

## 9. Inference HTTP Contract

**R9.1** — The HTTP contract is **frozen**. Endpoints: `POST /generate_accompaniment`, `POST /inject_notes`, `POST /clear_history`, `GET /injection_status`, `GET /health`. Full schemas in `INFERENCE_SPEC.md`.

**R9.2** — `HttpInferenceClient` and `scripts/fake_inference_server.py` must agree on the schema. Any change updates both in a single commit.

---

## 10. Observability

**R10.1** — `output_stats()` reports per-inference `round_trip_ms`, `server_process_ms`, and (optionally) `network_latency_ms`.

**R10.2** — Output sinks implementing `log_inference(...)` receive the full inference payload when available.

**R10.3** — `output_status(state, message)` surfaces state changes (`"error"`, debug counters for dropped/merged requests, etc.).

---

## 11. Configuration Surface

**R11.1** — Five frozen config dataclasses: `TempoConfig`, `InputConfig`, `OutputConfig`, `InferenceConfig`, `ApplicationConfig`. Fields and defaults enumerated in `APPLICATION_SPEC.md`.

**R11.2** — All configs are `@dataclass(frozen=True)`. Construct new instances; never mutate.

---

## 12. Testing

**R12.1** — `uv run pytest tests/` completes with zero failures, zero xfails, zero skips at the audited commit.

**R12.2** — Unit tests cover every domain entity, every input/output implementation, serialization, HTTP client, Stanley adapter, Lekai adapter, and the CLI config parser.

**R12.3** — Integration tests cover: CLI startup path, MIDI simulator output, Lekai runtime info.

---

## Non-Goals (Explicit Out-of-Scope)

- Model training, checkpointing, and hyperparameter tuning — handled in a separate repository.
- Dataset preparation and MIDI preprocessing at scale — handled externally.
- Web UI / dashboard — this is a CLI application.
- Multi-session orchestration or multi-tenant server — the inference server serves one client at a time.
