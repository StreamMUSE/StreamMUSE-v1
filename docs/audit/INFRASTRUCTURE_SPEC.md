# INFRASTRUCTURE_SPEC.md — Infrastructure Layer

**Audited commit:** `05fc2fa`
**Location:** `src/streammuse/infrastructure/`
**Rule:** Each class in this layer implements a `domain/interfaces/` Protocol. All external I/O (OS, network, MIDI hardware, filesystem) lives here and nowhere else.

Inference-related infrastructure has its own spec: see `INFERENCE_SPEC.md`.

---

## 1. Input Sources (`infrastructure/input/`)

All implement `InputSource`: `read_events() -> Iterator[MusicalEvent]`, `close() -> None`.

### 1.1 `MidiDeviceInput`
File: `infrastructure/input/midi_device.py`

- Uses `mido` + `python-rtmidi` to open a hardware MIDI device.
- `read_events()` blocks on incoming MIDI messages; yields `MusicalEvent` with `tick=0` (the application layer assigns the real tick based on wall clock at arrival).
- Supports device selection by name; default resolves via `mido.get_input_names()`.
- `close()` closes the underlying `mido` input port.

### 1.2 `KeyboardInput`
File: `infrastructure/input/keyboard.py`

- Uses `pynput.keyboard.Listener` on a background thread.
- Built-in `DEFAULT_KEY_TO_PITCH` maps QWERTY row-based keys to MIDI pitches (`z`=C4, `x`=D4, `c`=E4, …).
- On key press / release, pushes a `MusicalEvent(NOTE_ON|NOTE_OFF)` to an internal queue.
- `read_events()` yields events by popping the internal queue; non-blocking iteration.
- `close()` stops the pynput listener.

### 1.3 `MidiFileInput`
File: `infrastructure/input/midi_file.py`

- Parses a MIDI file with `mido.MidiFile`.
- Converts delta-time ticks to `tick` values using the provided `Tempo`.
- Optional `delay_ticks` shifts the entire sequence forward to create a silent lead-in.
- `read_events()` yields events in tempo-accurate real time (sleeping between deliveries) so the service experiences the file as a live input stream.

### 1.4 `ListInput`
File: `infrastructure/input/list_input.py`

- Wraps a pre-built `list[MusicalEvent]` for tests / deterministic scenarios.
- No I/O; simply iterates the list.

---

## 2. Output Sinks (`infrastructure/output/`)

All implement `OutputSink`: `output_event`, `output_tick`, `output_stats`, `output_status`, `output_config`, `close`. Some additionally expose `log_inference(...)` which the service calls via `hasattr` check.

### 2.1 `ConsoleOutputSink`
File: `infrastructure/output/console.py`

- Prints events, ticks (every `ticks_per_beat` to reduce noise), stats (rolling averages), and status transitions to stdout.
- Zero external dependencies beyond Python stdlib.

### 2.2 `AudioOutputSink`
File: `infrastructure/output/audio.py`

- Opens a `mido` output port (default, or by name).
- Translates `MusicalEvent` → `mido.Message("note_on"|"note_off", note=pitch, velocity=velocity, channel=channel)` and sends immediately.
- The scheduler upstream ensures events fire at the correct wall-clock moment; this sink does not do its own timing.
- Optionally sends `program_change` on first note of a given program/channel combo.
- `close()` closes the mido output port.

### 2.3 `MidiFileOutputSink`
File: `infrastructure/output/midi_file.py`

- Accumulates events in memory, one `pretty_midi.Instrument` per (channel, program) combination.
- Closes on `close()` by pairing `NOTE_ON`/`NOTE_OFF` and writing via `pretty_midi.PrettyMIDI(...).write(path)`.
- When attached under `SessionLoggerOutputSink`, writes `session_dir / "combined.mid"`.

### 2.4 `WebSocketOutputSink`
File: `infrastructure/output/websocket.py`

- Maintains a WebSocket connection to a configured URL.
- Serializes events via the same JSON schema as the inference HTTP contract (see `INFERENCE_SPEC.md`) and pushes them.
- Used for UI / remote observer scenarios.

### 2.5 `JsonLoggerOutputSink`
File: `infrastructure/output/json_logger.py`

- Writes `session_dir / "events.jsonl"` (one `LogEvent.to_json()` per line).
- Writes `session_dir / "inferences.json"` on `close()` from an in-memory list of `InferenceEvent`.
- Exposes `log_inference(request_data, response_data, latency_ms, server_process_ms)` which the service calls.
- `inference_log_detail="summary"` elides large payloads; `"full"` keeps the raw request/response JSON.

### 2.6 `SessionLoggerOutputSink`
File: `infrastructure/output/session_logger.py`

- Combines JSON logging with MIDI recording.
- On `close()`, writes `performance.json` (latency p95/p99, event counts, music-analysis summary) and `statistics.csv`.
- Delegates to `SessionManager.save_summary()` for the human-readable `session_summary.txt`.

### 2.7 `CompositeOutputSink`
File: `infrastructure/output/composite.py`

- Holds a list of child sinks.
- Every method call fans out to each child in order; exceptions from one child do not halt the others (logged via `output_status("error", ...)`).
- Used when `--output-type composite` is combined with `--log-dir`: typically `[ConsoleOutputSink, SessionLoggerOutputSink(+MidiFileOutputSink)]`.

---

## 3. Storage (`infrastructure/storage/`)

### 3.1 `prompt_repository.py`

- Loads injection MIDI files from `prompts/<key>/pop909_XXX_{mel,acc}.mid`.
- Resolves melody → accompaniment siblings by name convention (`_mel.mid` ↔ `_acc.mid`).
- `prompts/metadata.json` indexes available prompts by key.

---

## 4. Inference (`infrastructure/inference/`)

**See `INFERENCE_SPEC.md`** for HTTP contract, Lekai pathway (primary), Stanley pathway (secondary), serialization, and model internals. That spec covers:

- `HttpInferenceClient`
- `StanleyInferenceEngine` (event↔note adapter)
- `LegacyInferenceEngineStanley` (RoFormer wrapper)
- `lekai_model/`, `lekai_http_backend.py`, `server_lekai.py`
- `serialization.py` (event_to_dict, event_from_dict, timing_info_from_dict)
- `runtime_device.py` (GPU/CPU/MPS selection)

---

## 5. Shared Conventions

- **Thread safety:** Every sink that maintains internal collections uses a `threading.Lock`. Factories are single-threaded (called once at startup).
- **Error propagation:** Infrastructure errors surface to the service via `output_sink.output_status("error", message)`. The service logs and continues. Fatal errors (e.g. inference server unreachable at startup) propagate as exceptions out of `start()`.
- **Close semantics:** `close()` is idempotent. The service calls it exactly once on shutdown via `atexit`/signal handlers. Infrastructure objects may additionally no-op if already closed.
- **Serialization boundary:** When crossing process boundaries (HTTP, WebSocket, JSON log), events are serialized via `infrastructure/inference/serialization.py` (`event_to_dict` / `event_from_dict`). Never roll your own JSON shape.
