# APPLICATION_SPEC.md — Application Layer

**Audited commit:** `05fc2fa`
**Location:** `src/streammuse/application/`
**Role:** Orchestration and composition. Wires domain protocols to infrastructure implementations. No I/O of its own.

---

## 1. Configuration (`application/config/models.py`)

All five configs are `@dataclass(frozen=True)`. Defaults shown.

### 1.1 `TempoConfig`
```python
bpm: float = 120.0
ticks_per_beat: int = 4
beats_per_bar: int = 4
```

### 1.2 `InputConfig`
```python
type: InputType = "midi_device"
  # Literal["midi_device", "keyboard", "midi_file", "list"]
midi_device_name: Optional[str] = None
midi_file_path: Optional[str] = None
midi_file_delay_ticks: int = 0
injection_file: Optional[str] = None
injection_length_ticks: int = 0
injection_acc_file: Optional[str] = None
```

Validation at CLI layer:
- `injection_file` requires `type == "midi_file"` and an existing file.
- `injection_length_ticks > 0` when injection is used.

### 1.3 `OutputConfig`
```python
type: OutputType = "console"
  # Literal["audio", "midi_file", "console", "websocket",
  #         "composite", "json_log", "session"]
midi_out_port: Optional[str] = None
midi_file_output_path: Optional[str] = None
inference_log_detail: InferenceLogDetail = "summary"
  # Literal["summary", "full"]
```

### 1.4 `InferenceConfig`
```python
type: InferenceType = "http"
  # Literal["http", "stanley"]
server_generate_url: str = "http://localhost:8000/generate_accompaniment"
timeout_s: float = 30.0
model_name: ModelName = "stanley"
  # Literal["stanley", "lekai"]
  # NOTE: project directive (2026-04-24) is "lekai is primary". This default
  # is flagged as DRIFT in docs/audit/progress.txt. Do not silently change;
  # plan a transition in developing-logs/.
inference_mode: str = "sliding_window"
checkpoint_path: Optional[str] = None
model_size: str = "0.12B"
model_max_seq_len_frames: int = 96
generation_length_frames: int = 20
generation_interval_ticks: int = 2
```

### 1.5 `ApplicationConfig`
Top-level composite:
```python
tempo: TempoConfig = TempoConfig()
input: InputConfig = InputConfig()
output: OutputConfig = OutputConfig()
inference: InferenceConfig = InferenceConfig()
```

---

## 2. Factories (`application/factories/`)

Each factory exposes a single `create(config, ...) -> <Protocol>` static/classmethod that inspects the relevant sub-config and constructs the matching infrastructure class.

### 2.1 `InputSourceFactory`
File: `application/factories/input_factory.py`

| `InputConfig.type` | Returns |
|---|---|
| `"midi_device"` | `MidiDeviceInput(device_name=config.midi_device_name)` |
| `"keyboard"` | `KeyboardInput(default key map)` |
| `"midi_file"` | `MidiFileInput(midi_file_path, delay_ticks=midi_file_delay_ticks, tempo)` |
| `"list"` | `ListInput(events)` |

### 2.2 `OutputSinkFactory`
File: `application/factories/output_factory.py`

| `OutputConfig.type` | Returns |
|---|---|
| `"console"` | `ConsoleOutputSink()` |
| `"audio"` | `AudioOutputSink(port_name=midi_out_port)` |
| `"midi_file"` | `MidiFileOutputSink(output_path, tempo)` |
| `"websocket"` | `WebSocketOutputSink(url)` |
| `"json_log"` | `JsonLoggerOutputSink(session_manager, detail=inference_log_detail)` |
| `"session"` | `SessionLoggerOutputSink(session_manager, tempo, detail=inference_log_detail)` |
| `"composite"` | `CompositeOutputSink([...])` (console + session when `--log-dir` is set) |

When a `SessionManager` is passed in, the factory may also auto-attach a `MidiFileOutputSink` targeting `session_dir / "combined.mid"`.

### 2.3 `InferenceEngineFactory`
File: `application/factories/inference_factory.py`

| `InferenceConfig.type` | Returns |
|---|---|
| `"http"` | `HttpInferenceClient(config)` |
| `"stanley"` | `StanleyInferenceEngine(config)` (in-process, loads checkpoint) |

For HTTP, `model_name` determines which remote model is addressed (`"stanley"` or `"lekai"`).

---

## 3. The Orchestrator: `RealTimeMusicService`

File: `application/services/real_time_music_service.py` (406 lines)

### 3.1 Construction

```python
RealTimeMusicService(
    input_source: InputSource,
    inference_engine: InferenceEngine,
    output_sink: OutputSink,
    tempo: Tempo,
    scheduler: PlaybackScheduler,
    generation_interval_ticks: int = 2,
    generation_length_frames: int = 20,
)
```

### 3.2 State

| Attribute | Purpose |
|---|---|
| `_event_q: Queue[MusicalEvent]` | input-worker → tick-loop (user events) |
| `_inference_request_queue: Queue[tuple[int, list[MusicalEvent]]]` | tick-loop → inference-worker |
| `_inference_response_queue: Queue[tuple[int, list[MusicalEvent], TimingInfo]]` | inference-worker → tick-loop |
| `_melody_history: list[MusicalEvent]` | authoritative user stream (guarded by `_melody_history_lock`) |
| `running: bool` | service active flag |
| `session_start: float` | wall-clock origin for tick assignment |

### 3.3 Thread Model — exactly three threads

#### Thread 1: `_input_worker`
```
for event in input_source.read_events():
    event.tick = tempo.seconds_to_tick(time.time() - session_start)
    event.source = "user"
    _event_q.put(event)
    with _melody_history_lock:
        _melody_history.append(event)
```

#### Thread 2: `_tick_loop`
Per tick (starting at `tick=0`):
1. **Wait** until wall-clock time `>= session_start + tempo.tick_to_seconds(tick)`.
2. **Emit tick**: `output_sink.output_tick(tick, bar, beat)` with `MusicalTime.from_tick`.
3. **Boot injection**: on `tick == 0`, if `_melody_history` is non-empty (injection pre-populated it), enqueue a full-history inference request.
4. **Buffer window**: sleep ~10 % of `seconds_per_tick` to collect events that arrived just after the tick boundary.
5. **Drain input queue**:
   - For each user event: `output_sink.output_event(event, "user")`
   - Buffer into the per-tick melody slice for the next inference trigger.
6. **Drain inference responses**: for each response:
   - `scheduler.clear_future_events(generation_start_tick, source="model")`
   - For each accompaniment event: compute `backup_level = event.tick - generation_start_tick`; `scheduler.schedule(event, event.tick)` (or reschedule to `current_tick + 1` if already past).
7. **Play scheduled events**: `for ev in scheduler.get_events_at_tick(tick): output_sink.output_event(ev, ev.source)`.
8. **Trigger inference** every `generation_interval_ticks` (default 2): enqueue `(tick + generation_length_frames // 2, buffered_melody_slice)`.
9. Increment `tick`.

#### Thread 3: `_inference_worker`
```
while running:
    request = _inference_request_queue.get(timeout=0.1)
    # Latest-only semantics:
    while not _inference_request_queue.empty():
        request = _inference_request_queue.get_nowait()   # newer supersedes
        # Melody events are merged across superseded requests
    acc_events, timing = inference_engine.generate_accompaniment(
        melody_events=merged_melody,
        generation_start_tick=request.tick,
        generation_length_frames=self.generation_length_frames,
    )
    _inference_response_queue.put((request.tick, acc_events, timing))
    output_sink.output_stats(round_trip_ms=..., server_process_ms=...)
    if hasattr(output_sink, "log_inference"):
        output_sink.log_inference(request_data, response_data, latency_ms, server_process_ms)
```

### 3.4 Control API
- `start(max_ticks: Optional[int] = None)` — launches all three threads; the main thread blocks on `time.sleep(0.1)` until SIGINT/SIGTERM or `max_ticks` reached.
- `stop()` — sets `running = False`, joins threads, calls `input_source.close()` and `output_sink.close()`.

---

## 4. Latest-Only Semantics (Load-Shedding Policy)

When inference falls behind the tick loop, the worker does **not** FIFO. Instead, on every wake-up it drains the queue, keeps the **newest** `generation_start_tick`, and **merges** the melody events from all superseded requests. Dropped/merged counts are surfaced via `output_status("debug", ...)`.

**Why this matters:** FIFO would let the system accumulate stale requests under load, eventually producing accompaniment that is seconds behind the user. Latest-only keeps the system responsive at the cost of occasional skipped frames.

Do not change this to FIFO without explicit user approval.

---

## 5. Model / History Invalidation on Inference Response

When a new accompaniment arrives for `generation_start_tick = T`:

1. All previously scheduled **model** events at ticks `>= T` are cleared.
2. User events at ticks `>= T` remain untouched.
3. The new model events are scheduled or rescheduled (if late).

This is what makes the system "streaming": each inference response is authoritative for its time window, and the scheduler always reflects the freshest model output.

---

## 6. What This Layer Does *Not* Do

- No network I/O — handled in `infrastructure/inference/http_client.py`.
- No MIDI I/O — handled in `infrastructure/input/` and `infrastructure/output/`.
- No model loading — handled in `infrastructure/inference/stanley_legacy.py` or the remote Lekai server.
- No argument parsing — handled in `presentation/cli/config_parser.py`.
