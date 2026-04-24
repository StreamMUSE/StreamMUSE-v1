# DOMAIN_SPEC.md — Domain Layer

**Audited commit:** `05fc2fa`
**Location:** `src/streammuse/domain/`
**Rule:** Domain has zero external dependencies and no I/O. Objects are immutable (`@dataclass(frozen=True)`).

---

## 1. Musical Model (`domain/musical/`)

### 1.1 `EventType` (enum)

```python
class EventType(Enum):
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
```

### 1.2 `MusicalEvent` — frozen dataclass

File: `domain/musical/events.py`

```python
@dataclass(frozen=True)
class MusicalEvent:
    tick: int                    # absolute musical time in ticks (>= 0)
    pitch: int                   # 0..127, or -1 when is_placeholder=True
    event_type: EventType        # NOTE_ON | NOTE_OFF
    velocity: int = 100          # 0..127
    channel: int = 0             # 0..15
    program: int = 0             # 0..127 (MIDI instrument)
    is_placeholder: bool = False # synthetic padding event
    source: str = "unknown"      # "user" | "model" | "unknown"
    backup_level: int = 0        # offset from generation_start_tick (late-arrival tracking)
```

Validation runs in `__post_init__`:
- `tick >= 0`
- `0 <= velocity <= 127`
- `0 <= channel <= 15`
- `0 <= program <= 127`
- if `is_placeholder`: `pitch == -1`
- else: `0 <= pitch <= 127`

### 1.3 `Note` — frozen dataclass

File: `domain/musical/events.py`

```python
@dataclass(frozen=True)
class Note:
    pitch: int
    tick: int
    duration: int            # > 0 (in ticks)
    velocity: int = 100
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False
```

Key methods:
- `to_events() -> list[MusicalEvent]` — returns `[NOTE_ON @ tick, NOTE_OFF @ tick+duration]`
- `Note.from_events(note_on, note_off)` — reverse

### 1.4 Converters

File: `domain/musical/converters.py`

**`events_to_notes(events, horizon_tick) -> list[Note]`** — canonical close-at-horizon policy:
1. Iterate events in chronological order.
2. Pair each `NOTE_ON` with the next `NOTE_OFF` of the same pitch.
3. If a `NOTE_ON` has no matching `NOTE_OFF` by end of stream, close it at `horizon_tick`.
4. If the same pitch retriggers before its prior `NOTE_OFF` (polyphonic re-attack), close the prior note at the new `NOTE_ON`'s tick.
5. Return notes sorted by `(tick, pitch)`.

**`musical_event_to_generic_event(event)`** / **`generic_event_to_musical_event(generic)`** — bridge to the `domain/events/generic.py` model (designed for future multi-modal expansion; not on the current hot path).

### 1.5 `sequence.py`

Declared for future use. Not imported by any module reachable from the CLI entry point. Do not remove without explicit approval (see `progress.txt`).

---

## 2. Timing Model (`domain/timing/`)

### 2.1 `Tempo` — frozen dataclass

File: `domain/timing/tempo.py`

```python
@dataclass(frozen=True)
class Tempo:
    bpm: float              # > 0
    ticks_per_beat: int     # > 0 (default 4 in TempoConfig)
    beats_per_bar: int      # > 0 (default 4 in TempoConfig)
```

Derived properties and methods:
- `seconds_per_tick` → `(60.0 / bpm) / ticks_per_beat`
- `tick_to_seconds(tick: int) -> float` → `tick * seconds_per_tick`
- `seconds_to_tick(seconds: float) -> int` → `int(seconds / seconds_per_tick)`
- `ticks_per_bar` → `ticks_per_beat * beats_per_bar`

### 2.2 `MusicalTime` — frozen dataclass

File: `domain/timing/tempo.py`

```python
@dataclass(frozen=True)
class MusicalTime:
    tick: int            # absolute
    bar: int             # 0-based
    beat: int            # 0-based within bar
    tick_in_beat: int    # 0-based within beat
```

Constructor:
- `MusicalTime.from_tick(tick, tempo)` decomposes an absolute tick into `(bar, beat, tick_in_beat)`.

### 2.3 `PlaybackScheduler`

File: `domain/timing/scheduler.py`. **Thread-safe** via an internal `threading.Lock()`. Not a dataclass.

API:
- `schedule(event: MusicalEvent, tick: int) -> None` — add event for playback at `tick`.
- `get_events_at_tick(tick: int) -> list[MusicalEvent]` — return and remove all events scheduled for that tick.
- `clear_future_events(from_tick: int, source: str | None = None) -> None` — remove events at ticks `>= from_tick`. When `source` is given, only matching events (e.g. `"model"`) are removed — user events are never touched by this call.

Internal representation: `dict[int, list[MusicalEvent]]` keyed by tick.

---

## 3. Protocols / Interfaces (`domain/interfaces/`)

Duck-typed `Protocol` classes. The four concrete infrastructure families must match these signatures exactly.

### 3.1 `InputSource`

File: `domain/interfaces/input.py`

```python
class InputSource(Protocol):
    def read_events(self) -> Iterator[MusicalEvent]: ...
    def close(self) -> None: ...
```

### 3.2 `OutputSink`

File: `domain/interfaces/output.py`

```python
class OutputSink(Protocol):
    def output_event(self, event: MusicalEvent, source: str) -> None: ...
    def output_tick(self, tick: int, bar: int, beat: int) -> None: ...
    def output_stats(
        self,
        hit_rate: Optional[float] = None,
        avg_backup_level: Optional[float] = None,
        round_trip_ms: Optional[float] = None,
        server_process_ms: Optional[float] = None,
        network_latency_ms: Optional[float] = None,
        total_hits: Optional[int] = None,
        total_ticks: Optional[int] = None,
    ) -> None: ...
    def output_status(self, state: str, message: str = "") -> None: ...
    def output_config(self, config: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...
```

Optional extensions (duck-typed, checked at call site):
- `log_inference(...)` — called by the service when a sink supports detailed inference logging (`JsonLoggerOutputSink`, `SessionLoggerOutputSink`).

### 3.3 `InferenceEngine`

File: `domain/interfaces/inference.py`

```python
class InferenceEngine(Protocol):
    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]: ...

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None: ...

    def set_injection_offset(self, offset_ticks: int) -> None: ...

    def clear_history(self) -> Dict[str, Any]: ...
```

### 3.4 `TimingInfo`

File: `domain/interfaces/timing_info.py`

```python
@dataclass(frozen=True)
class TimingInfo:
    request_arrival_time: float         # server wall clock at request receipt
    response_output_time: float         # server wall clock at response send
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float
    # Client-derived (optional):
    round_trip_time: Optional[float] = None
    server_processing_duration: Optional[float] = None
    total_network_latency: Optional[float] = None
```

All times are Unix epoch seconds (floats).

---

## 4. Logging Model (`domain/logging/`)

### 4.1 `LogEvent`

File: `domain/logging/event_types.py`

```python
@dataclass(frozen=True)
class LogEvent:
    timestamp: float
    tick: int
    event_type: EventType   # local enum: NOTE_ON, NOTE_OFF, INFERENCE_REQUEST,
                            # INFERENCE_RESPONSE, TICK, STATUS
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

### 4.2 `InferenceEvent`

```python
@dataclass(frozen=True)
class InferenceEvent:
    inference_id: str
    timestamp_request: float
    timestamp_response: float
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    latency_ms: float
    server_process_ms: float

    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

### 4.3 `SessionManager`

File: `domain/logging/session_manager.py`

```python
class SessionManager:
    def __init__(self, base_log_dir: str = "logs", session_id: Optional[str] = None)
    def create_session_directory(self) -> Path
    def save_config(self, config: Dict[str, Any]) -> None   # writes session_config.json
    def save_summary(self, summary: Dict[str, Any]) -> None # writes session_summary.txt
    def get_session_dir(self) -> Path
    def get_session_id(self) -> str
```

Directory policy:
- Primary: `base_log_dir / YYYY-MM-DD / session_HHMMSS/`
- Legacy fallback: `base_log_dir / session_HHMMSS/` (preserved when that path already exists)

### 4.4 `metrics_calculator.py`

Declared for future use. Not on the current hot path. Do not remove without approval (see `progress.txt`).

---

## 5. Generic Event Model (`domain/events/`)

File: `domain/events/generic.py`

Declared for future multi-modal event streams (non-musical signals). Not imported by the CLI runtime at audit time.

---

## Invariants Enforced by the Domain

1. `MusicalEvent` and `Note` are **immutable**. Construct new instances; never mutate.
2. `tick` is always non-negative and strictly monotonic per source within a session.
3. `PlaybackScheduler` is the **only** place events cross from past-scheduled to present-playable, and the only place that knows about tick ordering for scheduled items.
4. Unpaired `NOTE_ON`s are closed at the provided horizon — never dropped, never held open indefinitely.
5. Clearing future scheduled events respects source; user events survive model invalidation.
