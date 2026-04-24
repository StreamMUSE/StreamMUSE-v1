# PROCESS_FLOW.md — End-to-End Data Flow & State Transitions

**Audited commit:** `05fc2fa`
**Scope:** Follows a single session from `streammuse-cli` invocation to clean shutdown.

This document maps control flow, data flow, and state transitions exactly as the current code executes them. It does not propose changes.

---

## 1. Startup Sequence

```
$ streammuse-cli --input-mode keyboard --output-type composite --log-dir logs
```

1. **Argv → config**
   `presentation/cli/config_parser.py` builds an immutable `ApplicationConfig` from CLI args (env vars can override).
2. **Injection validation**
   If `--injection-file` is set: assert file exists, assert `--input-mode midi_file`, assert `--injection-length > 0`.
3. **Session directory**
   If output type ≠ `midi_file`, construct `SessionManager(base_log_dir=args.log_dir)`; it creates `logs/YYYY-MM-DD/session_HHMMSS/` and writes `session_config.json`.
4. **Factories**
   - `OutputSinkFactory.create(config, session_manager)` → concrete sink (possibly `CompositeOutputSink`)
   - `InferenceEngineFactory.create(config)` → `HttpInferenceClient` or `StanleyInferenceEngine`
   - `InputSourceFactory.create(config)` → concrete input
5. **Injection (optional)**
   `_perform_injection`:
   - Parse melody MIDI → `list[MusicalEvent]`
   - Parse accompaniment MIDI (if paired file exists) → `list[MusicalEvent]`
   - `inference_engine.inject_history(melody, acc, injection_length_ticks)`
   - Save `melody_history.json`, `accompaniment_history.json` to session dir
6. **Service construction**
   `RealTimeMusicService(input_source, inference_engine, output_sink, tempo, scheduler, generation_interval_ticks, generation_length_frames)`
7. **Cleanup wiring**
   Register `atexit` + `signal.signal(SIGINT, …)` + `signal.signal(SIGTERM, …)` handlers that call:
   - `inference_engine.clear_history()`
   - `output_sink.close()`
   - `session_manager.save_summary(...)`
8. **Start**
   `service.start(max_ticks)` launches the three worker threads and returns. The main thread enters `while service.running: time.sleep(0.1)` and waits for an interrupt.

---

## 2. Steady-State Loop (Per Tick)

At tick `T` (wall clock `session_start + T * seconds_per_tick`):

```
tick_loop iteration @ tick=T
│
├── wait until now() >= session_start + T * seconds_per_tick
│
├── output_sink.output_tick(T, bar, beat)
│
├── if T == 0 and _melody_history (injection pre-filled):
│       _inference_request_queue.put((T, copy(_melody_history)))
│
├── sleep(0.1 * seconds_per_tick)   # buffer window for late inputs
│
├── drain _event_q:
│       for user_event in all queued events:
│           output_sink.output_event(user_event, "user")
│           buffer_user_event_for_next_inference_trigger()
│
├── drain _inference_response_queue:
│       for (gen_tick, acc_events, timing) in responses:
│           scheduler.clear_future_events(gen_tick, source="model")
│           for ev in acc_events:
│               ev.backup_level = ev.tick - gen_tick
│               if ev.tick <= T:
│                   scheduler.schedule(replace(ev, tick=T+1), T+1)
│               else:
│                   scheduler.schedule(ev, ev.tick)
│
├── play scheduled:
│       for ev in scheduler.get_events_at_tick(T):
│           output_sink.output_event(ev, ev.source)
│
├── if T > 0 and (T % generation_interval_ticks) == 0:
│       _inference_request_queue.put(
│           (T + generation_length_frames // 2, buffered_melody_slice)
│       )
│
└── T := T + 1
```

### Parallel inference worker:

```
inference_worker loop
│
├── req = _inference_request_queue.get(timeout=0.1)
│
├── # latest-only coalescing
│   while not _inference_request_queue.empty():
│       newer = _inference_request_queue.get_nowait()
│       merged_melody += newer.melody_events
│       req = replace(req, tick=newer.tick, melody_events=merged_melody)
│       dropped_count += 1
│
├── if dropped_count > 0:
│       output_sink.output_status("debug", f"dropped={dropped_count}")
│
├── try:
│       acc_events, timing = inference_engine.generate_accompaniment(
│           req.melody_events, req.tick, generation_length_frames
│       )
│       _inference_response_queue.put((req.tick, acc_events, timing))
│       output_sink.output_stats(round_trip_ms, server_process_ms)
│       if hasattr(output_sink, "log_inference"):
│           output_sink.log_inference(request_data, response_data, ...)
│   except Exception as e:
│       output_sink.output_status("error", str(e))
```

### Parallel input worker:

```
input_worker loop
│
├── for ev in input_source.read_events():   # blocks on MIDI / key / file
│       tick_now = tempo.seconds_to_tick(time.time() - session_start)
│       ev2 = replace(ev, tick=tick_now, source="user")
│       _event_q.put(ev2)
│       with _melody_history_lock:
│           _melody_history.append(ev2)
```

---

## 3. State Transitions

### 3.1 Scheduler (per-tick bucket)

| Event                                     | Effect on scheduler                           |
|-------------------------------------------|-----------------------------------------------|
| Inference response for `gen_tick=T0`      | Remove all `source="model"` events at tick ≥ T0 |
| Accompaniment event at future tick T'     | Appended to bucket for T'                      |
| Accompaniment event at past tick (T' ≤ T) | Rescheduled to T+1, `backup_level` annotated   |
| `get_events_at_tick(T)`                   | Returns **and removes** bucket[T]              |
| User event                                | **Not** placed in scheduler; fired immediately |

### 3.2 Melody history

| Event                          | Effect on `_melody_history`                |
|--------------------------------|--------------------------------------------|
| User NOTE_ON / NOTE_OFF        | Appended (lock-guarded)                    |
| Inference boot (tick 0)        | Full history copy enqueued for inference   |
| `clear_history()` on shutdown  | Delegated to inference engine; local history is **not** cleared here |

### 3.3 Inference request queue

| Event                                   | Effect                                        |
|-----------------------------------------|-----------------------------------------------|
| Tick loop triggers (every N ticks)      | `put((tick, buffered_melody_slice))`          |
| Worker drains → finds newer after pop   | Merges melody events, keeps newest tick, increments `dropped` counter |

---

## 4. Shutdown Sequence

Triggered by SIGINT (Ctrl-C), SIGTERM, `max_ticks` reached, or uncaught exception in the main thread.

1. `service.stop()` sets `running = False`.
2. Each thread exits its loop at the next check-in (≤100 ms).
3. `input_source.close()` — stops pynput listener / closes mido port / closes MIDI file iterator.
4. `inference_engine.clear_history()` — resets server / in-process history.
5. `output_sink.close()`:
   - `MidiFileOutputSink` pairs outstanding notes and writes `combined.mid`.
   - `JsonLoggerOutputSink` writes `inferences.json`.
   - `SessionLoggerOutputSink` writes `performance.json`, `statistics.csv`.
   - `CompositeOutputSink` calls `close()` on each child in order.
6. `session_manager.save_summary(...)` — writes human-readable `session_summary.txt`.
7. `atexit` handlers fire; process exits 0.

Any exception raised during shutdown is logged to stderr and the process still exits 0 (best-effort cleanup). The session directory is preserved even on crash-out so logs can be inspected.

---

## 5. Error Paths

| Error                                                 | Path                                                                                         |
|-------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Inference HTTP timeout                                | `HttpInferenceClient` raises → caught in `_inference_worker` → `output_status("error", ...)` → worker continues |
| Inference HTTP server unreachable at startup          | First call propagates exception → `service.start()` still runs; subsequent calls keep erroring until server recovers |
| Invalid `MusicalEvent` (e.g. pitch 128)               | `__post_init__` raises `ValueError` at construction site — never reaches the queue           |
| MIDI device disconnect                                | `mido` raises → `_input_worker` thread exits; service keeps running with no new user input   |
| Keyboard input: missing pynput backend                | `KeyboardInput.__init__` raises → surfaces in CLI startup                                    |
| MIDI file not found for injection                     | Validation in `cli.main` raises before service starts                                        |
| Output sink `close()` raises                          | Error logged to stderr; other children of `CompositeOutputSink` still close                  |

---

## 6. Timing Budget (Default Configuration)

At BPM=120, `ticks_per_beat=4`:
- `seconds_per_tick` = `(60 / 120) / 4` = **0.125 s** (125 ms)
- `generation_interval_ticks=2` → inference triggered every **250 ms**
- `generation_length_frames=20` with 96-frame context → generates **10 ticks = 1.25 s** of accompaniment per call

For the system to keep up, round-trip inference latency must be `< generation_interval_ticks * seconds_per_tick` = **250 ms**. When it is not, the latest-only semantics in the inference worker absorb the backlog at the cost of occasional dropped intermediate requests.

---

## 7. Invariants (Runtime)

1. **Wall clock is authoritative for user events.** Tick assignment at input time uses `time.time() - session_start`.
2. **`generation_start_tick` is authoritative for model events.** The scheduler keys on it to invalidate stale predictions.
3. **User events are never cleared by inference updates.** `clear_future_events(..., source="model")` filters on source.
4. **The scheduler is drained exactly once per tick.** `get_events_at_tick(T)` removes the bucket atomically.
5. **Session directory is created before any event is produced**, so no log writes race against directory creation.
