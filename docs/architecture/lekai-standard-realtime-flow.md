# Lekai Standard Realtime Flow

This note records the current standard Lekai realtime client and engine flow.
It is meant as the baseline for comparing why `prompt_continuation` can miss
realtime playback deadlines under the same BPM and generation settings.

## Scope

Mode:

```text
--continuation-mode standard
--model-name lekai
```

Main files:

```text
src/streammuse/application/services/real_time_music_service.py
src/streammuse/infrastructure/inference/http_client.py
src/streammuse/infrastructure/inference/server_lekai.py
src/streammuse/infrastructure/inference/lekai_http_backend.py
```

## Client Flow

```text
InputSource
  |
  | input worker stamps incoming user events with current realtime tick
  v
RealTimeMusicService
  |
  | every tick:
  |   sleep to tick wall-clock boundary
  |   output tick / metronome
  |   wait 10% tick buffer for near-boundary events
  |   drain user events
  |   output user events immediately
  |
  | at beat tail:
  |   enqueue request for next beat:
  |     generation_start_tick = tick + 1
  |     melody_events = user events since last request
  v
Inference worker
  |
  | latest-only behavior:
  |   if multiple requests are waiting, keep newest generation_start_tick
  |   merge skipped melody events into newest request
  v
HTTP /generate_accompaniment
  |
  | response returns one generated segment
  v
Tick loop
  |
  | clear future model events from generation_start_tick
  | schedule returned events:
  |   if event.tick >= current_tick -> schedule at event.tick
  |   if event.tick < current_tick  -> recover at current_tick
  v
OutputSink
```

## Engine Flow

```text
LekaiHttpBackend.generate
  |
  | append request melody_events to server melody_history
  | configure runtime:
  |   generation_interval_ticks
  |   generation_length_frames
  |   prompt_length_ticks = None
  v
Lekai continuation model
  |
  | generate accompaniment for generation_start_tick
  | duration is generation_length_frames
  v
Return one segment
```

Important behavior:

- The server receives melody increments but keeps full melody history.
- The client does not wait for a separate prompt stage.
- The client can drop stale queued requests and keep the newest request.
- A late model response can still be scheduled at `current_tick`.

## Timing Model

At BPM 90:

```text
1 beat = 60 / 90 = 0.6667 s
1 tick = 0.6667 / 4 = 0.1667 s
```

If generation is requested at every beat boundary, the practical deadline for a
one-beat segment is roughly one beat minus transport/client overhead:

```text
deadline_per_segment ~= 0.6667 s at BPM 90
```

If model latency exceeds the deadline, the segment is late. Standard mode still
has a recovery path:

```text
late event -> schedule at current_tick
```

This can sound rushed or compressed, but it avoids a hard startup barrier.

## Expected Result

Standard mode is best-effort realtime:

```text
new melody arrives
  -> next beat request is sent
  -> model returns one future segment
  -> stale requests may be collapsed
  -> late events can be recovered at current tick
```

It can be musically late, but it normally does not need to wait for an initial
multi-stage prompt/catch-up pipeline before producing any audible accompaniment.

## Failure Modes To Measure

Use this path as the baseline for:

- round-trip latency per `/generate_accompaniment`;
- server `inference_end_time - inference_start_time`;
- number of stale requests dropped by latest-only behavior;
- number of late model events recovered at the current tick;
- audible gap length before the first model accompaniment event.

The key question is whether standard mode remains acceptable because each call
only has to generate the next short segment, while prompt-continuation has to
finish a serialized prompt plus catch-up sequence before playback can begin.
