# Lekai Prompt-Continuation Backend Design

## Roles

`LekaiPromptContinuationBackend` is the request-facing boundary. It handles API contracts, request data, response shape, and lifecycle state exposed to the server.

`LekaiPromptContinuationEngine` is the inference orchestrator. It coordinates the prompt model and the continuation model.

`LekaiPromptEngine` generates the initial accompaniment prompt from user melody.

`LekaiContinuationEngine` continues accompaniment generation after a prompt accompaniment exists.

`LekaiPromptContinuationScheduler` is the single-process background coordinator. It serializes prompt and continuation model calls on one worker thread, while the backend thread can keep accepting user melody updates.

## Standard Runtime Scenario

The standard teacher-discussed flow is:

1. The user inputs 2 bars of melody.
2. The prompt model starts generating prompt accompaniment for those 2 bars.
3. While the prompt model is running, the user continues playing another `n` beats of melody.
4. Assume the prompt model finishes after those `n` beats have arrived.
5. The continuation model receives:
   - user melody history, including the original 2 bars and the later `n` beats,
   - generated prompt accompaniment history for the first 2 bars.
6. The continuation model keeps generating accompaniment while conditioning on the user melody history.
7. The system is considered caught up only when accompaniment history has reached the same beat length as melody history, and the model has also produced the next accompaniment beat that can be played.
8. Only after that catch-up point should the backend return playable accompaniment to the frontend.

In short: catching up does not mean the prompt model is done. Catching up means continuation has filled the accompaniment gap created while the user kept playing, and has produced the next accompaniment beat for playback.

## Beat-Length State Definition

Use beat counts as the coordination unit:

- `melody_history_beats`: number of melody beats observed from the user.
- `accompaniment_history_beats`: number of accompaniment beats already available, including generated prompt accompaniment and continuation output.
- `playable_next_beat`: the first accompaniment beat after histories are aligned.

The continuation stage is not ready to return playback if:

```text
accompaniment_history_beats < melody_history_beats + 1
```

The `+1` is intentional. If accompaniment only reaches the same beat count as melody, the system has caught up to the past but does not yet have the next accompaniment beat to play. Playback can start when the next beat exists.

## Example

If the user first provides 8 beats of melody:

```text
melody_history_beats = 8
accompaniment_history_beats = 0
```

The prompt model generates 8 beats of accompaniment:

```text
melody_history_beats = 8 + n
accompaniment_history_beats = 8
```

If the user continued for `n = 3` beats while the prompt model ran:

```text
melody_history_beats = 11
accompaniment_history_beats = 8
```

The continuation model must generate at least 4 accompaniment beats:

```text
beats_needed = melody_history_beats + 1 - accompaniment_history_beats
             = 11 + 1 - 8
             = 4
```

After generating 4 beats:

```text
melody_history_beats = 11
accompaniment_history_beats = 12
```

Now beat 12 is available as the next playable accompaniment beat.

## Current Implementation Gap

The current implementation wires the backend, prompt engine, continuation engine, model loading, prompt-model warmup, a synchronous catch-up state tracker, and a single-process background scheduler.

Current behavior is still request/response oriented:

- prompt accompaniment can be generated,
- generated prompt accompaniment can be injected into continuation history,
- continuation generation can be called after that.
- catch-up status is exposed through runtime/status methods.
- background scheduling can accept more melody while prompt generation is running, then run continuation until the catch-up state is playback-ready.

Still needed for the standard scenario:

- decide whether the frontend should keep HTTP polling or move to WebSocket updates,
- return only the first playable accompaniment beat or playable chunk after catch-up.

## HTTP Polling Contract

The first integration contract uses HTTP polling rather than WebSocket streaming:

- `POST /prompt_continuation/start`
  - starts background prompt generation from the initial melody window,
  - sets `prompt_length_ticks`,
  - records `observed_until_tick` so rests near the boundary still count as played time.
- `POST /prompt_continuation/append_melody`
  - appends more user melody while prompt or continuation is running,
  - updates melody history length from `observed_until_tick`.
- `GET /prompt_continuation/status`
  - returns scheduler phase and catch-up state.
- `GET /prompt_continuation/playable`
  - returns accompaniment only when playback is ready; otherwise `accompaniment` is empty and status explains why.

Current scheduler phases:

```text
idle
prompt_running
catchup_running
ready
failed
```

## Design Notes

The prompt model should be loaded and warmed up during server startup. The measured prompt-model latency on GPU0 was roughly:

```text
model load: about 2.05 s
warmup prompt generation: about 0.9-1.1 s
steady prompt generation median: about 0.7 s
steady p90: about 1.0 s
```

So the server should not wait until the first realtime request to pay load and warmup cost.

## Open Questions

- Should the backend return only the first playable accompaniment beat after catch-up, or a short chunk of multiple beats?
- Should continuation generation block until catch-up, or should the frontend poll/subscribe to readiness?
- What is the maximum allowed catch-up delay before falling back to silence or hold-last accompaniment?
