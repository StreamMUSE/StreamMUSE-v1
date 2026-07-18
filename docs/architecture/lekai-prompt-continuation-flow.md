# Lekai Prompt-Continuation Flow And Deadline Risk

This note records the current Lekai prompt-continuation client and engine flow.
It is written to explain why prompt+continuation can appear to be completely too
late under the same BPM settings where standard Lekai continuation is still
usable.

## Scope

Mode:

```text
--continuation-mode prompt_continuation
--model-name lekai
```

This mode is opt-in. The default remains `--continuation-mode standard`.
`RuntimeSessionBuilder` is the shared CLI/web composition root and selects the
prompt-continuation service only for this explicit mode. Rap and
prompt-continuation cannot currently be enabled in the same session; the CLI
rejects that unverified combination during configuration.

Main files:

```text
src/streammuse/application/services/prompt_continuation_realtime_service.py
src/streammuse/application/runtime/builder.py
src/streammuse/infrastructure/inference/prompt_continuation_http_client.py
src/streammuse/infrastructure/inference/server_lekai.py
src/streammuse/infrastructure/inference/lekai_prompt_continuation/scheduler.py
src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py
src/streammuse/infrastructure/inference/lekai_prompt_continuation/continuation_engine.py
```

Optional variant:

```text
LEKAI_PROMPT_CONTINUATION_ENGINE=prompt_extension
LEKAI_PROMPT_CONTINUATION_EXTENSION_TICKS=4
```

## Client Flow

```text
InputSource
  |
  | input worker stamps user melody with realtime tick
  v
PromptContinuationRealtimeService
  |
  | ticks before prompt boundary:
  |   collect M0 into prompt_events
  |
  | observed_until_tick == prompt_length_ticks:
  |   POST /prompt_continuation/start
  |   body:
  |     melody_events = M0
  |     prompt_length_ticks
  |     generation_interval_ticks
  |
  | ticks after prompt boundary:
  |   collect M1, M2, ...
  |   at generation_interval boundary:
  |     POST /prompt_continuation/append_melody
  |
  | protocol worker:
  |   poll /prompt_continuation/status
  |   only fetch /playable after:
  |     start was sent
  |     at least one append after prompt was sent
  |     backend reports is_playback_ready
  v
/prompt_continuation/playable
  |
  | returns full accompaniment history, not only newest segment
  v
Local scheduling policy
  |
  | strict:
  |   pair notes, drop fully-past notes, clip sustaining notes
  |
  | recover_late:
  |   recover late events at current_tick
  |
  | bounded recover_late:
  |   drop note_on older than max recovery window
  v
OutputSink
```

## Engine Flow

```text
LekaiPromptContinuationScheduler.start
  |
  | background worker starts
  v
Prompt stage
  |
  | PromptEngine(M0, prompt_length_ticks)
  |   -> A0
  |
  | prompt_extension variant:
  |   PromptEngine(M0, prompt_length_ticks + extension_ticks)
  |     -> A0 plus prompt-generated extra beat(s)
  v
Seed continuation
  |
  | ContinuationEngine.inject_history(
  |   melody_history = all melody observed so far,
  |   accompaniment_events = prompt accompaniment,
  |   injection_length_ticks = actual prompt output length
  | )
  v
Catch-up loop
  |
  | while accompaniment_history_beats < melody_history_beats + lookahead:
  |   generation_start_tick = accompaniment_history_beats * 4
  |   melody_increment = melody events not yet sent to continuation
  |   ContinuationEngine.generate(one chunk)
  |   append result to accompaniment_history
  v
Ready
  |
  | playable when:
  |   accompaniment_history_beats >= melody_history_beats + 1
```

## Timing Model

At BPM 90:

```text
1 beat = 60 / 90 = 0.6667 s
8 beat prompt = 5.333 s
```

The prompt-continuation path has a different deadline shape from standard
continuation:

```text
first audible model output cannot be safely scheduled until:

prompt boundary is reached
  + prompt model latency
  + at least one append after prompt
  + continuation catch-up latency
  + status polling / playable fetch overhead
  + local scheduling policy
```

So even if continuation itself is fast, the first playable result can be late if
the prompt model finishes after the client has already advanced beyond the first
returned accompaniment ticks.

Example shape:

```text
T = 5.33s - epsilon:
  client has M0
  send start(M0)

T = 5.33s + prompt_latency:
  prompt returns A0
  client may already have M1
  send append(M1)

T = 5.33s + prompt_latency + continuation_latency:
  backend may have A0 + A1
  /playable returns full history

current realtime tick may already be past some or all of A0/A1
```

This is why prompt+continuation can look "completely too late": the client is
not merely waiting for the next short segment; it is waiting for a two-stage
pipeline to produce enough history plus lookahead.

## Difference From Standard Lekai

| Topic | Standard Lekai | Prompt-Continuation |
| --- | --- | --- |
| First model call | next short continuation segment | prompt model over the whole prompt window |
| Backend state | one continuation backend keeps history | prompt engine plus continuation engine plus scheduler |
| Client requests | one `/generate_accompaniment` stream | `/start`, `/append_melody`, `/status`, `/playable` |
| Response shape | one generated segment | full accompaniment history |
| Startup barrier | no prompt-stage barrier | must finish prompt stage and catch-up |
| Stale request handling | latest-only queue collapses stale requests | scheduler serializes prompt then catch-up |
| Late scheduling | can recover late returned segment | strict/recover/bounded policies needed because full history is returned |
| Main risk | late segment | first playable history is already in the past |

## Architecture-Level Hypotheses For "Too Late"

These are architecture hypotheses, not confirmed server measurements yet:

1. Prompt-stage startup barrier:
   prompt model inference starts only after the prompt window is observed. If
   prompt latency is large relative to beat duration, the client is already past
   the earliest playable ticks before the backend is ready.

2. Serialized model stages:
   prompt inference and continuation catch-up are serialized through one worker.
   Continuation cannot start until prompt accompaniment is generated and injected.

3. Catch-up requires lookahead:
   readiness requires `melody_history_beats + 1`, not merely equal history
   length. This intentionally avoids playing without future accompaniment, but
   it increases time-to-first-playable.

4. Full-history playable response:
   `/playable` returns prompt plus continuation history. If fetched late, many
   events can already be in the past. Standard mode usually returns only the
   requested future segment.

5. BPM makes the deadline tighter:
   at higher BPM, each beat is shorter. The same prompt latency consumes more
   musical time. For example, 0.7 seconds is more than one beat at BPM 90.

6. Melody send early can expose mismatch:
   the client can append post-prompt melody while the prompt model is still
   running. That is necessary for catch-up, but it means the backend may have to
   generate multiple continuation chunks before the frontend has anything safe
   to schedule.

7. Scheduling policy changes audible result:
   strict mode may drop old events; unbounded recovery may replay too much old
   history at the current tick; bounded recovery may drop too-old note_on events.
   This is why the recovery policies are switchable.

## What To Measure When Server Is Reachable

For the same MIDI, BPM, prompt length, and generation interval, run:

```text
standard
prompt_continuation standard engine + strict scheduling
prompt_continuation standard engine + unbounded recover_late
prompt_continuation standard engine + bounded recover_late
prompt_continuation prompt_extension engine
```

Collect:

- time from prompt boundary to `/start` response;
- prompt model `inference_end - inference_start`;
- time of first `/append_melody` after prompt;
- number of continuation calls before `is_playback_ready`;
- time from prompt boundary to first `/playable`;
- current client tick when `/playable` is scheduled;
- count of dropped past notes, recovered late events, and bounded dropped note_on
  events;
- first audible model event tick in `combined.mid`;
- raw model history in `prompt_continuation_raw_history.mid/json`.

The likely deciding metric is:

```text
time_to_first_playable - available_lead_time
```

If this is positive and large, prompt-continuation will sound late even if each
individual continuation call is fast.
