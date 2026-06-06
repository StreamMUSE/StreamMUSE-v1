# Lekai Prompt-Continuation Realtime System

This document describes the current StreamMUSE integration of Lekai's two-stage
prompt-continuation realtime accompaniment flow. It is intended as an engineering
handoff note: what the system does, which files are involved, what was changed,
and how to validate that the realtime path is still aligned with the RT offline
reference.

## Current Status

The `lekai_prompt_continuation` path is wired through `streammuse-cli` and the
Lekai HTTP server. It has been validated with real prompt and continuation
checkpoints under strict real-model mode, with fallback disabled.

The current validated behavior is:

- Prompt model output from `streammuse-cli` matches RT offline prompt-stage output
  exactly by event SHA for the tested pieces.
- Continuation uses the RT offline tokenizer/model layout, not the older
  StreamMUSE Lekai tokenizer.
- Realtime audible MIDI now preserves recoverable late model events instead of
  dropping most cross-beat notes.
- MIDI exports now write the configured time signature, so 2/4 pieces are not
  exported as 4/4.
- Debug exports retain raw prompt + continuation history even when the audible
  realtime output drops late notes.

Recent validation artifacts are local generated files and are intentionally not
tracked by git:

```text
realtime_runs/0509_unified_recover_late_author_top50_topp098_tempclamp120/
realtime_runs/0509_6217163_2-4_recover_late_comparison/
```

## Runtime Flow

```text
streammuse-cli
  -> MidiFileInput / realtime user input
  -> PromptContinuationRealtimeService
  -> PromptContinuationHttpClient
  -> server_lekai.py
  -> LekaiPromptContinuationBackend
  -> LekaiPromptContinuationEngine
  -> LekaiPromptContinuationScheduler
     -> LekaiPromptEngine
     -> LekaiContinuationEngine
```

Standard scenario:

1. User/client provides a melody prompt window, normally 8 beats.
2. Client sends that melody to `/prompt_continuation/start`.
3. Prompt model generates initial accompaniment for the prompt window.
4. While prompt generation is running, client continues sending melody chunks to
   `/prompt_continuation/append_melody`.
5. Continuation model consumes prompt accompaniment plus later melody and keeps
   generating accompaniment until playback is ready.
6. Client polls `/prompt_continuation/status`.
7. Once ready, client fetches `/prompt_continuation/playable`.
8. `PromptContinuationRealtimeService` schedules playable events into the local
   realtime `PlaybackScheduler`.

The frontend/client can be silent before playback is ready. The user does not
need to hear prompt-model accompaniment immediately. Prompt accompaniment is
still saved in debug history.

## Computation Flow

```text
Client realtime clock
  |
  | each tick:
  |   drain user melody events
  |   split by prompt window
  v
PromptContinuationRealtimeService
  |
  | observed_until_tick < prompt_length_ticks
  |   collect events into prompt_events
  |
  | observed_until_tick == prompt_length_ticks
  |   POST /prompt_continuation/start
  |   body:
  |     melody_events = prompt_events
  |     prompt_length_ticks
  |     generation_interval_ticks
  v
LekaiPromptContinuationScheduler
  |
  | Prompt stage:
  |   PromptEngine(prompt melody, prompt_length_ticks)
  |     -> prompt_accompaniment_history
  |
  | Continuation seed:
  |   ContinuationEngine.inject_history(
  |     melody_history,
  |     prompt_accompaniment_history,
  |     injection_length_ticks = actual_prompt_length_ticks
  |   )
  v
Catch-up loop
  |
  | while accompaniment_history_beats < melody_history_beats + lookahead_beats:
  |   generation_start_tick = accompaniment_history_beats * ticks_per_beat
  |   melody_increment = melody_history not yet sent to continuation
  |   ContinuationEngine.generate(melody_increment, generation_start_tick)
  |     -> append to accompaniment_history
  v
Playback readiness
  |
  | GET /prompt_continuation/status
  | ready when:
  |   accompaniment_history_beats >= melody_history_beats + 1
  |
  | GET /prompt_continuation/playable
  | returns full accompaniment_history, not only the newest segment
  v
Local audible scheduling
  |
  | default:
  |   pair note_on/note_off
  |   drop notes fully in the past
  |   clip sustaining notes to current_tick
  |
  | recover-late mode:
  |   recover late events at current_tick
  |   optional bounded policy can drop very old late note_on events
  |   late note_off is allowed to close already-sounding notes
  v
PlaybackScheduler -> Output sinks
```

Important timing distinction:

```text
Standard realtime continuation:
  request returns one future generation segment
  -> late recovery can safely reschedule that segment at current_tick

Prompt-continuation:
  /playable returns full prompt + continuation history
  -> unbounded late recovery would replay old history after slow prompt-model
     inference, especially on small GPUs
  -> strict paired scheduling, unbounded recovery, and bounded recovery are
     separate switchable policies for A/B diagnosis
```

## Catch-Up Rule

The system is not playback-ready when accompaniment merely reaches the same beat
length as melody history. It needs one extra beat of lookahead:

```text
playback_ready = accompaniment_history_beats >= melody_history_beats + 1
```

Equal history lengths mean the model has caught up to the past, but there is no
next accompaniment beat available to play.

## Prompt Length Units

Use beats as the user-facing unit.

With default `ticks_per_beat=4`:

```text
prompt_length_ticks = prompt_beats * 4
```

For normal 4/4 and 2/4 cases, the current tested prompt is 8 beats. For 3/4
pieces, we use 6 beats because RT prompt preparation requires:

```text
prompt_beats % beats_per_bar == 0
```

That is a prompt-model/data-preparation constraint, not a frontend display rule.

## Time-Signature Handling

The realtime path keeps the source piece's `beats_per_bar` instead of assuming
4/4.

Current prompt-length policy:

```text
4/4 piece -> 8 beat prompt
2/4 piece -> 8 beat prompt
3/4 piece -> 6 beat prompt
```

The 2/4 case still uses 8 beats because 8 is divisible by `beats_per_bar=2`,
so it satisfies the RT prompt-preparation constraint and gives the model the same
amount of beat-level context as the 4/4 demo cases. We do not convert 2/4 into
4/4 internally.

MIDI export must also preserve the original time signature. `MidiFileOutputConfig`
therefore carries `beats_per_bar`, and the session MIDI writer emits the matching
MIDI time-signature meta event. This avoids the previous bug where a 2/4 piece
was musically generated as 2/4 but exported with a misleading 4/4 grid.

## Playback Scheduling Policy

There are three histories to keep separate:

- `prompt_continuation_prompt_history.*`: prompt-model accompaniment only.
- `prompt_continuation_raw_history.*`: full prompt + continuation accompaniment
  history from the backend.
- `combined.mid`: audible realtime output after local scheduling policy.

The important realtime policy is in
`PromptContinuationRealtimeService._schedule_playable`.

Three scheduling policies are useful for diagnosis:

- Default historical strict mode: pair `note_on/note_off`, drop events whose
  original ticks are already in the past.
- Unbounded recover-late mode: schedule returned events event-by-event. If an
  event is late, schedule it at the current tick.
- Bounded recover-late mode: same event-by-event recovery, but drop late
  `note_on` events outside a configured recovery window. Late `note_off` events
  are still allowed so already-sounding notes can be closed.

Recover-late mode is enabled by:

```bash
export LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
```

This switch alone is intentionally unbounded. To test the bounded policy, enable
it separately:

```bash
export LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=1
```

If bounded recovery is enabled and no max is provided, the client uses
`generation_interval_ticks` as the cap. To set the cap explicitly:

```bash
export LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
```

Setting `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` also opts into the
bounded policy unless `LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=0` is set
explicitly. Raw debug history is not affected by either scheduling policy.

For demo-style runs, the current practical setting is:

```bash
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=1
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
```

## Main Files

`src/streammuse/infrastructure/inference/server_lekai.py`

HTTP entrypoint. Existing Lekai server file extended with prompt-continuation
routes:

```text
/prompt_continuation/start
/prompt_continuation/append_melody
/prompt_continuation/status
/prompt_continuation/playable
/prompt_continuation/raw_history
/prompt_continuation/prompt_history
/prompt_continuation/runtime_info
```

`src/streammuse/infrastructure/inference/prompt_continuation_http_client.py`

Small HTTP client used by the realtime service for the endpoints above.

`src/streammuse/application/services/prompt_continuation_realtime_service.py`

Realtime client-side service. It handles user input, sends prompt/append requests,
polls backend status, fetches playable accompaniment, and schedules audible model
events. Recent changes here added trace logging and recover-late scheduling.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/backend.py`

Request-facing backend wrapper. Delegates runtime work to
`LekaiPromptContinuationEngine`.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/engine.py`

Owns the prompt engine, continuation engine, and scheduler.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/scheduler.py`

Single-process background scheduler. It serializes prompt and continuation model
calls on one worker thread while HTTP requests can continue appending melody.

Phases:

```text
idle
prompt_running
catchup_running
ready
failed
```

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/catchup_state.py`

Pure beat-count logic for catch-up readiness.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py`

Loads and runs the Lekai prompt model. Converts melody events to prompt-model
condition tokens and decodes generated accompaniment events.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/continuation_engine.py`

Thin wrapper around the Lekai continuation runtime.

`src/streammuse/infrastructure/inference/lekai_prompt_continuation/token_conversion.py`

Shared request/event conversion helpers.

`src/streammuse/infrastructure/output/midi_file.py`

Writes session MIDI. Recent change: `MidiFileOutputConfig` includes
`beats_per_bar`, and MIDI exports now write the correct time signature.

`scripts/run_cli_prompt_alignment_batch.sh`

Strict validation runner for `streammuse-cli` prompt-continuation sessions.
It starts the real server, rejects fallback, runs CLI, and compares prompt output
against RT offline reference prompt output.

## Tokenizer Boundary

Prompt model tokenizer:

```text
src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_model/
```

Continuation model tokenizer:

```text
src/streammuse/infrastructure/inference/lekai_continuation_model/
```

Older StreamMUSE Lekai tokenizer:

```text
src/streammuse/infrastructure/inference/lekai_model/
```

Do not feed prompt-model beat tokens directly into continuation. The prompt model
and continuation model do not share the same sequence layout. Prompt output must
be decoded to events/piano-roll and re-encoded into the RT offline continuation
layout.

The continuation checkpoint follows RT offline acc-first ordering:

```text
[BOS][TS][BPM] [bar] [beat] acc...track_acc mel...track_mel ...
```

## Strict Real-Model Environment

Prompt model checkpoint:

```bash
export LEKAI_PROMPT_CHECKPOINT_PATH=/path/to/prompt/model.safetensors
```

Continuation model checkpoint:

```bash
export LEKAI_CONTINUATION_CHECKPOINT_PATH=/path/to/continuation/model.safetensors
```

Recommended strict mode:

```bash
export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1
export LEKAI_DISABLE_FALLBACK=1
```

With these set, missing checkpoints or model-load failures fail loudly instead
of silently producing rule-based fallback output.

## Verification Matrix

Lightweight tests do not import the full Lekai torch backend. They are suitable
for checking switch wiring, client-side realtime scheduling, catch-up arithmetic,
MIDI-file timing, and prompt-continuation HTTP client contracts in a minimal
developer environment:

```bash
PYTHONPATH=src python -m pytest \
  tests/unit/presentation/test_cli_config_parser.py \
  tests/unit/presentation/web/test_server.py \
  tests/integration/test_cli_entry_point.py \
  tests/unit/application/test_factories_and_service.py \
  tests/unit/application/test_prompt_continuation_realtime_service.py \
  tests/unit/infrastructure/input/test_midi_file_input.py \
  tests/unit/infrastructure/inference/test_prompt_continuation_http_client.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_catchup_state.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_scheduler.py \
  tests/integration/test_lekai_prompt_continuation_rt_midi.py
```

Backend and server tests import `torch` directly through `lekai_http_backend.py`.
That is intentional: the real server path depends on torch and should fail
clearly if the full environment is incomplete. Run these only after `uv sync`
or equivalent has installed the project dependencies:

```bash
PYTHONPATH=src python -m pytest \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_backend.py \
  tests/unit/infrastructure/inference/test_server_lekai.py \
  tests/unit/infrastructure/inference/test_lekai_http_backend.py
```

Important sampling knobs:

```bash
export LEKAI_PROMPT_TEMPERATURE=1.1
export LEKAI_PROMPT_TOP_K=0
export LEKAI_PROMPT_TOP_P=0.95

export LEKAI_RT_TEMPERATURE=0.8
export LEKAI_RT_TOP_K=50
export LEKAI_RT_TOP_P=0.98
export LEKAI_RT_REPETITION_PENALTY=1.2
```

The batch validation script accepts these as shell variables and forwards them
to server/CLI.

## How To Run A Strict CLI Test

Use `streammuse-cli` for realtime-path validation. Do not use a separate fake
runner as evidence for frontend/backend realtime behavior.

Recommended one-command demo/validation run:

```bash
scripts/run_lekai_prompt_continuation_realtime_demo.sh
```

This command is intentionally usable without editing. It defaults to:

```text
DEVICE=0
PORT=8001
IDS="6217163 5472152 5472153 6144911 332891 3329166 6303422"
OUT_ROOT=realtime_runs/lekai_prompt_continuation_demo
CLI_TEMPO=metadata
CLI_TEMPO_MAX=120
PROMPT_BEATS=auto
MAX_TICKS=432
PROMPT_TEMPERATURE=1.1
PROMPT_TOP_K=0
PROMPT_TOP_P=0.95
RT_TEMPERATURE=0.8
RT_TOP_K=50
RT_TOP_P=0.98
RT_REPETITION_PENALTY=1.2
RECOVER_LATE_EVENTS=1
BOUND_LATE_RECOVERY=1
RECOVER_LATE_MAX_TICKS=4
```

`PROMPT_BEATS=auto` means:

```text
4/4 -> 8 beat prompt
2/4 -> 8 beat prompt
3/4 -> 6 beat prompt
```

To change only the GPU or output folder, prefix environment variables without
editing the script:

```bash
DEVICE=5 OUT_ROOT=realtime_runs/my_prompt_continuation_demo \
  scripts/run_lekai_prompt_continuation_realtime_demo.sh
```

To smoke-test only one piece:

```bash
IDS="6217163" MAX_TICKS=96 \
  scripts/run_lekai_prompt_continuation_realtime_demo.sh
```

The lower-level batch script is still available for explicit parameter sweeps:

```bash
DEVICE=0 \
PORT=8102 \
IDS='6217163' \
OUT_ROOT=realtime_runs/example_prompt_continuation \
CLI_TEMPO=metadata \
CLI_TEMPO_MAX=120 \
PROMPT_BEATS=8 \
MAX_TICKS=432 \
PROMPT_TEMPERATURE=1.1 \
PROMPT_TOP_K=0 \
PROMPT_TOP_P=0.95 \
RT_TEMPERATURE=0.8 \
RT_TOP_K=50 \
RT_TOP_P=0.98 \
RT_REPETITION_PENALTY=1.2 \
RECOVER_LATE_EVENTS=1 \
BOUND_LATE_RECOVERY=1 \
RECOVER_LATE_MAX_TICKS=4 \
 scripts/run_cli_prompt_alignment_batch.sh
```

The script checks:

- server health;
- real prompt model loaded;
- real continuation model loaded;
- no fallback reason present;
- prompt-stage CLI output equals offline reference by event SHA.

## Session Output Files

A `--output-type session` run writes:

```text
combined.mid
```

Audible realtime output after local scheduling policy.

```text
prompt_continuation_prompt_history.mid/json
```

Prompt-model accompaniment only, plus melody track in the MIDI for inspection.
The JSON is the file used for prompt SHA comparison.

```text
prompt_continuation_raw_history.mid/json
```

Full backend prompt + continuation history before frontend scheduling policy.
Use this for debugging model output.

```text
prompt_continuation_*_status.json
```

Backend status snapshots for the corresponding debug histories.

```text
prompt_continuation_client_trace.jsonl
```

Optional client-side trace when `LEKAI_PROMPT_CONTINUATION_TRACE_PATH` is set.
The batch script sets this automatically. It records start/append/status/playable
fetches and scheduling counts.

## What Changed Recently

The current branch includes these key changes:

- Added Lekai prompt-continuation integration and HTTP endpoints.
- Copied packaged Lekai prompt-model code into `prompt_model/`.
- Switched continuation to RT offline tokenizer/model layout.
- Added strict prompt alignment comparison against RT offline reference.
- Split prompt sampling params from continuation sampling params.
- Added repeated status/playable polling so the client can continue fetching
  after more melody arrives.
- Preserved note pairs in MIDI output and closed same-pitch retriggers.
- Added recover-late event scheduling for prompt-continuation audible playback.
- Added switchable bounded late recovery for dropping very old audible `note_on`
  events while preserving raw history.
- Added correct MIDI time-signature export through `beats_per_bar`.
- Added local trace output for prompt-continuation client scheduling.
- Ignored generated realtime/test-set/user-sample directories in `.gitignore`.

## Current Caveats

- Prompt alignment can be exact, but continuation output is stochastic unless
  seeds/sampling are fixed and the same runtime path is used.
- Real frontend behavior still needs a clear playback policy: when to start
  sounding, how much late material to recover, and how to handle user timing
  jitter.
- `midi_file` tests simulate user input under controlled timing. Real MIDI-device
  latency should still be tested before claiming live performance reliability.
- `BOUND_LATE_RECOVERY` and `RECOVER_LATE_MAX_TICKS` are policy knobs, not a
  model fix. Tune them based on listening tests and keep unbounded recovery as
  an A/B condition.
- Generated outputs under `realtime_runs/` are not tracked by git.
