# Lekai Prompt Continuation

Temporary integration notes for the Lekai prompt-continuation realtime path.

This folder implements the special realtime flow where a prompt model first
generates initial accompaniment, then a continuation model catches up while the
user keeps playing melody.

## Status

This path is wired and covered by unit tests, but model-quality testing still
requires available GPU time and real checkpoints.

Current CPU/stub tests validate:

- HTTP protocol shape.
- Scheduler state transitions.
- Melody append while prompt generation is running.
- Melody append after prompt generation has already reached `ready`.
- Catch-up readiness accounting.

They do not validate musical quality.

## High-Level Runtime Flow

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

1. The user plays the first prompt window, currently 8 beats by default.
2. The client sends those melody events to `/prompt_continuation/start`.
3. `LekaiPromptEngine` generates accompaniment for the prompt window.
4. While prompt generation is running, the client continues sending later melody
   events to `/prompt_continuation/append_melody`.
5. After prompt accompaniment is available, `LekaiContinuationEngine` continues
   generating accompaniment until playback is safe.
6. The client polls `/prompt_continuation/status`.
7. Once ready, the client fetches `/prompt_continuation/playable`.

The prompt window is beat-based, not bar-based. With the current CLI default,
`prompt_length_ticks=32` and `ticks_per_beat=4`, so the prompt model receives
8 beats of melody condition and attempts to generate 8 beats of prompt
accompaniment. `LEKAI_PROMPT_CONDITION_BEATS` can override the condition length
for diagnostics; older bar-based overrides are kept only as compatibility
escape hatches.

## Catch-Up Rule

The system is not playback-ready when accompaniment merely reaches the same
length as melody history. It needs one extra beat of lookahead:

```text
playback_ready = accompaniment_history_beats >= melody_history_beats + 1
```

The extra beat is intentional. Equal history lengths mean the model has caught
up to the past, but there is no next accompaniment beat available to play.

## Playback-Ready Scheduling Policy

The frontend/client may be silent before playback is ready. The backend still
returns the generated accompaniment history through `/prompt_continuation/playable`.
When the client receives that ready signal, it does not schedule events at their
original historical ticks, because those ticks may already be in the past.

`PromptContinuationRealtimeService` therefore schedules only returned
accompaniment events whose original tick is still at or after the frontend's
current tick. Events before the current tick are dropped from audible playback.

This means the user does not hear the prompt-model accompaniment directly. The
initial audible behavior is silence while the user keeps playing melody; sound
starts only after the backend reports catch-up readiness. For debugging, a
separate diagnostic export can still save the full raw accompaniment history
including prompt accompaniment and catch-up accompaniment.

## Main Files

`backend.py`

Request-facing backend wrapper. It exposes the server contract and delegates
runtime work to `LekaiPromptContinuationEngine`.

`engine.py`

Owns the prompt engine, continuation engine, and scheduler. This is the
orchestrator for the two-stage inference flow.

`scheduler.py`

Single-process background scheduler. It serializes prompt and continuation model
calls on one worker thread while HTTP requests can keep appending melody.

Important phases:

```text
idle
prompt_running
catchup_running
ready
failed
```

`catchup_state.py`

Pure beat-count logic for deciding whether accompaniment has caught up enough
for playback.

`prompt_engine.py`

Loads and runs the Lekai prompt model. It converts melody events to prompt-model
tokens, calls generation, and decodes generated accompaniment events.

`continuation_engine.py`

Thin wrapper around the existing Lekai continuation runtime. It receives melody
increments and uses injected accompaniment history.

`token_conversion.py`

Small payload/request helpers used by the engine and scheduler.

`prompt_model/`

Copied Lekai prompt-model implementation and tokenizer code.

## Server Entry

The HTTP entrypoint is outside this folder:

```text
src/streammuse/infrastructure/inference/server_lekai.py
```

`server_lekai.py` already existed before this integration. The prompt-
continuation integration added:

- `LekaiPromptContinuationBackend` construction.
- `model_name == "lekai_prompt_continuation"` backend selection.
- `/prompt_continuation/start`.
- `/prompt_continuation/append_melody`.
- `/prompt_continuation/status`.
- `/prompt_continuation/playable`.
- `/prompt_continuation/raw_history`.
- `/prompt_continuation/prompt_history`.

## Client Entry

The CLI path is:

```text
src/streammuse/presentation/cli/cli.py
```

When `--model-name lekai_prompt_continuation` is selected, CLI creates:

- `PromptContinuationHttpClient`
- `PromptContinuationRealtimeService`

It does not use the normal per-window `InferenceEngineFactory` path.

## Useful Environment Variables

Prompt model checkpoint:

```bash
export LEKAI_PROMPT_CHECKPOINT_PATH=/path/to/lekai_prompt_model/model.safetensors
```

Continuation model checkpoint:

```bash
export LEKAI_CONTINUATION_CHECKPOINT_PATH=/path/to/continuation/model.safetensors
```

Server port:

```bash
export LEKAI_SERVER_PORT=8000
```

Prompt model sampling overrides:

```bash
export LEKAI_PROMPT_TOP_P=0.95
export LEKAI_PROMPT_TOP_K=0
export LEKAI_PROMPT_TEMPERATURE=1.1
```

Strict real-model mode:

```bash
export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1
export LEKAI_DISABLE_FALLBACK=1
```

With these set, missing checkpoints, model-load failures, or any rule-based
fallback path fail loudly instead of producing stub accompaniment.

## CLI-Only Prompt Verification Shape

For prompt-continuation validation, use `streammuse-cli` only. Do not use a
separate fake-offline runner as evidence for realtime behavior.

Start server:

```bash
export LEKAI_PROMPT_CHECKPOINT_PATH=/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors
export LEKAI_CONTINUATION_CHECKPOINT_PATH=/path/to/continuation/model.safetensors
export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1
export LEKAI_DISABLE_FALLBACK=1
export LEKAI_SERVER_PORT=8001
uv run python -m streammuse.infrastructure.inference.server_lekai
```

Run CLI with a melody MIDI:

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path path/to/melody.mid \
  --midi-file-trim-leading-rest \
  --tempo 240 \
  --ticks-per-beat 4 \
  --beats-per-bar 4 \
  --inference-type http \
  --server-url http://127.0.0.1:8000/generate_accompaniment \
  --model-name lekai_prompt_continuation \
  --prompt-length-ticks 32 \
  --generation-interval-ticks 4 \
  --generation-length-frames 16 \
  --output-type session \
  --log-dir realtime_runs/0509_prompt_continuation_cli \
  --max-ticks 80
```

The session directory saves:

- `combined.mid`: audible realtime output after catch-up scheduling.
- `prompt_continuation_prompt_history.mid`: prompt-model accompaniment only,
  plus the input melody track for inspection.
- `prompt_continuation_prompt_history.json`: prompt-model accompaniment events
  only. This is the file to compare against the RT offline prompt-stage output.
- `prompt_continuation_raw_history.mid`: prompt plus continuation history,
  before frontend scheduling drops past events.

If the strict env vars above are set, fallback/stub output is impossible: the
server or CLI request fails instead.

## Current Caveats

- `scripts/run_lekai_batch_client.py` still needs its model-name allowlist
  updated before it can batch-run `lekai_prompt_continuation`.
- Real prompt + continuation quality must be checked with GPU and checkpoints.
- The current protocol is HTTP polling. WebSocket streaming is still an open
  design choice.
