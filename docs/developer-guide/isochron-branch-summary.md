# Isochron Branch Summary

This branch is turning StreamMUSE into a broader realtime research system, not only a realtime music demo.

The main direction is:

```text
shared realtime runtime ideas
  -> music generation
  -> replay/debugging
  -> ProjectIsochron-style LLM tasks
  -> future local LLM / multimodal realtime experiments
```

## What Changed

### Replay Debugger

The branch adds a trace-first replay debugger for comparing an offline reference path against a realtime-shaped simulation path.

It records:

- `manifest.json`
- `trace.jsonl`
- `comparison.json`
- artifacts for event payloads, token/model summaries, scheduler state, and rendered MIDI

The current viewer is read-only. It loads a completed trace directory and shows:

- whether the replay contract held
- where the first mismatch occurred
- a pipeline map
- per-stage diagnosis
- event/token/scheduler/raw detail panes

Current command shape:

```bash
.venv/bin/python -m streammuse.presentation.debug.cli replay \
  --scenario lekai-prompt-continuation \
  --midi-file 01_5472152_input_melody.mid \
  --compare offline,sim \
  --output-dir debug_runs
```

Serve a trace:

```bash
.venv/bin/python -m streammuse.presentation.debug.server \
  --trace-dir debug_runs/<replay_dir> \
  --host 127.0.0.1 \
  --port 8011
```

### Shared Runtime Boundary

The branch starts extracting common runtime assembly into `RuntimeSessionBuilder` and `RuntimeSession`.

The goal is to avoid duplicated setup between:

- terminal CLI
- web entry point
- debugger/replay tools
- future task runtimes

Existing music services still run as before. This is a gradual migration, not a rewrite.

### ProjectIsochron Task Framework

ProjectIsochron is used as a reference, not imported directly.

This branch adds a StreamMUSE-native generic task framework for local-server LLM tasks:

- `RealtimeTask`
- `TaskState`
- `TaskTurn`
- `TaskRefereeResult`
- `TaskRuntime`
- `LocalChatModelClient`

The first reference task is `ZipZapZopTask`.

The framework supports two runner modes:

- `offline_benchmark`: run as fast as the local model server allows
- `realtime_loop`: run on a fixed tick/deadline schedule

Current command shape:

```bash
.venv/bin/python -m streammuse.presentation.task.cli run \
  --task zip_zap_zop \
  --runner offline_benchmark \
  --model-url http://localhost:8000/v1 \
  --model gemma \
  --max-turns 20 \
  --output-dir task_runs
```

## Important Caveats

- The current replay demo trace used fallback/rule-stub Lekai behavior, not real Lekai checkpoints.
- Real Lekai testing should pass real prompt and continuation checkpoints.
- The replay debugger currently compares major checkpoint stages; explicit per-cycle realtime stepping is a next improvement.
- The generic task framework is a first vertical slice. It does not replace `RealTimeMusicService` or `PromptContinuationRealtimeService`.
- Local LLM access is designed around local server APIs first, especially OpenAI-compatible `/chat/completions` servers.

## Why This Matters

This branch addresses the main realtime research problem: when offline and realtime behavior differ, we need to know which boundary caused the divergence.

The long-term goal is one modular system where music, LLM games, local model benchmarks, and future realtime tasks can share:

- timing concepts
- replay and trace artifacts
- deterministic validation
- local model server boundaries
- debugger and benchmark tooling

## Current Verification

The branch has been verified with:

```bash
.venv/bin/python -m pytest tests/ -q --tb=no
```

Latest observed result:

```text
267 passed, 3 skipped, 1 warning
```

