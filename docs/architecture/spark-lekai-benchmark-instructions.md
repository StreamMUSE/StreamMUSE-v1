# Spark Lekai Benchmark Instructions

This is the handoff for running the Lekai prompt-continuation timing test on a Spark/H200 machine.

## Files

- `scripts/run_spark_lekai_benchmark.sh`
- `scripts/benchmark_lekai_spark.py`

The bash script is the entrypoint. The Python script performs direct model measurements and parses public-client logs.

## Required checkpoints

Expected default layout:

```bash
models/hf/RT-accompanimentV2-checkpoints/lekai_prompt_model/model.safetensors
models/hf/RT-accompanimentV2-checkpoints/lekai_continuation_model/model.safetensors
```

If Spark stores them elsewhere, override:

```bash
PROMPT_CKPT=/path/to/lekai_prompt_model/model.safetensors \
CONT_CKPT=/path/to/lekai_continuation_model/model.safetensors \
scripts/run_spark_lekai_benchmark.sh
```

## One-command run

From repository root:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_$(date +%Y%m%d_%H%M%S)

scripts/run_spark_lekai_benchmark.sh
```

If the script is not executable:

```bash
bash scripts/run_spark_lekai_benchmark.sh
```

## Output layout

The output root is split intentionally:

```text
spark_runs/<run_name>/
  micro/
    direct_micro_and_scheduler.json
    direct_micro_and_scheduler.log
  public_client/
    realtime/
      original_console.log
      trim_console.log
    prompt_continuation/
      original_console.log
      original_trace.jsonl
      trim_console.log
      trim_trace.jsonl
  summary/
    public_client_first_sound_and_drops.json
  server_logs/
    realtime_server.out.log
    realtime_server.err.log
    prompt_continuation_server.out.log
    prompt_continuation_server.err.log
```

## What to report back

From `micro/direct_micro_and_scheduler.json`:

- `prompt_model.generate_ms.mean_ms`
- `prompt_model.generate_ms.median_ms`
- `prompt_model.generate_ms.p95_ms`
- `continuation_model.round_trip_ms_per_beat.mean_ms`
- `continuation_model.round_trip_ms_per_beat.median_ms`
- `continuation_model.round_trip_ms_per_beat.p95_ms`
- `continuation_model.server_inference_ms_per_beat.mean_ms`
- `prompt_continuation_scheduler[*].total_scheduler_ms`
- `prompt_continuation_scheduler[*].final_status.continuation_calls`

From `summary/public_client_first_sound_and_drops.json`:

- `realtime_trim.first_model_seconds_at_120bpm`
- `prompt_trim.first_ready_seconds_at_120bpm`
- `prompt_trim.first_model_seconds_at_120bpm`
- `prompt_trim.scheduled_zero_count`
- `prompt_trim.dropped_before_first_model`
- `prompt_trim.total_dropped_in_schedule_reports`

## Important benchmark meanings

Direct microbenchmark:

```text
prompt_model.generate_ms = prompt model generating 8 beats
continuation_model.round_trip_ms_per_beat = direct backend call for 1 continuation beat
prompt_continuation_scheduler = prompt model + history injection + continuation catch-up
```

Public client:

```text
first_ready_seconds = backend says prompt-continuation is playable
first_model_seconds = first actual [event] source=model printed by the CLI
scheduled_zero_count = generated/fetched playable batches that scheduled 0 events
dropped_before_first_model = stale events dropped before the first audible model event
```

The public-client numbers are the relevant ones for "why did I not hear anything yet?"

## Useful overrides

Skip public-client server/CLI runs and only run direct model benchmarks:

```bash
RUN_PUBLIC_CLIENT=0 scripts/run_spark_lekai_benchmark.sh
```

Use a different MIDI:

```bash
MIDI_FILE=prompts/B_minor/pop909_007_mel.mid scripts/run_spark_lekai_benchmark.sh
```

Increase sample counts:

```bash
PROMPT_REPEATS=10 CONT_REQUESTS=200 scripts/run_spark_lekai_benchmark.sh
```

Use a different tempo for public-client simulation:

```bash
BPM=100 scripts/run_spark_lekai_benchmark.sh
```

Use different ports:

```bash
RT_PORT=8114 PROMPT_PORT=8115 scripts/run_spark_lekai_benchmark.sh
```

## Expected comparison format

Please summarize Spark/H200 results like this:

```text
Prompt model 生成 8 beats: 约 A - B 秒
Continuation 生成 1 beat: 平均约 C 秒，中位数约 D 秒，p95 约 E 秒，最慢约 F 秒
Prompt-continuation public client first ready: 约 G 秒
Prompt-continuation public client first source=model: 约 H 秒
Scheduled 0 before first model: I 次，dropped before first model: J events
```
