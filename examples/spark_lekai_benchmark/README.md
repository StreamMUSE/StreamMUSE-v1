# Spark Lekai Benchmark Examples

This folder contains copy-paste examples for running the Lekai benchmark on Spark/H200.

Main docs:

- `docs/architecture/spark-lekai-benchmark-instructions.md`
- `scripts/run_spark_lekai_benchmark.sh`
- `scripts/benchmark_lekai_spark.py`

Included music examples:

- `music_examples/pop909_291_mel.mid`
- `music_examples/pop909_013_mel.mid`
- `music_examples/pop909_007_mel.mid`

Each melody has a matching reference accompaniment in the same folder.

## Example 1: Default Paths

Use this if the checkpoints are already at the repository defaults:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_$(date +%Y%m%d_%H%M%S)

bash scripts/run_spark_lekai_benchmark.sh
```

Default checkpoint paths:

```text
models/hf/RT-accompanimentV2-checkpoints/lekai_prompt_model/model.safetensors
models/hf/RT-accompanimentV2-checkpoints/lekai_continuation_model/model.safetensors
```

Default prompt-continuation public-client playback policy:

```bash
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=1
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES=1
LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP=1
LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS=0
```

This matches the H200 preset and tests bounded late recovery, not strict
drop-past scheduling. To intentionally test strict mode, set
`LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=0`.
To ablate active-note state recovery, set
`LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES=0`.
The representation loop checks that the server's decoded playable events and
the public client's decoded playable events have the same canonical digest.
Set `LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS=1` only when you need
full event-key traces for debugging.

## Example 2: Custom Checkpoint Paths

Use this if Spark stores the downloaded safetensors elsewhere:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_$(date +%Y%m%d_%H%M%S)

PROMPT_CKPT=/path/to/lekai_prompt_model/model.safetensors \
CONT_CKPT=/path/to/lekai_continuation_model/model.safetensors \
bash scripts/run_spark_lekai_benchmark.sh
```

## Example 3: Direct Model Speed Only

This skips public-client server/CLI runs and only measures direct prompt, continuation, and scheduler timing:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_micro_$(date +%Y%m%d_%H%M%S)

RUN_PUBLIC_CLIENT=0 \
bash scripts/run_spark_lekai_benchmark.sh
```

## Example 4: Use Included Music Example

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_music_$(date +%Y%m%d_%H%M%S)

MIDI_FILE=examples/spark_lekai_benchmark/music_examples/pop909_291_mel.mid \
bash scripts/run_spark_lekai_benchmark.sh
```

## Example 5: More Samples

Use this for a more stable latency estimate:

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=python
export OUT_ROOT=spark_runs/lekai_h200_long_$(date +%Y%m%d_%H%M%S)

PROMPT_REPEATS=10 \
CONT_REQUESTS=200 \
bash scripts/run_spark_lekai_benchmark.sh
```

## What To Send Back

Please send back:

```text
<OUT_ROOT>/micro/direct_micro_and_scheduler.json
<OUT_ROOT>/summary/public_client_first_sound_and_drops.json
<OUT_ROOT>/public_client/prompt_continuation/trim_console.log
<OUT_ROOT>/public_client/realtime/trim_console.log
```

If public-client runs are skipped, send only:

```text
<OUT_ROOT>/micro/direct_micro_and_scheduler.json
```
