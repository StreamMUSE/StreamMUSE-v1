#!/usr/bin/env bash
set -euo pipefail

STREAMMUSE_ROOT=${STREAMMUSE_ROOT:-/data/home/yuanxin/StreamMUSE-v1}
RT_ROOT=${RT_ROOT:-/data/home/yuanxin/RT-accompanimentV2}
STREAMMUSE_PYTHON=${STREAMMUSE_PYTHON:-$STREAMMUSE_ROOT/.venv/bin/python}
RT_PYTHON=${RT_PYTHON:-$RT_ROOT/.venv/bin/python}
CHECKPOINT=${CHECKPOINT:-$RT_ROOT/checkpoints-resume/epoch_15_0307_1858/model.safetensors}
PREPARED_ROOT=${PREPARED_ROOT:-$STREAMMUSE_ROOT/output/lekai_cross_style/task3_frozen20_input_preflight_20260802/prepared}
RUN_TAG=${RUN_TAG:-task3_sampling_frozen20_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$STREAMMUSE_ROOT/output/lekai_cross_style/$RUN_TAG}
GT_PREFIX_BEATS=${GT_PREFIX_BEATS:-8}
SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-1.1}
SAMPLING_TOP_K=${SAMPLING_TOP_K:-0}
SAMPLING_TOP_P=${SAMPLING_TOP_P:-0.95}
GPUS=${GPUS:-3,4,5,6}

RUNS=(greedy:0 sampling:0 sampling:1 sampling:2)
EXPECTED_FINGERPRINT=05533801a0f7bbce7f640d623bf456fc6caf630c
LOG_ROOT=$RUN_ROOT/logs
INFERENCE_ROOT=$RUN_ROOT/inference

export PYTHONPATH=$STREAMMUSE_ROOT/src:$STREAMMUSE_ROOT
mkdir -p "$LOG_ROOT"

for path in \
  "$CHECKPOINT" \
  "$PREPARED_ROOT/manifest.json" \
  "$PREPARED_ROOT/indices.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$PREPARED_ROOT/input_npz" ]]; then
  echo "Missing prepared input directory: $PREPARED_ROOT/input_npz" >&2
  exit 1
fi

run_condition() {
  local gpu=$1
  local decoder=$2
  local seed=$3
  local temperature=$SAMPLING_TEMPERATURE
  local top_k=$SAMPLING_TOP_K
  local top_p=$SAMPLING_TOP_P
  if [[ "$decoder" == "greedy" ]]; then
    temperature=1.0
    top_k=1
    top_p=1.0
  fi
  local output_dir=$INFERENCE_ROOT/$decoder/seed$seed
  cd "$RT_ROOT"
  CUDA_VISIBLE_DEVICES=$gpu "$RT_PYTHON" inference_new.py \
    --ckpt "$CHECKPOINT" \
    --device cuda \
    --use-fp16 \
    --data-dir "$PREPARED_ROOT/input_npz" \
    --num-samples 20 \
    --indices-file "$PREPARED_ROOT/indices.json" \
    --sampling-mode without_replacement \
    --dataset-mode test \
    --test-split-ratio 1.0 \
    --dataset-seed 0 \
    --seed "$seed" \
    --gt-prefix-beats "$GT_PREFIX_BEATS" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --repetition-penalty 1.0 \
    --export-gt-midi \
    --no-prepare-only \
    --output-dir "$output_dir"
}

IFS=',' read -ra gpu_ids <<< "$GPUS"
if [[ "${#gpu_ids[@]}" -ne "${#RUNS[@]}" ]]; then
  echo "GPUS must provide exactly ${#RUNS[@]} comma-separated GPU ids" >&2
  exit 1
fi

pids=()
for index in "${!RUNS[@]}"; do
  IFS=':' read -r decoder seed <<< "${RUNS[$index]}"
  gpu=${gpu_ids[$index]}
  (run_condition "$gpu" "$decoder" "$seed") \
    > "$LOG_ROOT/${decoder}_seed$seed.log" 2>&1 &
  pids+=("$!")
  printf 'prepared launch decoder=%s seed=%s gpu=%s pid=%s\n' \
    "$decoder" "$seed" "$gpu" "$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one Task 3 seed failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" experiments/lekai_cross_style/analyze_task3_outputs.py \
  --prepared-manifest "$PREPARED_ROOT/manifest.json" \
  --inference-root "$INFERENCE_ROOT" \
  --output-dir "$RUN_ROOT/analysis" \
  --ignore-prefix-beats "$GT_PREFIX_BEATS" \
  --expected-fingerprint "$EXPECTED_FINGERPRINT"

printf 'Task 3 output: %s\n' "$RUN_ROOT"
