#!/usr/bin/env bash
set -euo pipefail

STREAMMUSE_ROOT=${STREAMMUSE_ROOT:-/data/home/yuanxin/StreamMUSE-v1}
RT_ROOT=${RT_ROOT:-/data/home/yuanxin/RT-accompanimentV2}
STREAMMUSE_PYTHON=${STREAMMUSE_PYTHON:-$STREAMMUSE_ROOT/.venv/bin/python}
RT_PYTHON=${RT_PYTHON:-$RT_ROOT/.venv/bin/python}
INPUT_ROOT=${INPUT_ROOT:-$RT_ROOT/formal_pop909_sanity_20260710/inputs/scored19}
CHECKPOINT=${CHECKPOINT:-$RT_ROOT/checkpoints-resume/epoch_15_0307_1858/model.safetensors}
RUN_TAG=${RUN_TAG:-task1_v2_greedy_nucleus_new4_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$STREAMMUSE_ROOT/output/lekai_melody_timing_robustness/$RUN_TAG}
GT_PREFIX_BEATS=${GT_PREFIX_BEATS:-8}
GPUS=${GPUS:-4,5,6,7}

PIECE_IDS=(5889045 5817083 332891 5472153)
CONDITIONS=(baseline onset_tick1_pct10_seed0 onset_tick1_pct50_seed0)
SAMPLING_SEEDS=(0 1 2)

PREPARED_ROOT=$RUN_ROOT/prepared
SINGLETON_ROOT=$RUN_ROOT/singleton_inputs
INFERENCE_ROOT=$RUN_ROOT/inference
LOG_ROOT=$RUN_ROOT/logs

export PYTHONPATH=$STREAMMUSE_ROOT/src:$STREAMMUSE_ROOT
mkdir -p "$LOG_ROOT"

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" \
  experiments/lekai_melody_timing_robustness/prepare_task1_v2_dataset.py \
  --input-root "$INPUT_ROOT" \
  --output-root "$PREPARED_ROOT"

for piece_id in "${PIECE_IDS[@]}"; do
  for condition in "${CONDITIONS[@]}"; do
    stem=${piece_id}__${condition}
    input_dir=$SINGLETON_ROOT/$stem/input_npz
    mkdir -p "$input_dir"
    cp "$PREPARED_ROOT/npz/$stem.npz" "$input_dir/$stem.npz"
    printf '[0]\n' > "$SINGLETON_ROOT/$stem/indices.json"
  done
done

run_one() {
  local gpu=$1
  local decoder=$2
  local decoder_seed=$3
  local temperature=$4
  local top_k=$5
  local top_p=$6
  local piece_id=$7
  local condition=$8
  local stem=${piece_id}__${condition}
  local singleton=$SINGLETON_ROOT/$stem
  local output_dir=$INFERENCE_ROOT/$decoder/seed$decoder_seed/$piece_id/$condition

  cd "$RT_ROOT"
  CUDA_VISIBLE_DEVICES=$gpu "$RT_PYTHON" inference_new.py \
    --ckpt "$CHECKPOINT" \
    --device cuda \
    --use-fp16 \
    --data-dir "$singleton/input_npz" \
    --num-samples 1 \
    --indices-file "$singleton/indices.json" \
    --sampling-mode without_replacement \
    --dataset-mode test \
    --test-split-ratio 1.0 \
    --dataset-seed 0 \
    --seed "$decoder_seed" \
    --gt-prefix-beats "$GT_PREFIX_BEATS" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --repetition-penalty 1.0 \
    --export-gt-midi \
    --no-prepare-only \
    --output-dir "$output_dir"
}

run_piece() {
  local gpu=$1
  local piece_id=$2
  for condition in "${CONDITIONS[@]}"; do
    run_one "$gpu" greedy 0 1.0 1 1.0 "$piece_id" "$condition"
    for seed in "${SAMPLING_SEEDS[@]}"; do
      run_one "$gpu" sampling "$seed" 1.1 0 0.95 "$piece_id" "$condition"
    done
  done
}

IFS=',' read -ra gpu_ids <<< "$GPUS"
if [[ "${#gpu_ids[@]}" -ne "${#PIECE_IDS[@]}" ]]; then
  echo "GPUS must provide exactly ${#PIECE_IDS[@]} comma-separated GPU ids" >&2
  exit 1
fi

pids=()
for index in "${!PIECE_IDS[@]}"; do
  piece_id=${PIECE_IDS[$index]}
  gpu=${gpu_ids[$index]}
  (run_piece "$gpu" "$piece_id") > "$LOG_ROOT/$piece_id.log" 2>&1 &
  pids+=("$!")
  printf 'launched piece=%s gpu=%s pid=%s\n' "$piece_id" "$gpu" "$!"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one piece run failed; inspect $LOG_ROOT" >&2
  exit "$status"
fi

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" \
  experiments/lekai_melody_timing_robustness/analyze_task1_v2_outputs.py \
  --prepared-manifest "$PREPARED_ROOT/manifest.json" \
  --inference-root "$INFERENCE_ROOT" \
  --output-dir "$RUN_ROOT/analysis" \
  --ignore-prefix-beats "$GT_PREFIX_BEATS"

printf 'Task 1 v2 output: %s\n' "$RUN_ROOT"
