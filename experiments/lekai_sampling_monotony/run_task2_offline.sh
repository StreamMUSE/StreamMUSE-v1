#!/usr/bin/env bash
set -euo pipefail

STREAMMUSE_ROOT=${STREAMMUSE_ROOT:-/data/home/yuanxin/StreamMUSE-v1}
RT_ROOT=${RT_ROOT:-/data/home/yuanxin/RT-accompanimentV2}
STREAMMUSE_PYTHON=${STREAMMUSE_PYTHON:-$STREAMMUSE_ROOT/.venv/bin/python}
RT_PYTHON=${RT_PYTHON:-$RT_ROOT/.venv/bin/python}
CHECKPOINT=${CHECKPOINT:-$RT_ROOT/checkpoints-resume/epoch_15_0307_1858/model.safetensors}
SOURCE_NPZ_ROOT=${SOURCE_NPZ_ROOT:-$STREAMMUSE_ROOT/output/lekai_melody_timing_robustness/task1_grid_onset_gtprompt8_greedy_20260801/prepared/npz}
RUN_TAG=${RUN_TAG:-task2_greedy_vs_nucleus_sampling_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$STREAMMUSE_ROOT/output/lekai_sampling_monotony/$RUN_TAG}
GPU=${GPU:-7}
GT_PREFIX_BEATS=${GT_PREFIX_BEATS:-8}
SAMPLING_SEEDS=${SAMPLING_SEEDS:-0,1,2,3,4}
SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-1.1}
SAMPLING_TOP_K=${SAMPLING_TOP_K:-0}
SAMPLING_TOP_P=${SAMPLING_TOP_P:-0.95}

export PYTHONPATH=$STREAMMUSE_ROOT/src:$STREAMMUSE_ROOT
export CUDA_VISIBLE_DEVICES=$GPU

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" experiments/lekai_sampling_monotony/prepare_task2_inputs.py \
  --source-root "$SOURCE_NPZ_ROOT" \
  --output-root "$RUN_ROOT/prepared"

run_inference() {
  local mode=$1
  local seed=$2
  local temperature=$3
  local top_k=$4
  local top_p=$5
  local output_dir=$RUN_ROOT/inference/$mode/seed$seed

  cd "$RT_ROOT"
  "$RT_PYTHON" inference_new.py \
    --ckpt "$CHECKPOINT" \
    --device cuda \
    --use-fp16 \
    --data-dir "$RUN_ROOT/prepared/input_npz" \
    --num-samples 3 \
    --indices-file "$RUN_ROOT/prepared/indices.json" \
    --dataset-mode test \
    --test-split-ratio 1.0 \
    --dataset-seed 0 \
    --seed "$seed" \
    --gt-prefix-beats "$GT_PREFIX_BEATS" \
    --temperature "$temperature" \
    --top-k "$top_k" \
    --top-p "$top_p" \
    --repetition-penalty 1.0 \
    --output-dir "$output_dir"
}

run_inference greedy 0 1.0 1 1.0
IFS=',' read -ra seeds <<< "$SAMPLING_SEEDS"
for seed in "${seeds[@]}"; do
  run_inference sampling "$seed" "$SAMPLING_TEMPERATURE" "$SAMPLING_TOP_K" "$SAMPLING_TOP_P"
done

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" experiments/lekai_sampling_monotony/analyze_task2_outputs.py \
  --inference-root "$RUN_ROOT/inference" \
  --output-dir "$RUN_ROOT/analysis" \
  --ignore-prefix-beats "$GT_PREFIX_BEATS"

printf 'Task 2 output: %s\n' "$RUN_ROOT"
