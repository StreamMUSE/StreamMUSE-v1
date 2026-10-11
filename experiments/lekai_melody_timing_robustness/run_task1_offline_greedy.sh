#!/usr/bin/env bash
set -euo pipefail

STREAMMUSE_ROOT=${STREAMMUSE_ROOT:-/data/home/yuanxin/StreamMUSE-v1}
RT_ROOT=${RT_ROOT:-/data/home/yuanxin/RT-accompanimentV2}
STREAMMUSE_PYTHON=${STREAMMUSE_PYTHON:-$STREAMMUSE_ROOT/.venv/bin/python}
RT_PYTHON=${RT_PYTHON:-$RT_ROOT/.venv/bin/python}
INPUT_ROOT=${INPUT_ROOT:-$RT_ROOT/formal_pop909_sanity_20260710/inputs/scored19}
CHECKPOINT=${CHECKPOINT:-$RT_ROOT/checkpoints-resume/epoch_15_0307_1858/model.safetensors}
GPU=${GPU:-0}
RUN_TAG=${RUN_TAG:-task1_grid_onset_$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-$STREAMMUSE_ROOT/output/lekai_melody_timing_robustness/$RUN_TAG}
GT_PREFIX_BEATS=${GT_PREFIX_BEATS:-8}

PREPARED_ROOT=$RUN_ROOT/prepared
INFERENCE_ROOT=$RUN_ROOT/inference

export PYTHONPATH=$STREAMMUSE_ROOT/src:$STREAMMUSE_ROOT
export CUDA_VISIBLE_DEVICES=$GPU

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" experiments/lekai_melody_timing_robustness/prepare_task1_dataset.py \
  --input-root "$INPUT_ROOT" \
  --output-root "$PREPARED_ROOT" \
  --selection-rates "${SELECTION_RATES:-0.10,0.25,0.50}" \
  --seeds "${PERTURBATION_SEEDS:-0,1,2}" \
  --shift-steps "${SHIFT_STEPS:-1}"

sample_count=$(find "$PREPARED_ROOT/npz" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [[ "$sample_count" -le 0 ]]; then
  echo "No prepared NPZ files found" >&2
  exit 1
fi

cd "$RT_ROOT"
"$RT_PYTHON" inference_new.py \
  --ckpt "$CHECKPOINT" \
  --device cuda \
  --use-fp16 \
  --data-dir "$PREPARED_ROOT/npz" \
  --num-samples "$sample_count" \
  --sampling-mode without_replacement \
  --dataset-mode test \
  --test-split-ratio 1.0 \
  --dataset-seed 0 \
  --seed 0 \
  --gt-prefix-beats "$GT_PREFIX_BEATS" \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repetition-penalty 1.0 \
  --output-dir "$INFERENCE_ROOT"

cd "$STREAMMUSE_ROOT"
"$STREAMMUSE_PYTHON" experiments/lekai_melody_timing_robustness/analyze_task1_outputs.py \
  --prepared-manifest "$PREPARED_ROOT/manifest.json" \
  --inference-dir "$INFERENCE_ROOT" \
  --output-dir "$RUN_ROOT/analysis" \
  --ignore-prefix-beats "$GT_PREFIX_BEATS"

printf 'Task 1 output: %s\n' "$RUN_ROOT"
