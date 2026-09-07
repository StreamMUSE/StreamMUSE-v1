#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/data/home/yuanxin/experiments/human10_fullsong_trustedbase_2condition_3seed_20260907/manifests}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/data/home/yuanxin/experiments/human10_fullsong_trustedbase_2condition_3seed_20260907}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/yuanxin/experiments/human10_fullsong_single_unconstrained_3seed_20260907/full}"
PROMPT_CHECKPOINT="${PROMPT_CHECKPOINT:-/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors}"
CONTINUATION_CHECKPOINT="${CONTINUATION_CHECKPOINT:-/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT/worker_logs"

window_end_tick() {
  local piece="$1"
  "$PYTHON_BIN" -c \
    'import json, sys; print(json.load(open(sys.argv[1]))["evaluation_contract"]["window_end_tick_exclusive"])' \
    "$REFERENCE_ROOT/runs/single_n1/$piece/run_manifest.json"
}

run_piece() {
  local gpu="$1"
  local piece="$2"
  local end_tick
  end_tick="$(window_end_tick "$piece")"

  "$PYTHON_BIN" "$REPO_ROOT/scripts/run_matched_system_eval.py" \
    --cohort-manifest "$MANIFEST_ROOT/$piece.json" \
    --output-root "$OUTPUT_ROOT/runs/single_unconstrained_n1/$piece" \
    --prompt-checkpoint "$PROMPT_CHECKPOINT" \
    --continuation-checkpoint "$CONTINUATION_CHECKPOINT" \
    --seeds 0,1,2 \
    --systems streammuse_v2_prompt_continuation \
    --gpu "$gpu" \
    --python-bin "$PYTHON_BIN" \
    --playback-bpm 90 \
    --model-condition-bpm 80 \
    --ticks-per-beat 4 \
    --window-end-tick "$end_tick" \
    --prompt-beats 8 \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --prompt-temperature 1.1 \
    --prompt-top-p 0.95 \
    --prompt-top-k 50 \
    --prompt-repetition-penalty 1.0 \
    --continuation-temperature 1.1 \
    --continuation-top-p 0.95 \
    --continuation-top-k 50 \
    --continuation-repetition-penalty 1.0 \
    --continuation-tonal-constraint 0 \
    --continuation-empty-token-guard 0 \
    --prompt-selection-mode single \
    --prompt-batch-candidates 1 \
    --server-start-timeout-s 600 \
    --trial-timeout-s 900
}

run_worker() {
  local gpu="$1"
  shift
  local piece
  for piece in "$@"; do
    echo "[gpu $gpu] starting piece $piece"
    run_piece "$gpu" "$piece"
    echo "[gpu $gpu] completed piece $piece"
  done
}

run_worker 0 01 04 07 10 >"$OUTPUT_ROOT/worker_logs/gpu0.log" 2>&1 &
pid0=$!
run_worker 1 02 05 08 >"$OUTPUT_ROOT/worker_logs/gpu1.log" 2>&1 &
pid1=$!
run_worker 2 03 06 09 >"$OUTPUT_ROOT/worker_logs/gpu2.log" 2>&1 &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1

if [[ "$status" -ne 0 ]]; then
  echo "One or more workers failed; inspect $OUTPUT_ROOT/worker_logs" >&2
  exit "$status"
fi

echo "All human10 single-unconstrained trials completed: $OUTPUT_ROOT"
