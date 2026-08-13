#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-/data/home/yuanxin/runs/rule_s_v2_manual_001_005_20260813}"

PROMPT_CKPT="${PROMPT_CKPT:-/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors}"
CONT_CKPT="${CONT_CKPT:-/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors}"

declare -A INPUTS=(
  [001]="/data/home/yuanxin/runs/rule_s_npz_offline_smoke_001_20260813/input/npz/001.npz"
  [002]="/data/home/yuanxin/runs/rule_s_npz_offline_smoke_002_20260813/input/npz/002.npz"
  [003]="/data/home/yuanxin/runs/rule_s_npz_offline_manual_more_20260813/input/npz/003.npz"
  [004]="/data/home/yuanxin/runs/rule_s_npz_offline_manual_more_20260813/input/npz/004.npz"
  [005]="/data/home/yuanxin/runs/rule_s_npz_offline_manual_more_20260813/input/npz/005.npz"
)

mkdir -p "${OUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for piece_id in 001 002 003 004 005; do
  input_npz="${INPUTS[${piece_id}]}"
  if [[ ! -f "${input_npz}" ]]; then
    echo "Missing input NPZ: ${input_npz}" >&2
    exit 1
  fi
  echo "[Rule-S v2] piece=${piece_id} input=${input_npz}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_rule_s_npz_offline.py" \
    --npz-file "${input_npz}" \
    --output-dir "${OUT_ROOT}/${piece_id}/results" \
    --prompt-checkpoint "${PROMPT_CKPT}" \
    --continuation-checkpoint "${CONT_CKPT}" \
    --device cuda \
    --candidate-count 5 \
    --prompt-seed 20260813 \
    --continuation-seeds 0 \
    --prompt-prefix-beats 8 \
    --prompt-temperature 0.8 \
    --prompt-top-k 50 \
    --prompt-top-p 0.95 \
    --continuation-temperature 1.1 \
    --continuation-top-k 0 \
    --continuation-top-p 0.95 \
    --duration-weight 0.49 \
    --duration-expected-log-ratio 0.0 \
    --include-gt-accompaniment
done

echo "[Rule-S v2] complete: ${OUT_ROOT}"
