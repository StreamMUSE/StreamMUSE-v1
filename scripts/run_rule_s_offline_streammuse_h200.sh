#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
GPU="${GPU:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"

PROMPT_CKPT="${PROMPT_CKPT:-/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors}"
CONT_CKPT="${CONT_CKPT:-/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors}"
MIDI_DIR="${MIDI_DIR:-${REPO_ROOT}/prompts/old_input/mel}"
OUT_ROOT="${OUT_ROOT:-/data/home/yuanxin/runs/rule_s_offline_streammuse_${STAMP}}"

export CUDA_VISIBLE_DEVICES="${GPU}"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_rule_s_offline_streammuse_experiment.py" \
  --midi-dir "${MIDI_DIR}" \
  --output-dir "${OUT_ROOT}" \
  --prompt-checkpoint "${PROMPT_CKPT}" \
  --continuation-checkpoint "${CONT_CKPT}" \
  --candidate-count "${CANDIDATE_COUNT:-5}" \
  --continuation-seeds "${CONTINUATION_SEEDS:-0,1}" \
  --gpu "${GPU}" \
  "$@"
