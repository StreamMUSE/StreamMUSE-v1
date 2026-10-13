#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEKAI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RT_ROOT="$(cd "${LEKAI_ROOT}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${DEVICE:-7}"
PYTHON_BIN="${PYTHON_BIN:-${RT_ROOT}/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-${LEKAI_ROOT}/prompt_8_vs_9_runs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_prompt_8_vs_9.py" \
  --device cuda \
  --fp16 \
  --run-prefix "${RUN_PREFIX:-prompt_8_vs_9_seven}" \
  --npz-ids "${NPZ_IDS:-5889045,5817083,332891,43038,5472153,3410421,2581346}" \
  --prompt-seed-base "${PROMPT_SEED_BASE:-20260727}" \
  --max-prompt-attempts "${MAX_PROMPT_ATTEMPTS:-20}" \
  --continuation-seeds "${CONTINUATION_SEEDS:-0,1,2}" \
  --prompt-temperature "${PROMPT_TEMPERATURE:-0.8}" \
  --prompt-top-k "${PROMPT_TOP_K:-50}" \
  --prompt-top-p "${PROMPT_TOP_P:-0.95}" \
  --prompt-repetition-penalty "${PROMPT_REPETITION_PENALTY:-1.0}" \
  --continuation-temperature "${CONTINUATION_TEMPERATURE:-1.1}" \
  --continuation-top-k "${CONTINUATION_TOP_K:-0}" \
  --continuation-top-p "${CONTINUATION_TOP_P:-0.95}" \
  --continuation-repetition-penalty "${CONTINUATION_REPETITION_PENALTY:-1.0}" \
  --blind-seed "${BLIND_SEED:-20260727}" \
  --output-dir "${OUT_ROOT}" \
  "$@"
