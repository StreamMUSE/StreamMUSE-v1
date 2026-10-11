#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEKAI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RT_ROOT="$(cd "${LEKAI_ROOT}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES="${DEVICE:-2}"
PYTHON_BIN="${PYTHON_BIN:-${RT_ROOT}/.venv/bin/python}"
OUT_ROOT="${OUT_ROOT:-${LEKAI_ROOT}/q1_prompt_batch_selection_runs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_q1_pilot.py" \
  --device cuda \
  --fp16 \
  --run-prefix "${RUN_PREFIX:-q1_pilot}" \
  --npz-ids "${NPZ_IDS:-5889045,5472153,3410421}" \
  --prompt-seeds "${PROMPT_SEEDS:-0,1,2}" \
  --continuation-seeds "${CONTINUATION_SEEDS:-0,1,2,3}" \
  --prompt-prefix-beats "${PROMPT_PREFIX_BEATS:-8}" \
  --prompt-num-bars "${PROMPT_NUM_BARS:-2}" \
  --prompt-temperature "${PROMPT_TEMPERATURE:-1.1}" \
  --prompt-top-k "${PROMPT_TOP_K:-0}" \
  --prompt-top-p "${PROMPT_TOP_P:-0.95}" \
  --prompt-repetition-penalty "${PROMPT_REPETITION_PENALTY:-1.0}" \
  --continuation-temperature "${CONTINUATION_TEMPERATURE:-1.1}" \
  --continuation-top-k "${CONTINUATION_TOP_K:-0}" \
  --continuation-top-p "${CONTINUATION_TOP_P:-0.95}" \
  --continuation-repetition-penalty "${CONTINUATION_REPETITION_PENALTY:-1.0}" \
  --blind-seed "${BLIND_SEED:-20260722}" \
  --output-dir "${OUT_ROOT}" \
  "$@"
