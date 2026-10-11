#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEKAI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export RUN_PREFIX="${RUN_PREFIX:-q1_top7}"
export NPZ_IDS="${NPZ_IDS:-5889045,5817083,332891,43038,5472153,3410421,2581346}"
export PROMPT_SEEDS="${PROMPT_SEEDS:-0,1,2}"
export CONTINUATION_SEEDS="${CONTINUATION_SEEDS:-0,1,2,3}"
export BLIND_SEED="${BLIND_SEED:-20260724}"
export OUT_ROOT="${OUT_ROOT:-${LEKAI_ROOT}/q1_prompt_batch_selection_runs/top7}"

exec bash "${SCRIPT_DIR}/run_q1_pilot.sh" "$@"
