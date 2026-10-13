#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX="${RUN_PREFIX:-rule_s_best_of_5_new4_heldout}"
export NPZ_IDS="${NPZ_IDS:-5817083,332891,43038,2581346}"
export CANDIDATE_COUNT="${CANDIDATE_COUNT:-5}"
export PROMPT_BATCH_SEED="${PROMPT_BATCH_SEED:-20260727}"
export CONTINUATION_SEEDS="${CONTINUATION_SEEDS:-0,1}"
export DEVICE="${DEVICE:-7}"

exec bash "${SCRIPT_DIR}/run_rule_s_best_of_n.sh" "$@"
