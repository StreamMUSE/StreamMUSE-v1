#!/usr/bin/env bash
set -euo pipefail

STREAMMUSE_ROOT="${STREAMMUSE_ROOT:-/data/home/yuanxin/StreamMUSE-v1}"
NPZ_ROOT="${NPZ_ROOT:-/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/home/yuanxin/data/lekai_failcase_dataset_v2_trimmed_midi}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTION_JSONL="${SELECTION_JSONL:-${STREAMMUSE_ROOT}/experiments/lekai_failcase_dataset_v2/provisional_selection.jsonl}"

cd "${STREAMMUSE_ROOT}"
export PYTHONPATH="${STREAMMUSE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" experiments/lekai_failcase_dataset_v2/prepare_trimmed_melody_midi.py \
  --selection-jsonl "${SELECTION_JSONL}" \
  --npz-root "${NPZ_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
