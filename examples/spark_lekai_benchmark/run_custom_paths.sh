#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-spark_runs/lekai_h200_$(date +%Y%m%d_%H%M%S)}"

: "${PROMPT_CKPT:?Set PROMPT_CKPT=/path/to/lekai_prompt_model/model.safetensors}"
: "${CONT_CKPT:?Set CONT_CKPT=/path/to/lekai_continuation_model/model.safetensors}"

bash scripts/run_spark_lekai_benchmark.sh

