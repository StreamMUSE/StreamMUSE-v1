#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export OUT_ROOT="${OUT_ROOT:-spark_runs/lekai_h200_$(date +%Y%m%d_%H%M%S)}"

bash scripts/run_spark_lekai_benchmark.sh

