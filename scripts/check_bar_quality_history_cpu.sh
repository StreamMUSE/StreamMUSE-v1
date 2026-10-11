#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -B -m unittest discover \
  -s experiments/bar_quality_history/tests -p 'test_*.py' -v
"$python_bin" -B experiments/bar_quality_history/harness.py \
  experiments/bar_quality_history/example.synthetic.json
