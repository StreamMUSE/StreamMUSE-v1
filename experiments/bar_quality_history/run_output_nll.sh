#!/usr/bin/env bash
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
source=/data/home/yuanxin/experiments/bar_history_offline_user10_20261007/run_IkyAZT/results
base=/data/home/yuanxin/experiments/bar_history_offline_user10_20261007
run=$(mktemp -d "$base/nll_XXXXXX")
cd "$root"
export PYTHONPATH="$root:$root/src:$root/transformers/src"
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
python=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python
trap 'printf "%s\n" "$?" > "$run/exit_code.txt"' EXIT
env CUDA_VISIBLE_DEVICES= "$python" -m pytest -q \
  experiments/bar_quality_history/tests/test_output_nll.py > "$run/tests.log" 2>&1
"$python" -u -m experiments.bar_quality_history.evaluate_output_nll \
  --root "$source" --output "$run/results" --gpu 2 --batch-size 8 > "$run/run.log" 2>&1
