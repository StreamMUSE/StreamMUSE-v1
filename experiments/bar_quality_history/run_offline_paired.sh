#!/usr/bin/env bash
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
base=/data/home/yuanxin/experiments/bar_history_offline_user10_20261007
mkdir -p "$base"
run=$(mktemp -d "$base/run_XXXXXX")
cd "$root"
export PYTHONPATH="$root:$root/src:$root/transformers/src"
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
python=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python
trap 'printf "%s\n" "$?" > "$run/exit_code.txt"' EXIT
env CUDA_VISIBLE_DEVICES= "$python" -m pytest -q \
  tests/unit/infrastructure/inference/test_history_repair.py \
  tests/unit/infrastructure/inference/test_empty_token_guard.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_backend.py \
  experiments/bar_quality_history/tests > "$run/tests.log" 2>&1
"$python" -u -m experiments.bar_quality_history.run_offline_paired \
  --plan /data/home/yuanxin/experiments/formal_pc_beattail_stopfix_user10_test40_3seed_20260910/plan.json \
  --output "$run/results" --gpu 2 > "$run/run.log" 2>&1
"$python" -m experiments.bar_quality_history.analyze_offline_paired \
  --root "$run/results" --eval-root /data/home/yuanxin/StreamMUSE-eval-metrics-20260902 \
  > "$run/metrics.log" 2>&1
