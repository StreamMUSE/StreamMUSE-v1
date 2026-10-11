#!/usr/bin/env bash
# Strong-beat triad boost A/B/C on user10. Usage: run_law_ab.sh <gpu> <limit_pieces> [seeds...]
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
base=/data/home/yuanxin/experiments/chord_law_20261008
gpu="${1:?gpu}"; limit="${2:?limit_pieces}"; shift 2
seeds="${*:-0 1 2}"
mkdir -p "$base"
run=$(mktemp -d "$base/run_XXXXXX")
cd "$root"
export PYTHONPATH="$root:$root/src:$root/transformers/src"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONDONTWRITEBYTECODE=1
python=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python
trap 'printf "%s\n" "$?" > "$run/exit_code.txt"' EXIT
env CUDA_VISIBLE_DEVICES= "$python" -m pytest -q \
  tests/unit/infrastructure/inference/test_tonal_constraint.py \
  tests/unit/infrastructure/inference/test_empty_token_guard.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_backend.py \
  > "$run/tests.log" 2>&1
"$python" -u -m experiments.chord_law.run_law_ab \
  --plan /data/home/yuanxin/experiments/formal_pc_beattail_stopfix_user10_test40_3seed_20260910/plan.json \
  --output "$run/results" --gpu "$gpu" --limit-pieces "$limit" --seeds $seeds > "$run/run.log" 2>&1
"$python" -u -m experiments.chord_law.score_law_ab \
  --root "$run/results" --output "$run/scores.json" > "$run/scores.log" 2>&1
cat "$run/scores.log"
