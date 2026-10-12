#!/usr/bin/env bash
# Oracle arm (GT-history continuation). Usage:
#   run_oracle.sh <gpu> <limit_pieces> <gt_manifest> <plan> <cohort> [seeds...]
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
base=/data/home/yuanxin/experiments/oracle_20261012
gpu="${1:?gpu}"; limit="${2:?limit_pieces}"; manifest="${3:?gt_manifest}"
plan="${4:?plan}"; cohort="${5:?cohort}"; shift 5
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
"$python" -u -m experiments.oracle.run_oracle \
  --plan "$plan" --cohort "$cohort" \
  --gt-manifest "$manifest" \
  --output "$run/results" --gpu "$gpu" --limit-pieces "$limit" --seeds $seeds > "$run/run.log" 2>&1
"$python" -u -m experiments.chord_law.score_law_ab \
  --root "$run/results" --output "$run/scores.json" > "$run/scores.log" 2>&1
cat "$run/scores.log"
