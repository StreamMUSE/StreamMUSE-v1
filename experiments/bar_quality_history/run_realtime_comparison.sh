#!/usr/bin/env bash
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
policy=${1:-strict}
case "$policy" in
  strict) base=/data/home/yuanxin/experiments/bar_history_realtime_guarded_20261007 ;;
  preserve_cross_boundary) base=/data/home/yuanxin/experiments/bar_history_partial_repair_20261007 ;;
  *) printf 'Unknown repair policy: %s\n' "$policy" >&2; exit 2 ;;
esac
mkdir -p "$base"
run=$(mktemp -d "$base/run_XXXXXX")
cd "$root"
export PYTHONPATH="$root:$root/src:$root/transformers/src"
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
python=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python
set +e
env CUDA_VISIBLE_DEVICES= "$python" -m pytest -q \
  tests/unit/infrastructure/inference/test_history_repair.py \
  tests/unit/infrastructure/inference/test_empty_token_guard.py \
  tests/unit/infrastructure/inference/test_lekai_prompt_continuation_backend.py \
  experiments/bar_quality_history/tests \
  > "$run/tests.log" 2>&1
status=$?
if [ "$status" -ne 0 ]; then
  printf '%s\n' "$status" > "$run/exit_code.txt"
  exit "$status"
fi
"$python" -u -m experiments.bar_quality_history.run_realtime_comparison \
  --plan /data/home/yuanxin/experiments/formal_pc_beattail_stopfix_user10_test40_3seed_20260910/plan.json \
  --root "$run/results" --gpu 2 --repair-policy "$policy" > "$run/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run/exit_code.txt"
exit "$status"
