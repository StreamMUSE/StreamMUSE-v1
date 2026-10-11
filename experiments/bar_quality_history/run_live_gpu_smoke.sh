#!/usr/bin/env bash
set -euo pipefail
root=/data/home/yuanxin/StreamMUSE-bar-quality-history-experiment
base=/data/home/yuanxin/experiments/bar_history_live_20261007
mkdir -p "$base"
run=$(mktemp -d "$base/run_XXXXXX")
printf '%s\n' "$run"
cd "$root"
export PYTHONPATH="$root:$root/src:$root/transformers/src"
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
python=/data/home/yuanxin/StreamMUSE-v1/.venv/bin/python
set +e
"$python" -u -m experiments.bar_quality_history.live_gpu_smoke \
  --config /data/home/yuanxin/experiments/bar_history_probe_20261007/tonghua_seed0/config.json \
  --melody "$base/inputs/melody.mid" --history "$base/inputs/history.mid" \
  --output "$run/results" --gpu 2 --playback-bpm 90 > "$run/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run/exit_code.txt"
exit "$status"
