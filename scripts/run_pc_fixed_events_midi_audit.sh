#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
root=${1:?Usage: run_pc_fixed_events_midi_audit.sh RESULT_ROOT GPU}
gpu=${2:?GPU must be 0, 1 or 2}
case "$gpu" in 0|1|2) ;; *) exit 2 ;; esac
cd "$repo"
export PYTHONPATH="$repo/src"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
exec /data/home/yuanxin/StreamMUSE-v1/.venv/bin/python \
  scripts/pc_fixed_events_midi_audit.py worker --root "$root" --gpu "$gpu"
