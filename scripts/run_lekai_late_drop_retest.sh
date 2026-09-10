#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$repo/src:$repo"
export PYTHONUNBUFFERED=1
exec /data/home/yuanxin/StreamMUSE-v1/.venv/bin/python \
  scripts/run_lekai_late_drop_retest.py "$@"
