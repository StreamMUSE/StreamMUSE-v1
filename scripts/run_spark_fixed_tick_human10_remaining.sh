#!/usr/bin/env bash
set -euo pipefail

REPO=/home/xiaosongma/StreamMUSE-same-tick-fix-acceptance-20260909
PYTHON=/home/xiaosongma/StreamMUSE-v1/.venv/bin/python
PILOT=/home/xiaosongma/experiments/human10_fixed_midi_ticks_pilot_20260909
OUT=/home/xiaosongma/experiments/human10_fixed_midi_ticks_full_20260909

test -f "$PILOT/cases/01_tong_hua/comparison.json"
if test -e "$OUT"; then
  printf 'Output already exists; refusing to overwrite: %s\n' "$OUT" >&2
  exit 1
fi
mkdir "$OUT"
cp -a "$PILOT/inputs" "$PILOT/plan.json" "$OUT/"
mkdir "$OUT/pilot_01_tong_hua"
cp -a "$PILOT/cases/01_tong_hua" "$PILOT/server" "$PILOT/executed_scripts" \
  "$PILOT/README.md" "$PILOT/progress.json" "$OUT/pilot_01_tong_hua/"
cp "$0" "$OUT/launch_command.sh"
cd "$REPO"
export PYTHONPATH="$REPO/src"

# Tong Hua's two completed repeats are retained above, not regenerated.
"$PYTHON" -u scripts/fixed_tick_midi_replay.py launch \
  --root "$OUT" --continue-on-difference \
  --pieces 02_night_dancer 03_qun_qing 04_lemon 05_cruel_angel \
  06_gao_bai_qi_qiu 07_qing_hua_ci 08_xiao_xing_yun \
  09_yue_liang_dai_biao_wo_de_xin 10_castle_in_the_sky \
  > "$OUT/launch.log" 2>&1
