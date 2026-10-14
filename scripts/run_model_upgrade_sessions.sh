#!/bin/bash
# A3/B2 online sessions (plan 2026-10-14): for each writer x TTS combination, start the stack on the
# H200, then record a 90 BPM 96-bar session with 8 live edits and a 120 BPM 48-bar session with 4.
# usage: scripts/run_model_upgrade_sessions.sh <gpu> <combo>...   (combo = writer:tts, e.g. qwen38:mosslocal)
set -u
cd "$(dirname "$0")/.."
gpu=$1; shift
O=output/model_upgrade_20261014/sessions; mkdir -p $O
EDITS90="12:topic=volcano;20@1:template=baseline_straight_9;28:topic=city nights,template=baseline_syncopated_9;36@1:topic=ocean;52:topic=desert;60@1:topic=robots,template=baseline_staggered_9;76:template=baseline_straight_9;84@1:topic=rain"
EDITS120="12:topic=volcano;20@1:template=baseline_straight_9;28:topic=city nights,template=baseline_syncopated_9;36@1:topic=ocean"
for combo in "$@"; do
  writer=${combo%%:*}; tts=${combo##*:}
  echo "== $combo $(date)"
  ssh -o ConnectTimeout=20 mbzuai-masdar "cd ~/mbzuai-projects/StreamMUSE-v1 && git pull -q && bash scripts/model_upgrade_stack.sh start $writer $tts $gpu" | tail -2
  curl -s -m 5 -o /dev/null -w "health %{http_code}\n" http://127.0.0.1:18620/health
  for spec in "90 96 $EDITS90" "120 48 $EDITS120"; do
    read tempo bars edits <<< "$spec"
    PATH=/opt/homebrew/bin:$PATH .venv/bin/streammuse-rap-demo --rap-audio-renderer moss_aligned_remote --rap-protocol v3 \
      --rap-audio-transport opus --rap-render-url http://127.0.0.1:18620 --rap-render-startup-timeout 120 \
      --audio-output wav --tempo $tempo --max-bars $bars --log-dir $O/$writer-$tts-$tempo --terminal-detail summary --no-web \
      --rap-scripted-edits "$edits" > $O/$writer-$tts-$tempo.out 2>&1
    echo "session $writer-$tts-$tempo exit=$?"
  done
done
ssh -o ConnectTimeout=20 mbzuai-masdar "cd ~/mbzuai-projects/StreamMUSE-v1 && bash scripts/model_upgrade_stack.sh stop"
echo SESSIONS_DONE
