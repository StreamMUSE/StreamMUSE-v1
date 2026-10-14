#!/bin/bash
# TTS candidates for the model-upgrade plan (2026-10-14), served by SGLang-Omni 0.1.8.
# usage: model_upgrade_tts.sh start <model-path> <gpu> [sgl-omni serve args...] | stop
# Serves on 127.0.0.1:8501; one candidate at a time. Stops only the recorded process group.
set -u
H=/data/home/bowenzheng/mbzuai-projects
S=$H/model-envs/sglang-omni
B=$H/StreamMUSE-v1/logs/sglang_moss_benchmark_20260911
E=$H/StreamMUSE-v1/logs/model_upgrade_20261014/tts; mkdir -p $E
stop() {
  [ -f $E/server.pid ] || return 0; read pid pgid owner < $E/server.pid
  if [ "$owner" = "$(id -un)" ] && kill -0 -$pgid 2>/dev/null; then
    kill -INT -$pgid; for i in $(seq 1 60); do kill -0 -$pgid 2>/dev/null || break; sleep 1; done
    kill -0 -$pgid 2>/dev/null && kill -TERM -$pgid; sleep 2; kill -0 -$pgid 2>/dev/null && kill -KILL -$pgid
  fi; rm -f $E/server.pid; echo "stopped candidate tts"
}
if [ "$1" = stop ]; then stop; exit 0; fi
model=$2; gpu=$3; shift 3
stop
tag=$(echo $model | tr '/' '_')
if [ "$model" = IndexTeam/IndexTTS-2.5 ]; then
  # Not served by SGLang-Omni: its own Python 3.11 uv environment and a small wrapper server.
  I=$H/model-envs/index-tts
  cd $I
  setsid env HF_HOME=$H/models/huggingface CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$I \
    $I/.venv/bin/python $H/StreamMUSE-v1/scripts/indextts_server.py \
    --model-dir $(ls -d $H/models/huggingface/hub/models--IndexTeam--IndexTTS-2.5/snapshots/*/ | head -1) \
    --port 8501 > $E/$tag.log 2>&1 < /dev/null &
else
  cd $S
  setsid env HF_HOME=$H/models/huggingface CUDA_VISIBLE_DEVICES=$gpu SGLANG_OMNI_STARTUP_TIMEOUT=2400 SGLANG_OMNI_WARMUP_TIMEOUT=2400 \
    PATH=$S/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    $S/.venv/bin/sgl-omni serve --model-path $model --host 127.0.0.1 --port 8501 \
    --allowed-local-media-path $B/media "$@" > $E/$tag.log 2>&1 < /dev/null &
fi
pid=$!; echo "$pid $(ps -o pgid= -p $pid | tr -d ' ') $(id -un)" > $E/server.pid
for i in $(seq 1 540); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8501/health)" = 200 ] && { echo "ready $model"; exit 0; }
  kill -0 $pid 2>/dev/null || { echo "server exited"; tail -40 $E/$tag.log; exit 1; }; sleep 5
done
echo "timeout"; tail -40 $E/$tag.log; exit 1
