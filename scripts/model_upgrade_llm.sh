#!/bin/bash
# Candidate lyric LLMs for the model-upgrade plan (2026-10-14).
# usage: llm.sh start <model> <gpu> <gpu_memory_utilization> [extra vllm args...] | stop
# Serves on 127.0.0.1:8401 as "cand"; one candidate at a time. Stops only the recorded process group.
set -u
H=/data/home/bowenzheng/mbzuai-projects
E=$H/StreamMUSE-v1/logs/model_upgrade_20261014/llm; mkdir -p $E
stop() {
  [ -f $E/vllm.pid ] || return 0; read pid pgid owner < $E/vllm.pid
  if [ "$owner" = "$(id -un)" ] && kill -0 -$pgid 2>/dev/null; then
    kill -INT -$pgid; for i in $(seq 1 60); do kill -0 -$pgid 2>/dev/null || break; sleep 1; done
    kill -0 -$pgid 2>/dev/null && kill -TERM -$pgid; sleep 2; kill -0 -$pgid 2>/dev/null && kill -KILL -$pgid
  fi; rm -f $E/vllm.pid; echo "stopped candidate vllm"
}
if [ "$1" = stop ]; then stop; exit 0; fi
model=$2; gpu=$3; util=$4; shift 4
stop
tag=$(echo $model | tr '/' '_')
setsid env HF_HOME=$H/models/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
  PATH=$H/llm-serving/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  $H/llm-serving/.venv/bin/vllm serve $model --host 127.0.0.1 --port 8401 --served-model-name cand \
  --max-model-len 4096 --max-num-seqs 48 --gpu-memory-utilization $util "$@" > $E/$tag.log 2>&1 < /dev/null &
pid=$!; echo "$pid $(ps -o pgid= -p $pid | tr -d ' ') $(id -un)" > $E/vllm.pid
for i in $(seq 1 180); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8401/v1/models)" = 200 ] && { echo "ready $model"; exit 0; }
  kill -0 $pid 2>/dev/null || { echo "vllm exited"; tail -30 $E/$tag.log; exit 1; }; sleep 5
done
echo "timeout"; tail -30 $E/$tag.log; exit 1
