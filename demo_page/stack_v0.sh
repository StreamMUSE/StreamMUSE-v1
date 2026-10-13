#!/bin/bash
# Demo page stacks for the old versions (plan 2026-10-12-rap-version-demo-page-plan.md).
# usage: stack_v0.sh <v0.0|v0.1|v0.2> start <gpu> | stop
# Ports: v0.0 render 8120 / vLLM 8101; v0.1 render 8220 / vLLM 8201 / SGLang 8230;
#        v0.2 render 8320 / vLLM 8301 / SGLang 8330.
# Each process runs in its own process group; stop only kills groups recorded here and owned by this user.
set -u
version=$1; action=$2
H=/data/home/bowenzheng/mbzuai-projects
M=$H/StreamMUSE-v1
B=$M/logs/sglang_moss_benchmark_20260911; P=$M/logs/profiling_20260925; L=$M/logs/latency_patch_20261001
W=$H/rap-versions/$version
E=$M/logs/rap_demo_page_2026-10/stack-$version; mkdir -p $E
case $version in
  v0.0) render_port=8120; vllm_port=8101; sglang_port= ;    seqs=8;  warp=all_onsets_r3 ;;
  v0.1) render_port=8220; vllm_port=8201; sglang_port=8230; seqs=8;  warp=all_onsets_r3 ;;
  v0.2) render_port=8320; vllm_port=8301; sglang_port=8330; seqs=32; warp=gentle_sparse_r3 ;;
  *) echo "unknown version $version"; exit 1 ;;
esac
launch() {  # name, then command...
  local name=$1; shift
  setsid "$@" > $E/$name.log 2>&1 < /dev/null &
  local pid=$!; echo "$pid $(ps -o pgid= -p $pid | tr -d ' ') $(id -un)" > $E/$name.pid
}
wait_http() {  # url pidfile seconds
  read pid pgid owner < $2
  for i in $(seq 1 $(( $3 / 5 ))); do
    [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' $1)" = 200 ] && return 0
    kill -0 $pid 2>/dev/null || return 1; sleep 5; done; return 1
}
stop_all() {
  for name in render sglang vllm; do
    [ -f $E/$name.pid ] || continue; read pid pgid owner < $E/$name.pid
    if [ "$owner" = "$(id -un)" ] && kill -0 -$pgid 2>/dev/null; then
      kill -INT -$pgid; for i in $(seq 1 40); do kill -0 -$pgid 2>/dev/null || break; sleep 1; done
      kill -0 -$pgid 2>/dev/null && kill -TERM -$pgid; sleep 2; kill -0 -$pgid 2>/dev/null && kill -KILL -$pgid
    fi; rm -f $E/$name.pid; echo "stopped $version $name"
  done
}
if [ "$action" = stop ]; then stop_all; exit 0; fi
gpu=$3
git -C $W rev-parse HEAD > $E/revision.txt

# vLLM as each version documented it: --max-num-seqs 8 (v0.0, v0.1) or 32 (v0.2).
launch vllm env HF_HOME=$H/models/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
  PATH=$H/llm-serving/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  $H/llm-serving/.venv/bin/vllm serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port $vllm_port \
  --served-model-name qwen-rap --max-model-len 2048 --max-num-seqs $seqs --gpu-memory-utilization 0.25
wait_http http://127.0.0.1:$vllm_port/v1/models $E/vllm.pid 600 || { echo "vllm failed"; stop_all; exit 1; }

common="--host 127.0.0.1 --port $render_port --artifact-root $E/render_artifacts \
  --vllm-url http://127.0.0.1:$vllm_port/v1 --vllm-model qwen-rap \
  --moss-reference-wav $B/media/reference.wav --aligner-device cuda:0 --candidate-profile realtime \
  --moss-warp-policy $warp --wire-audio-codec opus"
# v0.2 adds the 2026-10-01 latency work: patched SGLang-Omni, concurrent bar generation, Opus level 5.
sglang_path=; extra=
if [ $version = v0.2 ]; then
  sglang_path="PYTHONPATH=$L/patched"
  extra="--concurrent-bar-generation --opus-compression-level 5 \
    --moss-runtime-patch-sha256 $(sha256sum $M/patches/sglang-omni-0.1.4-af3ab61-moss-latency.patch | cut -d' ' -f1)"
fi
if [ $version = v0.0 ]; then
  launch render bash -c "source $P/env.sh; export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$P/overlay:$W/src; cd $W; \
    exec $B/baseline-env/bin/python -u -m streammuse.presentation.rap_render_server $common \
    --moss-model $B/model/cdd3b911b1585e3f2dbc7775ef10f9926f58850a --moss-device cuda:0"
else
  # Unpatched SGLang-Omni for v0.1 (as it shipped); the latency-patched copy for v0.2.
  launch sglang bash -c "source $P/env.sh; export CUDA_VISIBLE_DEVICES=$gpu $sglang_path; exec $B/env/bin/sgl-omni serve --config $P/moss_tts.yaml \
    --model-path $B/model/cdd3b911b1585e3f2dbc7775ef10f9926f58850a --host 127.0.0.1 --port $sglang_port \
    --model-name OpenMOSS-Team/MOSS-TTS-v1.5 --allowed-local-media-path $B/media"
  wait_http http://127.0.0.1:$sglang_port/health $E/sglang.pid 600 || { echo "sglang failed"; stop_all; exit 1; }
  launch render bash -c "source $P/env.sh; export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$P/overlay:$W/src; cd $W; \
    exec $B/baseline-env/bin/python -u -m streammuse.presentation.rap_render_server $common \
    --moss-model OpenMOSS-Team/MOSS-TTS-v1.5 --moss-serving-backend sglang-omni \
    --moss-sglang-url http://127.0.0.1:$sglang_port --moss-reference-text-file $B/reference.txt \
    --moss-sglang-reference-uri file://$B/media/reference.wav \
    --moss-sglang-reference-sha256 af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389 \
    --moss-request-timeout-s 120 --moss-cancellation-grace-s 2 --moss-model-revision cdd3b911b1585e3f2dbc7775ef10f9926f58850a \
    --moss-runtime-version 0.1.4 --moss-runtime-revision af3ab61a665427ff3fe3d67e32df6223524ed129 \
    --moss-sglang-version 0.5.18 --moss-sglang-revision pypi-0.5.18 \
    --moss-runtime-environment-sha256 dc726e26bf9d515a3893bfdaa8207895aaffe005f577c7cfab7d0b28c25e2b53 \
    --moss-runtime-config $P/moss_tts.yaml --moss-runtime-config-sha256 3d575d2b73aa83bd608ef62df0a0f9b8fdbf3c7a99b362f79864d873cba568b9 $extra"
fi
wait_http http://127.0.0.1:$render_port/health $E/render.pid 900 || { echo "render failed"; tail -20 $E/render.log; stop_all; exit 1; }
echo "stack $version ready on GPU $gpu"
