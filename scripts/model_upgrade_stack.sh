#!/bin/bash
# Protocol v3 stacks for the model-upgrade sessions (plan 2026-10-14, A3/B2), all on one GPU.
# usage: model_upgrade_stack.sh start <qwen25|qwen38> <moss014|mosslocal> <gpu> [writer-gpu] | stop
#   writer-gpu: put the lyric model on another GPU (default: the same one)
#   qwen25:    Qwen2.5-7B-Instruct (v1.0's lyric model)
#   qwen38:    Qwen3.8-27B-FP8, thinking disabled
#   moss014:   MOSS-TTS v1.5 on SGLang-Omni 0.1.4 with the latency and continuation patches (v1.0)
#   mosslocal: MOSS-TTS-Local-Transformer v1.5 on SGLang-Omni 0.1.8 with the ported continuation patch
# Ports: vLLM 8601, TTS 8630, render server 8620. Each process runs in its own process group;
# stop only kills groups recorded here and owned by this user.
set -u
H=/data/home/bowenzheng/mbzuai-projects
R=$H/StreamMUSE-v1
B=$R/logs/sglang_moss_benchmark_20260911; P=$R/logs/profiling_20260925
S=$H/model-envs/sglang-omni
E=$R/logs/model_upgrade_20261014/stack; mkdir -p $E

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
  for name in render tts vllm; do
    [ -f $E/$name.pid ] || continue; read pid pgid owner < $E/$name.pid
    if [ "$owner" = "$(id -un)" ] && kill -0 -$pgid 2>/dev/null; then
      kill -INT -$pgid; for i in $(seq 1 40); do kill -0 -$pgid 2>/dev/null || break; sleep 1; done
      kill -0 -$pgid 2>/dev/null && kill -TERM -$pgid; sleep 2; kill -0 -$pgid 2>/dev/null && kill -KILL -$pgid
    fi; rm -f $E/$name.pid; echo "stopped $name"
  done
}
if [ "$1" = stop ]; then stop_all; exit 0; fi
writer=$2; tts=$3; gpu=$4; wgpu=${5:-$4}
stop_all
echo "$writer $tts gpu=$gpu writer_gpu=$wgpu $(git -C $R rev-parse HEAD)" > $E/current.txt

case $writer in
  qwen25) model=Qwen/Qwen2.5-7B-Instruct; vargs="--gpu-memory-utilization 0.25"; payload= ;;
  qwen38) model=Qwen/Qwen3.8-27B-FP8; vargs="--gpu-memory-utilization 0.45 --limit-mm-per-prompt {\"image\":0,\"video\":0}"
          payload='{"chat_template_kwargs": {"enable_thinking": false}}' ;;
  *) echo "unknown writer $writer"; exit 1 ;;
esac
launch vllm env HF_HOME=$H/models/huggingface HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$wgpu \
  PATH=$H/llm-serving/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  $H/llm-serving/.venv/bin/vllm serve $model --host 127.0.0.1 --port 8601 --served-model-name qwen-rap \
  --max-model-len 2048 --max-num-seqs 48 $vargs
wait_http http://127.0.0.1:8601/v1/models $E/vllm.pid 900 || { echo "vllm failed"; stop_all; exit 1; }

case $tts in
  moss014)
    launch tts bash -c "source $P/env.sh; export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$R/logs/lyric_stream_20261012/continuation; \
      exec $B/env/bin/sgl-omni serve --config $P/moss_tts.yaml --model-path $B/model/cdd3b911b1585e3f2dbc7775ef10f9926f58850a \
      --host 127.0.0.1 --port 8630 --model-name OpenMOSS-Team/MOSS-TTS-v1.5 --allowed-local-media-path $B/media"
    moss="--moss-model OpenMOSS-Team/MOSS-TTS-v1.5 --moss-model-revision cdd3b911b1585e3f2dbc7775ef10f9926f58850a \
      --moss-runtime-version 0.1.4 --moss-runtime-revision af3ab61a665427ff3fe3d67e32df6223524ed129 \
      --moss-sglang-version 0.5.18 --moss-sglang-revision pypi-0.5.18 \
      --moss-runtime-environment-sha256 dc726e26bf9d515a3893bfdaa8207895aaffe005f577c7cfab7d0b28c25e2b53 \
      --moss-runtime-config $P/moss_tts.yaml --moss-runtime-config-sha256 3d575d2b73aa83bd608ef62df0a0f9b8fdbf3c7a99b362f79864d873cba568b9 \
      --moss-runtime-patch-sha256 $(sha256sum $R/patches/sglang-omni-0.1.4-af3ab61-moss-continuation.patch | cut -d' ' -f1)" ;;
  mosslocal)
    launch tts env HF_HOME=$H/models/huggingface CUDA_VISIBLE_DEVICES=$gpu SGLANG_OMNI_STARTUP_TIMEOUT=2400 \
      PATH=$S/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin bash -c "cd $S; exec $S/.venv/bin/sgl-omni serve \
      --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --config examples/configs/moss_tts_local_non_streaming.yaml \
      --tts_engine.gpu_memory_fraction 0.3 --tts_engine.engine.mem_fraction_static 0.3 \
      --host 127.0.0.1 --port 8630 --allowed-local-media-path $B/media"
    moss="--moss-model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --moss-model-revision be7766a6735b98bd793f7c79fb720b4d0f5d13b8 \
      --moss-runtime-version 0.1.8 --moss-runtime-revision 870d77446b3fae6e4f4f2a6964f7eae2f75e663a \
      --moss-sglang-version 0.5.21 --moss-sglang-revision pypi-0.5.21 \
      --moss-runtime-environment-sha256 93cbc4829a7c209c8ae89a68ade1935bd2c08c25ecc5decf2eff5d82cf67a4d9 \
      --moss-runtime-config $S/examples/configs/moss_tts_local_non_streaming.yaml \
      --moss-runtime-config-sha256 1b75998a3bb92badb9617429bc8a854e5ac923c25e6677c42d924d76c5b15ed7 \
      --moss-runtime-patch-sha256 $(sha256sum $R/patches/sglang-omni-0.1.8-moss-local-continuation.patch | cut -d' ' -f1) \
      --moss-sglang-resample-output" ;;
  *) echo "unknown tts $tts"; stop_all; exit 1 ;;
esac
wait_http http://127.0.0.1:8630/health $E/tts.pid 2400 || { echo "tts failed"; stop_all; exit 1; }

extra=; [ -n "$payload" ] && extra="--vllm-extra-payload '$payload'"
launch render bash -c "source $P/env.sh; export CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$P/overlay:$R/src; cd $R; \
  exec $B/baseline-env/bin/python -u -m streammuse.presentation.rap_render_server --host 127.0.0.1 --port 8620 \
  --artifact-root $E/render_artifacts --vllm-url http://127.0.0.1:8601/v1 --vllm-model qwen-rap $extra \
  --concurrent-bar-generation --moss-serving-backend sglang-omni --moss-reference-wav $B/media/reference.wav \
  --moss-sglang-url http://127.0.0.1:8630 --moss-reference-text-file $B/reference.txt \
  --moss-sglang-reference-uri file://$B/media/reference.wav \
  --moss-sglang-reference-sha256 af3ac928e98cdb171c9abdc611813f2cc40c52b71167790b785628104f81f389 \
  --moss-request-timeout-s 120 --moss-cancellation-grace-s 2 $moss \
  --aligner-device cuda:0 --candidate-profile realtime --moss-warp-policy gentle_sparse_r3 \
  --wire-audio-codec opus --opus-compression-level 5 --lyric-stream --lyric-candidates 32 --lyric-continuation"
wait_http http://127.0.0.1:8620/health $E/render.pid 900 || { echo "render failed"; tail -30 $E/render.log; stop_all; exit 1; }
echo "stack $writer (GPU $wgpu) + $tts (GPU $gpu) ready"
