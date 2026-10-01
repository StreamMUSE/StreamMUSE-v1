#!/usr/bin/env bash
# Start the complete Mac-local rap stack on loopback, then (optionally) the demo.
#
#   Qwen chat   scripts/mlx_chat_server.py   envs/rap-mlx-chat   127.0.0.1:8001
#   MOSS TTS    scripts/mlx_moss_server.py   envs/rap-mlx-moss   127.0.0.1:8030
#   renderer    streammuse-rap-render-server  main uv env         127.0.0.1:8020
#
# Usage:
#   scripts/run_rap_local_mac.sh            # services only; Ctrl-C (or kill <pid>) stops them
#   scripts/run_rap_local_mac.sh --demo     # services, then streammuse-rap-demo
#
# Every pin below is a default that can be overridden from the environment. The
# render server refuses to start when the MOSS service reports a different
# runtime identity, so a reconverted model needs its new weights hash here.
#
# Model weights and caches are expected on the external SSD (see ~/.zshenv); the
# script stops instead of letting anything download to the internal disk.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_DEMO=0
for arg in "$@"; do
  case "$arg" in
    --demo) RUN_DEMO=1 ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

SSD_ROOT="${SSD_ROOT:-/Volumes/ZBW-SSD1}"
MODELS_ROOT="${MODELS_ROOT:-$SSD_ROOT/models}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-$MODELS_ROOT/qwen2.5-7b-instruct-a09a3545-mlx-q4g64}"
MOSS_MODEL_PATH="${MOSS_MODEL_PATH:-$MODELS_ROOT/moss-tts-v1.5-cdd3b911-mlx-q8g64}"
MOSS_QUANTIZATION="${MOSS_QUANTIZATION:-affine-q8-g64}"
MOSS_WEIGHTS_SHA256="${MOSS_WEIGHTS_SHA256:-2453a27307dfca327a32081f460493c5e471183da5bc60ab01725960271a0f06}"
MOSS_MODEL_REVISION="${MOSS_MODEL_REVISION:-cdd3b911b1585e3f2dbc7775ef10f9926f58850a}"
MOSS_TOKENIZER_REVISION="${MOSS_TOKENIZER_REVISION:-3cd226ba2947efa357ef453bcad111b6eafba782}"
MLX_VERSION="${MLX_VERSION:-0.32.3}"
MLX_AUDIO_VERSION="${MLX_AUDIO_VERSION:-0.5.7}"
MLX_AUDIO_COMMIT="${MLX_AUDIO_COMMIT:-94c7716212b2228f178d2f9c7619a591fd1b0b78}"
MOSS_REFERENCE_WAV="${MOSS_REFERENCE_WAV:-$SSD_ROOT/assets/rap-voices/0011_000001.wav}"
# all_onsets_r3 until the gentle_sparse public-anchor contract is settled; the
# Mac client rejects gentle_sparse chunks whose targets drift from its schedule.
MOSS_WARP_POLICY="${MOSS_WARP_POLICY:-all_onsets_r3}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/output/rap_local_mac_run}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$RUN_DIR/artifacts}"

fail() { echo "error: $*" >&2; exit 1; }

# --- preflight -----------------------------------------------------------------
[[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]] || fail "Apple Silicon macOS only"
[[ -d "$SSD_ROOT" ]] || fail "external SSD $SSD_ROOT is not mounted"
for name in HF_HOME TORCH_HOME UV_CACHE_DIR; do
  value="${!name:-}"
  [[ "$value" == "$SSD_ROOT"/* ]] || fail "$name must point under $SSD_ROOT (got '${value}'); see ~/.zshenv"
done
for path in "$QWEN_MODEL_PATH" "$MOSS_MODEL_PATH" "$MOSS_MODEL_PATH/audio_tokenizer"; do
  [[ -d "$path" ]] || fail "missing model directory $path"
done
[[ -f "$MOSS_REFERENCE_WAV" ]] || fail "missing reference voice $MOSS_REFERENCE_WAV"
export PATH="/opt/homebrew/bin:$PATH"
for tool in uv rubberband ffmpeg; do
  command -v "$tool" >/dev/null || fail "$tool is not installed (brew install rubberband ffmpeg; uv from astral.sh)"
done
# Never fetch unpinned files at runtime: everything must already be on the SSD.
export HF_HUB_OFFLINE=1

mkdir -p "$RUN_DIR" "$ARTIFACT_ROOT"
PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  rm -f "$RUN_DIR"/*.pid
}
trap cleanup EXIT INT TERM

wait_for() {
  local url="$1" pid="$2" name="$3" timeout="${4:-300}" waited=0
  until curl -sf "$url" >/dev/null; do
    kill -0 "$pid" 2>/dev/null || fail "$name exited during startup; see $RUN_DIR/$name.log"
    (( waited >= timeout )) && fail "$name did not become ready within ${timeout}s"
    sleep 2
    waited=$(( waited + 2 ))
  done
  echo "ready: $name ($url)"
}

start() {
  local name="$1"; shift
  "$@" >"$RUN_DIR/$name.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  echo "$pid" >"$RUN_DIR/$name.pid"
  echo "started: $name pid $pid (log $RUN_DIR/$name.log)" >&2
  LAST_PID="$pid"
}

# --- services ------------------------------------------------------------------
uv sync --project envs/rap-mlx-chat --frozen >/dev/null 2>&1 || uv sync --project envs/rap-mlx-chat
uv sync --project envs/rap-mlx-moss --frozen >/dev/null 2>&1 || uv sync --project envs/rap-mlx-moss
uv sync >/dev/null

start mlx-chat uv run --project envs/rap-mlx-chat python scripts/mlx_chat_server.py \
  --model-path "$QWEN_MODEL_PATH" --served-model-name qwen-rap --port 8001
CHAT_PID="$LAST_PID"
start mlx-moss uv run --project envs/rap-mlx-moss python scripts/mlx_moss_server.py \
  --model-path "$MOSS_MODEL_PATH" --model-revision "$MOSS_MODEL_REVISION" \
  --quantization "$MOSS_QUANTIZATION" --audio-tokenizer-revision "$MOSS_TOKENIZER_REVISION" \
  --allowed-media-root "$(dirname "$MOSS_REFERENCE_WAV")" --port 8030
MOSS_PID="$LAST_PID"
wait_for http://127.0.0.1:8001/v1/models "$CHAT_PID" mlx-chat
wait_for http://127.0.0.1:8030/v1/models "$MOSS_PID" mlx-moss

start render-server uv run streammuse-rap-render-server --port 8020 \
  --artifact-root "$ARTIFACT_ROOT" \
  --vllm-url http://127.0.0.1:8001/v1 --vllm-model qwen-rap --concurrent-bar-generation \
  --moss-model OpenMOSS-Team/MOSS-TTS-v1.5 --moss-serving-backend mlx \
  --moss-mlx-url http://127.0.0.1:8030 --moss-reference-wav "$MOSS_REFERENCE_WAV" \
  --moss-model-revision "$MOSS_MODEL_REVISION" \
  --mlx-version "$MLX_VERSION" --mlx-audio-version "$MLX_AUDIO_VERSION" \
  --mlx-audio-commit "$MLX_AUDIO_COMMIT" --mlx-moss-quantization "$MOSS_QUANTIZATION" \
  --mlx-moss-weights-sha256 "$MOSS_WEIGHTS_SHA256" \
  --mlx-moss-audio-tokenizer-revision "$MOSS_TOKENIZER_REVISION" \
  --aligner-cache "$TORCH_HOME/hub" --moss-warp-policy "$MOSS_WARP_POLICY" \
  --wire-audio-codec pcm
wait_for http://127.0.0.1:8020/health "$LAST_PID" render-server 300

if (( RUN_DEMO )); then
  uv run streammuse-rap-demo \
    --rap-audio-renderer moss_aligned_remote \
    --rap-audio-transport pcm \
    --rap-render-url http://127.0.0.1:8020 \
    --rap-render-profile realtime \
    --rap-render-startup-timeout 120 \
    --rap-render-rolling-timeout 5.0 \
    --audio-output composite \
    --tempo 90 \
    --lookahead-bars 2 \
    --max-bars 0 \
    --log-dir "$RUN_DIR/demo-logs" \
    --terminal-layout split \
    --terminal-detail full \
    --host 127.0.0.1 \
    --port 8012
else
  echo "all services ready; Ctrl-C to stop"
  wait
fi
