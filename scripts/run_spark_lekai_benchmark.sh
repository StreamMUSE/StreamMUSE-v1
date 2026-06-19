#!/usr/bin/env bash
set -euo pipefail

# Spark/H200 Lekai benchmark entrypoint.
#
# Copy/edit only the block below for a new machine. The output is separated into:
#   ${OUT_ROOT}/micro                direct model + scheduler JSON
#   ${OUT_ROOT}/public_client        CLI/server logs for audible first-event behavior
#   ${OUT_ROOT}/summary              parsed summary JSON

# ===== Spark machine configuration =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PYTHONPATH:-src}"

PROMPT_CKPT="${PROMPT_CKPT:-models/hf/RT-accompanimentV2-checkpoints/lekai_prompt_model/model.safetensors}"
CONT_CKPT="${CONT_CKPT:-models/hf/RT-accompanimentV2-checkpoints/lekai_continuation_model/model.safetensors}"
MIDI_FILE="${MIDI_FILE:-prompts/A_major/pop909_291_mel.mid}"

OUT_ROOT="${OUT_ROOT:-spark_runs/lekai_benchmark_$(date +%Y%m%d_%H%M%S)}"
BPM="${BPM:-120}"
PROMPT_LENGTH_TICKS="${PROMPT_LENGTH_TICKS:-32}"
GEN_INTERVAL_TICKS="${GEN_INTERVAL_TICKS:-4}"
GEN_LENGTH_FRAMES="${GEN_LENGTH_FRAMES:-4}"

# Direct microbenchmark repeat counts.
PROMPT_REPEATS="${PROMPT_REPEATS:-3}"
CONT_REQUESTS="${CONT_REQUESTS:-80}"
OBSERVED_UNTIL_TICKS="${OBSERVED_UNTIL_TICKS:-48,64}"

# Public client simulation. Set RUN_PUBLIC_CLIENT=0 to skip server/CLI runs.
RUN_PUBLIC_CLIENT="${RUN_PUBLIC_CLIENT:-1}"
RT_PORT="${RT_PORT:-8014}"
PROMPT_PORT="${PROMPT_PORT:-8015}"
PUBLIC_MAX_TICKS_ORIGINAL="${PUBLIC_MAX_TICKS_ORIGINAL:-240}"
PUBLIC_MAX_TICKS_TRIM="${PUBLIC_MAX_TICKS_TRIM:-160}"

# Runtime settings.
export LEKAI_DEVICE="${LEKAI_DEVICE:-cuda}"
export LEKAI_PROMPT_DEVICE="${LEKAI_PROMPT_DEVICE:-cuda}"
export LEKAI_DTYPE="${LEKAI_DTYPE:-float16}"
export LEKAI_PROMPT_DTYPE="${LEKAI_PROMPT_DTYPE:-float16}"
export LEKAI_DISABLE_FALLBACK="${LEKAI_DISABLE_FALLBACK:-1}"
export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS="${LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS:-1}"
export LEKAI_PROMPT_WARMUP="${LEKAI_PROMPT_WARMUP:-0}"
export LEKAI_USE_CACHE="${LEKAI_USE_CACHE:-1}"
export LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS="${LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS:-1}"
export LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY="${LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY:-1}"
export LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS="${LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS:-4}"
export LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES="${LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES:-1}"
export LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP="${LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP:-1}"
export LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS="${LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS:-0}"

mkdir -p \
  "${OUT_ROOT}/micro" \
  "${OUT_ROOT}/public_client/realtime" \
  "${OUT_ROOT}/public_client/prompt_continuation" \
  "${OUT_ROOT}/summary" \
  "${OUT_ROOT}/server_logs"

echo "[spark-bench] OUT_ROOT=${OUT_ROOT}"
echo "[spark-bench] PROMPT_CKPT=${PROMPT_CKPT}"
echo "[spark-bench] CONT_CKPT=${CONT_CKPT}"
echo "[spark-bench] MIDI_FILE=${MIDI_FILE}"
echo "[spark-bench] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

echo "[spark-bench] running direct microbenchmark"
"${PYTHON_BIN}" scripts/benchmark_lekai_spark.py micro \
  --prompt-checkpoint "${PROMPT_CKPT}" \
  --continuation-checkpoint "${CONT_CKPT}" \
  --midi-file "${MIDI_FILE}" \
  --trim-leading-rest \
  --prompt-length-ticks "${PROMPT_LENGTH_TICKS}" \
  --generation-interval-ticks "${GEN_INTERVAL_TICKS}" \
  --generation-length-frames "${GEN_LENGTH_FRAMES}" \
  --prompt-repeats "${PROMPT_REPEATS}" \
  --continuation-requests "${CONT_REQUESTS}" \
  --observed-until-ticks "${OBSERVED_UNTIL_TICKS}" \
  --output "${OUT_ROOT}/micro/direct_micro_and_scheduler.json" \
  2>&1 | tee "${OUT_ROOT}/micro/direct_micro_and_scheduler.log"

wait_for_health() {
  local port="$1"
  local deadline="$2"
  "${PYTHON_BIN}" - "$port" "$deadline" <<'PY'
import sys
import time
import requests

port = int(sys.argv[1])
deadline = time.time() + float(sys.argv[2])
url = f"http://127.0.0.1:{port}/health"
last_error = None
while time.time() < deadline:
    try:
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            print(f"ready:{url}")
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(1)
print(f"not_ready:{url}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

SERVER_PID=""
stop_server() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap stop_server EXIT

run_cli_case() {
  local mode="$1"
  local port="$2"
  local trim_flag="$3"
  local max_ticks="$4"
  local log_path="$5"
  local trace_path="$6"

  local trim_args=()
  if [[ "${trim_flag}" == "trim" ]]; then
    trim_args+=(--midi-file-trim-leading-rest)
  fi

  if [[ "${mode}" == "prompt_continuation" ]]; then
    LEKAI_PROMPT_CONTINUATION_TRACE_PATH="${trace_path}" \
    "${PYTHON_BIN}" -m streammuse.presentation.cli.cli \
      --input-mode midi_file \
      --midi-file-path "${MIDI_FILE}" \
      "${trim_args[@]}" \
      --continuation-mode prompt_continuation \
      --inference-type http \
      --server-url "http://127.0.0.1:${port}/generate_accompaniment" \
      --model-name lekai \
      --prompt-length-ticks "${PROMPT_LENGTH_TICKS}" \
      --generation-interval-ticks "${GEN_INTERVAL_TICKS}" \
      --generation-length-frames "${GEN_LENGTH_FRAMES}" \
      --timeout-s 240 \
      --output-type console \
      --max-ticks "${max_ticks}" \
      --tempo "${BPM}" \
      2>&1 | tee "${log_path}"
  else
    "${PYTHON_BIN}" -m streammuse.presentation.cli.cli \
      --input-mode midi_file \
      --midi-file-path "${MIDI_FILE}" \
      "${trim_args[@]}" \
      --continuation-mode standard \
      --inference-type http \
      --server-url "http://127.0.0.1:${port}/generate_accompaniment" \
      --model-name lekai \
      --generation-interval-ticks "${GEN_INTERVAL_TICKS}" \
      --generation-length-frames "${GEN_LENGTH_FRAMES}" \
      --timeout-s 240 \
      --output-type console \
      --max-ticks "${max_ticks}" \
      --tempo "${BPM}" \
      2>&1 | tee "${log_path}"
  fi
}

if [[ "${RUN_PUBLIC_CLIENT}" == "1" ]]; then
  echo "[spark-bench] running public client standard realtime"
  LEKAI_SERVER_HOST=127.0.0.1 \
  LEKAI_SERVER_PORT="${RT_PORT}" \
  LEKAI_CHECKPOINT_PATH="${CONT_CKPT}" \
  LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=0 \
  "${PYTHON_BIN}" -m streammuse.infrastructure.inference.server_lekai \
    >"${OUT_ROOT}/server_logs/realtime_server.out.log" \
    2>"${OUT_ROOT}/server_logs/realtime_server.err.log" &
  SERVER_PID="$!"
  wait_for_health "${RT_PORT}" 180

  run_cli_case standard "${RT_PORT}" original "${PUBLIC_MAX_TICKS_ORIGINAL}" \
    "${OUT_ROOT}/public_client/realtime/original_console.log" \
    "${OUT_ROOT}/public_client/realtime/original_trace.jsonl"
  run_cli_case standard "${RT_PORT}" trim "${PUBLIC_MAX_TICKS_TRIM}" \
    "${OUT_ROOT}/public_client/realtime/trim_console.log" \
    "${OUT_ROOT}/public_client/realtime/trim_trace.jsonl"

  stop_server

  echo "[spark-bench] running public client prompt-continuation"
  echo "[spark-bench] prompt-continuation late recovery: recover=${LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS} bound=${LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY} max_ticks=${LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS} rehydrate=${LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES}"
  echo "[spark-bench] prompt-continuation representation loop: strict=${LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP} trace_keys=${LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS}"
  LEKAI_SERVER_HOST=127.0.0.1 \
  LEKAI_SERVER_PORT="${PROMPT_PORT}" \
  LEKAI_PROMPT_CHECKPOINT_PATH="${PROMPT_CKPT}" \
  LEKAI_CONTINUATION_CHECKPOINT_PATH="${CONT_CKPT}" \
  LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1 \
  "${PYTHON_BIN}" -m streammuse.infrastructure.inference.server_lekai \
    >"${OUT_ROOT}/server_logs/prompt_continuation_server.out.log" \
    2>"${OUT_ROOT}/server_logs/prompt_continuation_server.err.log" &
  SERVER_PID="$!"
  wait_for_health "${PROMPT_PORT}" 240

  run_cli_case prompt_continuation "${PROMPT_PORT}" original "${PUBLIC_MAX_TICKS_ORIGINAL}" \
    "${OUT_ROOT}/public_client/prompt_continuation/original_console.log" \
    "${OUT_ROOT}/public_client/prompt_continuation/original_trace.jsonl"
  run_cli_case prompt_continuation "${PROMPT_PORT}" trim "${PUBLIC_MAX_TICKS_TRIM}" \
    "${OUT_ROOT}/public_client/prompt_continuation/trim_console.log" \
    "${OUT_ROOT}/public_client/prompt_continuation/trim_trace.jsonl"

  stop_server

  echo "[spark-bench] parsing public client logs"
  "${PYTHON_BIN}" scripts/benchmark_lekai_spark.py analyze-console \
    --case realtime_original="${OUT_ROOT}/public_client/realtime/original_console.log" \
    --case realtime_trim="${OUT_ROOT}/public_client/realtime/trim_console.log" \
    --case prompt_original="${OUT_ROOT}/public_client/prompt_continuation/original_console.log" \
    --case prompt_trim="${OUT_ROOT}/public_client/prompt_continuation/trim_console.log" \
    --output "${OUT_ROOT}/summary/public_client_first_sound_and_drops.json"
fi

echo "[spark-bench] done"
echo "[spark-bench] summary files:"
find "${OUT_ROOT}" -maxdepth 3 -type f \( -name "*.json" -o -name "*.log" -o -name "*.jsonl" \) | sort
