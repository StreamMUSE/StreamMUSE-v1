#!/usr/bin/env bash
set -euo pipefail

# Parallel ZipZapZop memory matrix wrapper.
# It preserves scripts/run_zzz_full_matrix.sh's experiment cells, but schedules
# independent model jobs across GPU waves.

ROOT_DIR=${ROOT_DIR:-"task_runs/zzz_memory_full_matrix_parallel_$(date +%Y%m%d-%H%M%S)"}
HOST=${HOST:-"127.0.0.1"}
VLLM=${VLLM:-"$HOME/mbzuai-projects/llm-serving/.venv/bin/vllm"}
UV=${UV:-"uv"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"32"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.60"}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-"14400"}
HF_HOME=${HF_HOME:-"$HOME/mbzuai-projects/models/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}
VLLM_BIN_DIR=$(dirname "$VLLM")
GPU_QWEN3_8B=${GPU_QWEN3_8B:-"0"}
GPU_QWEN36_27B=${GPU_QWEN36_27B:-"1"}
GPU_GEMMA_27B=${GPU_GEMMA_27B:-"2"}
GPU_QWEN36_35B=${GPU_QWEN36_35B:-"6"}
GPU_GPT_OSS_120B=${GPU_GPT_OSS_120B:-"0"}
GPU_LLAMA_70B=${GPU_LLAMA_70B:-"2,6"}
GPU_QWEN235B=${GPU_QWEN235B:-"0,1"}

export HF_HOME HF_HUB_CACHE
export PATH="$VLLM_BIN_DIR:$PATH"
if [[ -z "${HF_TOKEN:-}" && -s "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi

mkdir -p "$ROOT_DIR/logs"

safe_name() {
  echo "$1" | sed 's#[/:]#_#g'
}

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log_main() {
  echo "[$(timestamp)] [parallel-matrix] $*"
}

start_server() {
  local model="$1"
  local label="$2"
  local port="$3"
  local gpus="$4"
  local tp="$5"
  shift 5
  local extra_args=("$@")
  local log_path="$ROOT_DIR/logs/$(safe_name "$model")_${label}_vllm.log"

  echo "[$(timestamp)] starting vLLM model=$model label=$label port=$port tp=$tp gpus=$gpus"
  if [[ "$tp" == "2" ]]; then
    CUDA_VISIBLE_DEVICES="$gpus" "$VLLM" serve "$model" \
      --host "$HOST" \
      --port "$port" \
      --served-model-name "$model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size 2 \
      "${extra_args[@]}" > "$log_path" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpus" "$VLLM" serve "$model" \
      --host "$HOST" \
      --port "$port" \
      --served-model-name "$model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      "${extra_args[@]}" > "$log_path" 2>&1 &
  fi
  VLLM_PID=$!
  echo "$VLLM_PID" > "$ROOT_DIR/logs/$(safe_name "$model")_${label}.pid"
  echo "[$(timestamp)] vLLM pid=$VLLM_PID log=$log_path"
}

stop_server() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[$(timestamp)] stopping vLLM pid=$VLLM_PID"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  VLLM_PID=""
  sleep 5
}

wait_ready() {
  local port="$1"
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  local url="http://$HOST:$port/v1/models"
  echo "[$(timestamp)] waiting for $url"
  until curl -fsS "$url" >/dev/null 2>&1; do
    if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
      echo "[$(timestamp)] vLLM exited before ready: pid=$VLLM_PID url=$url" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[$(timestamp)] server did not become ready before timeout: $url" >&2
      return 1
    fi
    sleep 10
  done
  echo "[$(timestamp)] server ready: $url"
}

run_single_sweep() {
  local model="$1"
  local port="$2"
  local out_dir="$3"
  shift 3
  mkdir -p "$out_dir"
  "$UV" run python scripts/run_zzz_memory_sweep.py \
    --model "$model" \
    --model-url "http://$HOST:$port/v1" \
    --sweep-root "$out_dir" \
    "$@"
}

smoke_model() {
  local model="$1"
  local port="$2"
  local out_dir="$3"
  local csv_path="$out_dir/per_turn.csv"
  echo "[$(timestamp)] smoke $model"
  run_single_sweep "$model" "$port" "$out_dir" \
    --pilot \
    --turns 3 \
    --history-limits 0 \
    --temperatures 0 \
    --repeats-nonzero-temp 1 \
    --no-oracle
  python3 - "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"missing smoke per_turn.csv: {path}")
with path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"empty smoke per_turn.csv: {path}")
bad = [row for row in rows if str(row.get("strict_valid")).lower() != "true"]
if bad:
    preview = [
        {
            "turn": row.get("turn_id"),
            "response": row.get("response"),
            "expected": row.get("expected"),
        }
        for row in bad[:3]
    ]
    raise SystemExit(f"smoke produced invalid answers: {preview}")
PY
}

assert_gpt_oss_ngram_matches_baseline() {
  local model_dir="$1"
  "$UV" run python - "$model_dir" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
base = root / "apc_on" / "per_turn.csv"
ngram = root / "ngram_apc_on" / "per_turn.csv"
if not base.exists() or not ngram.exists():
    raise SystemExit(0)

def greedy_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("temperature") in {"0.0", "0"} and row.get("oracle") == "False"
        ]

base_keyed = {(row["config"], row["turn_id"]): row["response"] for row in greedy_rows(base)}
for row in greedy_rows(ngram):
    key = (row["config"], row["turn_id"])
    if key in base_keyed and row["response"] != base_keyed[key]:
        raise SystemExit(
            f"gpt-oss ngram mismatch at {key}: {row['response']!r} != {base_keyed[key]!r}"
        )
PY
}

run_model_job() {
  local model="$1"
  local gpus="$2"
  local tp="$3"
  local port="$4"
  local safe
  safe=$(safe_name "$model")
  local model_dir="$ROOT_DIR/$safe"
  local done_marker="$model_dir/done"
  local failed_marker="$model_dir/failed"
  VLLM_PID=""
  trap stop_server EXIT

  mkdir -p "$model_dir"
  echo "[$(timestamp)] job start model=$model gpus=$gpus port=$port tp=$tp"
  if [[ -f "$done_marker" ]]; then
    echo "[$(timestamp)] skip completed model=$model"
    return 0
  fi

  start_server "$model" "apc_on" "$port" "$gpus" "$tp"
  wait_ready "$port"
  if ! smoke_model "$model" "$port" "$model_dir/smoke"; then
    echo "[$(timestamp)] smoke failed for $model; marking failed"
    touch "$failed_marker"
    stop_server
    return 1
  fi
  run_single_sweep "$model" "$port" "$model_dir/apc_on" --apc-label on
  stop_server

  start_server "$model" "apc_off" "$port" "$gpus" "$tp" --no-enable-prefix-caching
  wait_ready "$port"
  run_single_sweep "$model" "$port" "$model_dir/apc_off" \
    --apc-label off \
    --history-limits 32,all \
    --temperatures 0 \
    --repeats-nonzero-temp 1 \
    --no-oracle
  stop_server

  if [[ "$model" == "openai/gpt-oss-120b" ]]; then
    start_server "$model" "ngram_apc_on" "$port" "$gpus" "$tp" \
      --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'
    wait_ready "$port"
    run_single_sweep "$model" "$port" "$model_dir/ngram_apc_on" --apc-label on --spec-label ngram
    stop_server

    start_server "$model" "ngram_apc_off" "$port" "$gpus" "$tp" \
      --no-enable-prefix-caching \
      --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'
    wait_ready "$port"
    run_single_sweep "$model" "$port" "$model_dir/ngram_apc_off" \
      --apc-label off \
      --spec-label ngram \
      --history-limits 32,all \
      --temperatures 0 \
      --repeats-nonzero-temp 1 \
      --no-oracle
    stop_server
    assert_gpt_oss_ngram_matches_baseline "$model_dir"
  fi

  touch "$done_marker"
  rm -f "$failed_marker"
  echo "[$(timestamp)] job done model=$model"
}

wait_wave() {
  local wave_name="$1"
  shift
  local pids=("$@")
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  "$UV" run python scripts/merge_zzz_memory_summaries.py --root "$ROOT_DIR" || true
  if [[ "$status" != "0" ]]; then
    log_main "$wave_name completed with failures; see $ROOT_DIR/logs"
  else
    log_main "$wave_name completed"
  fi
  return "$status"
}

launch_job() {
  local model="$1"
  local gpus="$2"
  local tp="$3"
  local port="$4"
  local safe
  safe=$(safe_name "$model")
  run_model_job "$model" "$gpus" "$tp" "$port" > "$ROOT_DIR/logs/$safe.job.log" 2>&1 &
  LAUNCHED_PID="$!"
}

log_main "root=$ROOT_DIR"
log_main "HF_HOME=$HF_HOME"
log_main "GPU waves use configured CUDA_VISIBLE_DEVICES: qwen8=$GPU_QWEN3_8B qwen27=$GPU_QWEN36_27B gemma=$GPU_GEMMA_27B qwen35=$GPU_QWEN36_35B gptoss=$GPU_GPT_OSS_120B llama=$GPU_LLAMA_70B qwen235=$GPU_QWEN235B"

wave1_pids=()
launch_job "Qwen/Qwen3-8B" "$GPU_QWEN3_8B" "1" "8100"
wave1_pids+=("$LAUNCHED_PID")
launch_job "Qwen/Qwen3.6-27B" "$GPU_QWEN36_27B" "1" "8101"
wave1_pids+=("$LAUNCHED_PID")
launch_job "google/gemma-3-27b-it" "$GPU_GEMMA_27B" "1" "8102"
wave1_pids+=("$LAUNCHED_PID")
launch_job "Qwen/Qwen3.6-35B-A3B" "$GPU_QWEN36_35B" "1" "8103"
wave1_pids+=("$LAUNCHED_PID")
wait_wave "wave1" "${wave1_pids[@]}" || true

wave2_pids=()
launch_job "openai/gpt-oss-120b" "$GPU_GPT_OSS_120B" "1" "8110"
wave2_pids+=("$LAUNCHED_PID")
launch_job "meta-llama/Llama-3.3-70B-Instruct" "$GPU_LLAMA_70B" "2" "8111"
wave2_pids+=("$LAUNCHED_PID")
wait_wave "wave2" "${wave2_pids[@]}" || true

wave3_pids=()
launch_job "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" "$GPU_QWEN235B" "2" "8120"
wave3_pids+=("$LAUNCHED_PID")
wait_wave "wave3" "${wave3_pids[@]}" || true

"$UV" run python scripts/merge_zzz_memory_summaries.py --root "$ROOT_DIR" || true
log_main "done: $ROOT_DIR"
