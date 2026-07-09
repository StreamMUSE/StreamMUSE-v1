#!/usr/bin/env bash
set -euo pipefail

# Full ZipZapZop memory matrix wrapper. It starts one vLLM server at a time,
# runs the single-model sweep, and merges summaries at the end.

ROOT_DIR=${ROOT_DIR:-"task_runs/zzz_memory_full_matrix_$(date +%Y%m%d-%H%M%S)"}
MODEL_URL=${MODEL_URL:-"http://127.0.0.1:8000/v1"}
HOST=${HOST:-"127.0.0.1"}
PORT=${PORT:-"8000"}
VLLM=${VLLM:-"$HOME/mbzuai-projects/llm-serving/.venv/bin/vllm"}
UV=${UV:-"uv"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.85"}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-"900"}
STREAMMUSE_MATRIX_GPUS=${STREAMMUSE_MATRIX_GPUS:-"0,1,2"}
HF_HOME=${HF_HOME:-"$HOME/mbzuai-projects/models/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"$HF_HOME/hub"}

export HF_HOME HF_HUB_CACHE

MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3.6-27B"
  "google/gemma-3-27b-it"
  "Qwen/Qwen3.6-35B-A3B"
  "openai/gpt-oss-120b"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
)

mkdir -p "$ROOT_DIR/logs"

safe_name() {
  echo "$1" | sed 's#[/:]#_#g'
}

model_tp() {
  case "$1" in
    "meta-llama/Llama-3.3-70B-Instruct"|"Qwen/Qwen3-235B-A22B-Instruct-2507-FP8") echo "2" ;;
    *) echo "1" ;;
  esac
}

model_gpus() {
  local tp="$1"
  if [[ "$tp" == "2" ]]; then
    echo "$(echo "$STREAMMUSE_MATRIX_GPUS" | cut -d',' -f1,2)"
  else
    echo "$(echo "$STREAMMUSE_MATRIX_GPUS" | cut -d',' -f1)"
  fi
}

start_server() {
  local model="$1"
  local label="$2"
  shift 2
  local extra_args=("$@")
  local tp
  local gpus
  local log_path
  tp=$(model_tp "$model")
  gpus=$(model_gpus "$tp")
  log_path="$ROOT_DIR/logs/$(safe_name "$model")_${label}_vllm.log"

  echo "[matrix] starting vLLM model=$model label=$label tp=$tp gpus=$gpus"
  if [[ "$tp" == "2" ]]; then
    CUDA_VISIBLE_DEVICES="$gpus" "$VLLM" serve "$model" \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name "$model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size 2 \
      "${extra_args[@]}" > "$log_path" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpus" "$VLLM" serve "$model" \
      --host "$HOST" \
      --port "$PORT" \
      --served-model-name "$model" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      "${extra_args[@]}" > "$log_path" 2>&1 &
  fi
  VLLM_PID=$!
  echo "$VLLM_PID" > "$ROOT_DIR/logs/vllm_${label}.pid"
}

stop_server() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "[matrix] stopping vLLM pid=$VLLM_PID"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  VLLM_PID=""
  sleep 5
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT_S))
  echo "[matrix] waiting for $MODEL_URL/models"
  until curl -fsS "$MODEL_URL/models" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[matrix] server did not become ready before timeout" >&2
      return 1
    fi
    sleep 5
  done
}

smoke_model() {
  local model="$1"
  local out_dir="$2/smoke"
  mkdir -p "$out_dir"
  echo "[matrix] smoke $model"
  "$UV" run streammuse-task run \
    --task zip_zap_zop \
    --runner offline_benchmark \
    --model-url "$MODEL_URL" \
    --model "$model" \
    --max-turns 3 \
    --max-tokens 8 \
    --temperature 0 \
    --output-dir "$out_dir"
}

run_single_sweep() {
  local model="$1"
  local out_dir="$2"
  shift 2
  mkdir -p "$out_dir"
  "$UV" run python scripts/run_zzz_memory_sweep.py \
    --model "$model" \
    --model-url "$MODEL_URL" \
    --sweep-root "$out_dir" \
    "$@"
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
    with path.open(newline='', encoding='utf-8') as handle:
        return [row for row in csv.DictReader(handle) if row.get('temperature') in {'0.0', '0'} and row.get('oracle') == 'False']
base_rows = greedy_rows(base)
ngram_rows = greedy_rows(ngram)
base_keyed = {(r['config'], r['turn_id']): r['response'] for r in base_rows}
for row in ngram_rows:
    key = (row['config'], row['turn_id'])
    if key in base_keyed and row['response'] != base_keyed[key]:
        raise SystemExit(f"gpt-oss ngram mismatch at {key}: {row['response']!r} != {base_keyed[key]!r}")
PY
}

trap stop_server EXIT

for model in "${MODELS[@]}"; do
  safe=$(safe_name "$model")
  model_dir="$ROOT_DIR/$safe"
  done_marker="$model_dir/done"
  mkdir -p "$model_dir"
  if [[ -f "$done_marker" ]]; then
    echo "[matrix] skip completed $model"
    continue
  fi

  start_server "$model" "apc_on"
  wait_ready
  if ! smoke_model "$model" "$model_dir"; then
    echo "[matrix] smoke failed for $model; skipping" | tee "$model_dir/failed"
    stop_server
    continue
  fi
  run_single_sweep "$model" "$model_dir/apc_on" --apc-label on
  stop_server

  start_server "$model" "apc_off" --no-enable-prefix-caching
  wait_ready
  run_single_sweep "$model" "$model_dir/apc_off" \
    --apc-label off \
    --history-limits 32,all \
    --temperatures 0 \
    --repeats-nonzero-temp 1 \
    --no-oracle
  stop_server

  if [[ "$model" == "openai/gpt-oss-120b" ]]; then
    start_server "$model" "ngram_apc_on" --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'
    wait_ready
    run_single_sweep "$model" "$model_dir/ngram_apc_on" --apc-label on --spec-label ngram
    stop_server

    start_server "$model" "ngram_apc_off" --no-enable-prefix-caching --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3}'
    wait_ready
    run_single_sweep "$model" "$model_dir/ngram_apc_off" \
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
  "$UV" run python scripts/merge_zzz_memory_summaries.py --root "$ROOT_DIR" || true
done

"$UV" run python scripts/merge_zzz_memory_summaries.py --root "$ROOT_DIR"
echo "[matrix] done: $ROOT_DIR"
