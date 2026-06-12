#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/home/yuanxin/StreamMUSE-v1
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

IDS_TEXT=${IDS:-"6217163 5472152 5472153 6144911"}
read -r -a IDS <<< "$IDS_TEXT"
PORT=${PORT:-8001}
OUT_ROOT=${OUT_ROOT:-realtime_runs/0509_prompt_alignment_batch}
PROMPT_CKPT=/data/home/yuanxin/RT-accompanimentV2/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors
CONT_CKPT=/data/home/yuanxin/RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors
DEVICE=${DEVICE:-0}
PROMPT_CONTINUATION_ENGINE=${PROMPT_CONTINUATION_ENGINE:-standard}
PROMPT_CONTINUATION_EXTENSION_TICKS=${PROMPT_CONTINUATION_EXTENSION_TICKS:-${PROMPT_CONTINUATION_BRIDGE_TICKS:-4}}
MAX_TICKS=${MAX_TICKS:-96}
PROMPT_BEATS=${PROMPT_BEATS:-8}
CLI_TEMPO=${CLI_TEMPO:-240}
CLI_TEMPO_MAX=${CLI_TEMPO_MAX:-}
SAMPLING_TEMPERATURE=${SAMPLING_TEMPERATURE:-1.1}
SAMPLING_TOP_K=${SAMPLING_TOP_K:-0}
SAMPLING_TOP_P=${SAMPLING_TOP_P:-0.95}
PROMPT_TEMPERATURE=${PROMPT_TEMPERATURE:-$SAMPLING_TEMPERATURE}
PROMPT_TOP_K=${PROMPT_TOP_K:-$SAMPLING_TOP_K}
PROMPT_TOP_P=${PROMPT_TOP_P:-$SAMPLING_TOP_P}
RT_TEMPERATURE=${RT_TEMPERATURE:-$SAMPLING_TEMPERATURE}
RT_TOP_K=${RT_TOP_K:-$SAMPLING_TOP_K}
RT_TOP_P=${RT_TOP_P:-$SAMPLING_TOP_P}
RT_REPETITION_PENALTY=${RT_REPETITION_PENALTY:-1.0}
RECOVER_LATE_EVENTS=${RECOVER_LATE_EVENTS:-1}
BOUND_LATE_RECOVERY=${BOUND_LATE_RECOVERY:-}
RECOVER_LATE_MAX_TICKS=${RECOVER_LATE_MAX_TICKS:-}

mkdir -p "$OUT_ROOT/server"

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup_server EXIT

for id in "${IDS[@]}"; do
  beats_per_bar=$(python - <<PY
import numpy as np
p='/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore/$id.npz'
d=np.load(p, allow_pickle=True)
print(int(d['measure_0'].shape[2]//4))
PY
)
  prompt_beats="$PROMPT_BEATS"
  if [[ "$prompt_beats" == "auto" ]]; then
    if [[ "$beats_per_bar" == "3" ]]; then
      prompt_beats=6
    else
      prompt_beats=8
    fi
  fi

  echo "=== $id: prepare/reference ==="
  python scripts/prepare_and_compare_lekai_prompt_alignment.py \
    --npz-id "$id" \
    --output-root "$OUT_ROOT" \
    --prepare \
    --reference \
    --device cuda \
    --seed 42 \
    --prompt-beats "$prompt_beats" \
    --top-p "$PROMPT_TOP_P" \
    --top-k "$PROMPT_TOP_K" \
    --temperature "$PROMPT_TEMPERATURE"

  meta_json="$OUT_ROOT/$id/input_manifest.json"
  bpm=$(python - <<PY
import json
m=json.load(open('$meta_json'))
print(int(m['bpm'] or 120))
PY
)
  ts_idx=$(python - <<PY
import json
m=json.load(open('$meta_json'))
print(int(m['time_signature_idx']))
PY
)
  cli_tempo="$CLI_TEMPO"
  if [[ "$cli_tempo" == "metadata" ]]; then
    cli_tempo="$bpm"
  fi
  if [[ -n "$CLI_TEMPO_MAX" ]]; then
    cli_tempo=$(python - <<PY
tempo=float("$cli_tempo")
limit=float("$CLI_TEMPO_MAX")
print(int(min(tempo, limit)))
PY
)
  fi

  echo "=== $id: start strict server bpm=$bpm ts_idx=$ts_idx beats_per_bar=$beats_per_bar prompt_beats=$prompt_beats cli_tempo=$cli_tempo tempo_max=${CLI_TEMPO_MAX:-none} prompt_continuation_engine=$PROMPT_CONTINUATION_ENGINE prompt_extension_ticks=$PROMPT_CONTINUATION_EXTENSION_TICKS recover_late_events=$RECOVER_LATE_EVENTS bound_late_recovery=${BOUND_LATE_RECOVERY:-auto} recover_late_max_ticks=${RECOVER_LATE_MAX_TICKS:-none} ==="
  cleanup_server
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti tcp:$PORT | xargs -r kill || true
  fi
  server_log="$OUT_ROOT/server/${id}_server.log"
  (
    export CUDA_VISIBLE_DEVICES="$DEVICE"
    export LEKAI_SERVER_PORT="$PORT"
    export LEKAI_PROMPT_CHECKPOINT_PATH="$PROMPT_CKPT"
    export LEKAI_CONTINUATION_CHECKPOINT_PATH="$CONT_CKPT"
    export LEKAI_DEVICE=cuda
    export LEKAI_PROMPT_DEVICE=cuda
    export LEKAI_DTYPE=float16
    # Keep prompt model precision aligned with the RT offline reference.
    # FP16 changes sampled logits for some pieces even with the same seed.
    export LEKAI_PROMPT_DTYPE=float32
    export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1
    export LEKAI_PROMPT_CONTINUATION_ENGINE="$PROMPT_CONTINUATION_ENGINE"
    export LEKAI_PROMPT_CONTINUATION_EXTENSION_TICKS="$PROMPT_CONTINUATION_EXTENSION_TICKS"
    export LEKAI_DISABLE_FALLBACK=1
    export LEKAI_PROMPT_SEED=42
    export LEKAI_SEED=42
    export LEKAI_PROMPT_CONDITION_BEATS="$prompt_beats"
    export LEKAI_PROMPT_TOP_P="$PROMPT_TOP_P"
    export LEKAI_PROMPT_TOP_K="$PROMPT_TOP_K"
    export LEKAI_PROMPT_TEMPERATURE="$PROMPT_TEMPERATURE"
    export LEKAI_RT_TOP_P="$RT_TOP_P"
    export LEKAI_RT_TOP_K="$RT_TOP_K"
    export LEKAI_RT_TEMPERATURE="$RT_TEMPERATURE"
    export LEKAI_RT_REPETITION_PENALTY="$RT_REPETITION_PENALTY"
    export LEKAI_PROMPT_REPETITION_PENALTY=1.0
    export LEKAI_PROMPT_MAX_NEW_TOKENS=1024
    export LEKAI_PROMPT_TIME_SIGNATURE_INDEX="$ts_idx"
    export LEKAI_PROMPT_BPM="$bpm"
    exec python -u -m streammuse.infrastructure.inference.server_lekai
  ) > "$server_log" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server exited for $id"
      tail -160 "$server_log" || true
      exit 1
    fi
    sleep 2
  done
  if [[ "$ready" != 1 ]]; then
    echo "server timeout for $id"
    tail -160 "$server_log" || true
    exit 1
  fi

  runtime_json="$OUT_ROOT/$id/prompt_continuation_runtime_info.json"
  curl -fsS "http://127.0.0.1:$PORT/prompt_continuation/runtime_info" > "$runtime_json"
  python - <<PY
import json, sys
r=json.load(open('$runtime_json'))
if not (r.get('prompt_has_real_model') and r.get('has_real_model')):
    print(json.dumps(r, indent=2))
    sys.exit('prompt-continuation real model check failed')
if r.get('prompt_fallback_reason') or r.get('fallback_reason'):
    print(json.dumps(r, indent=2))
    sys.exit('prompt-continuation fallback reason present')
print('runtime ok:', r.get('prompt_mode'), r.get('mode'))
PY

  echo "=== $id: streammuse-cli ==="
  LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS="$RECOVER_LATE_EVENTS" \
  LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY="$BOUND_LATE_RECOVERY" \
  LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS="$RECOVER_LATE_MAX_TICKS" \
  LEKAI_PROMPT_CONTINUATION_TRACE_PATH="$OUT_ROOT/$id/prompt_continuation_client_trace.jsonl" \
  uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path "$OUT_ROOT/$id/${id}_npz_melody_input.mid" \
    --tempo "$cli_tempo" \
    --ticks-per-beat 4 \
    --beats-per-bar "$beats_per_bar" \
    --continuation-mode prompt_continuation \
    --inference-type http \
    --server-url "http://127.0.0.1:$PORT/generate_accompaniment" \
    --model-name lekai \
    --inference-mode sliding_window \
    --prompt-length-ticks "$((prompt_beats * 4))" \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --timeout-s 120 \
    --output-type session \
    --log-dir "$OUT_ROOT/$id/cli" \
    --max-ticks "$MAX_TICKS"

  session_dir=$(find "$OUT_ROOT/$id/cli" -path '*/session_*' -type d | sort | tail -1)
  echo "session_dir=$session_dir"

  echo "=== $id: compare ==="
  python scripts/prepare_and_compare_lekai_prompt_alignment.py \
    --npz-id "$id" \
    --output-root "$OUT_ROOT" \
    --compare \
    --cli-prompt-json "$session_dir/prompt_continuation_prompt_history.json"

  cp "$OUT_ROOT/$id/prompt_alignment_compare.json" "$OUT_ROOT/$id/prompt_alignment_compare_${id}.json"
  cleanup_server
  SERVER_PID=""
  echo "=== $id: PASS ==="
done

OUT_ROOT="$OUT_ROOT" python - <<'PY'
import json
import os
from pathlib import Path
root=Path(os.environ["OUT_ROOT"])
rows=[]
for p in sorted(root.glob('*/prompt_alignment_compare.json')):
    d=json.load(open(p))
    rows.append({
        'id': p.parent.name,
        'equal': d['events_equal_ignoring_velocity'],
        'ref_count': d['reference_event_count'],
        'cli_count': d['cli_event_count'],
        'sha': d['reference_sha256'],
    })
print(json.dumps(rows, indent=2))
if not all(r['equal'] for r in rows):
    raise SystemExit(2)
PY
