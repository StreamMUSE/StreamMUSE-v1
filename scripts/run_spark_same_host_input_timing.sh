#!/usr/bin/env bash
set -euo pipefail

REPO=/home/xiaosongma/StreamMUSE-same-tick-fix-acceptance-20260909
PYTHON=/home/xiaosongma/StreamMUSE-v1/.venv/bin/python
SOURCE=/home/xiaosongma/experiments/human6_timed_input_plain_lekai_20260909
ROOT=${OUT_ROOT:-/home/xiaosongma/experiments/human2_input_timing_spark_local_20260909}
CHECKPOINT=/home/xiaosongma/models/streammuse_prompt_continuation/continuation/model.safetensors
PORT=18845
PIECES=("$@")
if [[ ${#PIECES[@]} -eq 0 ]]; then
    PIECES=(01_tong_hua 02_night_dancer)
fi
cd "$REPO"
export PYTHONPATH="$REPO/src"

if [[ -e "$ROOT" ]]; then
    echo "Refusing to overwrite $ROOT" >&2
    exit 1
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "GPU has an active compute process; refusing to start" >&2
    exit 1
fi
printf '%s  %s\n' d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271 "$CHECKPOINT" | sha256sum -c -
"$PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",18845)); s.close()'
mkdir "$ROOT"
cp "$SOURCE/plan.json" "$ROOT/plan.json"
cp -a "$SOURCE/inputs" "$ROOT/inputs"
mkdir "$ROOT/executed_scripts"
cp scripts/plain_lekai_device_replay_acceptance.py scripts/human6_timed_input_replay.py scripts/device_replay_acceptance.py "$0" "$ROOT/executed_scripts/"
nvidia-smi > "$ROOT/gpu_before.txt"

"$PYTHON" -u scripts/plain_lekai_device_replay_acceptance.py server \
    --out "$ROOT/server" --gpu 0 --port "$PORT" --host 127.0.0.1 \
    --checkpoint-path "$CHECKPOINT" --system-ref ec843a79 > "$ROOT/server.log" 2>&1 &
BACKEND_PID=$!
cleanup() {
    if kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill -TERM "$BACKEND_PID"
    fi
    wait "$BACKEND_PID" || true
}
trap cleanup EXIT
"$PYTHON" - "$BACKEND_PID" <<'PY'
import os, sys, time, urllib.request, json
pid = int(sys.argv[1])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:18845/runtime_info', timeout=2) as r:
            print('Backend ready:', json.load(r), flush=True)
        break
    except OSError:
        time.sleep(1)
else:
    raise TimeoutError('Backend did not become ready')
PY
CUDA_VISIBLE_DEVICES='' "$PYTHON" -u scripts/human6_timed_input_replay.py run \
    --root "$ROOT" --server "http://127.0.0.1:$PORT" --input-only \
    --pieces "${PIECES[@]}" > "$ROOT/client.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/human6_timed_input_replay.py check-inputs \
    --root "$ROOT" > "$ROOT/input_checks.json"
echo "Completed: $ROOT"
