#!/usr/bin/env bash
# Run StreamMUSE CLI against the lekai HTTP backend.
#
# Usage:
#   bash scripts/run_lekai.sh                    # live keyboard input (default)
#   INPUT_MODE=midi_file bash scripts/run_lekai.sh prompts/old_input/mel/001.mid
#
# Env overrides:
#   INPUT_MODE                 (default midi_device; also: midi_file)
#   MIDI_INPUT_PORT            (auto-detect by MIDI_INPUT_MATCH if unset)
#   MIDI_INPUT_MATCH           (default "Keystation")
#   SERVER_URL                 (default http://127.0.0.1:8988)
#   GENERATION_INTERVAL_TICKS  (default 4)
#   GENERATION_LENGTH_FRAMES   (default 4)
#   MAX_TICKS                  (default 128; ignored if INPUT_MODE=midi_device unless set)
#   TEMPO                      (default 30 BPM)
#   OUTPUT_TYPE                (default audio   -- set to "console" to disable sound)
#   MIDI_OUT_PORT              (auto-detect FLUID Synth if unset)
#   MIDI_PORT_MATCH            (default "FLUID Synth")
#   TICKS_PER_BEAT             (default 4)
#   BEATS_PER_BAR              (default 4)
#   PROMPT_LENGTH_TICKS        (default 32)
#   ACC_VELOCITY_SCALE         (default 0.5; multiplier on model note velocities)
#   LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS  (default 1; preserves catchup events that the clip+pair algorithm would otherwise drop at the tick boundary)
#   STREAMMUSE_VERBOSE         (default 1; set to 0 to silence per-tick/event console logs)
#   TOP_K                      (override lekai sampling top_k; restarts server if differs)

set -euo pipefail

cd "$(dirname "$0")/.."

# TOP_K handling: restart lekai server if requested top_k differs from current.
TOP_K="${TOP_K:-}"
if [[ -n "$TOP_K" ]]; then
  current_top_k=""
  pid_file="$HOME/RTAS_demo/pids/lekai.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if [[ -r "/proc/$pid/environ" ]]; then
      current_top_k="$(tr '\0' '\n' </proc/$pid/environ | sed -n 's/^LEKAI_RT_TOP_K=//p')"
    fi
  fi
  if [[ "$current_top_k" != "$TOP_K" ]]; then
    echo "Restarting lekai server: LEKAI_RT_TOP_K ${current_top_k:-<none>} -> $TOP_K"
    kill "$(cat "$pid_file" 2>/dev/null)" 2>/dev/null || true
    sleep 2
    rm -f "$pid_file"
    LEKAI_RT_TOP_K="$TOP_K" bash "$HOME/RTAS_demo/services.sh" start >/dev/null
  else
    echo "lekai already running with LEKAI_RT_TOP_K=$TOP_K"
  fi
fi

# Locate uv (may not be on PATH in non-interactive shells).
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
  UV_BIN="$HOME/.cargo/bin/uv"
else
  echo "uv not found on PATH or in standard locations" >&2
  exit 1
fi

# Smart default: file arg passed but INPUT_MODE unset -> use midi_file.
if [[ -n "${1:-}" && -z "${INPUT_MODE:-}" ]]; then
  INPUT_MODE="midi_file"
fi
INPUT_MODE="${INPUT_MODE:-midi_device}"
MIDI_FILE="${1:-prompts/old_input/mel/001.mid}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8988}"
GENERATION_INTERVAL_TICKS="${GENERATION_INTERVAL_TICKS:-4}"
GENERATION_LENGTH_FRAMES="${GENERATION_LENGTH_FRAMES:-4}"
RAW_MAX_TICKS="${MAX_TICKS:-}"
MAX_TICKS="${MAX_TICKS:-128}"
TEMPO="${TEMPO:-30}"
OUTPUT_TYPE="${OUTPUT_TYPE:-audio}"
MIDI_PORT_MATCH="${MIDI_PORT_MATCH:-FLUID Synth}"
MIDI_INPUT_MATCH="${MIDI_INPUT_MATCH:-Keystation}"
TICKS_PER_BEAT="${TICKS_PER_BEAT:-4}"
BEATS_PER_BAR="${BEATS_PER_BAR:-4}"
PROMPT_LENGTH_TICKS="${PROMPT_LENGTH_TICKS:-32}"

# Auto-detect MIDI INPUT port if requested mode is midi_device.
if [[ "$INPUT_MODE" == "midi_device" ]]; then
  if [[ -z "${MIDI_INPUT_PORT:-}" ]]; then
    MIDI_INPUT_PORT="$("$UV_BIN" run --quiet python - <<PY
import sys, mido
match = ${MIDI_INPUT_MATCH@Q}
for n in mido.get_input_names():
    if match in n and "Transport" not in n:  # skip Transport sub-port
        print(n); sys.exit(0)
sys.exit(1)
PY
)" || {
      echo "ERROR: no MIDI input device matching '$MIDI_INPUT_MATCH' was found." >&2
      echo "Available inputs:" >&2
      "$UV_BIN" run --quiet python -c 'import mido; [print(" -", repr(n)) for n in mido.get_input_names()]' >&2 || true
      exit 1
    }
  fi
  echo "Reading MIDI from: $MIDI_INPUT_PORT"
fi

# Auto-detect MIDI OUTPUT port for audio sink.
if [[ "$OUTPUT_TYPE" == "audio" || "$OUTPUT_TYPE" == "composite" ]]; then
  if [[ -z "${MIDI_OUT_PORT:-}" ]]; then
    MIDI_OUT_PORT="$("$UV_BIN" run --quiet python - <<PY
import sys, mido
match = ${MIDI_PORT_MATCH@Q}
for n in mido.get_output_names():
    if match in n:
        print(n)
        sys.exit(0)
sys.exit(1)
PY
)" || {
      echo "ERROR: no MIDI output port matching '$MIDI_PORT_MATCH' was found." >&2
      echo "Is fluidsynth running?  Try: bash ~/RTAS_demo/services.sh start" >&2
      exit 1
    }
  fi
  echo "Routing MIDI to: $MIDI_OUT_PORT"
fi

ARGS=(
  --input-mode "$INPUT_MODE"
  --inference-type http
  --model-name lekai_prompt_continuation
  --server-url "$SERVER_URL"
  --generation-interval-ticks "$GENERATION_INTERVAL_TICKS"
  --generation-length-frames "$GENERATION_LENGTH_FRAMES"
  --tempo "$TEMPO"
  --output-type "$OUTPUT_TYPE"
  --ticks-per-beat "$TICKS_PER_BEAT"
  --beats-per-bar "$BEATS_PER_BAR"
  --prompt-length-ticks "$PROMPT_LENGTH_TICKS"
)

if [[ "$INPUT_MODE" == "midi_file" ]]; then
  ARGS+=(--midi-file-path "$MIDI_FILE" --max-ticks "$MAX_TICKS")
elif [[ "$INPUT_MODE" == "midi_device" ]]; then
  ARGS+=(--midi-device-name "$MIDI_INPUT_PORT")
  # midi_device runs indefinitely until Ctrl+C; only pass --max-ticks if user set it.
  if [[ -n "${RAW_MAX_TICKS:-}" ]]; then
    ARGS+=(--max-ticks "$MAX_TICKS")
  fi
fi

if [[ -n "${MIDI_OUT_PORT:-}" ]]; then
  ARGS+=(--midi-out-port "$MIDI_OUT_PORT")
fi

ACC_VELOCITY_SCALE="${ACC_VELOCITY_SCALE:-0.5}"
export ACC_VELOCITY_SCALE
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS="${LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS:-1}"
export LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS
STREAMMUSE_VERBOSE="${STREAMMUSE_VERBOSE:-1}"
export STREAMMUSE_VERBOSE
echo "Accompaniment velocity scale: $ACC_VELOCITY_SCALE"

exec "$UV_BIN" run streammuse-cli "${ARGS[@]}"
