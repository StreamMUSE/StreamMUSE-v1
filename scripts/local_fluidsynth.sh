#!/usr/bin/env bash
set -euo pipefail

ROOT="${STREAMMUSE_FLUIDSYNTH_ROOT:?set STREAMMUSE_FLUIDSYNTH_ROOT to the extracted rootfs}"
LIB_DIR="$ROOT/usr/lib/x86_64-linux-gnu"
PULSE_LIB_DIR="$LIB_DIR/pulseaudio"

if [[ ! -x "$ROOT/usr/bin/fluidsynth" ]]; then
  echo "missing local FluidSynth executable: $ROOT/usr/bin/fluidsynth" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$LIB_DIR:$PULSE_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/usr/bin/fluidsynth" "$@"
