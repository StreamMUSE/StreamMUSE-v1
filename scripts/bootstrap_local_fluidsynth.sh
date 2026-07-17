#!/usr/bin/env bash
set -euo pipefail

# Install a reproducible-enough FluidSynth runtime without root privileges.
# The formal listening freeze records hashes for every extracted package,
# executable, shared library, and soundfont; this bootstrap never modifies the
# host package database.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ROOT="${1:-${STREAMMUSE_FLUIDSYNTH_ROOT:-$REPO_ROOT/task_runs/toolchains/fluidsynth-2.3.4}}"
PACKAGES_DIR="$ROOT/packages"

command -v apt >/dev/null
command -v dpkg-deb >/dev/null
command -v sha256sum >/dev/null

mkdir -p "$PACKAGES_DIR" "$ROOT/rootfs"

packages=(
  fluidsynth
  libfluidsynth3
  libinstpatch-1.0-2
  libpipewire-0.3-0t64
  libsdl2-2.0-0
  libjack-jackd2-0
  libpulse0
  libsndfile1
  fluid-soundfont-gm
  libsamplerate0
  libdecor-0-0
  libflac12t64
  libvorbis0a
  libvorbisenc2
  libopus0
  libogg0
  libmpg123-0t64
  libmp3lame0
  libasyncns0
)

if [[ ! -x "$ROOT/rootfs/usr/bin/fluidsynth" || \
      ! -f "$ROOT/rootfs/usr/share/sounds/sf2/FluidR3_GM.sf2" ]]; then
  (
    cd "$PACKAGES_DIR"
    apt download "${packages[@]}"
  )
  for package in "$PACKAGES_DIR"/*.deb; do
    dpkg-deb -x "$package" "$ROOT/rootfs"
  done
fi

find "$PACKAGES_DIR" -maxdepth 1 -type f -name '*.deb' -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$ROOT/package_sha256.txt"

export STREAMMUSE_FLUIDSYNTH_ROOT="$ROOT/rootfs"
"$SCRIPT_DIR/local_fluidsynth.sh" --version

echo "STREAMMUSE_FLUIDSYNTH_ROOT=$STREAMMUSE_FLUIDSYNTH_ROOT"
echo "FLUIDSYNTH_BINARY=$STREAMMUSE_FLUIDSYNTH_ROOT/usr/bin/fluidsynth"
echo "FLUIDSYNTH_LAUNCHER=$SCRIPT_DIR/local_fluidsynth.sh"
echo "SOUNDFONT=$STREAMMUSE_FLUIDSYNTH_ROOT/usr/share/sounds/sf2/FluidR3_GM.sf2"
