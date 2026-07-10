#!/usr/bin/env bash
#
# 为 StreamMUSE 构建一个低延迟的 FluidSynth。
#
# 平台范围：这个补丁是 **macOS / CoreAudio 专用** 的。它解决的是 FluidSynth 的
# CoreAudio 驱动从不设置设备 IO 缓冲区、导致延迟固定在 ~10.7ms 的问题。
#
#   - macOS：下载 FluidSynth 2.5.6 源码 → 打补丁 → 编译。产出低延迟二进制。
#   - Linux：CoreAudio 补丁不适用。ALSA 驱动的 audio.period-size 本来就直接控制
#            硬件缓冲，无需补丁 —— 脚本会说明如何直接低延迟运行系统 FluidSynth。
#
# 用法：
#   ./apply_patch.sh              # 自动检测平台并执行
#   ./apply_patch.sh --build-dir /path/to/dir   # 指定构建目录（默认 ./build-fluidsynth）

set -euo pipefail

FS_VERSION="2.5.6"
PATCH_FILE="$(cd "$(dirname "$0")" && pwd)/fluidsynth-${FS_VERSION}-lowlatency-coreaudio.patch"
BUILD_DIR="$(pwd)/build-fluidsynth"
TARBALL_URL="https://github.com/FluidSynth/fluidsynth/archive/refs/tags/v${FS_VERSION}.tar.gz"

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

info() { printf '\033[1;34m[patch]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

[ -f "$PATCH_FILE" ] || { err "找不到补丁文件: $PATCH_FILE"; exit 1; }

OS="$(uname -s)"

# ---------------------------------------------------------------------------
if [ "$OS" = "Linux" ]; then
  warn "检测到 Linux。"
  cat <<'EOF'

  这个补丁修改的是 FluidSynth 的 CoreAudio 驱动（fluid_coreaudio.c），
  只对 macOS 有意义。Linux 不使用 CoreAudio。

  Linux 上无需补丁即可低延迟运行：ALSA 驱动的 audio.period-size 直接映射到
  硬件周期。直接用系统 fluidsynth：

      fluidsynth -a alsa -o audio.period-size=64 -o audio.periods=2 \
        -o synth.sample-rate=48000 \
        -m alsa_seq your_soundfont.sf2

  （period-size=64 @48kHz ≈ 1.3ms/周期。低于 64 在 ALSA 上容易 xrun，通常没必要。）
  若用 JACK，延迟由 JACK 的 buffer 决定：jackd -d alsa -p 64。

  StreamMUSE 侧：用 --midi-out-port 指向该 fluidsynth 的 MIDI 端口。

EOF
  exit 0
fi

# ---------------------------------------------------------------------------
if [ "$OS" != "Darwin" ]; then
  err "不支持的平台: $OS（本补丁仅用于 macOS）"
  exit 1
fi

info "平台：macOS。将构建打补丁的低延迟 FluidSynth ${FS_VERSION}。"

# 依赖检查
MISSING=""
for tool in cmake pkg-config curl tar patch; do
  command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
if [ -n "$MISSING" ]; then
  err "缺少构建工具:$MISSING"
  echo "  用 Homebrew 安装： brew install cmake pkg-config" >&2
  exit 1
fi
for lib in glib libsndfile; do
  brew --prefix "$lib" >/dev/null 2>&1 || { err "缺少库 $lib —— 运行： brew install $lib"; exit 1; }
done

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

SRC="fluidsynth-${FS_VERSION}"
if [ ! -d "$SRC" ]; then
  info "下载 FluidSynth ${FS_VERSION} 源码…"
  curl -fsSL "$TARBALL_URL" -o "fluidsynth-${FS_VERSION}.tar.gz"
  tar xzf "fluidsynth-${FS_VERSION}.tar.gz"
fi

cd "$SRC"

# 幂等：已打过补丁就跳过
if grep -q "kAudioDevicePropertyBufferFrameSize" src/drivers/fluid_coreaudio.c; then
  info "源码已包含补丁，跳过 apply。"
else
  info "应用补丁…"
  patch -p1 --forward < "$PATCH_FILE"
fi

info "配置并编译…"
export PKG_CONFIG_PATH="$(brew --prefix glib)/lib/pkgconfig:$(brew --prefix libsndfile)/lib/pkgconfig:$(brew --prefix gettext)/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Denable-network=ON >/dev/null
cmake --build . -j"$(sysctl -n hw.ncpu)"

BIN="$(pwd)/src/fluidsynth"
LIB="$(pwd)/src"

# 验证：period-size 下限已降到 16
if "$BIN" -a coreaudio -o help 2>&1 | grep -q "audio.period-size.*min=16"; then
  info "验证通过：period-size 下限 = 16。"
else
  warn "补丁编译完成，但 period-size 下限未确认为 16，请检查。"
fi

cat <<EOF

\033[1;32m完成。\033[0m 低延迟 FluidSynth：
  $BIN

以 16 帧缓冲运行（把 <设备名> 和 <soundfont> 换成你的）：

  DYLD_LIBRARY_PATH="$LIB" "$BIN" -s -i -g 0.5 \\
    -a coreaudio -o audio.coreaudio.device='<设备名>' \\
    -o audio.period-size=16 -o audio.periods=2 \\
    -o synth.sample-rate=48000 \\
    -m coremidi -o midi.portname=StreamMUSE -o midi.autoconnect=0 \\
    <soundfont>.sf2

然后让 StreamMUSE 指向这个端口：

  uv run streammuse-cli --output-type audio --midi-out-port "StreamMUSE"

设备名可用以下命令列出：
  "$BIN" -a coreaudio -o help 2>&1 | grep audio.coreaudio.device
EOF
