# patch_fluidsynth — FluidSynth 低延迟补丁（macOS）

让 FluidSynth 在 macOS 上真正低延迟运行，供 StreamMUSE 在 Mac 上做实时演奏
（键盘 → StreamMUSE → FluidSynth → 扬声器）时使用。

> **完整的双平台演奏配置（Mac 与 Spark/Linux 两端各怎么弄）见 [`LOW_LATENCY_SETUP.md`](LOW_LATENCY_SETUP.md)。**
> 本文件只讲 macOS 补丁本身。

## 平台范围

**这个补丁只对 macOS 有意义。** 它修改的是 FluidSynth 的 CoreAudio 驱动。

| 平台 | 需要补丁吗 | 原因 |
|---|---|---|
| **macOS** | **是** | FluidSynth 的 CoreAudio 驱动从不设置设备 IO 缓冲，延迟被钉在 ~10.7 ms（512 帧），且 `audio.period-size` 无效。 |
| **Linux (ALSA)** | 否 | ALSA 驱动的 `audio.period-size` 直接映射到硬件周期，本来就能低延迟。 |
| **Linux (JACK)** | 否 | 延迟由 JACK 的 buffer 决定。 |

服务器端（Linux GPU 机器）跑推理后端，不做音频播放，用不到本补丁。

## 补丁做了什么（两处改动）

改自 FluidSynth **2.5.6**：

1. `src/drivers/fluid_coreaudio.c` — 在 `AudioUnitInitialize` 前，于 AUHAL 输出单元的
   global scope 上调用一次 `AudioUnitSetProperty(kAudioDevicePropertyBufferFrameSize,
   period_size)`。原版从不设这个属性，所以设备始终停在系统默认 512 帧；
   `MaximumFramesPerSlice` 只是上限，不是 IO 周期。补丁让整个设备以 `period-size`
   帧运行，`period-size` 才真正决定输出延迟。

2. `src/drivers/fluid_adriver.c` — 把 `audio.period-size` 下限从 64 放宽到 16
   （CoreAudio 设备实测支持到 15/16 帧）。

## 实测效果（Casio 键盘 + USB 扬声器，macOS，MIDI 驱动 → DAC）

| 缓冲区 | 补丁前 | 补丁后 |
|---|---|---|
| 512 帧（默认） | 18.3 ms | — |
| 16 帧 | 无法达到 | **2.6 ms** |

音色不受影响：同一个 SoundFont，补丁前后完全一致，只是延迟变小。
（注：内置 MacBook 扬声器有约 690 帧的 DSP 延迟无法消除，低延迟务必用 USB 音频接口。）

## 用法

```sh
# macOS：自动下载 2.5.6 源码、打补丁、编译
./apply_patch.sh
```

脚本会：
- 检测平台（Linux 会直接说明无需补丁并给出 ALSA 低延迟命令）；
- 下载 FluidSynth 2.5.6 源码到 `./build-fluidsynth/`；
- 应用本目录的 `.patch`（幂等，已打过会跳过）；
- 用 CMake 编译，产出 `build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth`；
- 验证 `period-size` 下限已变为 16。

依赖（macOS）：`brew install cmake pkg-config glib libsndfile`

## 接入 StreamMUSE

打补丁的 FluidSynth 用 `-m coremidi -o midi.portname=StreamMUSE` 会创建一个名为
`StreamMUSE` 的 CoreMIDI 端口。启动它（16 帧缓冲）：

```sh
DYLD_LIBRARY_PATH=build-fluidsynth/fluidsynth-2.5.6/build/src \
build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth -s -i -g 0.5 \
  -a coreaudio -o audio.coreaudio.device='USBAudio2.0' \
  -o audio.period-size=16 -o audio.periods=2 \
  -o synth.sample-rate=48000 \
  -m coremidi -o midi.portname=StreamMUSE -o midi.autoconnect=0 \
  your_soundfont.sf2
```

然后让 StreamMUSE 把生成的伴奏发到这个端口：

```sh
uv run streammuse-cli --output-type audio --midi-out-port "StreamMUSE"
```

列出可用的 CoreAudio 输出设备名：

```sh
build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth -a coreaudio -o help 2>&1 | grep audio.coreaudio.device
```

## 文件

| 文件 | 说明 |
|---|---|
| `fluidsynth-2.5.6-lowlatency-coreaudio.patch` | 统一 diff，`patch -p1` 应用于 FluidSynth 2.5.6 源码根 |
| `apply_patch.sh` | 平台感知的自动构建脚本 |
| `README.md` | 本文件 |

## 注意

- 补丁针对 FluidSynth **2.5.6**；升级到其他版本需重新适配（上游无此功能）。
- `period-size × periods`（= `MaximumFramesPerSlice`）必须 ≥ 设备 IO buffer，
  否则 AudioUnit 拒绝渲染、静默输出零。补丁把设备 IO buffer 设成 `period-size`，
  `periods ≥ 2`，所以上限自动够用，保持 `periods` 默认即可。
- 缓冲越小，回调越密（16 帧 @48k ≈ 3000 次/秒）。较弱机器或较重 SoundFont
  若出现 xrun/爆音，调回 32 或 64。
