# StreamMUSE 低延迟演奏设置（Mac 与 Spark 两端）

把键盘按键到扬声器发声的延迟压到最低。StreamMUSE 本身不发声——它通过
`--midi-out-port` 把 MIDI 发给一个外部合成器（fluidsynth）。这份文档说明在
**macOS 开发机**和 **Spark（Linux）** 上分别怎么配这个合成器。

信号链：

```
Casio 键盘 ──MIDI──▶ [ fluidsynth 合成 ] ──音频──▶ USB 扬声器
                          ▲
                   StreamMUSE 也把生成的伴奏 MIDI 发到同一个 fluidsynth
```

两端结论对照（均为实测，MIDI 到发声）：

| 平台 | 合成器音频后端 | 缓冲区 | 延迟 | 要不要补丁 |
|---|---|---|---|---|
| **macOS** | CoreAudio | 16 帧 | **≈ 2.6 ms** | **要**（见下）|
| **Spark (Linux)** | ALSA 直连 | 64 帧 ×2 | **≈ 2.67 ms** | 不要 |

> 关键：**低延迟必须用 USB 音频接口，不要用笔记本内置扬声器**——MacBook 内置扬声器有
> ~690 帧（14.4 ms）的 DSP 延迟无法消除。

---

## A. macOS（开发机，弹 Casio 的地方）

macOS 上 fluidsynth 的 CoreAudio 驱动**从不设置设备缓冲区**，所以延迟被钉在
~10.7 ms，`audio.period-size` 无效。必须用打补丁的 fluidsynth。

### 1. 构建打补丁的 fluidsynth（一次）

```sh
cd patch_fluidsynth
brew install cmake pkg-config glib libsndfile   # 依赖
./apply_patch.sh                                # 下载 2.5.6、打补丁、编译
```

产物：`patch_fluidsynth/build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth`

### 2. 找到 USB 扬声器的设备名

```sh
FS=patch_fluidsynth/build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth
DYLD_LIBRARY_PATH="$(dirname "$FS")" "$FS" -a coreaudio -o help 2>&1 | grep audio.coreaudio.device
# 例如输出里有 'USBAudio2.0'
```

### 3. 启动低延迟 fluidsynth（16 帧）

```sh
FS=patch_fluidsynth/build-fluidsynth/fluidsynth-2.5.6/build/src/fluidsynth
DYLD_LIBRARY_PATH="$(dirname "$FS")" "$FS" -s -i -g 0.5 \
  -a coreaudio -o audio.coreaudio.device='USBAudio2.0' \
  -o audio.period-size=16 -o audio.periods=2 \
  -o synth.sample-rate=48000 \
  -m coremidi -o midi.portname=StreamMUSE -o midi.autoconnect=0 \
  /path/to/soundfont.sf2
```

这会创建一个名为 `StreamMUSE` 的 CoreMIDI 端口。

### 4. 连 Casio + 启动 StreamMUSE

Casio 直接弹：CoreMIDI 会让 fluidsynth 收到（若未自动连，用 macOS「音频 MIDI 设置」
或让 StreamMUSE 转发）。让 StreamMUSE 把伴奏发到同一端口：

```sh
uv run streammuse-cli --output-type audio --midi-out-port "StreamMUSE"
```

---

## B. Spark（Linux 部署机，`xiaosongma@10.127.123.250`）

Linux 上 **ALSA 的 `audio.period-size` 直接控制硬件缓冲，无需补丁**。直连
`hw:1,0`（USB 扬声器）绕过 PipeWire，延迟最低。

### 1. 确认设备

```sh
export XDG_RUNTIME_DIR=/run/user/$(id -u)
aplay -l | grep -i usb                 # USB 扬声器的 card 号，例如 card 1 -> hw:1,0
aconnect -i | grep -i casio            # Casio 的 ALSA seq 客户端号，例如 client 24
```

### 2. 启动 fluidsynth（64 帧 + 实时优先级）

64 帧这么小的缓冲，**必须要实时调度**才不会爆音，所以用 `sudo`（否则 fluidsynth
拿不到 SCHED_FIFO，会打印 `Failed to set thread to high priority`）：

```sh
sudo fluidsynth -si -g 1.0 \
  -a alsa -o audio.alsa.device=hw:1,0 \
  -o audio.period-size=64 -o audio.periods=2 -o synth.sample-rate=48000 \
  -o audio.realtime-prio=70 \
  -m alsa_seq -o midi.alsa_seq.id=fsynth \
  /usr/share/sounds/sf2/FluidR3_GM.sf2 &
```

验证实时优先级拿到了（音频线程应为 `FF`/SCHED_FIFO）：

```sh
ps -eLo pid,cls,rtprio,comm | grep -i 'alsa-audio'   # 期望 cls=FF, rtprio=70
cat /proc/asound/card1/pcm0p/sub0/hw_params | grep -E 'period_size|buffer_size'
# 期望 period_size: 64 / buffer_size: 128  = 2.67 ms
```

### 3. 连 Casio → fluidsynth

```sh
FCLIENT=$(aconnect -o | grep -i 'FLUID Synth' | grep -oE 'client [0-9]+' | grep -oE '[0-9]+' | head -1)
aconnect 24:0 ${FCLIENT}:0        # 24 = Casio；按第 1 步查到的实际号替换
```

现在弹 Casio 即可出声。让 StreamMUSE 也发到这个 seq 端口：

```sh
uv run streammuse-cli --output-type audio --midi-out-port "FLUID Synth (fsynth)"
```

### Spark 上的物理极限

- 这个 USB 扬声器每 1 ms 一个数据包（USB isochronous），硬件最小 period = **48 帧 = 1 ms**。
  fluidsynth 软件下限是 64，所以实际最低到 64×2=128 帧 = 2.67 ms。
- 想榨到 48 帧（≈2.0 ms，省 0.67 ms）需要给 Spark 上的 fluidsynth 也打「period 下限 64→16」
  的补丁并重编译——收益很小，一般不值得。

---

## 排错

| 现象 | 处理 |
|---|---|
| macOS 打补丁后静音 | `period-size × periods` 必须 ≥ 设备 IO buffer；保持 `periods` 默认（≥2）。|
| Linux 爆音 / 卡顿 | 确认实时优先级（`cls=FF`）；用 `sudo` 启动；必要时调回 `period-size=128`。|
| `pactl/wpctl` 连不上 | SSH 会话要 `export XDG_RUNTIME_DIR=/run/user/$(id -u)`。|
| 听不到声 | 先用 `aplay -D hw:1,0 tone.wav`（Linux）或系统输出测试确认扬声器本身通。|
| Casio 没被识别 | Linux 看 `aconnect -i` 里的 `CASIO`；`/proc/asound/seq/clients` 也能确认。|
