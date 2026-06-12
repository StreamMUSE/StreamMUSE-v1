---
title: 输入模式
description: 四种输入源（keyboard、midi_device、midi_file、list）的使用说明
---

# 输入模式

StreamMUSE 支持四种输入模式，通过 `--input-mode` 参数指定。

---

## keyboard — 计算机键盘

使用计算机键盘模拟钢琴演奏，无需 MIDI 设备。

```bash
uv run streammuse-cli --input-mode keyboard
```

### 键位映射

```text
白键（底行）:  Z  X  C  V  B  N  M  ,  .  /
音符:         C4 D4 E4 F4 G4 A4 B4 C5 D5 E5

黑键（顶行）:  S  D  G  H  J  L  ;
音符:         C#4 D#4 F#4 G#4 A#4 C#5 D#5
```

按下按键发出 NOTE_ON，松开发出 NOTE_OFF。支持和弦。

---

## midi_device — MIDI 设备

连接实体 MIDI 键盘或控制器。

```bash
uv run streammuse-cli --input-mode midi_device

uv run streammuse-cli \
  --input-mode midi_device \
  --midi-device-name "Arturia KeyLab 61"
```

查看可用设备：

```python
import mido
print(mido.get_input_names())
```

---

## midi_file — MIDI 文件模拟

从 `.mid` 文件读取旋律，按配置 tempo 实时播放，适合自动化测试和演示。

```bash
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/C_major/pop909_216_mel.mid
```

### 相关参数

| 参数 | 说明 |
|---|---|
| `--midi-file-path` | MIDI 文件路径（必填） |
| `--midi-file-delay-ticks` | 文件开始前的延迟 ticks，默认 0 |
| `--tempo` | 播放速度，默认 120 |
| `--injection-file` | 可选：会话开始前注入的 melody MIDI 文件 |
| `--injection-length` | 可选：注入前多少 ticks |
| `--inject-acc-file` | 可选：注入 accompaniment MIDI 文件 |

使用 injection 时，`MidiFileInput` 会从 `injection_length` 后开始播放，避免重复输入已注入片段。

---

## list（测试/内部使用）

使用 Python 列表中的预定义事件，主要用于单元测试和集成测试。

```python
factory.create(config, list_events=[ev1, ev2, ...])
```

CLI 层虽然允许 `--input-mode list`，但命令行没有参数可直接注入 `list_events`；该模式主要用于测试代码通过工厂直接传入事件列表。

---

## count-in 对输入的影响

`--count-in-beats` 会推迟正式输入读取：

1. `RealTimeMusicService` 先输出 count-in metronome。
2. `_input_worker` 睡到 `timeline_start_time` 后才开始读取 input source。
3. 正式输入事件从 tick=0 起按 wall-clock 打 tick。

---

## 各模式特性对比

| 特性 | keyboard | midi_device | midi_file | list |
|---|---|---|---|---|
| 需要外部设备 | 否 | 是 | 否 | 否 |
| 实时演奏 | 是 | 是 | 模拟实时 | 否 |
| 适合自动化测试 | 否 | 否 | 是 | 是 |
| 支持 CLI injection | 否 | 否 | 是 | 否 |
| 依赖 `pynput` | 是 | 否 | 否 | 否 |
| 依赖 `mido` | 否 | 是 | 是 | 否 |

---

## 通用行为

InputSource 发出的原始 `MusicalEvent` 通常使用 `tick=0`；Application 层会根据正式时间线的 wall-clock 重新赋值实际 tick。
