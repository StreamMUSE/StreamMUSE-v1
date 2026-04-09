---
title: 输入模式
description: 四种输入源（keyboard、midi_device、midi_file、list）的使用说明
---

# 输入模式

StreamMUSE 支持四种输入模式，通过 `--input-mode` 参数指定。

---

## keyboard — 计算机键盘

使用计算机键盘模拟钢琴演奏，无需任何 MIDI 设备。

```bash
uv run streammuse-cli --input-mode keyboard
```

### 键位映射

```
白键（底行）:  Z  X  C  V  B  N  M  ,  .  /
音符:         C4 D4 E4 F4 G4 A4 B4 C5 D5 E5

黑键（顶行）:  S  D  G  H  J  L  ;
音符:         C#4 D#4 F#4 G#4 A#4 C#5 D#5
```

按下按键发出 NOTE_ON，松开发出 NOTE_OFF。支持和弦（多键同时按下）。

---

## midi_device — MIDI 设备

连接实体 MIDI 键盘或控制器。

```bash
# 自动选择第一个可用 MIDI 输入设备
uv run streammuse-cli --input-mode midi_device

# 指定设备名
uv run streammuse-cli --input-mode midi_device --midi-device-name "Arturia KeyLab 61"
```

查看可用设备：

```python
import mido
print(mido.get_input_names())
```

---

## midi_file — MIDI 文件模拟

从 `.mid` 文件读取旋律，按实际速度实时播放，适合自动化测试和演示。

```bash
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/C_major/pop909_216_mel.mid
```

### 相关参数

| 参数 | 说明 |
|---|---|
| `--midi-file-path` | MIDI 文件路径（必填） |
| `--midi-file-delay-ticks` | 文件开始前的延迟 ticks（默认 0） |
| `--tempo` | 播放速度（默认 120） |

系统会按配置的 BPM 播放 MIDI 文件，文件结束后输入流自动停止。

---

## list（测试/内部使用）

使用 Python 列表中的预定义事件，主要用于单元测试和集成测试。

```python
# 通过 InputSourceFactory 使用
factory.create(config, list_events=[ev1, ev2, ...])
```

CLI 层虽然允许 `--input-mode list`，但命令行并没有参数可直接注入 `list_events`；该模式主要用于测试代码通过工厂直接传入事件列表。

---

## 各模式特性对比

| 特性 | keyboard | midi_device | midi_file | list |
|---|---|---|---|---|
| 需要外部设备 | 否 | 是（MIDI 键盘） | 否 | 否 |
| 实时演奏 | 是 | 是 | 模拟（实时速率） | 否 |
| 适合自动化测试 | 否 | 否 | 是 | 是 |
| 依赖 `pynput` | 是 | 否 | 否 | 否 |
| 依赖 `mido` | 否 | 是 | 是 | 否 |

---

## 通用行为

所有输入模式发出的 `MusicalEvent` 都设置 `tick=0`，由 Application 层根据时间戳自动赋值实际 tick。
