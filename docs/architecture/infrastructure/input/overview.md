---
title: infrastructure/input — 输入适配器总览
description: 四种 InputSource 实现的比较与设计约定
---

# infrastructure/input — 输入适配器总览

**源文件**：`src/streammuse/infrastructure/input/`

所有输入适配器均实现 `InputSource` 协议（见 [domain/interfaces](../../domain/interfaces.md)），提供 `read_events()` 迭代器和 `close()` 方法。

---

## 共同设计约定

**`tick=0` 约定**：所有适配器发出的 `MusicalEvent` 中 `tick` 均设为 `0`，由 Application 层（`RealTimeMusicService._input_worker`）根据 `tempo.seconds_to_tick(elapsed)` 重新赋值实际 tick 值。

这一分离使适配器只管"读取和发出事件"，而不管"时间"，测试时也无需模拟时钟。

---

## 四种实现对比

| 类 | 输入源 | 使用场景 | 依赖 |
|---|---|---|---|
| `KeyboardInput` | 计算机键盘 | 开发调试、无 MIDI 硬件时演示 | `pynput` |
| `MidiDeviceInput` | 实体 MIDI 设备 | 生产环境，键盘演奏者 | `mido` |
| `MidiFileInput` | `.mid` 文件（实时模拟） | 基准测试、MIDI 文件回放 | `mido` |
| `ListInput` | Python `list` | 单元测试、预录序列回放 | 无 |

---

## 详细文档

- [keyboard.md](keyboard.md) — `KeyboardInput`：键盘映射与 pynput 集成
- [midi_device.md](midi_device.md) — `MidiDeviceInput`：MIDI 硬件输入
- [midi_file.md](midi_file.md) — `MidiFileInput`：MIDI 文件实时模拟
- [list_input.md](list_input.md) — `ListInput`：测试用内存事件列表
