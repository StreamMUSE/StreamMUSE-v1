---
title: MidiDeviceInput — MIDI 设备输入
description: 从实体 MIDI 设备读取 note_on/note_off 消息
---

# MidiDeviceInput — MIDI 设备输入

**源文件**：`src/streammuse/infrastructure/input/midi_device.py`

从实体 MIDI 键盘或其他 MIDI 输入设备读取事件，是生产环境的默认输入方式。

---

## `MidiDeviceInput`

### `__init__(device_name: str | None = None) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `device_name` | `str \| None` | MIDI 设备名称；`None` 时 `mido` 自动选择系统中第一个可用设备 |

---

### `read_events() -> Iterator[MusicalEvent]`

打开 MIDI 端口（首次调用时），轮询（`port.poll()`）消息。每条 `note_on`（velocity > 0）或 `note_off`（含 velocity=0 的 note_on）消息都会 yield 一个 `MusicalEvent`。

所有发出的事件均为：
- `tick=0`（Application 层负责赋值）
- `channel` 来源于 MIDI 消息
- `velocity=0` 对应 NOTE_OFF

使用轮询（`poll`）而非阻塞（`receive`），以便 `close()` 能及时停止迭代器，无需等待下一条消息到来。

### `close() -> None`

设置 `_running = False`，关闭 MIDI 端口。如果在 `read_events()` 阻塞时调用，迭代器在下次轮询时会检测到 `_running=False` 并退出。

---

## 设备名称查询

```python
import mido
print(mido.get_input_names())   # 列出所有可用 MIDI 输入设备名称
```

通过 CLI 的 `--midi-device-name` 参数指定设备名称。
