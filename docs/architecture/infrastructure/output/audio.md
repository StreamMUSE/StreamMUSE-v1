---
title: AudioOutputSink — 实时音频播放
description: 向 MIDI 输出端口发送音符，实现实时音频播放
---

# AudioOutputSink — 实时音频播放

**源文件**：`src/streammuse/infrastructure/output/audio.py`

通过 `mido` 向 MIDI 输出端口发送 `note_on`/`note_off` 消息，实现实时音频播放。

---

## `AudioOutputConfig`

```python
@dataclass(frozen=True)
class AudioOutputConfig:
    port_name: Optional[str] = None
    default_program: int = 0
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `port_name` | `Optional[str]` | `None` | MIDI 输出端口名称；`None` 时 `mido` 自动选择第一个可用端口 |
| `default_program` | `int` | `0` | 初始化时发送的 Program Change 值（乐器编号） |

---

## `AudioOutputSink`

### `output_event(event, source)`

处理 `NOTE_ON` 和 `NOTE_OFF` 事件：

- 跳过 `is_placeholder=True` 或 `pitch=-1` 的事件
- `NOTE_ON`（velocity > 0）：发送 `mido.Message("note_on", note, velocity, channel)`
- `NOTE_OFF` 或 `NOTE_ON`（velocity=0）：发送 `mido.Message("note_off", note, 0, channel)`

**端口延迟初始化**：首次 `output_event()` 调用时才打开 MIDI 端口（`_ensure_port()`），避免在 `AudioOutputSink` 创建时就占用端口资源。

### `output_tick`, `output_stats`, `output_status`, `output_config`

均为空操作（no-op）。Audio Sink 只关心实际音符事件。

### `close()`

关闭 MIDI 输出端口。

---

## 设备名称查询

```python
import mido
print(mido.get_output_names())   # 列出所有可用 MIDI 输出设备
```

通过 CLI 的 `--midi-out-port` 参数指定端口名称。
