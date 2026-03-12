---
title: MidiFileOutputSink — MIDI 文件录制
description: 将事件流录制为 MIDI 文件，使用 pretty_midi 重建音符
---

# MidiFileOutputSink — MIDI 文件录制

**源文件**：`src/streammuse/infrastructure/output/midi_file.py`

将实时事件流录制为 MIDI 文件。使用 `pretty_midi` 库，将 `NOTE_ON`/`NOTE_OFF` 事件对重建为完整音符（含开始时间、结束时间、力度）。

---

## `MidiFileOutputConfig`

```python
@dataclass(frozen=True)
class MidiFileOutputConfig:
    bpm: float
    ticks_per_beat: int
    output_path: Optional[str] = None
    user_program: int = 0
    model_program: int = 0
    user_track_name: str = "User"
    model_track_name: str = "Model"
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bpm` | `float` | 必填 | 录制 BPM（决定 tick→秒的映射） |
| `ticks_per_beat` | `int` | 必填 | 每拍 tick 数 |
| `output_path` | `Optional[str]` | `None` | 输出文件路径；`None` 时 `close()` 不保存 |
| `user_program` | `int` | `0` | 用户音轨的 MIDI 乐器编号 |
| `model_program` | `int` | `0` | 模型音轨的 MIDI 乐器编号 |
| `user_track_name` | `str` | `"User"` | 用户音轨名称 |
| `model_track_name` | `str` | `"Model"` | 模型音轨名称 |

`seconds_per_tick()` 方法：返回 `(60.0 / bpm) / ticks_per_beat`。

---

## `MidiFileOutputSink`

### 内部结构

MIDI 文件创建两个独立的 `pretty_midi.Instrument` 音轨：
- `_user`：录制 `source="user"` 的事件
- `_model`：录制 `source="model"` 的事件

各自维护活跃音符字典 `_active_user` / `_active_model`，格式为 `{pitch: {start, velocity}}`。

### `output_event(event, source)`

根据 `source` 分发到对应的 `_handle_event()`：

**`_handle_event()` 逻辑**：
1. 跳过 `is_placeholder=True` 或 `pitch=-1` 的事件
2. `NOTE_ON`（velocity > 0）：
   - 若该音高已在 `active` 中（未关闭的音符），先关闭前一个音符（retrigger 处理）
   - 将 `{start: t, velocity: v}` 加入 `active`
3. `NOTE_OFF`：
   - 从 `active` 移除该音高
   - 创建 `pretty_midi.Note`（start=旧值, end=当前时间, velocity=旧 velocity）

### `close()`

遍历所有仍在 `_active_user`/`_active_model` 中的未关闭音符，将其在最大时间处关闭，然后调用 `pretty_midi.write(output_path)` 保存文件。

---

## 音轨结构示例

生成的 MIDI 文件包含两个音轨：

```
Track 0: User  (乐器 user_program)
Track 1: Model (乐器 model_program)
```
