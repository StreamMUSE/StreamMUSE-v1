---
title: MidiFileOutputSink — MIDI 文件录制
description: 将事件流、可选 metronome 和 count-in 录制为 MIDI 文件
---

# MidiFileOutputSink — MIDI 文件录制

**源文件**：`src/streammuse/infrastructure/output/midi_file.py`

`MidiFileOutputSink` 使用 `pretty_midi` 将实时事件流录制为 MIDI 文件。它把 `NOTE_ON` / `NOTE_OFF` 事件对重建为完整音符，并可选记录 metronome 鼓轨。

---

## `MidiFileOutputConfig`

```python
@dataclass(frozen=True)
class MidiFileOutputConfig:
    bpm: float
    ticks_per_beat: int
    beats_per_bar: int = 4
    output_path: Optional[str] = None
    user_program: int = 0
    model_program: int = 0
    user_track_name: str = "Melody"
    model_track_name: str = "Accompaniment"
    record_metronome: bool = False
    metronome_track_name: str = "Metronome"
    metronome_beat_note: int = 77
    metronome_downbeat_note: int = 76
    metronome_velocity: int = 80
    metronome_downbeat_velocity: int = 110
    metronome_duration_ticks: int = 1
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `bpm` | 必填 | 录制 BPM |
| `ticks_per_beat` | 必填 | 每拍 tick 数 |
| `beats_per_bar` | `4` | 每小节拍数，用于 downbeat 判断 |
| `output_path` | `None` | 输出文件路径；`None` 时不保存 |
| `user_track_name` | `Melody` | 用户旋律音轨名 |
| `model_track_name` | `Accompaniment` | 模型伴奏音轨名 |
| `record_metronome` | `False` | 是否创建 `Metronome` 鼓轨 |
| `metronome_beat_note` | `77` | 普通拍 MIDI note |
| `metronome_downbeat_note` | `76` | 小节第一拍 MIDI note |
| `metronome_velocity` | `80` | 普通拍力度 |
| `metronome_downbeat_velocity` | `110` | 小节第一拍力度 |

---

## 音轨结构

默认生成：

```text
Track: Melody        # source="user"
Track: Accompaniment # source="model"
```

若 `record_metronome=True`，额外生成：

```text
Track: Metronome     # is_drum=True
```

---

## `output_event(event, source)`

根据 `source` 分发到 `Melody` 或 `Accompaniment`：

1. 跳过 `is_placeholder=True` 或 `pitch=-1` 的事件。
2. `NOTE_ON` 且 velocity > 0：记录 active note；若同 pitch 已 active，则先关闭前一个音符。
3. `NOTE_OFF`：关闭 active note 并创建 `pretty_midi.Note`。

---

## `output_metronome_tick(tick, bar, beat)`

```python
if self._metronome is None:
    return
self._observe_recording_tick(int(tick))
if int(tick) % int(self._config.ticks_per_beat) != 0:
    return
```

只在 beat 边界记录 metronome note。downbeat 判断使用：

```python
ticks_per_bar = ticks_per_beat * beats_per_bar
is_downbeat = ticks_per_bar > 0 and tick % ticks_per_bar == 0
```

---

## count-in 录制机制

count-in 阶段 service 会输出负 tick，例如 4 拍 count-in、`ticks_per_beat=4` 时为 `-16..-1`。`MidiFileOutputSink` 会观察负 tick 并设置录制偏移：

```python
def _observe_recording_tick(self, tick: int) -> None:
    if int(tick) < 0:
        self._recording_tick_offset = max(self._recording_tick_offset, -int(tick))

def _time(self, tick: int) -> float:
    return float(int(tick) + int(self._recording_tick_offset)) * self._sp_tick
```

因此 MIDI 文件中 count-in click 会出现在开头，正式 tick=0 的音乐会向后平移。这个偏移只存在于 MIDI 录制层，不影响 service 内部 tick 和推理请求。

---

## `close()`

关闭时会把未结束的 active notes 截断到 `_max_time`，然后写入 `output_path`。
