---
title: MidiFileInput — MIDI 文件实时模拟
description: 解析 MIDI 文件并按实际速度实时发出事件，用于基准测试与回放
---

# MidiFileInput — MIDI 文件实时模拟

**源文件**：`src/streammuse/infrastructure/input/midi_file.py`

从 `.mid` 文件中读取音符，按照配置的 BPM 以实时速度模拟演奏，用于自动化测试、基准测试和演示演示。

---

## `MidiFileInputConfig`

```python
@dataclass(frozen=True)
class MidiFileInputConfig:
    bpm: float
    ticks_per_beat: int
    delay_ticks: int = 0
    min_pitch: int = 0
    max_pitch: int = 127
    program: Optional[int] = None
    max_tick: Optional[int] = None
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bpm` | `float` | 必填 | 播放速度（每分钟拍数） |
| `ticks_per_beat` | `int` | 必填 | 每拍的 tick 数，需与系统配置一致 |
| `delay_ticks` | `int` | `0` | 文件开始前的延迟 ticks |
| `min_pitch` | `int` | `0` | 音高下限过滤（包含） |
| `max_pitch` | `int` | `127` | 音高上限过滤（包含） |
| `program` | `Optional[int]` | `None` | 仅读取特定 MIDI 乐器编号的音轨（`None` 表示不过滤） |
| `max_tick` | `Optional[int]` | `None` | 截断到指定 tick（`None` 表示读取全部） |

`seconds_per_tick()` 方法：返回 `(60.0 / bpm) / ticks_per_beat`，即每 tick 的秒数。

---

## `MidiFileInput`

### `__init__(...)`

```python
def __init__(
    self,
    midi_file_path: str,
    *,
    config: MidiFileInputConfig,
    velocity_default: int = 64,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
```

| 参数 | 说明 |
|---|---|
| `midi_file_path` | MIDI 文件路径 |
| `config` | `MidiFileInputConfig` 配置对象 |
| `velocity_default` | 没有力度信息时的默认 velocity（默认 64） |
| `now` | 时间函数（可在测试中替换） |
| `sleep` | 睡眠函数（可在测试中替换） |

---

### `read_events() -> Iterator[MusicalEvent]`

1. 调用私有方法 `_midi_to_notes()` 解析 .mid 文件，返回 `[{pitch, tick, duration}]` 列表
2. 将每个 note 转换为两个 `MusicalEvent`（`NOTE_ON` 和 `NOTE_OFF`），并按 tick 排序
3. 按 `seconds_per_tick * tick_offset` 计算实际发出时间，睡眠直到对应时间点
4. 依次 yield `NOTE_ON`（tick=0）和 `NOTE_OFF`（tick=0）

**注意**：与其他适配器相同，发出的 `tick=0`。Application 层根据时间戳重新赋值实际 tick。

---

### `_midi_to_notes(...)` — 静态方法

```python
@staticmethod
def _midi_to_notes(
    midi_path: str,
    *,
    beat_div: int,
    min_pitch: int,
    max_pitch: int,
    program: Optional[int],
    max_tick: Optional[int],
) -> Tuple[List[Dict[str, int]], int, int]:
```

读取 MIDI 文件，将 `note_on`/`note_off` 对转换为带 `{pitch, tick, duration}` 的字典列表。

返回值：`(notes, resolution, actual_max_tick)`

- `notes`：按 tick 排序的音符字典列表
- `resolution`：MIDI 文件原始的 `ticks_per_beat`
- `actual_max_tick`：所有音符结束后的最大 tick

`beat_div` 参数决定输出 tick 的分辨率（即 `ticks_per_beat`），内部通过 `ticks_per_output_tick = resolution / beat_div` 进行缩放。

---

### `close() -> None`

设置 `_closed = True`，终止 `read_events()` 的内部循环。
