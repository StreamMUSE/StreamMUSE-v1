---
title: musical — 音乐数据类型
description: MusicalEvent、EventType、Note、converters、MusicalSequence 的完整说明
---

# musical — 音乐数据类型

**源文件**：`src/streammuse/domain/musical/`

本模块定义系统中流动的所有音乐数据结构，以及 `MusicalEvent` 与基于时长的 `Note` 之间的转换函数。

---

## `EventType`

**源文件**：`musical/events.py`

```python
class EventType(Enum):
    NOTE_ON  = "note_on"
    NOTE_OFF = "note_off"
```

| 值 | 含义 |
|---|---|
| `NOTE_ON` | 音符按下（开始发声） |
| `NOTE_OFF` | 音符抬起（停止发声） |

---

## `MusicalEvent`

**源文件**：`musical/events.py`

系统中音乐数据的基本单元。`frozen=True`，创建后不可修改。

```python
@dataclass(frozen=True)
class MusicalEvent:
    tick: int
    pitch: int
    event_type: EventType
    velocity: int = 100
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False
    source: str = "unknown"
```

### 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `tick` | `int` | 必填 | 绝对音乐时间（ticks），必须 ≥ 0 |
| `pitch` | `int` | 必填 | MIDI 音高（0–127）；placeholder 时为 `-1` |
| `event_type` | `EventType` | 必填 | `NOTE_ON` 或 `NOTE_OFF` |
| `velocity` | `int` | `100` | 力度（0–127） |
| `channel` | `int` | `0` | MIDI 通道（0–15） |
| `program` | `int` | `0` | MIDI 乐器编号（0–127） |
| `is_placeholder` | `bool` | `False` | 若为 `True`，则 `pitch` 必须为 `-1`，表示占位事件 |
| `source` | `str` | `"unknown"` | 事件来源：`"user"` 或 `"model"` |

### 验证规则（`__post_init__`）

- `tick` 必须 ≥ 0
- `velocity` 必须在 0–127 之间
- `is_placeholder=True` 时，`pitch` 必须为 `-1`
- `is_placeholder=False` 时，`pitch` 必须在 0–127 之间
- `channel` 必须在 0–15 之间
- `program` 必须在 0–127 之间

违反规则时抛出 `ValueError`。

---

## `Note`

**源文件**：`musical/events.py`

基于时长的音符表示，供 Stanley 等非事件流引擎使用。

```python
@dataclass(frozen=True)
class Note:
    pitch: int
    tick: int
    duration: int
    velocity: int = 100
    channel: int = 0
    program: int = 0
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `pitch` | `int` | MIDI 音高（0–127） |
| `tick` | `int` | 音符开始时刻（绝对 tick） |
| `duration` | `int` | 音符时长（ticks） |
| `velocity` | `int` | 力度（0–127） |
| `channel` | `int` | MIDI 通道 |
| `program` | `int` | MIDI 乐器编号 |

---

## `converters.py` — 转换函数

**源文件**：`musical/converters.py`

### `events_to_notes(events, horizon_tick)`

将事件流（`NOTE_ON`/`NOTE_OFF` 对）转换为基于时长的 `Note` 列表。

```python
def events_to_notes(
    events: List[MusicalEvent],
    horizon_tick: int,
) -> List[Note]:
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `events` | `List[MusicalEvent]` | 按时间顺序排列的事件列表 |
| `horizon_tick` | `int` | 截止 tick：未配对的 `NOTE_ON` 将在此处关闭 |

**返回**：`List[Note]`，配对完成的音符（或在 `horizon_tick` 关闭的未配对音符）

**"close-at-horizon" 策略**：若某 `NOTE_ON` 在事件列表末尾没有对应的 `NOTE_OFF`，则将其在 `horizon_tick` 处关闭。这样下游引擎始终能收到完整的音符（有开始有结束），而当真实的 `NOTE_OFF` 到来时，下一次请求会包含正确的时长，实现自动修正，无需额外的 correction API。

**不支持**：同时存在两个同音高的 `NOTE_ON`（polyphony for same pitch）；第一个 `NOTE_ON` 会先配对第一个 `NOTE_OFF`。

---

## `MusicalSequence`

**源文件**：`musical/sequence.py`

不可变的音乐事件集合，附带 `Tempo` 信息。

```python
@dataclass(frozen=True)
class MusicalSequence:
    events: tuple[MusicalEvent, ...]
    tempo: Tempo
```

### 方法

#### `get_events_in_range(start_tick, end_tick) -> MusicalSequence`

提取 tick 在 `[start_tick, end_tick)` 区间内的事件，返回新的 `MusicalSequence`。

```python
def get_events_in_range(self, start_tick: int, end_tick: int) -> MusicalSequence:
```

- 区间为**左闭右开**：`start_tick <= event.tick < end_tick`
- 保留原始 `Tempo`

#### `quantize(quantization_ticks) -> MusicalSequence`

将所有事件的 tick 量化到 `quantization_ticks` 的整数倍（向下取整）。

```python
def quantize(self, quantization_ticks: int) -> MusicalSequence:
```

| 参数 | 类型 | 说明 |
|---|---|---|
| `quantization_ticks` | `int` | 量化步长，必须 > 0，否则抛出 `ValueError` |

量化公式：`new_tick = (event.tick // quantization_ticks) * quantization_ticks`
