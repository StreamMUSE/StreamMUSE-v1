---
title: timing — 时间与调度
description: Tempo、MusicalTime、PlaybackScheduler 的完整说明
---

# timing — 时间与调度

**源文件**：`src/streammuse/domain/timing/`

本模块提供实时音乐系统的所有时间相关抽象：速度（BPM）换算、音乐时间位置表示，以及线程安全的事件调度器。

---

## `Tempo`

**源文件**：`timing/tempo.py`

不可变的速度配置，作为系统的核心时间基准。

```python
@dataclass(frozen=True)
class Tempo:
    bpm: float
    ticks_per_beat: int
    beats_per_bar: int
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `bpm` | `float` | 每分钟拍数（Beats Per Minute），必须 > 0 |
| `ticks_per_beat` | `int` | 每拍包含的 tick 数，必须 > 0（默认为 4） |
| `beats_per_bar` | `int` | 每小节拍数，必须 > 0（默认为 4，即 4/4 拍） |

### 属性与方法

#### `seconds_per_tick` （property）

单个 tick 对应的实时时长（秒）。

$$\text{seconds\_per\_tick} = \frac{60.0}{\text{bpm} \times \text{ticks\_per\_beat}}$$

#### `ticks_per_bar` （property）

每小节的总 tick 数：

$$\text{ticks\_per\_bar} = \text{ticks\_per\_beat} \times \text{beats\_per\_bar}$$

#### `tick_to_seconds(tick) -> float`

将绝对 tick 转换为实时秒数：

$$\text{seconds} = \text{tick} \times \text{seconds\_per\_tick}$$

#### `seconds_to_tick(seconds) -> int`

将实时秒数转换为 tick（截断为整数）：

$$\text{tick} = \lfloor \frac{\text{seconds}}{\text{seconds\_per\_tick}} \rfloor$$

---

## `MusicalTime`

**源文件**：`timing/tempo.py`

音乐时间位置表示，用于显示「第几小节第几拍」和 metronome（节拍器）回调。

```python
@dataclass(frozen=True)
class MusicalTime:
    tick: int
    bar: int
    beat: int
    tick_in_beat: int
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `tick` | `int` | 原始绝对 tick |
| `bar` | `int` | 小节编号（从 0 开始） |
| `beat` | `int` | 小节内的拍编号（从 0 开始） |
| `tick_in_beat` | `int` | 拍内的 tick 偏移（从 0 开始） |

### 类方法

#### `from_tick(tick, tempo) -> MusicalTime`

从绝对 tick 和 `Tempo` 计算音乐时间位置。

```python
@classmethod
def from_tick(cls, tick: int, tempo: Tempo) -> MusicalTime:
```

计算公式：

```
ticks_per_bar = tempo.ticks_per_beat * tempo.beats_per_bar
bar           = tick // ticks_per_bar
tick_in_bar   = tick % ticks_per_bar
beat          = tick_in_bar // tempo.ticks_per_beat
tick_in_beat  = tick_in_bar % tempo.ticks_per_beat
```

**示例**（`ticks_per_beat=4`, `beats_per_bar=4`）：

| tick | bar | beat | tick_in_beat |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 4 | 0 | 1 | 0 |
| 16 | 1 | 0 | 0 |
| 19 | 1 | 0 | 3 |

---

## `PlaybackScheduler`

**源文件**：`timing/scheduler.py`

线程安全的音乐事件调度器，按 tick 管理待播放事件队列，并支持取消过时事件（如推理响应替换旧的 model 输出）。

```python
class PlaybackScheduler:
    def __init__(self) -> None: ...
```

内部维护 `Dict[int, List[MusicalEvent]]`（tick → 事件列表）以及一个 `threading.Lock`，所有方法均在锁保护下运行。

### 方法

#### `schedule(event, tick) -> None`

将事件安排在指定 tick 播放。

```python
def schedule(self, event: MusicalEvent, tick: int) -> None:
```

若该 tick 尚无事件，自动初始化列表。

#### `get_events_at_tick(tick) -> List[MusicalEvent]`

取出并删除指定 tick 上的所有事件。

```python
def get_events_at_tick(self, tick: int) -> List[MusicalEvent]:
```

- 若该 tick 无事件，返回空列表
- 调用后该 tick 的事件从调度器中移除（consume 语义）

#### `clear_future_events(from_tick, source=None) -> None`

删除从 `from_tick` 起（含）的所有未来事件。

```python
def clear_future_events(
    self,
    from_tick: int,
    source: str | None = None,
) -> None:
```

| 参数 | 说明 |
|---|---|
| `from_tick` | 起始 tick（含），清除该 tick 及之后的事件 |
| `source` | 若指定（如 `"model"`），只清除该 source 的事件；`None` 表示清除所有 |

**典型用途**：收到新的推理响应时，先调用 `clear_future_events(generation_start_tick, source="model")` 清除旧的 model 事件，再调度新事件，避免新旧 model 输出混放。
