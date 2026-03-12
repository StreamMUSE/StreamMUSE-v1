---
title: KeyboardInput — 计算机键盘输入
description: 使用 pynput 将键盘按键映射为 MIDI 音符事件
---

# KeyboardInput — 计算机键盘输入

**源文件**：`src/streammuse/infrastructure/input/keyboard.py`

使用计算机键盘模拟钢琴演奏，无需任何 MIDI 硬件。

---

## `KeyboardInputConfig`

```python
@dataclass(frozen=True)
class KeyboardInputConfig:
    velocity: int = 100
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `velocity` | `int` | `100` | 按下键时发出的力度值（0–127） |

---

## 键位映射

`DEFAULT_KEY_TO_PITCH` 将键盘按键映射到 MIDI 音符编号（C4 = 60）：

```
白键（底行）:  z=C4  x=D4  c=E4  v=F4  b=G4  n=A4  m=B4  ,=C5  .=D5  /=E5
黑键（顶行）:  s=C#4 d=D#4 g=F#4 h=G#4 j=A#4 l=C#5 ;=D#5
```

可通过 `__init__` 的 `key_to_pitch` 参数替换为自定义映射。

---

## `KeyboardInput`

### `__init__(...)`

```python
def __init__(
    self,
    *,
    key_to_pitch: Optional[dict[str, int]] = None,
    config: KeyboardInputConfig | None = None,
    now: Callable[[], float] = time.time,
) -> None:
```

| 参数 | 说明 |
|---|---|
| `key_to_pitch` | 自定义键→音高映射，默认使用 `DEFAULT_KEY_TO_PITCH` |
| `config` | 速度配置 |
| `now` | 时间函数（可在测试中替换） |

---

### `read_events() -> Iterator[MusicalEvent]`

启动 `pynput.keyboard.Listener`，监听键盘事件。按下键发出 `NOTE_ON`，释放键发出 `NOTE_OFF`。

所有发出的事件均为 `tick=0`，`source` 字段未设置（由 Application 层赋值 `"user"`）。

内部使用 `queue.Queue` 实现生产者-消费者模式：`pynput` 监听器在独立线程中运行，`read_events()` 在调用线程中阻塞式取出事件。

**键位重复处理**：内部维护 `_pressed: set[str]`，当键已按下时不会重复发出 `NOTE_ON`。

### `_handle_key_down(char: str) -> None`

直接测试入口：将指定键处理为按下事件。单元测试应直接调用此方法，避免依赖 GUI。

### `_handle_key_up(char: str) -> None`

直接测试入口：将指定键处理为释放事件。

### `close() -> None`

停止 `pynput` 监听器，向队列放入 `None` sentinel 使 `read_events()` 退出。
