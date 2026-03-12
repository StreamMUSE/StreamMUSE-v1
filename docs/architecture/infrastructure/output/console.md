---
title: ConsoleOutputSink — 控制台输出
description: 将事件和统计信息打印到终端，用于调试和 CLI 监控
---

# ConsoleOutputSink — 控制台输出

**源文件**：`src/streammuse/infrastructure/output/console.py`

最简单的输出适配器，将所有事件、tick 信息和统计数据格式化后打印到标准输出。

---

## `ConsoleOutputConfig`

```python
@dataclass(frozen=True)
class ConsoleOutputConfig:
    show_events: bool = True
    show_ticks: bool = True
    show_stats: bool = True
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `show_events` | `True` | 是否打印音符事件 |
| `show_ticks` | `True` | 是否打印每 tick 的进度信息 |
| `show_stats` | `True` | 是否打印推理统计信息 |

---

## 方法

### `output_event(event, source)`

打印格式：
```
[event] source=user tick=12 type=note_on pitch=60
```

若 `show_events=False`，不输出任何内容。

### `output_tick(tick, bar, beat)`

打印格式：
```
[tick] tick=8 bar=1 beat=2
```

若 `show_ticks=False`，不输出任何内容。

### `output_stats(...)`

打印格式：
```
[stats] hit_rate=None avg_backup_level=None round_trip_ms=45.2 server_process_ms=40.1 ...
```

若 `show_stats=False`，不输出任何内容。

### `output_status(state, message="")`

始终打印：
```
[status] state=running message=
```

### `output_config(config)`

始终打印：
```
[config] {'bpm': 120.0, ...}
```

### `close()`

空操作（no-op）。
