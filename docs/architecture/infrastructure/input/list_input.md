---
title: ListInput — 内存列表输入
description: 从预定义 MusicalEvent 列表中依次发出事件，用于测试和回放
---

# ListInput — 内存列表输入

**源文件**：`src/streammuse/infrastructure/input/list_input.py`

最简单的 `InputSource` 实现，用于单元测试和预录制序列的确定性回放。

---

## `ListInput`

### `__init__(events: List[MusicalEvent]) -> None`

| 参数 | 类型 | 说明 |
|---|---|---|
| `events` | `List[MusicalEvent]` | 将要依次发出的事件列表（内部会复制一份） |

---

### `read_events() -> Iterator[MusicalEvent]`

依次 yield 列表中的每个 `MusicalEvent`，遍历完后迭代器自然结束。

若 `close()` 已被调用，立即返回（不发出任何事件）。

与其他适配器不同，`ListInput` 保留事件原有的 `tick` 值（不强制设为 0）。这允许测试代码精确控制事件时序，而不依赖 Application 层的时间戳赋值。

---

### `close() -> None`

设置 `_closed = True`。已关闭的实例调用 `read_events()` 时立即返回。

---

## 使用示例

```python
from streammuse.domain.musical import MusicalEvent, EventType
from streammuse.infrastructure.input.list_input import ListInput

events = [
    MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=80),
    MusicalEvent(tick=4, pitch=60, event_type=EventType.NOTE_OFF, velocity=0),
]
source = ListInput(events)

for ev in source.read_events():
    print(ev)  # 依次打印两个事件
```

---

## 设计意图

`ListInput` 主要用途：
1. **单元测试**：向 `RealTimeMusicService` 注入确定性事件序列
2. **工厂测试**：`InputSourceFactory.create()` 接受 `list_events` 参数并返回 `ListInput` 实例
3. **集成测试**：构建完整的 CLI 场景而无需任何外部输入设备
