---
title: 新增 InputSource
description: 如何为 StreamMUSE 添加新的输入源适配器
---

# 新增 InputSource

本指南说明如何为 StreamMUSE 添加一个新的输入源（如 OSC 输入、网络 MIDI 等）。

---

## 步骤概览

1. 在 `infrastructure/input/` 中创建新文件
2. 实现 `InputSource` 协议
3. 在 `InputSourceFactory` 中注册新类型
4. 在 `InputConfig.InputType` 中添加新的 Literal 值
5. 在 `config_parser.py` 中添加 CLI 参数（可选）
6. 编写测试

---

## 第一步：实现 InputSource 协议

```python
# src/streammuse/infrastructure/input/osc_input.py
from __future__ import annotations
from typing import Iterator
from streammuse.domain.musical import MusicalEvent, EventType

class OscInput:
    """从 OSC 消息读取音符事件。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        self._host = host
        self._port = port
        self._running = False

    def read_events(self) -> Iterator[MusicalEvent]:
        """阻塞式生成器，直到 close() 被调用。"""
        self._running = True
        # ... 建立 OSC 监听器 ...
        while self._running:
            # 从 OSC 取消息
            msg = self._receive_next()   # 自定义内部方法
            if msg is None:
                continue
            yield MusicalEvent(
                tick=0,            # 必须设为 0！Application 层负责赋值 tick
                pitch=msg.pitch,
                event_type=EventType.NOTE_ON if msg.velocity > 0 else EventType.NOTE_OFF,
                velocity=msg.velocity,
            )

    def close(self) -> None:
        """停止 read_events() 并清理资源。"""
        self._running = False
        # ... 关闭 OSC 监听器 ...
```

**关键约定**：
- `read_events()` 使用 `yield`（生成器），不是回调
- `tick=0`，Application 层负责赋值实际 tick
- `close()` 必须能让 `read_events()` 的循环安全退出

---

## 第二步：注册到 Factory

```python
# src/streammuse/application/factories/input_factory.py
from streammuse.infrastructure.input.osc_input import OscInput

class InputSourceFactory:
    @staticmethod
    def create(app_config, *, list_events=None):
        cfg = app_config.input
        ...
        elif cfg.type == "osc":
            return OscInput(
                host=cfg.osc_host or "0.0.0.0",
                port=cfg.osc_port or 8001,
            )
        ...
```

---

## 第三步：扩展配置

```python
# src/streammuse/application/config/models.py
InputType = Literal["midi_device", "keyboard", "midi_file", "list", "osc"]

@dataclass(frozen=True)
class InputConfig:
    ...
    osc_host: Optional[str] = None
    osc_port: int = 8001
```

---

## 第四步：添加 CLI 参数（可选）

```python
# src/streammuse/presentation/cli/config_parser.py
parser.add_argument("--osc-host", default=None)
parser.add_argument("--osc-port", type=int, default=8001)
```

---

## 第五步：编写测试

```python
# tests/unit/infrastructure/test_osc_input.py
from streammuse.infrastructure.input.osc_input import OscInput

def test_close_stops_iteration():
    source = OscInput()
    # 使用 _handle_message() 等内部方法（类似 keyboard 的 _handle_key_down）
    # 或 mock 网络层
```
