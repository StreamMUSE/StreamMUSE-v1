---
title: 新增 OutputSink
description: 如何为 StreamMUSE 添加新的输出 Sink 适配器
---

# 新增 OutputSink

本指南说明如何为 StreamMUSE 添加一个新的输出 Sink（如 HTTP 推送、数据库写入等）。

---

## 步骤概览

1. 在 `infrastructure/output/` 中创建新文件
2. 实现 `OutputSink` 协议（6 个方法）
3. 如有需要，额外实现 `log_inference()`
4. 在 `OutputSinkFactory` 中注册新类型
5. 在 `OutputConfig.OutputType` 中添加新的 Literal 值
6. 编写测试

---

## 第一步：实现 OutputSink 协议

```python
# src/streammuse/infrastructure/output/http_push.py
from __future__ import annotations
from typing import Any, Dict, Optional
from streammuse.domain.musical import MusicalEvent

class HttpPushOutputSink:
    """将事件通过 HTTP POST 推送到外部服务。"""

    def __init__(self, endpoint_url: str) -> None:
        self._url = endpoint_url

    def output_event(self, event: MusicalEvent, source: str) -> None:
        import requests
        requests.post(self._url + "/event", json={
            "tick": event.tick,
            "pitch": event.pitch,
            "type": event.event_type.value,
            "source": source,
        }, timeout=1.0)

    def output_tick(self, tick: int, bar: int, beat: int) -> None:
        pass   # 不需要推送每个 tick

    def output_stats(self, *, hit_rate=None, round_trip_ms=None, **kwargs) -> None:
        pass   # 不需要推送统计

    def output_status(self, state: str, message: str = "") -> None:
        import requests
        requests.post(self._url + "/status", json={"state": state, "message": message})

    def output_config(self, config: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass
```

**注意**：
- 6 个方法全部必须实现（即使是空操作）
- 不关心的方法写 `pass` 即可
- `output_stats()` 参数较多，用 `**kwargs` 接收可选参数

---

## 第二步：实现 `log_inference()`（可选）

如果需要记录推理信息，可额外实现此方法（无需在 Protocol 中声明，服务层通过 `hasattr` 检测）：

```python
def log_inference(
    self,
    request: Dict[str, Any],
    response: Dict[str, Any],
    latency_ms: float,
    server_process_ms: float,
) -> None:
    # 推送推理记录
    import requests
    requests.post(self._url + "/inference", json={
        "latency_ms": latency_ms,
        "server_process_ms": server_process_ms,
    })
```

---

## 第三步：注册到 Factory

```python
# src/streammuse/application/factories/output_factory.py
from streammuse.infrastructure.output.http_push import HttpPushOutputSink

class OutputSinkFactory:
    @staticmethod
    def create(app_config, session_manager=None):
        cfg = app_config.output
        ...
        elif cfg.type == "http_push":
            if cfg.http_push_url is None:
                raise ValueError("http_push mode requires http_push_url")
            return HttpPushOutputSink(cfg.http_push_url)
        ...
```

---

## 第四步：扩展配置

```python
# src/streammuse/application/config/models.py
OutputType = Literal["audio", "midi_file", "console", "websocket", "composite", "json_log", "session", "http_push"]

@dataclass(frozen=True)
class OutputConfig:
    ...
    http_push_url: Optional[str] = None
```

---

## 第五步：编写测试

```python
# tests/unit/infrastructure/test_http_push_output.py
from streammuse.infrastructure.output.http_push import HttpPushOutputSink
from streammuse.domain.musical import MusicalEvent, EventType

def test_output_event_posts_to_endpoint(requests_mock):
    sink = HttpPushOutputSink("http://localhost:9000")
    event = MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=80)
    requests_mock.post("http://localhost:9000/event")
    sink.output_event(event, "user")
    assert requests_mock.called
```
