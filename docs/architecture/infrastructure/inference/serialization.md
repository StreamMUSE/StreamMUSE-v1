---
title: serialization — 序列化辅助函数
description: MusicalEvent 和 TimingInfo 的 HTTP wire format 序列化/反序列化
---

# serialization — 序列化辅助函数

**源文件**：`src/streammuse/infrastructure/inference/serialization.py`

提供三个辅助函数，用于 `MusicalEvent` 和 `TimingInfo` 与 HTTP JSON wire format 之间的转换。

---

## `event_to_dict(event: MusicalEvent) -> dict`

将 `MusicalEvent` 序列化为 JSON 字典，用于在请求 payload 中传输旋律事件：

```python
{
    "type": "note_on",        # event.event_type.value
    "pitch": 60,              # int
    "tick": 12,               # int
    # 以下字段仅在非默认值时包含：
    "velocity": 80,           # 可选
    "channel": 1,             # 可选（非 0 时）
    "program": 5,             # 可选（非 0 时）
    "is_placeholder": True,   # 可选（True 时）
}
```

`None` 和默认值字段不会出现在输出字典中，保持 wire format 简洁。

---

## `event_from_dict(data: dict) -> MusicalEvent`

将 JSON 字典反序列化为 `MusicalEvent`，用于解析服务器响应中的伴奏事件：

```python
MusicalEvent(
    tick=data["tick"],
    pitch=data["pitch"],
    event_type=EventType(data["type"]),   # 从字符串值构建
    velocity=data.get("velocity", 100),
    channel=data.get("channel", 0),
    program=data.get("program", 0),
    is_placeholder=data.get("is_placeholder", False),
    source=data.get("source", "model"),
)
```

`EventType` 通过其 `.value` 字符串（`"note_on"` / `"note_off"`）实例化。

---

## `timing_info_from_dict(data: dict) -> TimingInfo`

将服务器响应中的 `timings` 字段反序列化为 `TimingInfo` dataclass，映射全部 9 个字段：

| `TimingInfo` 字段 | `data` 字典键 |
|---|---|
| `request_arrival_time` | `"request_arrival_time"` |
| `inference_start_time` | `"inference_start_time"` |
| `inference_end_time` | `"inference_end_time"` |
| `response_send_time` | `"response_send_time"` |
| `preprocess_start_time` | `"preprocess_start_time"` |
| `postprocess_start_time` | `"postprocess_start_time"` |
| `client_request_send_time` | `"client_request_send_time"`（可选，默认 0.0） |
| `client_response_receive_time` | `"client_response_receive_time"`（可选，默认 0.0） |
| `round_trip_ms` | `"round_trip_ms"`（可选，默认 0.0） |

服务器端填充前 6 个字段（必填），后 3 个字段由客户端填充或用 0.0 占位。
