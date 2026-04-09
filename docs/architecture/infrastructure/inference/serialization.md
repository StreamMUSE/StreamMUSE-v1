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
    # 以下字段在值非 None 时包含（当前 MusicalEvent 默认都非 None）：
    "velocity": 80,
    "channel": 1,
    "program": 5,
    "is_placeholder": True,   # 可选（True 时）
}
```

`is_placeholder` 仅在 `True` 时发送；其余数值字段在值为 `None` 时才会被省略。

---

## `event_from_dict(data: dict) -> MusicalEvent`

将 JSON 字典反序列化为 `MusicalEvent`，用于解析服务器响应中的伴奏事件：

```python
MusicalEvent(
    tick=int(data.get("tick", 0)),
    pitch=int(data.get("pitch", -1)),
    event_type=EventType(str(data.get("type"))),
    velocity=int(data.get("velocity")) if data.get("velocity") is not None else 100,
    channel=int(data.get("channel")) if data.get("channel") is not None else 0,
    program=int(data.get("program")) if data.get("program") is not None else 0,
    is_placeholder=bool(data.get("is_placeholder", False)),
)
```

`event_from_dict` 会显式处理 `null` 值：当 `velocity/channel/program` 为 `null` 或缺失时，回退到默认值 `100/0/0`。

---

## `timing_info_from_dict(data: dict) -> TimingInfo`

将服务器响应中的 `timings` 字段反序列化为 `TimingInfo` dataclass。

| `TimingInfo` 字段 | `data` 字典键 |
|---|---|
| `request_arrival_time` | `"request_arrival_time"` |
| `response_output_time` | `"response_output_time"` |
| `preprocess_start_time` | `"preprocess_start_time"` |
| `inference_start_time` | `"inference_start_time"` |
| `inference_end_time` | `"inference_end_time"` |
| `postprocess_start_time` | `"postprocess_start_time"` |
| `round_trip_time` | `"round_trip_time"`（可选） |
| `server_processing_duration` | `"server_processing_duration"`（可选） |
| `total_network_latency` | `"total_network_latency"`（可选） |

前 6 个字段是必填服务端时间戳；后 3 个字段是可选扩展字段，缺失时会被解析为 `None`。
