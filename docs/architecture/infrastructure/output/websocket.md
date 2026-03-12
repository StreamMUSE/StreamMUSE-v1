---
title: WebSocketOutputSink — WebSocket 推送
description: 将事件序列化为 JSON 入队，供 WebSocket 服务器消费
---

# WebSocketOutputSink — WebSocket 推送

**源文件**：`src/streammuse/infrastructure/output/websocket.py`

将音符事件序列化为 JSON 字符串并放入线程安全队列，供外部 WebSocket 服务器从队列中取出并推送给前端客户端。

---

## `WebSocketOutputConfig`

```python
@dataclass(frozen=True)
class WebSocketOutputConfig:
    include_timestamps: bool = True
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `include_timestamps` | `True` | 是否在每条消息中附加 `timestamp` 字段 |

---

## `WebSocketOutputSink`

### `message_queue`

```python
self.message_queue: queue.Queue[str]
```

公开的线程安全队列。WebSocket 服务器（或其他消费者）可直接访问此队列取出消息。

### `output_event(event, source)`

将 `NOTE_ON`/`NOTE_OFF` 事件序列化为 JSON 字符串放入队列：

**NOTE_ON 消息格式**：
```json
{
  "type": "note",
  "event": "on",
  "pitch": 60,
  "velocity": 80,
  "tick": 12,
  "duration": 0,
  "source": "user",
  "timestamp": 1700000000.123
}
```

**NOTE_OFF 消息格式**：
```json
{
  "type": "note",
  "event": "off",
  "pitch": 60,
  "tick": 16,
  "source": "user",
  "timestamp": 1700000000.456
}
```

跳过 `is_placeholder=True` 的事件。

### `output_tick`, `output_status`, `output_config`

各自生成对应类型的 JSON 消息（`type: "tick"` / `type: "status"` / `type: "config"`）入队。

### `output_stats(...)`

生成 `type: "stats"` 的 JSON 消息入队，包含所有统计字段。

### `close()`

空操作（no-op）。

---

## WebSocket 服务器集成

WebSocket 服务器需从 `message_queue` 轮询消息并推送：

```python
sink = WebSocketOutputSink()

async def ws_handler(websocket):
    while True:
        try:
            msg = sink.message_queue.get_nowait()
            await websocket.send(msg)
        except queue.Empty:
            await asyncio.sleep(0.01)
```
