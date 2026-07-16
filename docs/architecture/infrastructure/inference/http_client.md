---
title: HttpInferenceClient — HTTP 推理客户端
description: 通过 REST API 向远端推理服务器发送推理请求
---

# HttpInferenceClient — HTTP 推理客户端

**源文件**：`src/streammuse/infrastructure/inference/http_client.py`

将 `InferenceEngine` 协议的调用转换为 HTTP 请求，发送到 FastAPI 推理服务器。

---

## `HttpInferenceClientConfig`

```python
@dataclass(frozen=True)
class HttpInferenceClientConfig:
    generate_url: str
    timeout_s: float = 30.0
    model_name: str = "stanley"
    inference_mode: str = "sliding_window"
    generation_interval_ticks: int = 2
    checkpoint_path: Optional[str] = None
    bpm: Optional[int] = None
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `generate_url` | `str` | 必填 | 推理服务器的 generate 端点，如 `http://localhost:8000/generate_accompaniment` |
| `timeout_s` | `float` | `30.0` | HTTP 请求超时时间（秒） |
| `model_name` | `str` | `"stanley"` | 透传给服务器的模型名称（如 `stanley`/`lekai`） |
| `inference_mode` | `str` | `"sliding_window"` | 透传给服务器的推理模式提示 |
| `generation_interval_ticks` | `int` | `2` | 透传给服务器的触发间隔参数 |
| `checkpoint_path` | `Optional[str]` | `None` | 可选透传 checkpoint 路径 |
| `bpm` | `Optional[int]` | `None` | 透传给模型的条件 BPM；不是客户端 playback wall-clock tempo |

---

## `HttpInferenceClient`

### 端点推导规则

所有端点通过替换 `generate_url` 中的路径段派生：

| 方法 | 端点 |
|---|---|
| `generate_accompaniment` | `generate_url`（原始 URL） |
| `inject_history` | `generate_url` 中 `/generate_accompaniment` → `/inject_notes` |
| `clear_history` | `generate_url` 中 `/generate_accompaniment` → `/clear_history` |
| `get_injection_status` | `generate_url` 中 `/generate_accompaniment` → `/injection_status` |

---

### `generate_accompaniment(melody_events, generation_start_tick, generation_length_frames, ...) -> tuple[List[MusicalEvent], TimingInfo]`

**请求 Payload**（POST JSON）：

```json
{
  "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}, ...],
  "generation_start_tick": 12,
  "client_request_send_time": 1700000000.123,
  "generation_length_frames": 20,
  "generation_interval_ticks": 2,
  "model_name": "stanley",
  "inference_mode": "sliding_window",
  "checkpoint_path": null,
  "bpm": 120,
  "prompt_length_ticks": null
}
```

`null` 字段在发送前被移除（`{k: v for k, v in payload.items() if v is not None}`）。

**响应处理**：
- 从 `data["accompaniment"]` 解析 `MusicalEvent` 列表（via `event_from_dict()`）
- 从 `data["timings"]` 解析 `TimingInfo`（via `timing_info_from_dict()`）

---

### `inject_history(melody_events, accompaniment_events, injection_length_ticks) -> None`

POST 到 `/inject_notes`，将历史旋律和伴奏注入服务器端模型历史。

---

### `set_injection_offset(offset_ticks) -> None`

**仅客户端操作**，不发送 HTTP 请求。将 `_injection_offset_ticks` 存储在本地，可在后续请求中使用（如需要）。

---

### `clear_history() -> dict`

POST 到 `/clear_history`，解析并返回 server 的清理结果，典型字段：

- `success`
- `message`
- `melody_history`
- `accompaniment_history`

---

### `get_injection_status() -> dict`

GET `/injection_status`，返回服务器端的注入状态 JSON（用于调试）。

---

## 错误处理

所有 HTTP 请求调用 `resp.raise_for_status()`，HTTP 错误（4xx/5xx）将抛出 `requests.HTTPError`。服务层（`_inference_worker`）对异常进行全局捕获并调用 `output_status("error", str(e))`。
