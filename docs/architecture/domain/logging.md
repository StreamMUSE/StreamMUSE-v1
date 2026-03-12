---
title: logging — 会话日志领域对象
description: LogEvent、InferenceEvent、SessionManager、MetricsCalculator 的完整说明
---

# logging — 会话日志领域对象

**源文件**：`src/streammuse/domain/logging/`

本模块定义会话日志系统的领域对象，不涉及任何文件写入操作（文件 I/O 在 Infrastructure 层的 `JsonLoggerOutputSink` 和 `SessionLoggerOutputSink` 中完成）。

---

## `EventType`（日志用）

**源文件**：`logging/event_types.py`

注意：此 `EventType` 与 `domain/musical/` 中的 `EventType` 不同，专用于日志系统。

```python
class EventType(str, Enum):
    NOTE_ON             = "note_on"
    NOTE_OFF            = "note_off"
    INFERENCE_REQUEST   = "inference_request"
    INFERENCE_RESPONSE  = "inference_response"
    TICK                = "tick"
    STATUS              = "status"
```

---

## `LogEvent`

**源文件**：`logging/event_types.py`

记录单个音乐或系统事件的不可变日志条目。

```python
@dataclass(frozen=True)
class LogEvent:
    timestamp: float
    tick: int
    event_type: EventType
    data: Dict[str, Any]
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | `float` | Unix 时间戳（秒） |
| `tick` | `int` | 事件发生时的绝对 tick |
| `event_type` | `EventType` | 事件类型（见上方枚举） |
| `data` | `Dict[str, Any]` | 事件附加数据（如 pitch、velocity、source 等） |

### 方法

#### `to_dict() -> Dict[str, Any]`

将 `LogEvent` 序列化为字典，`event_type` 值为字符串形式。

#### `to_json() -> str`

将 `LogEvent` 序列化为 JSON 字符串。

---

## `InferenceEvent`

**源文件**：`logging/event_types.py`

记录单次完整推理请求-响应的不可变日志条目。

```python
@dataclass(frozen=True)
class InferenceEvent:
    inference_id: str
    timestamp_request: float
    timestamp_response: float
    request_data: Dict[str, Any]
    response_data: Dict[str, Any]
    latency_ms: float
    server_process_ms: float
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `inference_id` | `str` | 推理唯一 ID（通常为 UUID 或递增整数字符串） |
| `timestamp_request` | `float` | 请求发出时的 Unix 时间戳 |
| `timestamp_response` | `float` | 响应接收时的 Unix 时间戳 |
| `request_data` | `Dict[str, Any]` | 请求内容摘要（generation_start_tick、melody_notes_count 等） |
| `response_data` | `Dict[str, Any]` | 响应内容摘要（accompaniment_notes_count 等） |
| `latency_ms` | `float` | 客户端往返延迟（ms） |
| `server_process_ms` | `float` | 服务端处理时间（ms） |

### 方法

#### `to_dict() -> Dict[str, Any]`

序列化为字典，格式：

```json
{
  "id": "...",
  "timestamp_request": 1234567890.0,
  "timestamp_response": 1234567890.1,
  "request_data": {...},
  "response_data": {...},
  "latency_ms": 100.0,
  "server_process_ms": 80.0
}
```

#### `to_json() -> str`

序列化为 JSON 字符串。

---

## `SessionManager`

**源文件**：`logging/session_manager.py`

管理单个会话的目录创建、配置保存和摘要写出。不涉及推理/事件数据的写入（由 `JsonLoggerOutputSink` 负责）。

```python
class SessionManager:
    def __init__(self, base_log_dir: str = "logs") -> None:
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `base_log_dir` | `str` | `"logs"` | 日志根目录路径 |

初始化时自动生成 `session_id`，并设置 `session_dir` 路径（但不立即创建目录）。

### 方法

#### `_generate_session_id() -> str`（私有）

生成 `YYYYMMDD-HHMMSS` 格式的会话 ID，基于当前时间。

#### `create_session_directory() -> Path`

在 `<base_log_dir>/session_<session_id>/` 创建会话目录（`parents=True, exist_ok=True`）。返回会话目录的 `Path` 对象。通常在 CLI 启动时调用一次。

#### `save_config(config: Dict[str, Any]) -> None`

将配置字典以 JSON 格式写入 `session_dir/session_config.json`（indent=2）。

#### `save_summary(summary: Dict[str, Any]) -> None`

将摘要字典以纯文本格式写入 `session_dir/session_summary.txt`，每行格式为 `key: value`。

#### `get_session_dir() -> Path`

返回当前会话目录路径（`Path` 对象）。

#### `get_session_id() -> str`

返回当前会话 ID 字符串。

---

## `MetricsCalculator`

**源文件**：`logging/metrics_calculator.py`

累积并计算会话统计数据，生成 `performance.json` 内容。不直接写文件；写操作由 `JsonLoggerOutputSink` 完成。

```python
class MetricsCalculator:
    def __init__(self) -> None:
```

初始化内部累积器：事件列表、推理事件列表等。

### 方法

#### `add_event(event: LogEvent) -> None`

向统计器中添加一个日志事件，用于后续 `calculate_event_stats()` 和 `calculate_music_stats()` 计算。

#### `add_inference(inf: InferenceEvent) -> None`

向统计器中添加一条推理记录，用于后续 `calculate_latency_stats()` 计算。

#### `calculate_latency_stats() -> Dict[str, float]`

基于已收集的所有 `InferenceEvent` 计算延迟统计。

**返回键**：`p50_ms`、`p95_ms`、`p99_ms`、`mean_ms`、`min_ms`、`max_ms`

若无推理记录，返回空字典。

#### `calculate_event_stats() -> Dict[str, Any]`

统计事件数量分布。

**返回键**：`total_events`、`user_events`、`model_events`、`note_on_count`、`note_off_count`

#### `calculate_music_stats() -> Dict[str, Any]`

音乐特征统计。

**返回键**：`unique_pitches`、`pitch_range`（最高音 - 最低音）、`avg_velocity`、`note_density`（events per tick）等

#### `generate_performance_json(session_config: Dict[str, Any]) -> Dict[str, Any]`

聚合所有统计，生成完整的 `performance.json` 结构。

```python
def generate_performance_json(self, session_config: Dict[str, Any]) -> Dict[str, Any]:
```

返回结构示例：

```json
{
  "session_config": {...},
  "latency": {
    "p50_ms": 80.0,
    "p95_ms": 150.0,
    "p99_ms": 200.0
  },
  "events": {
    "total_events": 500,
    "user_events": 200,
    "model_events": 300
  },
  "music": {
    "unique_pitches": 12,
    "pitch_range": 24
  }
}
```

#### `generate_statistics_csv() -> str`

生成 CSV 格式的汇总字符串（含表头行），便于在 `statistics.csv` 中写入。
