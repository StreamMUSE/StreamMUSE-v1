---
title: interfaces — Domain 接口 Protocol
description: InputSource、OutputSink、InferenceEngine、TimingInfo 的完整契约说明
---

# interfaces — Domain 接口 Protocol

**源文件**：`src/streammuse/domain/interfaces/`

本模块用 Python `Protocol` 定义系统的三个核心接口契约。Infrastructure 层中的所有具体实现类均需满足这些契约，但无需显式继承（structural subtyping）。

---

## `InputSource`

**源文件**：`interfaces/input.py`

音乐输入源的协议，由键盘、MIDI 设备、MIDI 文件等具体实现。

```python
class InputSource(Protocol):
    def read_events(self) -> Iterator[MusicalEvent]: ...
    def close(self) -> None: ...
```

### `read_events() -> Iterator[MusicalEvent]`

**阻塞式生成器**：持续产出 `MusicalEvent`，直到以下任一条件满足：

- `close()` 被调用（通常来自另一个线程）
- 输入源自然结束（如 MIDI 文件播放完毕）

**重要**：`_input_worker` 线程运行在此生成器的循环中，因此 `read_events()` 必须是可中断的——`close()` 调用后应在毫秒级别内使 `read_events()` 退出。

### `close() -> None`

释放底层资源，并使正在阻塞的 `read_events()` 退出。通常通过设置停止标志或关闭底层端口实现。可被多次安全调用（幂等）。

---

## `OutputSink`

**源文件**：`interfaces/output.py`

音乐输出和 UI 更新的协议，由控制台打印、音频播放、MIDI 录制等具体实现。

```python
class OutputSink(Protocol):
    def output_event(self, event: MusicalEvent, source: str) -> None: ...
    def output_tick(self, tick: int, bar: int, beat: int) -> None: ...
    def output_stats(
        self,
        hit_rate: Optional[float] = None,
        avg_backup_level: Optional[float] = None,
        round_trip_ms: Optional[float] = None,
        server_process_ms: Optional[float] = None,
        network_latency_ms: Optional[float] = None,
        total_hits: Optional[int] = None,
        total_ticks: Optional[int] = None,
    ) -> None: ...
    def output_status(self, state: str, message: str = "") -> None: ...
    def output_config(self, config: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...
```

### 方法说明

#### `output_event(event, source) -> None`

处理单个音乐事件。根据 Sink 类型，可能是：播放 MIDI 声音、写入文件、推送 WebSocket 消息、打印到终端等。

| 参数 | 类型 | 说明 |
|---|---|---|
| `event` | `MusicalEvent` | 要输出的音乐事件 |
| `source` | `str` | `"user"` 或 `"model"`，标识事件来源 |

#### `output_tick(tick, bar, beat) -> None`

每 tick 由 `_tick_loop` 调用，传递当前时间位置。用于显示进度、metronome 回调等。简单 Sink（如 `AudioOutputSink`）通常实现为 no-op。

#### `output_stats(...) -> None`

`_inference_worker` 在每次推理完成后调用，传递延迟统计。所有参数均为可选（`Optional`），未知的字段可传 `None`。

| 参数 | 说明 |
|---|---|
| `round_trip_ms` | 客户端往返延迟（ms） |
| `server_process_ms` | 服务端处理时间（ms） |
| `hit_rate` | 推理命中率（伴奏按时就绪的比例） |

#### `output_status(state, message="") -> None`

系统状态变更通知。

| `state` 可能取值 | 含义 |
|---|---|
| `"running"` | 服务已启动 |
| `"stopping"` | 服务正在关闭 |
| `"error"` | 发生错误，附带 `message` |

#### `output_config(config) -> None`

在服务启动时（`start()` 内）调用一次，传递会话配置快照。用于 WebSocket / JSON log 在会话开始时记录配置。

#### `close() -> None`

刷新缓冲区，关闭文件句柄或 MIDI 端口。服务停止时调用。

---

## 关于 `log_inference()`

`log_inference()` **不在** `OutputSink` Protocol 中定义。这是面向日志 Sink 的扩展方法，通过 `hasattr` 检测调用：

```python
# 在 RealTimeMusicService._inference_worker() 中
if hasattr(self._output, "log_inference"):
    self._output.log_inference(request=..., response=..., latency_ms=..., server_process_ms=...)
```

实现了 `log_inference()` 的类：`JsonLoggerOutputSink`、`SessionLoggerOutputSink`、`CompositeOutputSink`。

---

## `InferenceEngine`

**源文件**：`interfaces/inference.py`

推理引擎的协议，由 `HttpInferenceClient` 和 `StanleyInferenceEngine` 实现。

```python
class InferenceEngine(Protocol):
    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]: ...

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None: ...

    def set_injection_offset(self, offset_ticks: int) -> None: ...

    def clear_history(self) -> None: ...
```

### `generate_accompaniment(...)  -> tuple[List[MusicalEvent], TimingInfo]`

核心推理接口，根据旋律事件生成伴奏。

| 参数 | 类型 | 说明 |
|---|---|---|
| `melody_events` | `List[MusicalEvent]` | 当前上下文内的所有旋律事件（`source="user"`） |
| `generation_start_tick` | `int` | 伴奏生成的起始 tick |
| `generation_length_frames` | `int` | 生成帧数（1 帧 = 0.5 tick，默认 20 帧 = 10 ticks） |
| `prompt_length_ticks` | `int \| None` | 用于上下文截断的提示长度，`None` 表示不截断 |

**返回**：`(List[MusicalEvent], TimingInfo)` 元组。

**契约**：
- 返回的 `MusicalEvent` 的 `tick` 均相对于系统绝对 tick（非相对值）
- 引擎内部维护历史状态（`_inference_worker` 会重复使用同一引擎实例）

### `inject_history(melody_events, accompaniment_events, injection_length_ticks) -> None`

预填充模型历史，用于 MIDI prompt 注入（`--injection-file` 选项）。在 `service.start()` 时调用，`generate_accompaniment()` 调用之前完成。

### `set_injection_offset(offset_ticks) -> None`

设置注入偏移量（tick），影响后续 `generate_accompaniment()` 的 tick 对齐。

### `clear_history() -> None`

清空引擎内部历史状态和注入状态，使引擎恢复到初始状态。

---

## `TimingInfo`

**源文件**：`interfaces/timing_info.py`

推理调用的时序数据，由推理引擎返回。

```python
@dataclass(frozen=True)
class TimingInfo:
    request_arrival_time: float
    response_output_time: float
    preprocess_start_time: float
    inference_start_time: float
    inference_end_time: float
    postprocess_start_time: float
    round_trip_time: Optional[float] = None
    server_processing_duration: Optional[float] = None
    total_network_latency: Optional[float] = None
```

### 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_arrival_time` | `float` | 服务端收到请求时的时间戳（Unix 秒） |
| `response_output_time` | `float` | 服务端开始发送响应时的时间戳 |
| `preprocess_start_time` | `float` | 服务端预处理开始时间 |
| `inference_start_time` | `float` | 模型推理开始时间 |
| `inference_end_time` | `float` | 模型推理结束时间 |
| `postprocess_start_time` | `float` | 服务端后处理开始时间 |
| `round_trip_time` | `Optional[float]` | 客户端往返时间（秒），由客户端在接收响应后设置 |
| `server_processing_duration` | `Optional[float]` | 服务端处理时长（秒），由服务端计算 |
| `total_network_latency` | `Optional[float]` | 纯网络延迟估算（秒） |
