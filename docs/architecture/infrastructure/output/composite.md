---
title: CompositeOutputSink — 组合输出
description: Fan-out 模式，将所有 OutputSink 调用分发给多个子 Sink
---

# CompositeOutputSink — 组合输出

**源文件**：`src/streammuse/infrastructure/output/composite.py`

实现 fan-out 模式，将所有 `OutputSink` 方法调用依次转发给内部的每个子 Sink。

---

## `CompositeOutputSink`

```python
@dataclass(frozen=True)
class CompositeOutputSink:
    sinks: List[Any]
```

`CompositeOutputSink` 是一个 `frozen=True` 的 dataclass，持有 `sinks: List[Any]`（子 Sink 列表，类型为 `Any` 以允许混合类型）。

---

## 方法行为

### `output_event`, `output_tick`, `output_stats`, `output_status`, `output_config`

对所有子 Sink 依次调用对应方法（串行 fan-out）：

```python
def output_event(self, event, source):
    for s in self.sinks:
        s.output_event(event, source)
```

---

### `log_inference(request, response, latency_ms, server_process_ms)`

> 此方法本身不在 `OutputSink` Protocol 中，`CompositeOutputSink` 额外实现以支持日志聚合。

对每个子 Sink，先检查 `hasattr(s, "log_inference")`，再调用：

```python
def log_inference(self, request, response, latency_ms, server_process_ms):
    for s in self.sinks:
        if hasattr(s, "log_inference"):
            s.log_inference(request, response, latency_ms, server_process_ms)
```

这样不支持 `log_inference` 的子 Sink（如 `ConsoleOutputSink`）不会出错。

---

### `close()`

依次调用所有子 Sink 的 `close()` 方法，顺序与 `sinks` 列表一致。

---

## 典型用法

由 `OutputSinkFactory` 在 `composite` 模式下构建：

```python
# 有 session_manager 时
sinks = [
    ConsoleOutputSink(ConsoleOutputConfig()),
    SessionLoggerOutputSink(session_dir=..., include_midi=True, include_json=True),
]
composite = CompositeOutputSink(sinks=sinks)

# 无 session_manager 时
sinks = [
    ConsoleOutputSink(ConsoleOutputConfig()),
    WebSocketOutputSink(),
]
composite = CompositeOutputSink(sinks=sinks)
```

也可以手动组合任意数量的 Sink：

```python
composite = CompositeOutputSink(sinks=[
    ConsoleOutputSink(),
    AudioOutputSink(),
    MidiFileOutputSink(config),
])
```

---

## 注意事项

- 子 Sink 中任何一个抛出异常，后续子 Sink 将不会被调用（因为是串行遍历，无异常捕获）
- `sinks` 列表在 dataclass 初始化后不可修改（`frozen=True`）
