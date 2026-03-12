---
title: SessionLoggerOutputSink — 会话日志输出
description: 组合 MidiFileOutputSink 和 JsonLoggerOutputSink，写入完整会话记录
---

# SessionLoggerOutputSink — 会话日志输出

**源文件**：`src/streammuse/infrastructure/output/session_logger.py`

将 MIDI 录制（`MidiFileOutputSink`）和 JSON 日志（`JsonLoggerOutputSink`）组合在一起，在单一会话目录中保存完整的演奏记录。

---

## `SessionLoggerOutputSink`

### `__init__(...)`

```python
def __init__(
    self,
    session_dir: Path,
    include_midi: bool = True,
    include_json: bool = True,
) -> None:
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `session_dir` | `Path` | 必填 | 会话目录（由 `SessionManager` 创建） |
| `include_midi` | `bool` | `True` | 是否创建 MIDI 录制 |
| `include_json` | `bool` | `True` | 是否创建 JSON 日志 |

初始化时（如已启用）：
- 创建 `MidiFileOutputSink`，配置 `bpm=120.0`，`ticks_per_beat=4`，`output_path=session_dir/combined.mid`
- 创建 `JsonLoggerOutputSink(session_dir)`

---

### `output_event`, `output_tick`, `output_stats`, `output_status`, `output_config`

将调用委托给 `midi_sink` 和 `json_sink`（如已初始化）。

---

### `log_inference(request, response, latency_ms, server_process_ms)`

> 此方法不在 `OutputSink` Protocol 中，由服务层通过 `hasattr` 检测调用。

仅委托给 `json_sink.log_inference()`（MIDI Sink 不处理推理信息）。

---

### `save_metrics(session_config: Dict) -> None`

委托给 `json_sink.save_metrics()`，生成 `performance.json` 和 `statistics.csv`。

---

### `close()`

依次调用 `midi_sink.close()` 和 `json_sink.close()`，保存 MIDI 文件。

---

## 输出文件结构

```
session_20241201-120000/
├── combined.mid       # 用户 + 模型的 MIDI 录制（两个音轨）
├── events.jsonl       # 每行一个事件
├── inferences.json    # 推理记录
├── performance.json   # 性能报告
└── statistics.csv     # 摘要 CSV
```

`combined.mid` 包含两个音轨：`User`（用户演奏）和 `Model`（模型伴奏）。

---

## 与 CompositeOutputSink 的关系

在 `--output-type composite --log-dir logs` 模式下，`OutputSinkFactory` 创建：
```
CompositeOutputSink([
    ConsoleOutputSink,        # 终端实时显示
    SessionLoggerOutputSink,  # 完整会话录制
])
```
