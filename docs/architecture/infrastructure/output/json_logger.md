---
title: JsonLoggerOutputSink — JSON 日志输出
description: 将事件写入 events.jsonl，将推理记录写入 inferences.json
---

# JsonLoggerOutputSink — JSON 日志输出

**源文件**：`src/streammuse/infrastructure/output/json_logger.py`

将音符事件和推理信息持久化为 JSON 格式，用于事后分析和性能评估。

---

## `JsonLoggerOutputSink`

### `__init__(session_dir: Path) -> None`

| 参数 | 说明 |
|---|---|
| `session_dir` | 会话目录（通常为 `logs/session_YYYYMMDD-HHMMSS/`） |

初始化时创建以下对象：
- `events_file = session_dir / "events.jsonl"` ← 逐行追加
- `inferences_file = session_dir / "inferences.json"` ← 最终写入
- `MetricsCalculator` 实例（实时累积统计）
- `inferences: list[InferenceEvent]`（内存中累积）

---

### `output_event(event, source)`

将 `MusicalEvent` 转换为 `LogEvent` 并：
1. 追加到 `MetricsCalculator`（`add_event()`）
2. 以 JSON Lines 格式写入 `events.jsonl`

写入格式（每行一个 JSON 对象）：
```json
{"timestamp": 1700000000.123, "tick": 12, "event_type": "note_on", "data": {"pitch": 60, "source": "user", "velocity": 80}}
```

---

### `log_inference(request, response, latency_ms, server_process_ms)`

> 此方法不在 `OutputSink` Protocol 中，由服务层通过 `hasattr` 检测调用。

创建 `InferenceEvent` 并：
1. 追加到 `self.inferences` 列表（内存）
2. 追加到 `MetricsCalculator`（`add_inference()`）

**不立即写文件**，在 `close()` 或 `save_inferences()` 时集中写入，保证 `inferences.json` 是完整的 JSON 数组。

---

### `save_metrics(session_config: Dict) -> None`

调用 `MetricsCalculator` 生成并写入：
- `performance.json`：延迟百分位、事件计数、音乐分析
- `statistics.csv`：摘要指标

---

### `save_inferences() -> None`

将 `self.inferences` 序列化为 JSON 数组写入 `inferences.json`。

---

### `output_tick`, `output_stats`, `output_status`, `output_config`

均为空操作（no-op）。

### `close()`

调用 `save_inferences()`（如 `close()` 中包含此逻辑时）。注意：CLI 中通过 `cleanup()` 函数显式调用 `save_metrics()` 和 `save_inferences()`。

---

## 文件输出示例

运行结束后，会话目录中包含：

```
session_20241201-120000/
├── events.jsonl       # 每行一个事件，实时追加
├── inferences.json    # 所有推理记录，退出时写入
├── performance.json   # MetricsCalculator 生成
└── statistics.csv     # 摘要 CSV
```
