---
title: 会话日志与分析
description: 使用 session/composite 模式记录演奏并分析性能数据
---

# 会话日志与分析

使用 `--output-type session` 或 `--output-type composite` 启用会话日志。

说明：
1. `session` 模式会生成完整日志（含 `performance.json`、`statistics.csv`）。
2. 当前实现中，`composite` 模式默认会生成 `events.jsonl`、`inferences.json`、`session_config.json`、`combined.mid`，但不会自动生成 `performance.json` 和 `statistics.csv`。

---

## 启动会话日志

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --output-type session \
    --log-dir logs
```

每次运行会在 `logs/` 目录下创建一个带时间戳的子目录：

```
logs/
└── session_20241201-120000/
    ├── events.jsonl
    ├── inferences.json
    ├── performance.json
    ├── statistics.csv
    ├── session_config.json
    ├── session_summary.txt
    └── combined.mid
```

若使用 `--output-type composite`，通常目录中会有：

```
logs/
└── session_20241201-120000/
    ├── events.jsonl
    ├── inferences.json
    ├── session_config.json
    └── combined.mid
```

---

## 文件说明

### `events.jsonl`

每行是一个 JSON 对象，记录每个音符事件：

```json
{"timestamp": 1700000000.123, "tick": 12, "event_type": "note_on", "data": {"pitch": 60, "source": "user", "velocity": 80}}
{"timestamp": 1700000000.456, "tick": 16, "event_type": "note_off", "data": {"pitch": 60, "source": "user", "velocity": 0}}
```

### `inferences.json`

JSON 数组，记录每次推理请求和响应：

```json
[
  {
    "inference_id": "inf_001",
    "timestamp_request": 1700000000.1,
    "timestamp_response": 1700000000.15,
    "latency_ms": 50.0,
    "server_process_ms": 45.2,
    "request_data": {...},
    "response_data": {...}
  }
]
```

### `performance.json`

`MetricsCalculator` 生成的性能报告，包含：

- **延迟指标**：mean/median/min/max（样本足够时附加 p95/p99）
- **事件统计**：用户/模型音符数量
- **音乐分析**：音符密度等

### `statistics.csv`

性能指标的 CSV 摘要，便于与其他工具（pandas、Excel）对比分析。

> 仅 `session` 模式在当前实现中保证自动生成。

### `session_config.json`

保存本次会话的完整配置（BPM、输入类型、推理参数等）。

### `combined.mid`

包含两个音轨（User 和 Model）的 MIDI 录制，可用 MuseScore、Logic Pro、GarageBand 等 DAW 打开检查。

---

## 日志分析示例

```python
import json
import pandas as pd

# 读取推理延迟
with open("logs/session_20241201-120000/inferences.json") as f:
    inferences = json.load(f)

latencies = [inf["latency_ms"] for inf in inferences]
print(f"平均延迟: {sum(latencies)/len(latencies):.1f} ms")
print(f"最大延迟: {max(latencies):.1f} ms")

# 读取事件
events = []
with open("logs/session_20241201-120000/events.jsonl") as f:
    for line in f:
        events.append(json.loads(line))

user_notes = [e for e in events if e["data"]["source"] == "user" and e["event_type"] == "note_on"]
model_notes = [e for e in events if e["data"]["source"] == "model" and e["event_type"] == "note_on"]
print(f"用户音符: {len(user_notes)}, 模型音符: {len(model_notes)}")
```

---

## 基准测试

利用现有基准脚本对 HTTP 推理接口做延迟测试：

```bash
# 启动 Lekai 或 fake 推理服务器（任选其一）
uv run python scripts/fake_inference_server.py

# 运行基准并导出 JSON
uv run python scripts/benchmark_lekai_http.py \
    --url http://127.0.0.1:8000/generate_accompaniment \
    --num-requests 20 \
    --warmup-requests 3 \
    --output results/benchmark_http.json
```
