---
title: 会话日志与分析
description: 使用 session/composite 模式记录演奏并分析性能数据
---

# 会话日志与分析

使用 `--output-type session` 或 `--output-type composite` 可启用完整会话日志。当前 CLI 还会为 `console` / `audio` / `websocket` 自动创建 session 目录并写 `combined.mid`。

---

## 启动会话日志

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --output-type session \
    --log-dir logs
```

每次运行会在 `logs/` 下创建日期和会话子目录：

```text
logs/
└── 2026-06-11/
    └── session_120000/
        ├── events.jsonl
        ├── inferences.json
        ├── performance.json
        ├── statistics.csv
        ├── session_config.json
        ├── session_summary.txt
        ├── melody_history.json
        ├── accompaniment_history.json
        └── combined.mid
```

`melody_history.json` 和 `accompaniment_history.json` 来自 CLI 退出时调用 server `clear_history()` 返回的历史。

---

## 文件说明

### `session_config.json`

保存本次会话配置。当前包含 tempo、input/output type、metronome、count-in 和推理参数等字段，例如：

```json
{
  "tempo_bpm": 120.0,
  "ticks_per_beat": 4,
  "beats_per_bar": 4,
  "input_type": "midi_file",
  "output_type": "console",
  "metronome_enabled": true,
  "metronome_port": null,
  "metronome_channel": 9,
  "count_in_beats": 4,
  "inference_type": "http",
  "generation_interval_ticks": 4,
  "generation_length_frames": 4
}
```

### `events.jsonl`

每行是一个 JSON 对象，记录 tick、event type、source、pitch、velocity 等事件信息。

### `inferences.json`

JSON 数组，记录每次推理请求和响应。`--inference-log-detail summary` 只记录摘要；`full` 会记录完整 melody/accompaniment 事件。

### `performance.json` 和 `statistics.csv`

`SessionLoggerOutputSink` 关闭时保存性能报告和 CSV 摘要。`json_log` 模式当前不保证生成这两个文件。

### `combined.mid`

MIDI 录制文件。默认包含：

- `Melody`：用户输入旋律
- `Accompaniment`：模型伴奏

如果启用 `--enable-metronome`，还会包含：

- `Metronome`：鼓轨，记录 beat/downbeat click

如果同时启用 `--count-in-beats`，count-in click 会出现在 MIDI 文件开头；正式音乐 tick=0 会在录制时间线上向后平移。这个平移只影响 MIDI 录制，不改变服务层推理 tick。

---

## 日志分析示例

```python
import json

with open("logs/2026-06-11/session_120000/inferences.json") as f:
    inferences = json.load(f)

latencies = [inf["latency_ms"] for inf in inferences]
print(f"平均延迟: {sum(latencies)/len(latencies):.1f} ms")
print(f"最大延迟: {max(latencies):.1f} ms")
```

```python
import json

events = []
with open("logs/2026-06-11/session_120000/events.jsonl") as f:
    for line in f:
        events.append(json.loads(line))

user_notes = [e for e in events if e["data"].get("source") == "user" and e["event_type"] == "note_on"]
model_notes = [e for e in events if e["data"].get("source") == "model" and e["event_type"] == "note_on"]
print(f"用户音符: {len(user_notes)}, 模型音符: {len(model_notes)}")
```

---

## HTTP 基准测试

```bash
uv run python scripts/fake_inference_server.py

uv run python scripts/benchmark_lekai_http.py \
    --url http://127.0.0.1:8000/generate_accompaniment \
    --num-requests 20 \
    --warmup-requests 3 \
    --output results/benchmark_http.json
```
