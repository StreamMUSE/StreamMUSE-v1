---
title: SessionLoggerOutputSink — 会话日志输出
description: 组合 MidiFileOutputSink 和 JsonLoggerOutputSink，写入完整会话记录
---

# SessionLoggerOutputSink — 会话日志输出

**源文件**：`src/streammuse/infrastructure/output/session_logger.py`

`SessionLoggerOutputSink` 将 MIDI 录制和 JSON 日志组合在一起，在同一会话目录中保存完整演奏记录。

---

## `__init__(...)`

```python
def __init__(
    self,
    session_dir: Path,
    include_midi: bool = True,
    include_json: bool = True,
    inference_log_detail: str = "summary",
    bpm: float = 120.0,
    ticks_per_beat: int = 4,
    beats_per_bar: int = 4,
    record_metronome: bool = False,
) -> None:
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `session_dir` | 必填 | 会话目录 |
| `include_midi` | `True` | 是否创建 `MidiFileOutputSink` |
| `include_json` | `True` | 是否创建 `JsonLoggerOutputSink` |
| `inference_log_detail` | `summary` | 推理日志粒度 |
| `bpm` | `120.0` | MIDI 录制 BPM |
| `ticks_per_beat` | `4` | MIDI 录制 tick 分辨率 |
| `beats_per_bar` | `4` | downbeat 判断 |
| `record_metronome` | `False` | 是否记录 `Metronome` 鼓轨 |

初始化时：

- `MidiFileOutputSink` 写 `session_dir/combined.mid`。
- `JsonLoggerOutputSink` 写 JSON 日志。

---

## 方法委托

| 方法 | 委托目标 |
|---|---|
| `output_event` | MIDI + JSON |
| `output_tick` | MIDI + JSON |
| `output_metronome_tick` | MIDI |
| `output_stats` | MIDI + JSON |
| `output_status` | MIDI + JSON |
| `output_config` | MIDI + JSON |
| `log_inference` | JSON |

`output_metronome_tick` 只委托给 `midi_sink`，因此 session 日志中 metronome 体现为 MIDI 鼓轨，不进入 `events.jsonl`。

---

## 输出文件结构

```text
session_HHMMSS/
├── combined.mid       # Melody + Accompaniment，可选 Metronome
├── events.jsonl       # 每行一个事件
├── inferences.json    # 推理记录
├── performance.json   # 性能报告
└── statistics.csv     # 摘要 CSV
```

`combined.mid` 默认包含 `Melody` 和 `Accompaniment` 两轨。开启 `--enable-metronome` 后额外包含 `Metronome` 鼓轨；如果启用了 `--count-in-beats`，count-in click 会出现在 MIDI 开头。

---

## 与 CompositeOutputSink 的关系

在 `--output-type composite --log-dir logs` 模式下，`OutputSinkFactory` 创建：

```python
CompositeOutputSink([
    ConsoleOutputSink(...),
    SessionLoggerOutputSink(...),
])
```

如果同时启用 metronome，外层还会附加 `MetronomeOutputSink`。
