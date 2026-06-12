---
title: infrastructure/output — 输出适配器总览
description: OutputSink 实现、自动 MIDI 录制、metronome 与组合关系
---

# infrastructure/output — 输出适配器总览

**源文件**：`src/streammuse/infrastructure/output/`

所有输出适配器均实现 `OutputSink` 协议（见 [domain/interfaces](../../domain/interfaces.md)）。部分适配器额外实现 `log_inference()` 或 `output_metronome_tick()`，服务层通过 `hasattr` 检测调用。

---

## 用户可选输出类型

| 类 | CLI 类型 | 说明 |
|---|---|---|
| `ConsoleOutputSink` | `console` | 将事件和统计信息打印到终端 |
| `AudioOutputSink` | `audio` | 向 MIDI 输出端口发送模型音符，实现实时播放 |
| `MidiFileOutputSink` | `midi_file` | 将事件流录制为 MIDI 文件 |
| `WebSocketOutputSink` | `websocket` | 将事件序列化为 JSON，入队供 WebSocket 推送 |
| `JsonLoggerOutputSink` | `json_log` | 写入 `events.jsonl` 和 `inferences.json` |
| `SessionLoggerOutputSink` | `session` | 组合 MIDI 录制和 JSON 日志，写入会话目录 |
| `CompositeOutputSink` | `composite` | 将调用 fan-out 到多个子 sink |

`MetronomeOutputSink` 是辅助 sink，由 `--enable-metronome` 附加，不作为 `--output-type` 的可选值。

---

## 组合关系

`--output-type composite` 且有 session manager 时：

```text
CompositeOutputSink
 ├── ConsoleOutputSink
 └── SessionLoggerOutputSink
      ├── MidiFileOutputSink  → combined.mid
      └── JsonLoggerOutputSink → events.jsonl, inferences.json
```

`console` / `audio` / `websocket` 且有 session manager 时，工厂会自动附加 MIDI 录制：

```text
CompositeOutputSink
 ├── <base sink>
 └── MidiFileOutputSink → combined.mid
```

开启 `--enable-metronome` 后，工厂会继续附加：

```text
CompositeOutputSink
 ├── ...existing sinks...
 └── MetronomeOutputSink → live MIDI click
```

同时，所有 `MidiFileOutputSink` 会用 `record_metronome=True` 记录 `Metronome` 鼓轨。

---

## 扩展方法

### `log_inference()`

以下 sink 支持推理日志：

- `JsonLoggerOutputSink`
- `SessionLoggerOutputSink`
- `CompositeOutputSink`（fan-out）

### `output_metronome_tick()`

以下 sink 支持 metronome tick：

- `MetronomeOutputSink`：实时发送 MIDI drum note
- `MidiFileOutputSink`：写入 MIDI `Metronome` 轨
- `SessionLoggerOutputSink`：委托给内部 `midi_sink`
- `CompositeOutputSink`：fan-out

---

## 详细文档

- [console.md](console.md) — `ConsoleOutputSink`
- [audio.md](audio.md) — `AudioOutputSink`
- [midi_file.md](midi_file.md) — `MidiFileOutputSink`
- [metronome.md](metronome.md) — `MetronomeOutputSink`
- [websocket.md](websocket.md) — `WebSocketOutputSink`
- [json_logger.md](json_logger.md) — `JsonLoggerOutputSink`
- [session_logger.md](session_logger.md) — `SessionLoggerOutputSink`
- [composite.md](composite.md) — `CompositeOutputSink`
