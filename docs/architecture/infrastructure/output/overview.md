---
title: infrastructure/output — 输出适配器总览
description: 七种 OutputSink 实现的比较与组合关系
---

# infrastructure/output — 输出适配器总览

**源文件**：`src/streammuse/infrastructure/output/`

所有输出适配器均实现 `OutputSink` 协议（见 [domain/interfaces](../../domain/interfaces.md)）。部分适配器还额外实现 `log_inference()` 方法（非 Protocol 要求，通过 `hasattr` 检测调用）。

---

## 七种实现对比

| 类 | 类型 | 说明 |
|---|---|---|
| `ConsoleOutputSink` | 调试 | 将所有事件和统计信息打印到终端 |
| `AudioOutputSink` | 播放 | 向 MIDI 输出端口发送音符，实现实时音频播放 |
| `MidiFileOutputSink` | 录制 | 将事件流录制为 MIDI 文件 |
| `WebSocketOutputSink` | 推送 | 将事件序列化为 JSON，入队供 WebSocket 服务器推送 |
| `JsonLoggerOutputSink` | 日志 | 写入 `events.jsonl` + `inferences.json` |
| `SessionLoggerOutputSink` | 日志 | 组合 MIDI 录制 + JSON 日志，写入会话目录 |
| `CompositeOutputSink` | 组合 | 将所有调用 fan-out 到多个子 Sink |

---

## 组合关系

```
CompositeOutputSink
 ├── ConsoleOutputSink
 └── SessionLoggerOutputSink
      ├── MidiFileOutputSink  → combined.mid
      └── JsonLoggerOutputSink → events.jsonl, inferences.json
```

这是 `--output-type composite --log-dir logs` 时的默认组合（有 `session_manager`）。

---

## `log_inference()` 模式

以下 Sink 额外实现了 `log_inference()`：
- `JsonLoggerOutputSink`（直接实现）
- `SessionLoggerOutputSink`（委托给内部的 `json_sink`）
- `CompositeOutputSink`（通过 `hasattr` fan-out 给所有子 Sink）

服务层（`RealTimeMusicService._inference_worker`）通过 `hasattr(output_sink, "log_inference")` 检测后调用，不要求 Sink 强制实现。

---

## 详细文档

- [console.md](console.md) — `ConsoleOutputSink`
- [audio.md](audio.md) — `AudioOutputSink`
- [midi_file.md](midi_file.md) — `MidiFileOutputSink`
- [websocket.md](websocket.md) — `WebSocketOutputSink`
- [json_logger.md](json_logger.md) — `JsonLoggerOutputSink`
- [session_logger.md](session_logger.md) — `SessionLoggerOutputSink`
- [composite.md](composite.md) — `CompositeOutputSink`
