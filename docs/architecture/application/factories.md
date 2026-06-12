---
title: application/factories — 工厂类
description: InputSourceFactory、OutputSinkFactory、InferenceEngineFactory 的实现细节
---

# application/factories — 工厂类

**源文件**：`src/streammuse/application/factories/`

三个 Factory 类将 `ApplicationConfig` 中的类型字符串转换为对应的 Domain 接口实现。Factory 模块是当前代码中的主要 composition root。

---

## `InputSourceFactory`

**源文件**：`factories/input_factory.py`

### `create(app_config, *, list_events=None) -> InputSource`

| `app_config.input.type` | 返回类 | 备注 |
|---|---|---|
| `"midi_device"` | `MidiDeviceInput` | 使用 `midi_device_name`，可为 `None` |
| `"keyboard"` | `KeyboardInput` | 无需额外参数 |
| `"midi_file"` | `MidiFileInput` | 需要 `midi_file_path` 非空 |
| `"list"` | `ListInput` | 使用 `list_events`，主要用于测试 |

`midi_file` 模式会构建：

```python
MidiFileInputConfig(
    bpm=float(tempo.bpm),
    ticks_per_beat=int(tempo.ticks_per_beat),
    delay_ticks=int(cfg.midi_file_delay_ticks),
    start_tick=(int(cfg.injection_length_ticks) if cfg.injection_file else 0),
)
```

这意味着 CLI injection 成功后，正式 MIDI 文件输入会跳过已注入的前 `injection_length_ticks`。

---

## `OutputSinkFactory`

**源文件**：`factories/output_factory.py`

### `create(app_config, session_manager=None) -> OutputSink`

| `app_config.output.type` | 基础 sink | 备注 |
|---|---|---|
| `"console"` | `ConsoleOutputSink` | 若有 session manager，会自动附加 `MidiFileOutputSink` |
| `"audio"` | `AudioOutputSink` | 使用 `midi_out_port`；若有 session manager，会自动附加 `MidiFileOutputSink` |
| `"midi_file"` | `MidiFileOutputSink` | 需要 `midi_file_output_path` 非空 |
| `"websocket"` | `WebSocketOutputSink` | 若有 session manager，会自动附加 `MidiFileOutputSink` |
| `"json_log"` | `JsonLoggerOutputSink` | 需要 `session_manager` |
| `"session"` | `SessionLoggerOutputSink` | 需要 `session_manager` |
| `"composite"` | `CompositeOutputSink` | CLI 中通常为 Console + SessionLogger |

### 自动 MIDI 录制

`console` / `audio` / `websocket` 在有 `session_manager` 时会被包装为：

```python
CompositeOutputSink([base_sink, MidiFileOutputSink(...combined.mid...)])
```

自动 MIDI 的 `record_metronome` 来自 `app_config.output.metronome_enabled`。

### metronome 附加

所有 output type 最后都会经过 `_attach_metronome_if_needed()`：

```python
if cfg.metronome_enabled:
    metronome_sink = MetronomeOutputSink(
        MetronomeOutputConfig(
            port_name=cfg.metronome_port or cfg.midi_out_port,
            ticks_per_beat=tempo.ticks_per_beat,
            beats_per_bar=tempo.beats_per_bar,
            channel=cfg.metronome_channel,
        )
    )
```

因此 `--enable-metronome` 有两个效果：

1. 增加实时 MIDI click 输出。
2. 对所有写 MIDI 的模式，额外记录 `Metronome` 鼓轨。

---

## `InferenceEngineFactory`

**源文件**：`factories/inference_factory.py`

| `app_config.inference.type` | 返回类 | 备注 |
|---|---|---|
| `"http"` | `HttpInferenceClient` | 使用 `server_generate_url`、`timeout_s`，并透传 `model_name`、`inference_mode`、`generation_interval_ticks`、`checkpoint_path` |
| `"stanley"` | `StanleyInferenceEngine` | 需要 `checkpoint_path` 非空 |

HTTP 模式是 Lekai server 和 fake server 的主路径；本地 Stanley 模式不需要 HTTP server。
