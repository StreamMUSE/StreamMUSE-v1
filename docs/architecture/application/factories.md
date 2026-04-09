---
title: application/factories — 工厂类
description: InputSourceFactory、OutputSinkFactory、InferenceEngineFactory 的实现细节
---

# application/factories — 工厂类

**源文件**：`src/streammuse/application/factories/`

三个 Factory 类各自实现一个静态方法 `create()`，将 `ApplicationConfig` 中的类型字符串转换为对应的 Domain 接口实现。Factory 模块是当前代码中的主要组合入口（composition root）。

---

## `InputSourceFactory`

**源文件**：`factories/input_factory.py`

### `create(app_config, *, list_events=None) -> InputSource`

```python
@staticmethod
def create(
    app_config: ApplicationConfig,
    *,
    list_events: list[MusicalEvent] | None = None,
) -> InputSource:
```

根据 `app_config.input.type` 创建对应的 `InputSource` 实现：

| `type` 值 | 返回类 | 备注 |
|---|---|---|
| `"midi_device"` | `MidiDeviceInput` | 使用 `app_config.input.midi_device_name`（可为 `None`） |
| `"keyboard"` | `KeyboardInput` | 无需额外参数 |
| `"midi_file"` | `MidiFileInput` | 需要 `app_config.input.midi_file_path` 非 `None` |
| `"list"` | `ListInput` | 使用 `list_events` 参数（主要用于测试） |

**异常**：
- `midi_file` 模式且 `midi_file_path=None` 时，抛出 `ValueError`
- 未知 `type` 时，抛出 `ValueError: Unknown input type: <type>`

`MidiFileInput` 会用 `MidiFileInputConfig` 封装速度参数（`bpm`、`ticks_per_beat`、`delay_ticks`）。

---

## `OutputSinkFactory`

**源文件**：`factories/output_factory.py`

### `create(app_config, session_manager=None) -> OutputSink`

```python
@staticmethod
def create(
    app_config: ApplicationConfig,
    session_manager: Optional[SessionManager] = None,
) -> OutputSink:
```

根据 `app_config.output.type` 创建对应的 `OutputSink` 实现：

| `type` 值 | 返回类 | 备注 |
|---|---|---|
| `"console"` | `ConsoleOutputSink` | 使用默认 `ConsoleOutputConfig` |
| `"audio"` | `AudioOutputSink` | 使用 `midi_out_port`（可为 `None`） |
| `"midi_file"` | `MidiFileOutputSink` | 需要 `midi_file_output_path` 非 `None` |
| `"websocket"` | `WebSocketOutputSink` | 使用默认配置 |
| `"json_log"` | `JsonLoggerOutputSink` | 需要 `session_manager` 非 `None` |
| `"session"` | `SessionLoggerOutputSink` | 需要 `session_manager` 非 `None` |
| `"composite"` | `CompositeOutputSink` | 见下文 |

**`composite` 模式的构建逻辑**：

- 若有 `session_manager`：`CompositeOutputSink([ConsoleOutputSink, SessionLoggerOutputSink])`
- 若无 `session_manager`：`CompositeOutputSink([ConsoleOutputSink, WebSocketOutputSink])`

**异常**：
- `midi_file` 模式且 `midi_file_output_path=None` 时，抛出 `ValueError`
- `json_log` 或 `session` 模式且 `session_manager=None` 时，抛出 `ValueError`
- 未知 `type` 时，抛出 `ValueError: Unknown output type: <type>`

**`session_manager` 参数**：仅在 `json_log`、`session`、`composite`（有 `session_manager` 时）模式下使用。

---

## `InferenceEngineFactory`

**源文件**：`factories/inference_factory.py`

### `create(app_config) -> InferenceEngine`

```python
@staticmethod
def create(app_config: ApplicationConfig) -> InferenceEngine:
```

根据 `app_config.inference.type` 创建对应的 `InferenceEngine` 实现：

| `type` 值 | 返回类 | 备注 |
|---|---|---|
| `"http"` | `HttpInferenceClient` | 使用 `server_generate_url`、`timeout_s`，并透传 `model_name`、`inference_mode`、`generation_interval_ticks`、`checkpoint_path` |
| `"stanley"` | `StanleyInferenceEngine` | 需要 `checkpoint_path` 非 `None` |

**`stanley` 模式**：创建 `StanleyInferenceConfig`（包含 `checkpoint_path`、`model_size`、`model_max_seq_len_frames`、`generation_length_frames`），传入 `StanleyInferenceEngine`。

**异常**：
- `stanley` 模式且 `checkpoint_path=None` 时，抛出 `ValueError`
- 未知 `type` 时，抛出 `ValueError: Unknown inference type: <type>`

**扩展性说明**：如需添加新的推理引擎类型（如 Lekai），只需：
1. 在 `InferenceType` Literal 中新增类型字符串
2. 在 `InferenceConfig` 中新增必要配置字段
3. 在 `InferenceEngineFactory.create()` 中新增分支
