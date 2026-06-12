---
title: application/config — 配置数据模型
description: TempoConfig、InputConfig、OutputConfig、InferenceConfig、ApplicationConfig 字段说明
---

# application/config — 配置数据模型

**源文件**：`src/streammuse/application/config/models.py`

所有配置对象均为 `frozen=True` 的 dataclass，在服务启动后不可修改。顶层对象为 `ApplicationConfig`。

---

## `TempoConfig`

```python
@dataclass(frozen=True)
class TempoConfig:
    bpm: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bpm` | `float` | `120.0` | 每分钟拍数 |
| `ticks_per_beat` | `int` | `4` | 每拍 tick 数 |
| `beats_per_bar` | `int` | `4` | 每小节拍数 |

---

## `InputConfig`

```python
@dataclass(frozen=True)
class InputConfig:
    type: InputType = "midi_device"
    midi_device_name: Optional[str] = None
    midi_file_path: Optional[str] = None
    midi_file_delay_ticks: int = 0
    injection_file: Optional[str] = None
    injection_length_ticks: int = 0
    injection_acc_file: Optional[str] = None
```

`InputType = Literal["midi_device", "keyboard", "midi_file", "list"]`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `InputType` | `"midi_device"` | 输入源类型 |
| `midi_device_name` | `Optional[str]` | `None` | MIDI 设备名称；`None` 时自动选择第一个可用设备 |
| `midi_file_path` | `Optional[str]` | `None` | MIDI 文件路径（`midi_file` 模式使用） |
| `midi_file_delay_ticks` | `int` | `0` | MIDI 文件开始前的延迟 ticks |
| `injection_file` | `Optional[str]` | `None` | 注入历史用的 melody MIDI 文件 |
| `injection_length_ticks` | `int` | `0` | 注入历史长度 ticks |
| `injection_acc_file` | `Optional[str]` | `None` | 可选 accompaniment MIDI 文件 |

---

## `OutputConfig`

```python
@dataclass(frozen=True)
class OutputConfig:
    type: OutputType = "console"
    midi_out_port: Optional[str] = None
    midi_file_output_path: Optional[str] = None
    inference_log_detail: InferenceLogDetail = "summary"
    metronome_enabled: bool = False
    metronome_port: Optional[str] = None
    metronome_channel: int = 9
```

`OutputType = Literal["audio", "midi_file", "console", "websocket", "composite", "json_log", "session"]`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `OutputType` | `"console"` | 输出类型 |
| `midi_out_port` | `Optional[str]` | `None` | MIDI 输出端口名称 |
| `midi_file_output_path` | `Optional[str]` | `None` | MIDI 文件保存路径（`midi_file` 模式使用） |
| `inference_log_detail` | `"summary" \| "full"` | `"summary"` | 推理日志粒度 |
| `metronome_enabled` | `bool` | `False` | 是否启用实时 MIDI click 和 MIDI metronome 录制 |
| `metronome_port` | `Optional[str]` | `None` | metronome 专用 MIDI 输出端口 |
| `metronome_channel` | `int` | `9` | metronome MIDI channel |

---

## `InferenceConfig`

```python
@dataclass(frozen=True)
class InferenceConfig:
    type: InferenceType = "http"
    server_generate_url: str = "http://localhost:8000/generate_accompaniment"
    timeout_s: float = 30.0
    model_name: ModelName = "stanley"
    inference_mode: str = "sliding_window"
    checkpoint_path: Optional[str] = None
    model_size: str = "0.12B"
    model_max_seq_len_frames: int = 96
    generation_length_frames: int = 20
    generation_interval_ticks: int = 2
```

`InferenceType = Literal["http", "stanley"]`，`ModelName = Literal["stanley", "lekai"]`。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `InferenceType` | `"http"` | 推理引擎类型 |
| `server_generate_url` | `str` | `"http://localhost:8000/generate_accompaniment"` | HTTP generate 端点 URL |
| `timeout_s` | `float` | `30.0` | HTTP 请求超时时间 |
| `model_name` | `ModelName` | `"stanley"` | HTTP server 端模型选择 |
| `inference_mode` | `str` | `"sliding_window"` | 透传给 server 的推理模式提示 |
| `checkpoint_path` | `Optional[str]` | `None` | 模型 checkpoint 路径 |
| `model_size` | `str` | `"0.12B"` | Stanley 模型规模 |
| `model_max_seq_len_frames` | `int` | `96` | 模型最大序列长度 |
| `generation_length_frames` | `int` | `20` | 每次推理生成帧数 |
| `generation_interval_ticks` | `int` | `2` | 透传给 HTTP server 和日志；当前 tick loop 不用它决定触发时刻 |

---

## `ApplicationConfig`

```python
@dataclass(frozen=True)
class ApplicationConfig:
    tempo: TempoConfig = TempoConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    inference: InferenceConfig = InferenceConfig()
    count_in_beats: int = 0
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `tempo` | `TempoConfig` | 速度配置 |
| `input` | `InputConfig` | 输入源配置 |
| `output` | `OutputConfig` | 输出 Sink 配置 |
| `inference` | `InferenceConfig` | 推理引擎配置 |
| `count_in_beats` | `int` | 正式时间线前的 count-in 拍数 |

由 `config_parser.args_to_config()` 从 CLI 参数构建。`env_to_config()` 当前返回 `None`，因此 CLI 最终仍以命令行参数为准。
