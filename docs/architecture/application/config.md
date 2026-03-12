---
title: application/config — 配置数据模型
description: TempoConfig、InputConfig、OutputConfig、InferenceConfig、ApplicationConfig 字段说明
---

# application/config — 配置数据模型

**源文件**：`src/streammuse/application/config/models.py`

所有配置对象均为 `frozen=True` 的 dataclass，在服务启动后不可修改。顶层对象为 `ApplicationConfig`，由四个子配置组合而成。

---

## `TempoConfig`

速度与节拍配置。

```python
@dataclass(frozen=True)
class TempoConfig:
    bpm: float = 120.0
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `bpm` | `float` | `120.0` | 每分钟拍数（Beats Per Minute） |
| `ticks_per_beat` | `int` | `4` | 每拍的 tick 数（1 tick = 1/4 拍） |
| `beats_per_bar` | `int` | `4` | 每小节拍数，决定小节边界计算 |

---

## `InputConfig`

输入源配置。

```python
@dataclass(frozen=True)
class InputConfig:
    type: InputType = "midi_device"
    midi_device_name: Optional[str] = None
    midi_file_path: Optional[str] = None
    midi_file_delay_ticks: int = 0
```

其中 `InputType = Literal["midi_device", "keyboard", "midi_file", "list"]`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `InputType` | `"midi_device"` | 输入源类型 |
| `midi_device_name` | `Optional[str]` | `None` | MIDI 设备名称；`None` 时自动选择第一个可用设备 |
| `midi_file_path` | `Optional[str]` | `None` | MIDI 文件路径（仅 `midi_file` 模式使用） |
| `midi_file_delay_ticks` | `int` | `0` | MIDI 文件开始前的延迟 ticks |

---

## `OutputConfig`

输出 Sink 配置。

```python
@dataclass(frozen=True)
class OutputConfig:
    type: OutputType = "console"
    midi_out_port: Optional[str] = None
    midi_file_output_path: Optional[str] = None
```

其中 `OutputType = Literal["audio", "midi_file", "console", "websocket", "composite", "json_log", "session"]`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `OutputType` | `"console"` | 输出类型 |
| `midi_out_port` | `Optional[str]` | `None` | MIDI 输出端口名称（`audio` 模式使用） |
| `midi_file_output_path` | `Optional[str]` | `None` | MIDI 文件保存路径（`midi_file` 模式使用） |

---

## `InferenceConfig`

推理引擎配置。

```python
@dataclass(frozen=True)
class InferenceConfig:
    type: InferenceType = "http"
    server_generate_url: str = "http://localhost:8000/generate_accompaniment"
    timeout_s: float = 30.0
    checkpoint_path: Optional[str] = None
    model_size: str = "0.12B"
    model_max_seq_len_frames: int = 96
    generation_length_frames: int = 20
    generation_interval_ticks: int = 2
```

其中 `InferenceType = Literal["http", "stanley"]`

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `InferenceType` | `"http"` | 推理引擎类型：`http`（远端服务器）或 `stanley`（本地模型） |
| `server_generate_url` | `str` | `"http://localhost:8000/generate_accompaniment"` | HTTP 推理服务器的 generate 端点 URL |
| `timeout_s` | `float` | `30.0` | HTTP 请求超时时间（秒） |
| `checkpoint_path` | `Optional[str]` | `None` | 模型 checkpoint 路径（仅 `stanley` 类型使用） |
| `model_size` | `str` | `"0.12B"` | 模型规模标识（仅 `stanley` 类型使用） |
| `model_max_seq_len_frames` | `int` | `96` | 模型最大序列长度（帧数），即 context window |
| `generation_length_frames` | `int` | `20` | 每次推理生成的帧数（20 帧 = 10 ticks = 2.5 拍） |
| `generation_interval_ticks` | `int` | `2` | 触发推理的时间间隔（ticks），默认每 2 ticks = 每半拍触发一次 |

---

## `ApplicationConfig`

顶层配置对象，组合四个子配置。

```python
@dataclass(frozen=True)
class ApplicationConfig:
    tempo: TempoConfig = TempoConfig()
    input: InputConfig = InputConfig()
    output: OutputConfig = OutputConfig()
    inference: InferenceConfig = InferenceConfig()
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `tempo` | `TempoConfig` | 速度配置 |
| `input` | `InputConfig` | 输入源配置 |
| `output` | `OutputConfig` | 输出 Sink 配置 |
| `inference` | `InferenceConfig` | 推理引擎配置 |

由 `config_parser.args_to_config()` 从 CLI 参数构建，或由 `env_to_config()` 从环境变量构建。
