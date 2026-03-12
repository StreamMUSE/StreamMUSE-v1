---
title: presentation/config_parser — CLI 参数解析
description: parse_args()、args_to_config()、env_to_config() 的完整参数说明
---

# presentation/config_parser — CLI 参数解析

**源文件**：`src/streammuse/presentation/cli/config_parser.py`

提供三个函数将 CLI 参数和环境变量转换为 `ApplicationConfig`。

---

## `parse_args() -> argparse.Namespace`

解析所有 CLI 参数，返回 `argparse.Namespace`。

### 速度参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--bpm` | `120.0` | 每分钟拍数 |
| `--ticks-per-beat` | `4` | 每拍 tick 数 |
| `--beats-per-bar` | `4` | 每小节拍数 |

### 输入参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-mode` | `"keyboard"` | 输入类型：`keyboard`、`midi`（= midi_device）、`midi_file`、`list` |
| `--midi-device-name` | `None` | MIDI 设备名称（`midi` 模式） |
| `--midi-file` | `None` | MIDI 文件路径（`midi_file` 模式） |
| `--midi-file-delay-ticks` | `0` | MIDI 文件开始前的延迟 ticks |

### 输出参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-type` | `"console"` | 输出类型：`console`、`audio`、`midi_file`、`websocket`、`json_log`、`session`、`composite` |
| `--midi-out-port` | `None` | MIDI 输出端口名称（`audio` 模式） |
| `--midi-file-output` | `None` | MIDI 文件输出路径（`midi_file` 模式） |
| `--log-dir` | `None` | 日志目录（`json_log`/`session`/`composite` 模式必须指定） |

### 推理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server-url` | `"http://localhost:8000/generate_accompaniment"` | 推理服务器 URL |
| `--inference-timeout` | `30.0` | HTTP 请求超时（秒） |
| `--inference-type` | `"http"` | 推理类型：`http`、`stanley` |
| `--checkpoint-path` | `None` | 模型 checkpoint 路径（`stanley` 模式） |
| `--model-size` | `"0.12B"` | 模型规模（`stanley` 模式） |
| `--model-max-seq-len-frames` | `96` | 模型 context window 帧数 |
| `--generation-length-frames` | `20` | 每次推理生成帧数 |
| `--generation-interval-ticks` | `2` | 推理触发间隔 ticks |

### 其他参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-ticks` | `None` | 最大运行 ticks（`None` 表示不限） |
| `--injection-file` | `None` | 预置历史注入的 MIDI 文件路径 |
| `--injection-length` | `None` | 注入长度（ticks） |

---

## `args_to_config(args) -> ApplicationConfig`

将 `parse_args()` 返回的 Namespace 转换为 `ApplicationConfig`：

```python
def args_to_config(args: argparse.Namespace) -> ApplicationConfig:
    return ApplicationConfig(
        tempo=TempoConfig(bpm=args.bpm, ...),
        input=InputConfig(type=_map_input_mode(args.input_mode), ...),
        output=OutputConfig(type=args.output_type, ...),
        inference=InferenceConfig(type=args.inference_type, ...),
    )
```

`--input-mode midi` 被映射为 `InputConfig.type = "midi_device"`（CLI 使用更简短的别名）。

---

## `env_to_config() -> Optional[ApplicationConfig]`

从环境变量读取配置。当前实现返回 `None`（环境变量支持尚未完整实现）。

若返回非 `None`，则在 `main()` 中优先使用环境变量配置（`env_to_config() or args_to_config(args)`）。

相关环境变量（推理服务器侧使用）：
- `CHECKPOINT_PATH`：模型 checkpoint 路径
- `MODEL_MAX_SEQ_LEN_FRAMES`：context window 帧数
- `GENERATION_LENGTH_FRAMES`：生成帧数
- `MODEL_SIZE`：模型规模
