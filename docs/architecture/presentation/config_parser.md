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

### 节奏参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--tempo` | `120.0` | 每分钟拍数 |
| `--ticks-per-beat` | `4` | 每拍 tick 数 |
| `--beats-per-bar` | `4` | 每小节拍数 |

### 输入参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input-mode` | `"midi_device"` | 输入类型：`midi_device`、`keyboard`、`midi_file`、`list` |
| `--midi-device-name` | `None` | MIDI 设备名称（`midi_device` 模式） |
| `--midi-file-path` | `None` | MIDI 文件路径（`midi_file` 模式） |
| `--midi-file-delay-ticks` | `0` | MIDI 文件开始前的延迟 ticks |

### 输出参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-type` | `"console"` | 输出类型：`console`、`audio`、`midi_file`、`websocket`、`json_log`、`session`、`composite` |
| `--midi-out-port` | `None` | MIDI 输出端口名称（`audio` 模式） |
| `--midi-file-output-path` | `None` | MIDI 文件输出路径（`midi_file` 模式） |
| `--log-dir` | `"logs"` | 日志目录 |
| `--inference-log-detail` | `"summary"` | 推理日志粒度：`summary` / `full` |
| `--enable-performance-tracking` | `False` | 预留参数（当前未接入额外逻辑） |

### 推理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server-url` | `"http://localhost:8000/generate_accompaniment"` | 推理服务器 URL |
| `--timeout-s` | `30.0` | HTTP 请求超时（秒） |
| `--inference-type` | `"http"` | 推理类型：`http`、`stanley` |
| `--model-name` | `"stanley"` | HTTP server 端模型：`stanley` / `lekai` |
| `--inference-mode` | `"sliding_window"` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `None` | 模型 checkpoint 路径（`stanley` 模式） |
| `--model-size` | `"0.12B"` | 模型规模（`stanley` 模式） |
| `--model-max-seq-len-frames` | `96` | 模型 context window 帧数 |
| `--generation-length-frames` | `20` | 每次推理生成帧数 |
| `--generation-interval-ticks` | `2` | 推理触发间隔 ticks |

### 其他参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-ticks` | `None` | 最大运行 ticks（`None` 表示不限） |

说明：当前 CLI 未暴露 `--injection-file` / `--injection-length`。

---

## `args_to_config(args) -> ApplicationConfig`

将 `parse_args()` 返回的 Namespace 转换为 `ApplicationConfig`：

```python
def args_to_config(args: argparse.Namespace) -> ApplicationConfig:
    return ApplicationConfig(
        tempo=TempoConfig(bpm=args.tempo, ...),
        input=InputConfig(type=args.input_mode, ...),
        output=OutputConfig(type=args.output_type, ...),
        inference=InferenceConfig(type=args.inference_type, ...),
    )
```

`OutputConfig` 会接收：
1. `midi_out_port`
2. `midi_file_output_path`
3. `inference_log_detail`

`InferenceConfig` 会接收：
1. `server_generate_url`
2. `timeout_s`
3. `model_name`
4. `inference_mode`
5. `generation_length_frames`
6. `generation_interval_ticks`

---

## `env_to_config() -> Optional[ApplicationConfig]`

从环境变量读取配置。当前实现直接返回 `None`。

在 `main()` 中，当前逻辑仍会调用 `args_to_config(args)` 构建最终配置。

相关环境变量（推理服务器侧）请参考：
1. `docs/user-guide/running-realtime.md`
2. `docs/reference/cli-reference.md`
