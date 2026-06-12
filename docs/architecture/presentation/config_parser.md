---
title: presentation/config_parser — CLI 参数解析
description: parse_args()、args_to_config()、env_to_config() 的完整参数说明
---

# presentation/config_parser — CLI 参数解析

**源文件**：`src/streammuse/presentation/cli/config_parser.py`

该模块负责把命令行参数转换为 `ApplicationConfig`。

---

## `parse_args() -> argparse.Namespace`

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
| `--midi-device-name` | `None` | MIDI 设备名称 |
| `--midi-file-path` | `None` | MIDI 文件路径 |
| `--midi-file-delay-ticks` | `0` | MIDI 文件开始前的延迟 ticks |
| `--injection-file` | `None` | 注入历史用 melody MIDI 文件 |
| `--injection-length` | `0` | 注入历史长度 ticks |
| `--inject-acc-file` | `None` | 可选 accompaniment MIDI 文件 |

### 输出参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output-type` | `"console"` | 输出类型：`console`、`audio`、`midi_file`、`websocket`、`json_log`、`session`、`composite` |
| `--midi-out-port` | `None` | MIDI 输出端口名称 |
| `--midi-file-output-path` | `None` | MIDI 文件输出路径 |
| `--enable-metronome` | `False` | 是否启用 MIDI click |
| `--metronome-port` | `None` | metronome 专用 MIDI 输出端口 |
| `--metronome-channel` | `9` | metronome MIDI channel |
| `--log-dir` | `"logs"` | 日志目录 |
| `--inference-log-detail` | `"summary"` | 推理日志粒度：`summary` / `full` |
| `--enable-performance-tracking` | `False` | 预留参数 |

### 推理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server-url` | `"http://localhost:8000/generate_accompaniment"` | 推理服务器 URL |
| `--timeout-s` | `30.0` | HTTP 请求超时 |
| `--inference-type` | `"http"` | 推理类型：`http`、`stanley` |
| `--model-name` | `"stanley"` | HTTP server 端模型：`stanley` / `lekai` |
| `--inference-mode` | `"sliding_window"` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `None` | 模型 checkpoint 路径 |
| `--model-size` | `"0.12B"` | Stanley 模型规模 |
| `--model-max-seq-len-frames` | `96` | 模型 context window 帧数 |
| `--generation-length-frames` | `20` | 每次推理生成帧数 |
| `--generation-interval-ticks` | `2` | 透传给 server 和日志的间隔参数 |

### 运行时参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--count-in-beats` | `0` | 正式输入和推理前空转的拍数 |
| `--max-ticks` | `None` | 最大运行 ticks |

---

## `args_to_config(args) -> ApplicationConfig`

核心映射：

```python
return ApplicationConfig(
    tempo=TempoConfig(...),
    input=InputConfig(
        type=args.input_mode,
        midi_file_path=args.midi_file_path,
        injection_file=getattr(args, "injection_file", None),
        injection_length_ticks=int(getattr(args, "injection_length", 0) or 0),
        injection_acc_file=getattr(args, "inject_acc_file", None),
    ),
    output=OutputConfig(
        type=args.output_type,
        inference_log_detail=getattr(args, "inference_log_detail", "summary"),
        metronome_enabled=bool(getattr(args, "enable_metronome", False)),
        metronome_port=getattr(args, "metronome_port", None),
        metronome_channel=int(getattr(args, "metronome_channel", 9)),
    ),
    inference=InferenceConfig(...),
    count_in_beats=max(0, int(getattr(args, "count_in_beats", 0) or 0)),
)
```

注意：`count_in_beats` 在解析到配置时会 clamp 到非负数；`RealTimeMusicService` 内部也会拒绝负值。

---

## `env_to_config() -> Optional[ApplicationConfig]`

当前实现直接返回 `None`：

```python
def env_to_config() -> Optional[ApplicationConfig]:
    return None
```

CLI 侧不读取环境变量构建运行配置。`LEKAI_*` 等环境变量属于 server 侧配置，请看 [运行实时系统](../../user-guide/running-realtime.md)。
