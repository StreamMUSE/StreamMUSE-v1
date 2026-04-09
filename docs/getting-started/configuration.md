---
title: 配置项
description: StreamMUSE CLI 所有参数说明
---

# 配置项

所有配置通过命令行参数传入，最终汇聚为 `ApplicationConfig` 对象。下文按功能分组列出所有参数。

---

## 节奏配置（Tempo）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tempo` | `float` | `120.0` | BPM（每分钟节拍数） |
| `--ticks-per-beat` | `int` | `4` | 每拍 tick 数（时间分辨率） |
| `--beats-per-bar` | `int` | `4` | 每小节拍数 |

> 1 tick = 1 / `ticks-per-beat` 拍。默认配置下 1 tick = 1/4 拍，BPM=120 时 1 tick ≈ 125ms。

---

## 输入配置（Input）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--input-mode` | `str` | `midi_device` | `midi_device` `keyboard` `midi_file` `list` | 输入源类型 |
| `--midi-device-name` | `str` | `None` | — | MIDI 输入设备名称；为 `None` 时自动选第一个可用设备 |
| `--midi-file-path` | `str` | `None` | — | MIDI 文件路径（`midi_file` 模式必填） |
| `--midi-file-delay-ticks` | `int` | `0` | — | MIDI 文件开始前的延迟 tick 数 |

---

## 输出配置（Output）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--output-type` | `str` | `console` | `console` `audio` `midi_file` `websocket` `composite` `json_log` `session` | 输出类型，详见[输出类型](../user-guide/output-types.md) |
| `--midi-out-port` | `str` | `None` | — | MIDI 输出端口名称（`audio` 模式可选） |
| `--midi-file-output-path` | `str` | `None` | — | MIDI 文件录制路径（`midi_file` 模式必填） |
| `--log-dir` | `str` | `logs` | — | Session 日志根目录（`composite`/`json_log`/`session` 模式使用） |
| `--inference-log-detail` | `str` | `summary` | `summary` `full` | 推理日志粒度（full 会增大日志体积） |
| `--enable-performance-tracking` | flag | `False` | — | 预留参数，当前版本未接入额外逻辑 |

---

## 推理配置（Inference）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--inference-type` | `str` | `http` | `http` `stanley` | 推理引擎类型 |
| `--server-url` | `str` | `http://localhost:8000/generate_accompaniment` | — | HTTP 推理服务器地址 |
| `--timeout-s` | `float` | `30.0` | — | HTTP 请求超时时间（秒） |
| `--model-name` | `str` | `stanley` | `stanley` `lekai` | HTTP server 端选择的模型 |
| `--inference-mode` | `str` | `sliding_window` | `sliding_window` `stateful` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `str` | `None` | — | 模型 checkpoint 路径（`stanley` 模式必填） |
| `--model-size` | `str` | `0.12B` | — | Stanley 模型大小 |
| `--model-max-seq-len-frames` | `int` | `96` | — | 模型最大上下文长度（帧数） |
| `--generation-length-frames` | `int` | `20` | — | 每次推理生成的帧数 |
| `--generation-interval-ticks` | `int` | `2` | — | 两次推理之间的间隔 tick 数 |

> **注意**：使用 `--model-name lekai` 时，`--generation-length-frames` 必须是 4 的倍数；`--generation-interval-ticks` 不受该约束。详见 [CLI 参考](../reference/cli-reference.md#lekai-模型参数约束)。

---

## 运行时选项

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--max-ticks` | `int` | `None` | 最大运行 tick 数（主要用于测试，不设则持续运行）|

---

## 配置组合示例

**键盘输入 + 控制台输出（开发调试）：**
```bash
uv run streammuse-cli --input-mode keyboard --output-type console
```

**MIDI 设备输入 + 实时音频输出：**
```bash
uv run streammuse-cli --input-mode midi_device --output-type audio
```

**MIDI 文件模拟 + Session 日志录制：**
```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/C_major/pop909_216_mel.mid \
  --output-type composite \
  --log-dir logs \
  --max-ticks 200
```

**本地 Stanley 引擎（无需服务器）：**
```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --inference-type stanley \
  --checkpoint-path path/to/model.ckpt
```

---

## 配置映射关系

所有参数最终通过 `config_parser.args_to_config()` 映射为嵌套的 `ApplicationConfig`：

```
ApplicationConfig
├── TempoConfig          ← --tempo, --ticks-per-beat, --beats-per-bar
├── InputConfig          ← --input-mode, --midi-device-name, --midi-file-path, ...
├── OutputConfig         ← --output-type, --midi-out-port, --midi-file-output-path
└── InferenceConfig      ← --inference-type, --server-url, --generation-length-frames, ...
```

参见 [Application 配置文档](../architecture/application/config.md) 了解每个配置类的字段详情。
