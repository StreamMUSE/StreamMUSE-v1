---
title: 配置项
description: StreamMUSE CLI 所有参数说明
---

# 配置项

所有配置通过命令行参数传入，最终汇聚为 `ApplicationConfig` 对象。下文按功能分组列出当前 CLI 支持的参数。

---

## 节奏配置（Tempo）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tempo` | `float` | `120.0` | BPM（每分钟节拍数） |
| `--ticks-per-beat` | `int` | `4` | 每拍 tick 数（时间分辨率） |
| `--beats-per-bar` | `int` | `4` | 每小节拍数 |

默认配置下 1 tick = 1/4 拍，BPM=120 时 1 tick = 0.125 秒。

---

## 输入配置（Input）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--input-mode` | `str` | `midi_device` | `midi_device` `keyboard` `midi_file` `list` | 输入源类型 |
| `--midi-device-name` | `str` | `None` | — | MIDI 输入设备名称；为 `None` 时自动选第一个可用设备 |
| `--midi-file-path` | `str` | `None` | — | MIDI 文件路径（`midi_file` 模式必填） |
| `--midi-file-delay-ticks` | `int` | `0` | — | MIDI 文件开始前的延迟 tick 数 |
| `--injection-file` | `str` | `None` | — | 会话开始前注入的旋律 MIDI 文件；仅支持 `--input-mode midi_file` |
| `--injection-length` | `int` | `0` | — | 注入前多少 ticks 的历史；使用 injection 时必须 > 0 |
| `--inject-acc-file` | `str` | `None` | — | 可选 accompaniment MIDI 文件；缺省时由 `/mel/` 替换为 `/acc/` 推导 |

使用 `--injection-file` 时，CLI 会先向 HTTP server 清空并注入历史，再创建 `MidiFileInput`。随后正式输入会从 `injection_length` 之后开始播放，避免重复发送已注入片段。

---

## 输出配置（Output）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--output-type` | `str` | `console` | `console` `audio` `midi_file` `websocket` `composite` `json_log` `session` | 输出类型，详见[输出类型](../user-guide/output-types.md) |
| `--midi-out-port` | `str` | `None` | — | MIDI 输出端口名称（`audio` 模式或 metronome 缺省端口使用） |
| `--midi-file-output-path` | `str` | `None` | — | MIDI 文件录制路径（`midi_file` 模式必填） |
| `--enable-metronome` | flag | `False` | — | 启用实时 MIDI click；若有 MIDI 录制也会记录 Metronome 轨 |
| `--metronome-port` | `str` | `None` | — | metronome 专用 MIDI 输出端口；缺省时复用 `--midi-out-port` 或系统默认输出 |
| `--metronome-channel` | `int` | `9` | — | metronome MIDI channel，默认 9（General MIDI percussion channel 10） |
| `--log-dir` | `str` | `logs` | — | Session 日志根目录 |
| `--inference-log-detail` | `str` | `summary` | `summary` `full` | 推理日志粒度（full 会显著增大 `inferences.json` 体积） |
| `--enable-performance-tracking` | flag | `False` | — | 预留参数，当前版本未接入额外逻辑 |

除 `midi_file` 外，CLI 会创建 session 目录。`console` / `audio` / `websocket` 会自动附加 `combined.mid` 录制；`json_log` 仍只写 JSON。

---

## 推理配置（Inference）

| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|---|---|---|---|---|
| `--inference-type` | `str` | `http` | `http` `stanley` | 推理引擎类型 |
| `--server-url` | `str` | `http://localhost:8000/generate_accompaniment` | — | HTTP 推理服务器地址 |
| `--timeout-s` | `float` | `30.0` | — | HTTP 请求超时时间（秒） |
| `--model-name` | `str` | `stanley` | `stanley` `lekai` | HTTP server 端选择的模型 |
| `--inference-mode` | `str` | `sliding_window` | `sliding_window` `stateful` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `str` | `None` | — | 模型 checkpoint 路径（`stanley` 模式必填，也可透传给 HTTP server） |
| `--model-size` | `str` | `0.12B` | — | Stanley 模型大小 |
| `--model-max-seq-len-frames` | `int` | `96` | — | 模型最大上下文长度（帧数） |
| `--generation-length-frames` | `int` | `20` | — | 每次推理生成的帧数 |
| `--generation-interval-ticks` | `int` | `2` | — | 透传给 HTTP server 和日志的间隔参数；当前客户端 tick loop 不用它决定触发时刻 |

当前客户端推理触发点是 tick=0 和每拍末尾（默认 tick=3、7、11...），不是每 `generation_interval_ticks` ticks 一次。

---

## 运行时选项

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--count-in-beats` | `int` | `0` | 正式输入和推理前空转的拍数；负数会被 clamp 到 0 |
| `--max-ticks` | `int` | `None` | 最大运行 tick 数（主要用于测试，不设则持续运行） |

`count-in` 阶段只输出 metronome，不读取正式输入，也不发送推理请求。若启用了 metronome MIDI 录制，count-in click 会被写入 MIDI 文件开头。

---

## 配置组合示例

**键盘输入 + 控制台输出：**

```bash
uv run streammuse-cli --input-mode keyboard --output-type console
```

**MIDI 文件 + Lekai HTTP + metronome + 4 拍 count-in：**

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --inference-type http \
  --model-name lekai \
  --server-url http://127.0.0.1:8001/generate_accompaniment \
  --generation-length-frames 4 \
  --generation-interval-ticks 4 \
  --output-type console \
  --enable-metronome \
  --count-in-beats 4 \
  --max-ticks 256
```

**MIDI 文件注入前 16 ticks 历史：**

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --injection-file prompts/inputs_lekai/mel/1.mid \
  --injection-length 16 \
  --inference-type http \
  --model-name lekai
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
├── InputConfig          ← --input-mode, --midi-file-path, --injection-file, ...
├── OutputConfig         ← --output-type, --midi-out-port, --enable-metronome, ...
├── InferenceConfig      ← --inference-type, --server-url, --generation-length-frames, ...
└── count_in_beats       ← --count-in-beats
```

参见 [Application 配置文档](../architecture/application/config.md) 了解每个配置类的字段详情。
