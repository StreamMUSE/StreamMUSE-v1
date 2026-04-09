---
title: CLI 参数完整参考
description: streammuse-cli 所有命令行参数的完整说明
---

# CLI 参数完整参考

本文基于当前实现（`src/streammuse/presentation/cli/config_parser.py`）整理。

## 用法

```
uv run streammuse-cli [参数...]
```

---

## 节奏参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tempo` | `float` | `120.0` | 每分钟拍数（BPM） |
| `--ticks-per-beat` | `int` | `4` | 每拍 tick 数（1 tick = 1/4 拍） |
| `--beats-per-bar` | `int` | `4` | 每小节拍数 |

---

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--input-mode` | `str` | `"midi_device"` | 输入模式：`midi_device`、`keyboard`、`midi_file`、`list` |
| `--midi-device-name` | `str` | `None` | MIDI 输入设备名称（`midi_device` 模式） |
| `--midi-file-path` | `str` | `None` | MIDI 文件路径（`midi_file` 模式必填） |
| `--midi-file-delay-ticks` | `int` | `0` | MIDI 文件开始前的延迟 ticks |

---

## 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--output-type` | `str` | `"console"` | 输出类型：`console`、`audio`、`midi_file`、`websocket`、`json_log`、`session`、`composite`。`console/audio/websocket/session/composite` 会生成 `combined.mid` |
| `--midi-out-port` | `str` | `None` | MIDI 输出端口名称（`audio` 模式） |
| `--midi-file-output-path` | `str` | `None` | MIDI 录制输出路径（`midi_file` 模式必填） |
| `--log-dir` | `str` | `"logs"` | session 根目录。除 `midi_file` 外都会创建 `logs/YYYY-MM-DD/session_HHMMSS/` |
| `--inference-log-detail` | `str` | `"summary"` | 推理日志粒度：`summary`（摘要）或 `full`（完整 request/response，体积更大） |
| `--enable-performance-tracking` | flag | `False` | 预留参数；当前版本未接入额外逻辑 |

### 输出类型与 MIDI 产物

| output_type | 是否写 `combined.mid` | 说明 |
|---|---|---|
| `console` | 是（自动附加） | 控制台 + 自动 MIDI |
| `audio` | 是（自动附加） | 实时播放 + 自动 MIDI |
| `websocket` | 是（自动附加） | WebSocket + 自动 MIDI |
| `session` | 是（原生） | SessionLogger 本身包含 MIDI |
| `composite` | 是（原生） | Composite 中包含 SessionLogger |
| `json_log` | 否 | 纯 JSON 日志 |
| `midi_file` | 否（不写 `combined.mid`） | 仅写 `--midi-file-output-path` |

---

## 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--server-url` | `str` | `"http://localhost:8000/generate_accompaniment"` | HTTP 推理服务器 URL |
| `--timeout-s` | `float` | `30.0` | HTTP 请求超时（秒） |
| `--inference-type` | `str` | `"http"` | 推理类型：`http`（远端服务器，主路径）、`stanley`（本地模型） |
| `--model-name` | `str` | `"stanley"` | HTTP server 端模型选择：`stanley` 或 `lekai` |
| `--inference-mode` | `str` | `"sliding_window"` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `str` | `None` | 模型 checkpoint 路径（`stanley` 模式必填） |
| `--model-size` | `str` | `"0.12B"` | 模型规模（`stanley` 模式） |
| `--model-max-seq-len-frames` | `int` | `96` | 模型 context window 帧数 |
| `--generation-length-frames` | `int` | `20` | 每次推理生成的帧数 |
| `--generation-interval-ticks` | `int` | `2` | 推理触发间隔（ticks） |

### Lekai 模型参数约束

使用 `--model-name lekai` 时，`--generation-length-frames` 必须是 **4 的倍数**（lekai 模型以 4 个 timestep 为一拍进行 tokenization）。
`--generation-interval-ticks` 仅控制客户端触发频率，不要求是 4 的倍数。

| 参数 | 约束 | 推荐值 |
|---|---|---|
| `--generation-length-frames` | 必须是 4 的倍数 | `20` |

该约束由 Lekai HTTP server 校验；不满足时会返回 HTTP 422 错误。

---

## 注入能力说明

当前 CLI **没有** `--injection-file` / `--injection-length` 参数。

如果需要注入历史，请使用 HTTP API（`/inject_notes`、`/injection_status`、`/clear_history`），详见用户文档：[music-injection](../user-guide/music-injection.md)。

### Lekai Server Runtime 环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LEKAI_CHECKPOINT_PATH` | `None` | Lekai checkpoint 路径；为空时使用 rule stub |
| `LEKAI_DEVICE` | `auto` | 设备策略：`auto` / `mps` / `cpu` / `cuda` |
| `LEKAI_DTYPE` | `auto` | 精度策略：`auto` / `float32` / `float16` |
| `LEKAI_ENABLE_MPS_FALLBACK` | `true` | `mps` 失败时是否自动回退 CPU |
| `LEKAI_USE_CACHE` | `true` | 生成时是否使用 KV cache |
| `LEKAI_WARMUP_STEPS` | `1` | server 启动后 warmup 步数 |
| `LEKAI_MAX_GENERATION_LENGTH_FRAMES` | `None` | 可选上限，限制单次生成长度 |
| `LEKAI_MAX_PROMPT_TICKS` | `None` | 可选上限，限制 prompt 长度 |
| `LEKAI_SERVER_HOST` | `0.0.0.0` | server 监听 host |
| `LEKAI_SERVER_PORT` | `8000` | server 监听端口 |

---

## 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--max-ticks` | `int` | `None` | 最大运行 tick 数（`None` 表示不限，持续运行直到 Ctrl+C） |

---

## 典型命令示例

```bash
# 最简启动（键盘 + 控制台）
uv run streammuse-cli --input-mode keyboard

# MIDI 设备 + 实时播放
uv run streammuse-cli --input-mode midi_device --output-type audio

# MIDI 文件 + 完整日志
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/C_major/pop909_216_mel.mid \
    --output-type composite \
    --log-dir logs/test \
    --inference-log-detail full

# HTTP 模式 + Lekai server
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type http \
    --model-name lekai \
    --inference-mode sliding_window \
    --generation-interval-ticks 4 \
    --generation-length-frames 20 \
    --server-url http://localhost:8000/generate_accompaniment

# 本地 Stanley 模型（不需要服务器）
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type stanley \
    --checkpoint-path checkpoints/model.ckpt

# 自定义速度参数
uv run streammuse-cli \
    --input-mode keyboard \
    --tempo 90 \
    --generation-interval-ticks 4 \
    --generation-length-frames 16
```
