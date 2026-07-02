---
title: CLI 参数完整参考
description: streammuse-cli 所有命令行参数的完整说明
---

# CLI 参数完整参考

本文基于当前实现（`src/streammuse/presentation/cli/config_parser.py`）整理。

## 用法

```bash
uv run streammuse-cli [参数...]
```

---

## 节奏参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--tempo` | `float` | `120.0` | 每分钟拍数（BPM） |
| `--ticks-per-beat` | `int` | `4` | 每拍 tick 数 |
| `--beats-per-bar` | `int` | `4` | 每小节拍数 |

---

## 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--input-mode` | `str` | `midi_device` | 输入模式：`midi_device`、`keyboard`、`midi_file`、`list` |
| `--midi-device-name` | `str` | `None` | MIDI 输入设备名称（`midi_device` 模式） |
| `--midi-file-path` | `str` | `None` | MIDI 文件路径（`midi_file` 模式必填） |
| `--midi-file-delay-ticks` | `int` | `0` | MIDI 文件开始前的延迟 ticks |
| `--injection-file` | `str` | `None` | 要注入为 prompt history 的旋律 MIDI 文件 |
| `--injection-length` | `int` | `0` | 注入前多少 ticks；使用 injection 时必须 > 0 |
| `--inject-acc-file` | `str` | `None` | 可选伴奏 MIDI 文件；缺省时尝试由 `/mel/` 替换为 `/acc/` 推导 |

Injection 当前仅支持 `--input-mode midi_file`。CLI 会先调用 server 的 `/clear_history` 和 `/inject_notes`，然后把 `MidiFileInput` 的起点设置为 `injection_length`，避免重复播放已注入片段。

---

## 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--output-type` | `str` | `console` | 输出类型：`console`、`audio`、`midi_file`、`websocket`、`json_log`、`session`、`composite` |
| `--midi-out-port` | `str` | `None` | MIDI 输出端口名称（`audio` 模式，也可作为 metronome 缺省端口） |
| `--midi-file-output-path` | `str` | `None` | MIDI 录制输出路径（`midi_file` 模式必填） |
| `--enable-metronome` | flag | `False` | 启用实时 MIDI click；若有 MIDI 录制则写入 Metronome 轨 |
| `--metronome-port` | `str` | `None` | metronome 专用 MIDI 输出端口 |
| `--metronome-channel` | `int` | `9` | metronome MIDI channel，默认 percussion channel |
| `--log-dir` | `str` | `logs` | session 根目录；除 `midi_file` 外都会创建日期/session 子目录 |
| `--inference-log-detail` | `str` | `summary` | 推理日志粒度：`summary` 或 `full` |
| `--session-artifact-tier` | `str` | `debug` | session artifact 档位：`debug` 保留完整日志，`normal` 只保留核心 MIDI/trace |
| `--enable-performance-tracking` | flag | `False` | 预留参数；当前版本未接入额外逻辑 |

退出行为说明（除 `midi_file` 外）：

1. CLI 结束时会调用一次 `InferenceEngine.clear_history()`。
2. `--session-artifact-tier debug` 时，若输出目录可用，会将返回的历史落盘为 `melody_history.json` 与 `accompaniment_history.json`。
3. `session/composite` 在 debug 档会生成 `performance.json` 与 `statistics.csv`；normal 档跳过完整 JSON 诊断文件。

### 输出类型与 MIDI 产物

| output_type | 是否写 `combined.mid` | 说明 |
|---|---|---|
| `console` | 是（自动附加） | 控制台 + 自动 MIDI |
| `audio` | 是（自动附加） | 实时播放 + 自动 MIDI |
| `websocket` | 是（自动附加） | WebSocket + 自动 MIDI |
| `session` | 是（原生） | SessionLogger 本身包含 MIDI |
| `composite` | 是（原生） | Console + SessionLogger |
| `json_log` | 否 | 纯 JSON 日志 |
| `midi_file` | 否（不写 `combined.mid`） | 仅写 `--midi-file-output-path` |

开启 `--enable-metronome` 后，所有写 MIDI 的模式都会额外写一个 `Metronome` 鼓轨。若同时设置 `--count-in-beats`，count-in click 也会被记录在 MIDI 文件开头。

`session/composite` 还会写 `model_schedule_trace.jsonl`；如果 trace 中有模型事件，关闭时会从 logical tick 重建 `theoretical_model.mid`。`combined.mid` 始终表示 actual scheduled playback。

---

## 推理参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--server-url` | `str` | `http://localhost:8000/generate_accompaniment` | HTTP 推理服务器 URL |
| `--timeout-s` | `float` | `30.0` | HTTP 请求超时（秒） |
| `--inference-type` | `str` | `http` | 推理类型：`http`（远端服务器）、`stanley`（本地模型） |
| `--model-name` | `str` | `stanley` | HTTP server 端模型选择：`stanley` 或 `lekai` |
| `--inference-mode` | `str` | `sliding_window` | 透传给 HTTP server 的推理模式提示 |
| `--checkpoint-path` | `str` | `None` | 模型 checkpoint 路径（`stanley` 模式必填，也可透传给 HTTP server） |
| `--model-size` | `str` | `0.12B` | 模型规模（`stanley` 模式） |
| `--model-max-seq-len-frames` | `int` | `96` | 模型 context window 帧数 |
| `--generation-length-frames` | `int` | `20` | 每次推理生成的帧数 |
| `--generation-interval-ticks` | `int` | `2` | 透传给 HTTP server 和日志的间隔参数 |

### 推理触发语义

当前 `RealTimeMusicService` 不再按 `generation_interval_ticks` 驱动 tick-loop 触发。实际触发点是：

1. `tick=0`：若已有 melody history（例如 injection），发送完整历史请求。
2. 每拍末尾：默认 `ticks_per_beat=4` 时，在 tick=3、7、11... 发送下一拍请求，`generation_start_tick=tick+1`。

`generation_interval_ticks` 仍保留在 HTTP payload 和日志中，供 server 或实验记录使用。

---

## Lekai Server Runtime 环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LEKAI_CHECKPOINT_PATH` | `None` | Lekai checkpoint 路径；为空时取决于 server 端实现 |
| `LEKAI_DEVICE` | `auto` | 设备策略：`auto` / `mps` / `cpu` / `cuda` |
| `LEKAI_DTYPE` | `auto` | 精度策略：`auto` / `float32` / `float16` |
| `LEKAI_ENABLE_MPS_FALLBACK` | `true` | `mps` 失败时是否自动回退 CPU |
| `LEKAI_USE_CACHE` | `true` | 生成时是否使用 KV cache |
| `LEKAI_WARMUP_STEPS` | `1` | server 启动后 warmup 步数 |
| `LEKAI_MAX_GENERATION_LENGTH_FRAMES` | `None` | 可选上限，限制单次生成长度 |
| `LEKAI_MAX_PROMPT_TICKS` | `None` | 可选上限，限制 prompt 长度 |
| `LEKAI_SERVER_HOST` | `0.0.0.0` | server 监听 host |
| `LEKAI_SERVER_PORT` | `8000` | server 监听端口 |

这些是 server 侧环境变量；CLI 的 `env_to_config()` 当前仍返回 `None`，最终配置来自命令行参数。

---

## 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--count-in-beats` | `int` | `0` | 正式输入和推理前空转的拍数 |
| `--max-ticks` | `int` | `None` | 最大运行 tick 数（`None` 表示不限，直到 Ctrl+C） |

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

# HTTP 模式 + Lekai server + metronome + count-in
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --inference-type http \
    --model-name lekai \
    --inference-mode sliding_window \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --output-type console \
    --enable-metronome \
    --count-in-beats 4 \
    --max-ticks 256

# CLI injection
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --injection-file prompts/inputs_lekai/mel/1.mid \
    --injection-length 16 \
    --inference-type http \
    --model-name lekai

# 本地 Stanley 模型（不需要服务器）
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type stanley \
    --checkpoint-path checkpoints/model.ckpt
```
