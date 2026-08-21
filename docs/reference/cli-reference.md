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
| `--tempo` | `float` | `120.0` | playback wall-clock BPM；控制 tick 调度速度，不必等于模型条件 BPM |
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
| `--model-condition-bpm` | `int` | `None` | HTTP 模型 prompt 的 BPM 条件 token；与 `--tempo` 独立。未指定时为兼容旧调用而回退到 playback tempo |

需要跨 playback tempo 比较生成内容时，必须显式固定 `--model-condition-bpm`。例如鲁棒性实验使用 `--tempo 60 --model-condition-bpm 120`；只固定 server 的 `LEKAI_DEFAULT_BPM` 不够，因为客户端会把有效 model-condition BPM 放进请求。

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
| `LEKAI_DEFAULT_BPM` | `120` | 请求没有 BPM 时的 server fallback；冻结实验仍应由 CLI 显式传 `--model-condition-bpm` |
| `LEKAI_MAX_GENERATION_LENGTH_FRAMES` | `None` | 可选上限，限制单次生成长度 |
| `LEKAI_MAX_PROMPT_TICKS` | `None` | 可选上限，限制 prompt 长度 |
| `LEKAI_SERVER_HOST` | `0.0.0.0` | server 监听 host |
| `LEKAI_SERVER_PORT` | `8000` | server 监听端口 |
| `LEKAI_ENABLE_DEBUG_RESET` | `false` | 是否开放会清空状态并换 seed/epoch 的 `/debug/reset_session`；仅应在 dedicated `127.0.0.1` 实验 server 上启用 |

这些是 server 侧环境变量；CLI 的 `env_to_config()` 当前仍返回 `None`，最终配置来自命令行参数。

---

## 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--count-in-beats` | `int` | `0` | 正式输入和推理前空转的拍数 |
| `--max-ticks` | `int` | `None` | `--run-stop-tick` 的兼容别名；右端点 exclusive |
| `--analysis-end-tick` | `int` | `None` | clean-reference 分析窗口的 exclusive 右端点 |
| `--last-input-note-off-tick` | `int` | `None` | 输入最后 note-off，写入 validity 并可用于 tail 公式 |
| `--request-cutoff-tick` | `int` | `None` | 最后允许的 generation-start tick；inclusive 且须按拍对齐 |
| `--run-stop-tick` | `int` | `None` | tick/playback 循环的 exclusive 右端点 |
| `--tail-beats` | `int` | `None` | 用 `ceil(last-note-off/beat)+tail` 推导 run stop |
| `--drain-timeout-s` | `float` | `10.0` | request/in-flight/response graceful drain 的 hard timeout |

实验运行应同时显式给出 analysis、last note-off、request cutoff 和 run stop。达到 cutoff 后服务进入 draining，不再产生请求，但会继续消费 response；`validity.json` 的 content/operational 双 verdict 是运行是否可用的权威来源。

---

## 交互式任务语音参数

以下参数属于 `streammuse-task play`，与音乐实时入口 `streammuse-cli` 的输入参数相互独立：

`play` 当前支持 `zip_zap_zop` 和 `animal_naming`。Animal Naming 使用 91 项 exact whitelist，`--max-turns` 不能超过 whitelist size；`--start-number` 对该任务没有作用。

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--human-input` | `terminal/voice` | `terminal` | 人类玩家输入方式 |
| `--voice-model` | `str` | `tiny.en` | faster-whisper 模型标识符或本地目录 |
| `--voice-device` | `str` | `cpu` | CTranslate2 执行设备 |
| `--voice-compute-type` | `str` | `int8` | CTranslate2 计算类型 |
| `--microphone-device` | 名称或索引 | 系统默认 | PortAudio 输入设备 |
| `--voice-model-cache` | 路径 | `None` | Hugging Face 模型下载/缓存根目录 |
| `--voice-model-revision` | `str` | `None` | 固定模型版本的 commit 或 tag |
| `--voice-local-files-only` | flag | `False` | 禁止联网下载模型 |
| `--voice-save-audio` | flag | `False` | 将每回合发言保存到运行产物目录 |
| `--voice-max-utterance-ms` | `float` | `5000` | 首个 VAD 语音帧之后的最大采集时长；Animal Naming 初始建议 1500 ms |

`streammuse-task voice-devices` 可以在不提供 `--task`、不连接聊天模型服务器且不加载 Whisper 的情况下列出麦克风。语音参数只能和 `--human-input voice` 一起使用。完整说明见[交互式游戏语音输入](../user-guide/voice-input.md)。

### 交互式任务语音输出参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--speech-output` | `off/audio` | `off` | 是否朗读 LLM 回答 |
| `--speech-backend` | `system/espeak_ng/kokoro/null` | `system` | TTS 后端 |
| `--speech-voice` | `str` | 后端默认 | 音色 |
| `--speech-rate` | `float` | `1.0` | 相对后端默认语速的倍率 |
| `--speaker-device` | 名称或索引 | 系统默认 | PortAudio 输出设备 |
| `--speech-model` | repo 或路径 | `None` | Kokoro 模型，仅 Kokoro |
| `--speech-model-cache` | 路径 | `None` | Kokoro snapshot 缓存 |
| `--speech-model-revision` | commit 或 tag | `None` | Kokoro 固定版本 |
| `--speech-local-files-only` | flag | `False` | Kokoro 禁止联网 |
| `--speech-synthesis-timeout-s` | `float` | `10.0` | 子进程合成上限 |
| `--speech-prewarm` / `--no-speech-prewarm` | flag | prewarm | 游戏前预合成词表 |
| `--speech-cache-miss` | `synthesize/skip` | `synthesize` | 未命中缓存的行为 |
| `--speech-cache-max-entries` | `int` | `512` | 缓存条数上限 |
| `--speech-cache-max-bytes` | `int` | `67108864` | 缓存字节上限 |
| `--speech-guard-ms` | `float` | `200` | 机器音频排空到开麦的间隔 |
| `--speech-on-error` | `fail/warn` | `fail` | 预期输出失败策略 |
| `--speech-save-audio` | flag | `False` | 保存每个 LLM 回合的合成 WAV |
| `--llm-deadline-basis` | `text/audio_end` | `text` | 机器回合计分口径 |

`streammuse-task speaker-devices` 不需要任务或模型服务器。关闭语音输出时禁止传入 `--speech-*` 参数；`audio_end` 必须同时启用音频输出。Kokoro 在当前版本必须显式提供模型和 revision。完整说明见[交互式游戏语音输出](../user-guide/speech-output.md)。

### 交互式任务 Web 观察界面

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--web-ui` | flag | `False` | 启用只读 Web UI；首个 viewer 完成首次 render 前不初始化游戏资源 |
| `--web-host` | `str` | `127.0.0.1` | Web server 监听地址 |
| `--web-port` | `int` | `8002` | Web server 端口 |
| `--web-allow-remote` | flag | `False` | 允许非 loopback 绑定；仍校验随机 token 和 Origin |

未启用 `--web-ui` 时不能单独传 `--web-*` 参数。启用后 Web server 与首个 `viewer_ready` 是开局前的 required dependency；开局后的断线不暂停对局。完整说明见[交互式任务 Web 观察界面](../user-guide/task-web-ui.md)。

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
    --tempo 60 \
    --model-condition-bpm 120 \
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

完整的扰动鲁棒性 staging、qualification、formal、analysis 和 listening 顺序见[旋律扰动鲁棒性实验工作流](../developer-guide/melody-perturbation-robustness.md)。
