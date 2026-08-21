---
title: 交互式游戏语音输入
description: 在 Zip-Zap-Zop 和 Animal Naming 中使用 faster-whisper tiny.en 进行本地语音输入
---

# 交互式游戏语音输入

`streammuse-task play` 可以在轮到人类玩家时打开麦克风，将一段发言转成文本，再交给现有游戏裁判判定。语音输入当前支持 `zip_zap_zop` 和 `animal_naming`，使用常驻的 `faster-whisper` `tiny.en` 模型，默认在 CPU 上以 `int8` 运行。

语音层不判断答案是否正确，也不会根据当前正确答案修改识别结果。它返回原始转录文本，再由 task-specific parser 做确定性的格式规范化，最后调用原有裁判逻辑。`ZipZapZopTask` 处理游戏词和英语数字；`AnimalNamingTask` 处理一个简短英文动物候选词，但 whitelist membership 和 repetition 只由 referee 判断。

识别器会把明显退化的模型输出标记为 `rejected_transcript`，而不是把它送给游戏 parser。当前防护覆盖超过 512 字符、同一 token 连续重复至少 16 次、任一 Whisper segment 的 compression ratio 超过 10，或 compression ratio 是非有限/不可解析值的输出。原始转录仍保留在 `metadata.human_input.raw_transcript`，触发原因和测量值记录在 ASR quality-gate metadata 中，便于排查而不会静默改写玩家说过的内容。

Zip-Zap-Zop 的口语整数语法范围为 `-999999999` 到 `999999999`。语音模式会在启动前检查本局从 `--start-number` 到 `--max-turns` 覆盖的范围；超出时直接拒绝配置，不会开始麦克风采集。Animal Naming 不使用 `--start-number`。

## 安装

语音依赖是可选的；普通终端输入不需要安装这些包。

```bash
uv sync --extra voice
```

该依赖组包含 `faster-whisper`、`sounddevice` 和 `webrtcvad-wheels`。SciPy 重采样器由主项目依赖提供。

## 检查麦克风

设备枚举不会加载 Whisper 模型，也不需要 `--task` 或模型服务器：

```bash
uv run --extra voice streammuse-task voice-devices
```

使用输出中的索引或完整设备名称选择麦克风：

```bash
--microphone-device 2
# 或
--microphone-device "Built-in Microphone"
```

macOS 首次访问麦克风时可能要求授权。若授权被拒绝，请在“系统设置 → 隐私与安全性 → 麦克风”中允许启动终端或 Python 的应用。Linux 需要可用的 PortAudio 输入设备；容器或远程会话通常不会自动暴露主机麦克风。

## 开始游戏

先启动 OpenAI 兼容的本地聊天模型服务，然后运行：

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

模型会在游戏开始前加载并预热一次。麦克风只在人类回合打开，每个回合检测到尾部静音后关闭；模型在整局游戏中保持常驻。

录音端点使用 20 ms WebRTC VAD frame。语音起点不会由单个 positive
frame 直接触发：默认观察最近 8 个 frame，并在至少 5 个 frame 被判为
voiced 后确认开始。未达到阈值的短噪声会被丢弃，系统继续等待用户开口；
确认起点后仍保留既有 100 ms pre-roll，避免截断单词开头。manifest 的
`human_input.microphone` 会记录 `vad_start_window_frames` 和
`vad_start_trigger_frames`。

Animal Naming 的 STT-only 示例：

```bash
uv run --frozen --extra voice streammuse-task play \
  --task animal_naming \
  --human-input voice \
  --human-first \
  --max-turns 20 \
  --microphone-device 2 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --voice-max-utterance-ms 1500 \
  --voice-save-audio \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-voice
```

Animal Naming 使用现有 91 项 exact whitelist。`capybara`、`donkey`、`polar bear` 等真实动物如果不在列表中，仍会得到 `UNKNOWN_ANIMAL`；V1 不自动合并 synonym 或 plural。完整规则和四种运行方式见 [Animal Naming 交互游戏](animal-naming.md)。

建议先使用 `soft` 和至少 3000 ms 的截止时间。等待开口、发言、端点静音和 ASR 都计入人类回合耗时。`hard`/`challenge` 会在采集阶段达到截止时间时停止录音，但进程内的 CTranslate2 转录不能安全取消，因此最终返回时间可能超过名义截止时间。

对于只接受单个短词的任务，可以限制从首个 VAD 语音帧开始的最大采集时长。Zip-Zap-Zop 可以从 1000 ms 开始测试：

```bash
--voice-max-utterance-ms 1000
```

未指定时默认仍为 5000 ms。该上限不是等待用户开口的时间；等待开口由独立的
start timeout 控制。如果环境噪声先被 VAD 误判为语音，过短的上限可能在用户
真正说完之前截断音频，因此应根据任务词长和录音审计结果设置。

Animal Naming 包含 `hippopotamus`、`rhinoceros` 等长词，初始建议为 1500 ms。若保存的 WAV 显示长词被截断，再提高到 1800–2000 ms。

## 模型缓存和离线运行

默认情况下，首次启动可能在游戏开始前从 Hugging Face 下载模型。可以显式指定缓存和模型版本：

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --voice-model tiny.en \
  --voice-model-cache /path/to/model-cache \
  --voice-model-revision <commit-or-tag> \
  --voice-local-files-only \
  --deadline-mode soft
```

`--voice-local-files-only` 禁止下载；缓存中没有请求的模型或版本时，游戏会在创建任何玩家回合之前失败。也可以把 `--voice-model` 直接设置为本地 faster-whisper 模型目录。

## 记录和隐私

默认不会保存麦克风音频。运行目录仍会记录：

- 原始 ASR 转录和规范化后的游戏响应；
- 等待语音、发言、端点静音和 ASR 延迟；
- 麦克风设备、采样率、模型配置和端点原因；
- 音频溢出、无语音和空转录状态。

只有显式添加 `--voice-save-audio` 时，才会把每个有界发言保存到运行目录的 `artifacts/turn/`。WAV 在采集和 ASR 计时结束后写入，文件 I/O 不参与回合截止时间计分。这些 WAV 文件可能包含敏感语音，不应默认提交或共享。

冻结语料库的准确率、混淆和延迟验收流程见[语音输入离线资格验证](../developer-guide/voice-input-qualification.md)。

## 故障诊断

| 现象 | 处理方式 |
|---|---|
| 提示缺少语音依赖 | 运行 `uv sync --extra voice` |
| 没有输入设备 | 运行 `voice-devices`，检查系统权限和 PortAudio 设备 |
| 设备只接受 44.1 kHz | 选择支持 8/16/32/48 kHz 的输入配置；首版不在 VAD 前做流式 44.1 kHz 重采样 |
| 离线模型未命中 | 去掉 `--voice-local-files-only` 完成一次预下载，或修正缓存/本地模型路径 |
| Zip/Zap 混淆 | 查看 `response_trace.jsonl` 中的原始转录；系统不会用预期答案自动纠错 |
| 真实动物被判 `UNKNOWN_ANIMAL` | 检查该名称是否在 Animal Naming exact whitelist；V1 不做 synonym 或 plural 推断 |
| 长动物名识别不完整 | 使用 `--voice-save-audio` 审计，并把 `--voice-max-utterance-ms` 从 1500 调到 1800–2000 |
| 一秒挑战经常超时 | 先使用 3000 ms `soft` 模式，依据目标设备的 p95 指标再缩短时间 |
| 采集期间需要退出 | 按 `Ctrl-C`；语音模式采集中不并发处理 `:quit` |

LLM 回答的可选 TTS 播放见[交互式游戏语音输出](speech-output.md)。STT、LLM、TTS
和 PortAudio 的分阶段耗时分析见[交互式语音延迟分解](voice-latency-breakdown.md)。
当前仍不包含声学回声消除。
