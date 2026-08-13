---
title: 交互式游戏语音输出
description: 让 Zip-Zap-Zop 的 LLM 回答通过 TTS 和扬声器播放
---

# 交互式游戏语音输出

`streammuse-task play` 可以在 LLM 生成答案后，将答案规范化为适合朗读的文本、合成音频并完整播放。TTS 只朗读 LLM 的原始答案语义：`ZipZap` 会读成 `Zip Zap`，数字保持原值，错误答案不会被纠正。

裁判结果、胜负、帮助、截止时间信息和人类回合提示仍然只显示在终端。系统不提供音频 cue、结果播报、声学回声消除或播放中抢答。

## 安装

输入和输出依赖彼此独立：

| 场景 | extra |
|---|---|
| 人说话，机器只打印 | `voice` |
| 人打字，机器使用 `say`/`espeak-ng` 说话 | `speech` |
| 双向语音，系统 TTS | `voice,speech` |
| 双向语音，Kokoro | `voice,tts-kokoro` |

```bash
uv sync --extra speech
```

macOS 的 `system` 后端使用系统 `say`。Linux 的 `system` 后端和显式的 `espeak_ng` 后端要求 `espeak-ng` 在 `PATH` 中。Kokoro 还要求系统提供 `espeak-ng`，并且必须显式指定模型和固定 revision。

## 检查扬声器

设备枚举不需要 `--task`、LLM 服务或 TTS 模型：

```bash
uv run --extra speech streammuse-task speaker-devices
```

使用索引或完整名称选择设备：

```bash
--speaker-device 2
# 或
--speaker-device "MacBook Pro Speakers"
```

## 开始游戏

人打字、机器说话：

```bash
uv run --extra speech streammuse-task play \
  --task zip_zap_zop \
  --human-input terminal \
  --speech-output audio \
  --speech-backend system \
  --deadline-mode soft \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

双向语音：

```bash
uv run --extra voice --extra speech streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --speech-output audio \
  --speech-backend system \
  --speech-guard-ms 200 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

建议双向语音时戴耳机。外放场景没有 AEC；系统只保证机器音频排空后等待一次 guard，再打印人类文字 prompt 并打开麦克风。人类先手、没有实际机器音频或使用终端输入时不等待 guard。

## 后端和缓存

- `system`：默认。macOS 使用 `say`，其他平台使用 `espeak-ng`。
- `espeak_ng`：显式使用 `espeak-ng`。
- `kokoro`：高质量可选后端，必须提供 `--speech-model` 和 `--speech-model-revision`。
- `null`：生成静音 PCM，只用于测试。

游戏开始前会预合成本局的有界词表。默认缓存最多 512 项和 64 MiB；命中时回合内 `synthesis_ms` 为 0。可以使用 `--no-speech-prewarm`，或用 `--speech-cache-miss skip` 在未命中时跳过播放。

Kokoro 示例：

```bash
uv run --extra tts-kokoro streammuse-task play \
  --task zip_zap_zop \
  --speech-output audio \
  --speech-backend kokoro \
  --speech-model hexgrad/Kokoro-82M \
  --speech-model-revision <commit> \
  --speech-model-cache .cache/tts-models \
  --speech-local-files-only \
  --deadline-mode soft
```

离线模式只接受本地已有的固定 snapshot，不会回退到 latest。

## 截止时间口径

默认 `--llm-deadline-basis text` 保持既有计分：LLM 文本就绪即停止机器回合计时，合成和播放只作为诊断信息。

显式使用 `--llm-deadline-basis audio_end` 时，计分包含文本就绪到完整播放排空的时间。如果没有完整音频，例如空答案、合成失败、跳过缓存未命中或播放被截断，该回合回退到 `text`。trace 会记录配置口径、实际口径和 fallback reason，summary 会统计回退次数。

## 错误策略

默认 `--speech-on-error fail`：合成、播放或可选 WAV 保存失败时，系统先写入该机器回合的裁判结果、trace 和 turn artifact，再以 `speech output error` 退出。

`--speech-on-error warn` 会在这些预期基础设施失败后继续对局。`KeyboardInterrupt`、`SystemExit` 和程序缺陷不受 `warn` 影响；它们同样先记录回合，然后以原异常退出。

## 记录和隐私

默认不保存合成音频。trace 会记录原始回答、朗读文本、后端/设备、缓存状态、合成耗时、首个 DAC 样本估计时间、排空时间和错误。

只有显式添加 `--speech-save-audio` 时才写入 `artifacts/turn/*_llm.wav`。文件在播放完成后原子写入，持久化耗时单独记录，不参与 `audio_end` 计分。合成 WAV 可能泄露模型输出，不应默认提交或共享。

## 故障诊断

| 现象 | 处理方式 |
|---|---|
| 缺少 `sounddevice` | 运行 `uv sync --extra speech` |
| 找不到扬声器 | 运行 `speaker-devices` 并检查 PortAudio |
| 找不到 `say`/`espeak-ng` | 安装对应系统命令并确认它在 `PATH` |
| Kokoro 缓存未命中 | 先联网下载固定 revision，或修正模型缓存/本地路径 |
| 外放触发麦克风 | 戴耳机；必要时增大 `--speech-guard-ms` |
| `audio_end` 经常回退 | 查看 trace 的 `deadline_basis_fallback_reason` 和播放错误 |
| 希望听到当前人类回合提示 | 当前设计只显示文字 prompt，不提供音频 cue |

STT、LLM、TTS、首个 DAC 样本和 drain 的分阶段统计见
[交互式语音延迟分解](voice-latency-breakdown.md)。
