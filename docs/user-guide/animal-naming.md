---
title: Animal Naming 交互游戏
description: Animal Naming 的 exact whitelist 规则、双向语音命令和 trace 判读
---

# Animal Naming 交互游戏

`animal_naming` 支持原有 `run` benchmark，也支持 Human 和 LLM 轮流参与的 `play` 模式。每个 turn 只能回答一个尚未成功使用过的英文动物名称；Human 和 LLM 共享同一份 `used_animals` 状态。

## 判定规则

答案会先转成小写、去除首尾标点和开头的 `a`、`an`、`the`，再交给 referee：

| 条件 | 结果 |
|---|---|
| 空答案 | `EMPTY_RESPONSE` |
| 不在 whitelist | `UNKNOWN_ANIMAL` |
| 已经成功使用 | `REPEATED_ANIMAL` |
| 在 whitelist 且尚未使用 | valid |

V1 使用代码中的 91 项 exact whitelist。它不会自动接受所有真实动物，也不会进行 synonym、plural 或 fuzzy matching。例如：

- `lion`、`rhinoceros` 和 `hippopotamus` 在 whitelist 中；
- `capybara`、`donkey` 和 `polar bear` 不在 whitelist 中；
- `rhino` 与 `rhinoceros` 当前是两个独立答案；
- `hippo` 不会自动映射到 `hippopotamus`；
- `lions` 不会自动转换成 `lion`。

当前 whitelist 定义在 `src/streammuse/domain/tasks/animal_naming.py` 的 `DEFAULT_ANIMALS`。

## 文字输入、无语音输出

```bash
uv run --frozen streammuse-task play \
  --task animal_naming \
  --human-input terminal \
  --human-first \
  --max-turns 10 \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-text
```

## 文字输入、LLM 使用 TTS

```bash
uv run --frozen --extra speech streammuse-task play \
  --task animal_naming \
  --human-input terminal \
  --human-first \
  --max-turns 10 \
  --speech-output audio \
  --speech-backend system \
  --speech-cache-miss synthesize \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-tts
```

Animal Naming 不会在启动时预合成完整 whitelist。LLM 的新动物名按需合成；当前 audio cache 仅存在于 session 内存中，不跨局持久化。因此正常运行应使用 `--speech-cache-miss synthesize`。

## 语音输入、无语音输出

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
  --output-dir /private/tmp/streammuse-animal-stt
```

Speech parser 只提取一个简短动物候选词，不决定它是否存在于 whitelist。比如 STT 得到 `Dragon.` 时，parser 返回 `dragon`，随后 referee 返回 `UNKNOWN_ANIMAL`。

## 完整双向语音

```bash
uv run --frozen --extra voice --extra speech streammuse-task play \
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
  --speech-output audio \
  --speech-backend system \
  --speech-cache-miss synthesize \
  --speech-guard-ms 200 \
  --model-url http://127.0.0.1:8101/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 8 \
  --timeout-s 30 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-animal-bidirectional
```

双向语音时建议戴耳机。系统会等待 LLM 音频播放排空并执行 speech guard 后，才打开下一轮 Human microphone，但当前不包含 acoustic echo cancellation。

## Trace 判读

Human voice turn 重点字段：

```text
metadata.human_input.raw_transcript
metadata.human_input.canonical_response
metadata.referee_metadata.normalized_animal
metadata.failure_reason
metadata.timing_breakdown
```

LLM turn 额外关注：

```text
metadata.speech_output.status
metadata.speech_output.spoken_text
metadata.speech_output.synthesis_ms
metadata.speech_output.cached
metadata.speech_output.playback_drained_offset_ms
```

运行 latency analyzer：

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  /private/tmp/streammuse-animal-bidirectional/<run-directory> \
  --output-dir /private/tmp/streammuse-animal-latency-analysis
```

Animal Naming 没有唯一 expected answer，因此 trace 的 `number` 和 `expected` 为 `null`。裁判依据 `normalized_animal`、whitelist 和 `used_animals` 判定；这些空字段不影响 latency analyzer。
