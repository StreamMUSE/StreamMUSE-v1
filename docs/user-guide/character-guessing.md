---
title: Character Guessing 交互游戏
description: 让 LLM 通过 yes/no questions 猜出玩家私下选择的历史人物
---

# Character Guessing 交互游戏

`character_guessing` 是一个按 logical round 运行的 Human/LLM 游戏。玩家先从
内置人物名单中私下选择一人，LLM 每轮提出一个 yes/no question 或正式猜一个
人物，玩家只回答 yes 或 no。

玩家不会把目标人物输入程序。目标不会进入 prompt、trace、manifest 或 WebSocket；
最终是否猜中，以玩家对一次正式 guess 的回答为准。

## 规则

- 游戏默认由 LLM 开始。
- LLM 的内部协议是单行 `ASK: ...?` 或 `GUESS: <candidate>`。
- `ASK` 只能询问属性，不能包含 catalog 人名或 alias；点名必须使用 `GUESS`。
- 一次 logical round 包含一个有效 LLM action 和一个有效 Human yes/no answer。
- 非法 LLM 输出由 LLM 重试；非法 Human 回答由 Human 重试。
- soft mode 下同一阶段连续三次非法会以 `retry_limit_reached` 结束。
- Human 对 guess 回答 yes 后立即结束。
- 达到 `--max-rounds` 时，只会在 Human 完成当前回答后结束，不会留下半轮。

V1 提供 `world_historical_figures_en` scope，包含 44 位英文历史人物。玩家可以
在 setup 阶段看到完整名单，但程序不维护人物事实数据库；Human 是本局的事实
oracle。

## 文字模式

```bash
uv run --frozen streammuse-task play \
  --task character_guessing \
  --character-scope world_historical_figures_en \
  --max-rounds 20 \
  --model-url http://127.0.0.1:8010/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 64 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-character
```

启动后从名单中选一个人物，只按 Enter 确认 ready，不要键入名字。随后每个
Human turn 输入 `yes` 或 `no`。也接受 `yeah`、`yep`、`nope`、`correct`、
`incorrect`、`right` 和 `wrong` 等受控短语；解释性长句和 `maybe` 不接受。

Character Guessing 不接受 `--human-first`、`--start-number` 或显式
`--max-turns`。默认 `--max-rounds 20`、LLM first 和 `--max-tokens 64`。

## 双向语音与 Web UI

```bash
uv run --frozen --extra voice --extra speech streammuse-task play \
  --task character_guessing \
  --character-scope world_historical_figures_en \
  --max-rounds 20 \
  --human-input voice \
  --microphone-device 2 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --voice-max-utterance-ms 1000 \
  --speech-output audio \
  --speech-backend system \
  --speech-cache-miss synthesize \
  --speech-guard-ms 200 \
  --web-ui \
  --web-port 8002 \
  --model-url http://127.0.0.1:8010/v1 \
  --model Qwen/Qwen3.6-27B \
  --temperature 0 \
  --max-tokens 64 \
  --deadline-mode soft \
  --deadline-ms 5000 \
  --output-dir /private/tmp/streammuse-character-voice
```

LLM question 是动态内容，因此启用 TTS 时必须使用
`--speech-cache-miss synthesize`。系统会等待 LLM 音频完整排空并执行 speech
guard，之后才打开 Human microphone。setup 选择阶段不打开麦克风，也不计入
deadline。

## 结果与分析

`run_summary.json` 和 `manifest.json` 会记录 logical round、question、guess、
rejected guess、human-confirmed guess 及猜中 round。每个 attempt 的 trace 还
包含 phase、action、canonical answer 和 `round_completed` 标记。

运行 latency analyzer 后，原有 `breakdown_turns.csv` 保持不变，并额外生成
`breakdown_rounds.csv`：

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  /private/tmp/streammuse-character-voice \
  --output-dir /private/tmp/streammuse-character-analysis
```

round CSV 只合并同一 logical round 中有效的 LLM 和 Human turns，非法 retry
不会被算进完整 round latency。
