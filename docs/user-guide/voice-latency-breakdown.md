---
title: 交互式语音延迟分解
description: 分析 STT、LLM、TTS、PortAudio 与双向对话关键路径耗时
---

# 交互式语音延迟分解

交互式任务的每个回合都会在 `response_trace.jsonl` 的
`metadata.timing_breakdown` 中记录单调时钟时间点。新增埋点不改变
`latency_ms`、deadline 判定、VAD endpoint、模型输入或播放完成判据。

## 记录范围

人类回合覆盖：

- guard、prompt 构造和终端输出；
- 麦克风流创建、首个 callback、首个/最后一个 VAD 语音帧；
- endpoint、重采样、关闭输入流；
- faster-whisper 模型调用、segment 消费、文本拼接和质量门；
- 游戏语音解析和裁判判定。

LLM 回合覆盖：

- prompt 构造和完整 HTTP round trip；
- HTTP 状态检查、JSON decode 和响应解析；
- spoken text 构造、TTS cache lookup 和可选合成；
- OutputStream 创建、首 callback、首个 DAC 样本和 drain；
- 游戏裁判判定。

当前 LLM 客户端使用非 streaming 请求，因此 `http_round_trip` 表示完整响应
时间，不等同于 TTFT。TTFT 必须通过 streaming 请求或带 request ID 的 vLLM
服务端 trace 单独测量。

## 分析一次运行

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  /private/tmp/streammuse-bidirectional-voice/<run-directory> \
  --output-dir /private/tmp/streammuse-voice-latency-analysis
```

也可以传入多个 run directory、单独的 `response_trace.jsonl`，或包含多个运行
目录的父目录：

```bash
uv run python scripts/analyze_interactive_voice_latency.py \
  task_runs/voice-baseline-a \
  task_runs/voice-baseline-b \
  --output-dir task_runs/voice-latency-summary
```

输出包括：

| 文件 | 内容 |
|---|---|
| `breakdown_turns.csv` | 每回合固定信息和所有可用阶段耗时 |
| `breakdown_summary.json` | 启动成本及各指标的 count/mean/p50/p90/p95/p99 |
| `breakdown_summary.md` | 便于人工审阅的汇总表 |

分析器严格拒绝负数、非有限值和未知 timing schema，避免静默生成错误统计。
它同时接受 Zip-Zap-Zop 的 numbered turns 和 Animal Naming 的 open-ended turns；后者在 CSV 中的 `number` 字段为空，不影响 timing 统计。

## 关键指标

优先关注：

- `human.last_voice_to_asr_text_ms`；
- `conversation.human_last_voice_to_llm_text_ms`；
- `conversation.human_last_voice_to_first_dac_ms`；
- `llm.request_to_audio_drained_ms`；
- `conversation.audio_drained_to_mic_callback_ms`。

人的反应和发言时间应与系统计算分开报告：

- `human.wait_for_speech_ms` 是从输入层启动到首个语音样本的观测等待时间，
  包含麦克风流启动开销；
- `human.voiced_utterance_ms` 是实际发言区间；
- `human.endpoint_delay_ms` 包含 VAD 的尾部静音策略；
- STT、LLM、TTS 和播放器字段才是系统处理路径。

`first_dac_sample` 来自 PortAudio `outputBufferDacTime` 映射，是设备 API
时间估计，不是外部麦克风测得的物理声学到达时间。

## 可复现实验

建议固定以下条件：

1. 固定 STT model/revision、LLM model、TTS backend 和设备编号。
2. vLLM 使用稳定的 HTTP keep-alive 设置。
3. 先运行 10 个 warm-up 回合，再收集至少 50 个测量回合。
4. cold start 的模型 resolution/load/warmup 和 TTS prewarm 单独报告。
5. 使用固定 WAV 的 `scripts/voice_microbench.py` 测纯 STT/TTS，再进行真人端到端测试。
6. 正式延迟测试默认不保存 WAV；需要审计 VAD 时才临时开启 `--voice-save-audio`。

Animal Naming qualification 建议完成 5 局、每局 20 turns，并把 Human intended animal 与 `human_input.raw_transcript`、`human_input.canonical_response` 和 `referee_metadata.normalized_animal` 对齐审计。
