# StreamMUSE 双向语音交互延迟分解报告

## 1. 报告信息

- 报告日期：2026-07-30
- 项目：StreamMUSE-v1
- 分支：`feature/voice`
- 任务：`zip_zap_zop`
- 实验规模：5 局，每局 10 turn
- 总样本数：50 turn
- Human turn：25
- LLM turn：25
- timing schema：`metadata.timing_breakdown.schema_version = 1`
- 分析器：`scripts/analyze_interactive_voice_latency.py`

本报告分析以下五次运行：

| 序号 | 运行目录 | run_id |
|---:|---|---|
| 1 | `zip_zap_zop_interactive_20260730-222732_a8a567b0` | `interactive-6f848b91` |
| 2 | `zip_zap_zop_interactive_20260730-222825_afd478ae` | `interactive-224e76b0` |
| 3 | `zip_zap_zop_interactive_20260730-222915_53778751` | `interactive-6dd0ba7e` |
| 4 | `zip_zap_zop_interactive_20260730-223003_66c2f4b3` | `interactive-28caa935` |
| 5 | `zip_zap_zop_interactive_20260730-223049_7dd3aea7` | `interactive-2bc27dfa` |

原始数据位于：

```text
/private/tmp/streammuse-bidirectional-voice/
```

分析器输出位于：

```text
/private/tmp/streammuse-breakdown-5x10-20260730/
├── breakdown_turns.csv
├── breakdown_summary.json
└── breakdown_summary.md
```

## 2. Executive Summary

本次实验成功验证了新加入的 latency breakdown：

1. 50/50 turn 均包含合法的 `timing_breakdown`。
2. 没有 turn 因缺失 timing schema 被分析器跳过。
3. Human、ASR、LLM HTTP、TTS cache、PortAudio 播放和跨回合时间均可分解。
4. 五局全部正常完成，没有基础设施错误或播放失败。

最重要的结论如下：

1. 当前主要延迟瓶颈是 VAD/capture，而不是 faster-whisper 推理。
   Human voice response 平均需要 4708 ms，其中 VAD 判定的 voiced utterance
   平均达到 3989 ms。对 Zip/Zap/Zop 这样的单词来说，这明显过长。

2. 25 个 Human turn 中有 24 个超过 3000 ms deadline。
   这些 deadline miss 基本由采集持续时间导致。LLM 的 25 个 turn 没有一次
   deadline miss。

3. faster-whisper 正常推理速度较快。
   ASR pipeline 的 p50 为 127 ms，p95 为 146 ms。唯一显著异常值为 725 ms，
   对应一次重复输出大量 `Zap` 的 hallucination。

4. LLM 文本延迟总体可接受。
   HTTP round trip 平均 389 ms、p50 341 ms、p95 685 ms。每局第一个 LLM
   请求明显更慢，平均 641 ms；后续请求平均约 309-350 ms。

5. TTS 回合内没有实时合成成本。
   25/25 回合全部命中预热缓存。LLM 文本返回后平均 41 ms 即到达首个 DAC
   样本。

6. 从系统判定的最后 voiced frame 到听见机器回答，平均约 924 ms。
   但由于 VAD 持续错误地把后续声音标成 voiced，这个 anchor 很可能晚于用户
   真实说完单词的时刻，不能直接解释为完整的主观交互延迟。

7. 启动阶段最大成本是 system TTS prewarm。
   每局预热 10 个 phrase 平均需要 8115 ms。STT snapshot resolution 平均
   2493 ms，但实际 trace 中 `local_files_only=false`。

## 3. 数据完整性

| 检查项 | 结果 |
|---|---:|
| Trace 文件数 | 5 |
| 每个 trace 的 turn 数 | 10 |
| 总 turn 数 | 50 |
| 包含 timing breakdown 的 turn | 50 |
| 被分析器跳过的 turn | 0 |
| timing schema 错误 | 0 |
| 负数或非有限 duration | 0 |
| 完成的游戏 | 5/5 |

因此，本次统计没有混入修改前的旧 trace，也没有将 20:00-21:00 的短测试纳入
五局汇总。

## 4. 实际运行配置

以下配置来自每局 `manifest.json`：

### 4.1 游戏与 deadline

- Task：`zip_zap_zop`
- Human first：`true`
- Max turns：`10`
- Deadline mode：`soft`
- Deadline：`3000 ms`
- Max LLM tokens：`8`
- Temperature：`0`

### 4.2 STT

- Model：`tiny.en`
- Device：`cpu`
- Compute type：`int8`
- Requested revision：
  `0d3d19a32d3338f10357c0889762bd8d64bbdeba`
- Resolved revision：
  `0d3d19a32d3338f10357c0889762bd8d64bbdeba`
- `local_files_only=false`
- VAD aggressiveness：`2`
- VAD frame：`20 ms`
- Start timeout：`5000 ms`
- End silence：`300 ms`
- Max utterance：`5000 ms`
- Pre-roll：`100 ms`
- Capture sample rate：`16000 Hz`
- 麦克风：设备 1，`MacBook Pro麦克风`

### 4.3 LLM

根据用户运行命令，本次服务目标为 Qwen 27B 服务，但当前 manifest 和
`response_trace.jsonl` 没有保存 LLM model identifier 或 model URL。

因此：

- 本报告可以验证 HTTP timing、token count 和响应文本；
- 不能仅依赖 artifact 独立证明服务端实际加载了哪个 model；
- 这是一个需要后续修正的可复现性缺口。

### 4.4 TTS 与播放

- Speech output：`audio`
- Backend：macOS `system` / `/usr/bin/say`
- Rate：`1.0`
- Speaker：设备 2，`MacBook Pro扬声器`
- Prewarm entry count：`10`
- Speech guard：`250 ms`
- LLM deadline basis：`text`
- Speech error policy：`fail`
- Save speech audio：`false`

需要特别注意：这批运行的实际 `llm_deadline_basis` 是 `text`，不是
`audio_end`。因此 LLM 的 `latency_ms` 和 deadline 判定不包含 TTS 播放时间。

## 5. 游戏结果

### 5.1 总体结果

| 指标 | 结果 |
|---|---:|
| 总 turn | 50 |
| Valid | 44 |
| Invalid | 6 |
| 总体正确率 | 88% |
| Deadline miss | 24 |
| 总体 deadline miss rate | 48% |

### 5.2 按 actor 分解

| Actor | Turn | Valid | Invalid | 正确率 | Deadline miss |
|---|---:|---:|---:|---:|---:|
| Human | 25 | 19 | 6 | 76% | 24 |
| LLM | 25 | 25 | 0 | 100% | 0 |

Human deadline miss rate 为 96%。LLM deadline miss rate 为 0%。

### 5.3 按游戏分解

| Game | Valid | Invalid | Deadline miss | Human mean latency | LLM mean latency |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 1 | 5 | 5007 ms | 378 ms |
| 2 | 9 | 1 | 5 | 4624 ms | 353 ms |
| 3 | 8 | 2 | 5 | 4545 ms | 371 ms |
| 4 | 9 | 1 | 5 | 5053 ms | 420 ms |
| 5 | 9 | 1 | 4 | 4310 ms | 426 ms |

五局之间的 Human mean latency 均处于 4.31-5.05 秒，说明长采集不是单次
偶发问题，而是稳定复现的问题。

## 6. Human Input Latency Breakdown

### 6.1 汇总

所有数值单位均为毫秒。

| 阶段 | count | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| 完整 response source | 25 | 4708 | 4884 | 5539 | 5686 |
| 输入层启动到首个语音样本 | 25 | 228 | 238 | 282 | 302 |
| VAD voiced utterance | 25 | 3989 | 4106 | 4985 | 4991 |
| 最后 voiced frame 到 endpoint | 25 | 231 | 302 | 307 | 307 |
| Endpoint 到输入流关闭 | 25 | 107 | 107 | 110 | 119 |
| 输入流关闭到 ASR start | 25 | 0.2 | 0.2 | 0.4 | 0.5 |
| ASR pipeline | 25 | 152 | 127 | 146 | 725 |
| 最后 voiced frame 到 ASR text | 25 | 490 | 529 | 550 | 1134 |
| 最后 voiced frame 到游戏判定 | 25 | 491 | 529 | 550 | 1134 |

### 6.2 典型 Human turn

使用中位数近似，一个典型 Human turn 可以分解为：

```text
response source 启动
  + 238 ms   等待首个 voiced sample，包含麦克风启动
  + 4106 ms  VAD 认为输入仍然是 voiced
  + 302 ms   trailing silence endpoint
  + 107 ms   关闭 PortAudio input stream
  + 127 ms   faster-whisper pipeline
  + < 1 ms   parse、terminal render 和游戏判定
  ≈ 4880 ms
```

其中 4106 ms 的 voiced interval 是绝对主导项。

### 6.3 VAD endpoint 结果

| Endpoint reason | Count | 比例 |
|---|---:|---:|
| `trailing_silence` | 17 | 68% |
| `max_utterance` | 8 | 32% |

`max_utterance` 配置为 5000 ms。8 次采集几乎持续到完整上限：

- 最大 voiced utterance：4991 ms
- voiced utterance p95：4985 ms
- 多次 endpoint silence 只有 0-160 ms

17 次 `trailing_silence` 表明 endpoint 逻辑在最终观察到 300 ms 静音后可以
正常停止。真正的问题是 VAD 在很长一段时间内没有连续观察到静音。

### 6.4 对 `last_voice` 指标的解释

`microphone.last_voiced` 是 VAD 标记的最后一个 voiced frame，不是外部声学
设备测得的“用户真实说完单词”。

当 VAD 把背景噪声、残余声音或其他输入误判为 voiced 时：

1. `last_voiced` 会持续向后移动；
2. `last_voice_to_asr_text` 会看起来只有约 0.5 秒；
3. 用户实际感受到的等待却接近完整 response source 的 4-5 秒。

因此，本次结果中：

- `last_voice_to_asr_text` 适合衡量 VAD endpoint 之后的系统处理；
- 不适合单独用作真人主观 end-to-end latency；
- 在 VAD 问题解决前，应同时报告完整 `response_source`。

## 7. ASR Breakdown

### 7.1 正常 ASR 性能

| ASR 阶段 | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| Input prepare | 0.033 ms | 0.031 ms | 0.051 ms | 0.054 ms |
| Model call | 4.1 ms | 3.8 ms | 7.5 ms | 9.3 ms |
| Segment iteration | 148 ms | 123 ms | 141 ms | 721 ms |
| Text assembly | 0.005 ms | 0.004 ms | 0.008 ms | 0.010 ms |
| Quality gate | 0.019 ms | 0.016 ms | 0.032 ms | 0.048 ms |
| Pipeline total | 152 ms | 127 ms | 146 ms | 725 ms |

faster-whisper 的 `transcribe()` 返回 lazy segment iterator。实际解码工作主要在
消费 iterator 时发生，因此 `segment_iteration` 大于初始 `model_call` 是正常
现象，不能把 4.1 ms 的 `model_call` 误解为完整 ASR 推理时间。

### 7.2 ASR 异常值

唯一显著 ASR latency outlier 出现在第一局、number 5：

- Expected：`Zop`
- Raw transcript：大量重复的 `Zap, Zap, Zap, ...`
- ASR pipeline：725 ms
- Segment iteration：721 ms
- Quality status：`rejected_transcript`
- Canonical response：空字符串
- 游戏判定：Invalid

quality gate 正确识别了连续 token 重复并拒绝该 transcript，因此 hallucination
没有被当成合法玩家输入。

### 7.3 识别准确率分布

| Number | Expected | 成功次数 | 总次数 | 观察到的错误 |
|---:|---|---:|---:|---|
| 1 | `1` | 5 | 5 | 无 |
| 3 | `Zip` | 4 | 5 | `Deep.` |
| 5 | `Zop` | 0 | 5 | `Zap`、重复 `Zap`、`It's up.` |
| 7 | `7` | 5 | 5 | 无 |
| 9 | `Zip` | 5 | 5 | 无 |

错误高度集中在 `Zop`：

- 3 次输出为 `Zap`
- 1 次输出重复 `Zap` 并被 quality gate 拒绝
- 1 次输出为 `It's up.`

这与 `tiny.en` 对短、近音词 `Zop/Zap` 的混淆一致。不过本次没有保存输入
WAV，因此不能仅凭 transcript 判断错误一定来自模型；也可能包含实际发音、
环境噪声、麦克风增益或其他声学因素。

## 8. LLM Latency Breakdown

### 8.1 HTTP pipeline

| 阶段 | count | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| Payload build | 25 | 0.003 ms | 0.002 ms | 0.006 ms | 0.006 ms |
| HTTP round trip | 25 | 389 ms | 341 ms | 685 ms | 843 ms |
| HTTP status check | 25 | 0.002 ms | 0.003 ms | 0.003 ms | 0.004 ms |
| JSON decode | 25 | 0.045 ms | 0.037 ms | 0.061 ms | 0.263 ms |
| Response parse | 25 | 0.017 ms | 0.015 ms | 0.027 ms | 0.069 ms |
| Client pipeline total | 25 | 389 ms | 341 ms | 685 ms | 843 ms |

客户端本地的 payload、JSON 和 parse 开销均可忽略。几乎全部 LLM latency 都在
HTTP round trip 内。

当前请求是非 streaming 请求，所以该指标表示完整 response latency，不是
TTFT。若需要真正的 token-level TTFT，必须改为 streaming，或者把 request ID
与 vLLM 服务端 trace 对齐。

### 8.2 每局首请求效应

| Number | Mean prompt tokens | Mean completion tokens | Mean LLM latency |
|---:|---:|---:|---:|
| 2，每局第一次 LLM 请求 | 99.0 | 2.0 | 641 ms |
| 4 | 113.8 | 3.0 | 350 ms |
| 6 | 127.4 | 2.0 | 318 ms |
| 8 | 142.4 | 3.0 | 309 ms |
| 10 | 149.4 | 3.0 | 329 ms |

第一条请求的 prompt 最短，但 latency 最高。这说明差异不是由 prompt 长度
造成，更符合以下一种或多种情况：

- server request path warm-up；
- CUDA graph/kernel 或 allocator warm-up；
- connection pool/HTTP keep-alive 首次建立；
- prefix/KV cache 尚未建立；
- 服务端同时存在其他请求。

正式 benchmark 应在每局开始前发送不计入统计的 warm-up request，或者把每局
第一条 LLM request 单独报告。

## 9. TTS 与 PortAudio Breakdown

### 9.1 正确性

| 检查项 | 结果 |
|---|---:|
| Speech status `ok` | 25/25 |
| Cache hit | 25/25 |
| 实时 synthesis | 0/25 |
| Completed normally | 25/25 |
| Playback error | 0 |
| Audio artifact save | 0 |

预热缓存实现了预期目标：TTS synthesis 没有进入任何回合的关键路径。

### 9.2 文本到播放完成

| 阶段 | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| Cache lookup | 0.0009 ms | 0.0010 ms | 0.0012 ms | 0.0014 ms |
| Text ready 到 first DAC | 41 ms | 40 ms | 54 ms | 58 ms |
| First DAC 到 stream inactive | 578 ms | 586 ms | 607 ms | 607 ms |
| Stream inactive 后清理并返回 | 106 ms | 106 ms | 108 ms | 109 ms |
| Speech output call total | 725 ms | 729 ms | 752 ms | 758 ms |

`text_to_first_dac` 是机器开始发声前最关键的 TTS/playback startup 指标。其
p95 只有 54 ms，表现稳定。

约 578 ms 是短答案实际播放长度，不应被视为纯计算开销。

### 9.3 Speaker 内部开销

| Speaker 阶段 | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|
| Audio prepare | 4.1 ms | 4.1 ms | 5.5 ms | 6.1 ms |
| Stream open | 1.9 ms | 1.8 ms | 2.4 ms | 2.5 ms |
| Stream start | 28.6 ms | 27.6 ms | 30.9 ms | 41.4 ms |
| Start 到 first callback | 29.5 ms | 27.9 ms | 39.6 ms | 47.2 ms |
| Start 到 first DAC | 35.2 ms | 33.5 ms | 45.2 ms | 52.8 ms |

音频 drain 之后仍有稳定的约 106 ms，发生在 player 返回到 speech sink 之前。
它主要对应 OutputStream close/cleanup 路径。它不影响已经完成的声音输出，但会
推迟下一回合的调度。

## 10. 跨阶段 End-to-End Latency

### 10.1 从 VAD last voiced frame 开始

| 指标 | count | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| Human last voice 到 LLM text | 25 | 883 ms | 869 ms | 1079 ms | 1484 ms |
| Human last voice 到 first DAC | 25 | 924 ms | 907 ms | 1121 ms | 1524 ms |
| Human last voice 到 audio drained | 25 | 1502 ms | 1481 ms | 1678 ms | 2084 ms |

平均关键路径近似为：

```text
VAD last voiced frame
  + 491 ms  endpoint、input stream close、ASR、parse、game validation
  + 390 ms  LLM HTTP response
  + 41 ms   TTS/player startup 到 first DAC
  = 约 922 ms 后开始听到机器声音

继续播放：
  + 578 ms  实际语音播放
  = 约 1500 ms 后音频 drain
```

数值与直接统计的 924 ms 和 1502 ms 基本一致，说明跨层 anchor 对齐正确。

### 10.2 从机器音频结束到下一次麦克风 callback

| 指标 | count | mean | p50 | p95 | max |
|---|---:|---:|---:|---:|---:|
| Audio drained 到下一次 mic callback | 20 | 466 ms | 465 ms | 476 ms | 483 ms |

该间隔可以近似分解为：

```text
106 ms  speaker stream close/cleanup
255 ms  实测 speech guard，配置值为 250 ms
103 ms  microphone open 到 first callback
  2 ms  其他调度和边界开销
--------------------------------------
466 ms  总计
```

该拆分与观测总值闭合，表明 timing breakdown 可以有效解释相邻回合间的空档。

## 11. Deadline 分析

### 11.1 Human

- 配置 deadline：3000 ms
- Deadline mode：`soft`
- Human deadline miss：24/25
- 唯一没有 miss 的 Human turn：约 1351 ms

`soft` mode 只记录 deadline miss，不会在 3000 ms 时强制终止采集。因此，
max utterance 仍可持续到 5000 ms，然后继续执行 stream close 和 ASR。

Human miss 的直接原因不是 ASR 需要数秒，而是 VAD/capture 多数时候在 3 秒时
仍然没有结束。

### 11.2 LLM

- LLM deadline miss：0/25
- 实际 deadline basis：`text`
- LLM request-to-text p95：685 ms
- LLM request-to-audio-drained p95：1277 ms

即使本次改为 `audio_end` 计分，25 个 LLM turn 的 p95 仍明显低于 3000 ms。
因此当前 deadline 风险集中在人类输入侧。

## 12. 启动成本

每一局是独立进程，因此以下启动工作执行了五次。

| 阶段 | mean | min | max |
|---|---:|---:|---:|
| STT snapshot resolution | 2493 ms | 2186 ms | 2856 ms |
| STT model load | 178 ms | 172 ms | 183 ms |
| STT warmup | 128 ms | 122 ms | 132 ms |
| TTS prewarm，10 entries | 8115 ms | 7966 ms | 8601 ms |

可见的测量启动成本合计平均约：

```text
2493 + 178 + 128 + 8115 = 10914 ms
```

### 12.1 STT resolution

虽然 snapshot 最终解析到正确的固定 revision，但 manifest 记录
`local_files_only=false`。这允许 Hugging Face resolution path 执行远端检查或
更完整的 cache resolution。

下一轮使用现有缓存时应显式添加：

```bash
--voice-local-files-only
```

然后比较 `model_resolution_ms` 是否显著下降。

### 12.2 TTS prewarm

system TTS 为 10 个 phrase 执行预合成，平均消耗约 8.1 秒，即每个 phrase
约 0.8 秒。收益是所有 25 个 LLM turn 均无回合内 synthesis 成本。

如果启动时间重要，可以进一步评估：

- 只预热实际由 LLM actor 使用的词表；
- 为 system TTS 增加跨进程磁盘 PCM cache；
- 复用常驻 TTS worker；
- 把 cold-start 与 steady-state 指标分开报告。

## 13. Root Cause Ranking

### P0：VAD 将短词延长为 3-5 秒 voiced interval

证据：

- voiced utterance mean：3989 ms
- voiced utterance p95：4985 ms
- 8/25 触发 max utterance
- Human response source mean：4708 ms
- 24/25 Human deadline miss

这是当前交互体验和 deadline 的主要问题。

### P1：`Zop` 的语音识别稳定性不足

证据：

- Expected `Zop`：0/5 成功
- 3 次被识别为 `Zap`
- 1 次重复 hallucination
- 1 次识别为 `It's up.`

需要使用保存的输入音频区分声学输入问题与 `tiny.en` 模型能力问题。

### P1：每局第一个 LLM 请求有明显 warm-up penalty

证据：

- 第一个 LLM turn mean：641 ms
- 后续 turn mean：309-350 ms
- 第一个请求的 prompt 反而最短

该问题影响 p95 和 max，但没有造成 deadline miss。

### P1：TTS cold-start 较大

证据：

- 10 phrase prewarm mean：8115 ms
- 五局重复执行

它不影响回合 latency，但影响开始游戏前的等待。

### P2：Playback drain 后存在约 106 ms cleanup

证据：

- 25 次均稳定复现
- min 103 ms，max 109 ms

这不是主要问题，但它占机器音频结束到下一轮麦克风 callback 的约 23%。

## 14. 局限性

1. 本次没有启用 `--voice-save-audio`。
   无法回放验证 VAD 为什么持续判定 voiced，也无法判断 `Zop/Zap` 错误来自模型、
   发音还是环境。

2. VAD last voiced frame 不是物理声学真值。
   因此 `human_last_voice_to_*` 只能表示系统内部 anchor 的关键路径。

3. 样本量只有 25 个 Human 和 25 个 LLM turn。
   p95/p99 对单个 outlier 很敏感，不应视为生产级 SLO。

4. 五局在同一台 Mac、同一组输入输出设备和相近时间内完成。
   结果不能直接推广到其他设备、房间噪声或 server load。

5. LLM 使用非 streaming API。
   当前无法测量真实 TTFT、token interval 或服务端 queue/model execution 分解。

6. Trace 没有保存 LLM model identifier 和 base URL。
   影响 artifact 的独立复现能力。

7. TTS 使用 system `say` 且全部缓存命中。
   本报告不能代表 Kokoro 或 cache miss synthesis 的回合性能。

## 15. 下一轮实验建议

### 15.1 第一优先级：VAD 审计

先进行一局受控测试：

- 使用安静环境和耳机；
- 启用 `--voice-save-audio`；
- 每个 human turn 只说一次目标词；
- 记录真实说话大致时刻；
- 回放 WAV，检查单词之后是否存在风扇、敲击、呼吸、扬声器残留或自动增益噪声。

WAV 包含敏感语音，应只保存在临时实验目录，不应提交到 Git。

### 15.2 暴露 VAD CLI 参数并执行 A/B

本次 trace 采集时，task CLI 尚未暴露以下 `VoiceInputConfig` 参数：

- `vad_aggressiveness`
- `end_silence_ms`
- `max_utterance_ms`
- `start_timeout_ms`

建议增加 CLI 参数后比较：

| 组别 | Aggressiveness | End silence | Max utterance |
|---|---:|---:|---:|
| A，当前基线 | 2 | 300 ms | 5000 ms |
| B，候选 | 3 | 200-300 ms | 1500-2000 ms |

候选值只用于实验，不应在没有 WAV 审计和漏检率数据时直接设为生产默认值。

报告完成后的第一项跟进已经增加 `--voice-max-utterance-ms`，同时保留通用默认
值 5000 ms。其余 VAD 参数仍未暴露。下一轮可以先用 1000 ms 和 1200 ms 做
短词任务 A/B，再决定是否需要调整 aggressiveness 或 end silence。

### 15.3 对比 STT model

在相同 WAV 上离线比较：

- `tiny.en int8`
- `base.en int8`
- 必要时 `small.en int8`

重点报告：

- `Zop` 与 `Zap` confusion matrix；
- transcript acceptance rate；
- ASR p50/p95；
- hallucination rate；
- 模型内存和启动成本。

不能使用游戏 expected answer 自动纠正 ASR，否则会污染游戏判定的独立性。

### 15.4 分离 LLM cold 与 warm

正式测量前：

1. 建立 SSH tunnel；
2. 请求 `/v1/models`；
3. 发送一次不计入游戏统计的 chat completion warm-up；
4. 再开始五局实验；
5. 同时保留 cold-run 数据作为单独报告。

### 15.5 改善复现信息

建议在 manifest 中增加：

- LLM base URL 的脱敏形式，例如 host/port；
- requested model identifier；
- server response 中返回的 model identifier；
- streaming/non-streaming；
- client retry count；
- 可选 request ID。

## 16. 建议验收标准

下一版 VAD/STT 实验可以暂定以下目标：

| 指标 | 当前 | 建议目标 |
|---|---:|---:|
| Human response source p95 | 5539 ms | < 1500 ms |
| Human deadline miss，3000 ms soft | 24/25 | 0/25 |
| Max utterance endpoint | 8/25 | 0/25 |
| ASR pipeline p95 | 146 ms | 保持 < 300 ms |
| Human valid rate | 76% | >= 95% |
| `Zop` success | 0/5 | >= 4/5 |
| LLM request-to-text p95 | 685 ms | 保持 < 1000 ms |
| Text-to-first-DAC p95 | 54 ms | 保持 < 100 ms |
| Speech playback success | 25/25 | 100% |

这些目标应在至少 50 个 Human turn 的样本上重新确认。

## 17. 最终结论

新增 breakdown instrumentation 已经达到预期目的：它不再只告诉我们“Human
turn 花了约 5 秒”，而是可以明确指出时间主要消耗在 VAD 认为输入持续 voiced，
而不是 faster-whisper 推理、游戏解析或终端输出。

当前系统的性能画像是：

- Human 输入：受 VAD/capture 主导，尚不满足 3000 ms deadline；
- ASR 推理：正常情况下约 127 ms，性能不是主要瓶颈；
- LLM：稳态约 0.3-0.35 秒，首请求存在 warm-up penalty；
- TTS startup：预热约 8.1 秒，但回合内全部 cache hit；
- 机器开始发声：LLM 文本后约 41 ms；
- 机器完整回答：LLM request 后平均约 1.0 秒 drain；
- 跨回合恢复麦克风：音频 drain 后平均约 466 ms。

下一步应优先完成带 WAV 的 VAD 审计和 VAD 参数 A/B，而不是先优化 ASR
推理或 LLM JSON 处理。
