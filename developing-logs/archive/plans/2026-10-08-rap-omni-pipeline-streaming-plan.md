# RAP 全流程进 SGLang-Omni 阶段图，以及 MOSS 流式改造：计划（2026-10-08）

> **2026-10-08 更新：写词部分已被[歌词流计划](2026-10-08-rap-lyric-stream-plan.md)替代。**
>
> - Qwen 写词不再做成 MOSS 流水线里的一个 stage，而是独立的会话服务，按 bar 连续往后写。D2 由它直接实现；chunk 改成 1 个 bar。
> - **并入新计划：** MOSS 续写（第 3 阶段，成为新计划的第 6 阶段）、0b 的续写效果测试（成为新计划的 0a），以及 Mac 端的对应实现。
> - **本计划中只剩下“把音频部分组织成 SGLang-Omni 阶段图”，作为以后的可选改造。**


## 背景与目标

对应组会 slides 讨论中的两个想法：

- **想法 2：** 做真正的流式，让 MOSS 边生成，MMS 和 R3 边对齐、边拉伸。
- **想法 3：** 按 omni 的阶段图来组织整条流程，Qwen 写词成为图里的一个新 stage，也就是 Thinker → Talker 的结构。

目标：

1. **Qwen 写词进阶段图。** render server 发一次请求，SGLang-Omni 流水线内部完成以下几步：
   - 写词；
   - 选词；
   - MOSS 合成；
   - MMS 对齐。
2. **改造 MOSS，支持续写。** 新一段人声接着上一段的音频往下说：
   - 两个 chunk 之间的嗓音和韵律能连上；
   - 一个 chunk 里可以按 bar 拆开合成。
3. **按 bar 流水线化。** bar 0 写完就开始合成，同时写 bar 1。服务器按 bar 分批返回，Mac 也按 bar 拉伸、混音，按 bar 兜底。
4. **Mac 端有对应的实现。** SGLang-Omni 不能在 Mac 上跑，所以要在 Mac 上用 MLX 实现同一张图、同一个接口。

先说清楚收益在哪里，避免期望错位：

- 现在 H200 准备一个两小节 chunk 只要 1.09 s，Mac 是 3.3 s，都远小于 5.333 s 的 chunk 周期。**稳态吞吐已经够了，流式化不是为了它。**
- 这次的收益在四个方面：
  - **第一小节更早可用：** H200 上预计从约 1.1 s 提前到约 0.7 s。
  - **按 bar 兜底：** bar 1 迟到时只替换 bar 1，不用丢掉整个 chunk。
  - **跨 chunk 连贯：** 现在每个 chunk 都只靠参考音色独立合成，接缝处的嗓音和韵律可能会跳。
  - **为缩短前瞻做准备：** 前瞻从 2 小节缩到 1 小节后，听众的输入（例如换主题）能更快体现在说唱里。
- **代价：** 一个 chunk 的总耗时会略增。按 bar 合成时，每次调用都要重新付一次 delay pattern 约 32 步的预热。

## 调研结论（2026-10-08，对照固定版本 SGLang-Omni 0.1.4 `af3ab61` 的源码和上游文档）

| 问题 | 结论 | 对计划的影响 |
|---|---|---|
| MOSS-TTS v1.5 能不能边收文本边说 | **不能。** 完整文本必须先放进 user 消息，生成音频时文本通道只出 `gen_slot` 和 `delay_slot` 两种控制 token（`model_runner.py:404-415`） | “边写边说”到 token 级在 v1.5 上做不到，退一步做按 bar（按行）的流水线 |
| MOSS 能不能接着上一段音频续写 | **模型和引擎支持，接口没有暴露。** 引擎会识别续写的 prompt（`is_continuation`、`assistant_prefix_rows`），vocoder 也会裁掉前缀（`delay_pattern.py:59-63`）。但 preprocessing 写死了 `mode="generation"`，只构造 user 消息 | 这就是“魔改 MOSS”的切入点：打补丁，把上一段的音频 codes 作为 assistant 前缀传进去 |
| MOSS 能不能流式输出音频 | **可以。** 0.1.4 支持 `stream=true` 加 PCM，vocoder 是滑动窗口重解码，codec 是因果的 | 不过 R3 需要一个 bar 的完整音频，所以按 bar 产出比按帧流式更有用 |
| 有没有能逐 token 接收文本的 MOSS | 有，是 **MOSS-TTS-Realtime**（1.7B，`push_text` 接口，文本和音频交错生成）。但它没有时长控制，也没有验证过说唱节奏；SGLang-Omni 不支持它 | 只做探索，不进主线（第 6 阶段） |
| 阶段图怎么定义 | stage 列表加 `next`、`stream_to`、`route_fn`、`wait_for`，只能用 Python 定义（YAML 只能覆盖已有 stage 的参数）。纯 CPU stage 用 `SimpleScheduler` 即可 | 我们在自己的包里写一个 `PipelineConfig`，通过 `register_config` 注册，不用 fork SGLang-Omni |
| 写词 LLM 能不能做成引擎 stage | 没有现成的纯文本 LLM stage，也不支持 `n`。可以参考的先例：`minimax_music3` 用 `Qwen3ForCausalLM` 加 `model_arch_override`；`MiniMaxMusic3Scheduler` 为一个 payload 生成多个兄弟请求 | 先用“CPU stage 调外部 vLLM”搭通拓扑，再换成引擎 stage |
| 同一张卡上放两个引擎 | 两个进程组各自声明 `gpu_memory_fraction`，加起来不超过 1。MOSS 引擎要用 `engine.mem_fraction_static` 控制显存 | 第 2 阶段要测显存怎么分 |
| 选词逻辑有多重 | 纯 CPU，只依赖 `pronouncing`，一个 chunk 约 40 个候选、评估约 7 ms | 可以原样搬进一个 CPU stage |
| 现在的缺口 | producer 指纹不包含写词模型、prompt、采样参数和 ScoreWeights | 写词进流水线时要一起补上 |

## 目标架构

```text
render server（变成薄客户端：协议、缓存、幂等、deadline、producer 指纹、交付给 Mac）
   │  一次请求：bars、flow、主题、context、seed、预算、续写引用
   ▼
SGLang-Omni 流水线 "rap_chunk"（H200 一张卡）
   lyric_engine ──► lyric_select ──► moss_preprocessing ──► moss_tts_engine ──► moss_vocoder ──► align ──► 结果
   Qwen2.5-7B        CPU：解析、        续写：上一 bar /        delay pattern       流式解码          MMS
   引擎 stage        评分、补救、       上一 chunk 的                                                 按 bar 产出
   每个 bar n=16     按 bar 选定       音频 codes 作前缀
        ▲                 │
        └── 补救 wave ─────┘  （route_fn 回环；不支持的话，lyric 合并成一个 stage）

按 bar 流水线（第 4 阶段）：
   bar 0：写词 → 选定 → MOSS（续写上一 chunk 的结尾）→ MMS → 先返回
   bar 1：      写词（context 带上已选定的 bar 0）→ 选定 → MOSS（续写 bar 0）→ MMS → 再返回
```

Mac 端：R3 拉伸、混音和兜底都保持在 Mac 上，与 v2 相同，只是从按 chunk 处理改为按 bar 处理。

### H200 上的时间预估（按 10-01 的实测数据推算）

| | 现在（两小节一次合成） | 按 bar 流水线 |
|---|---:|---:|
| 写词 | 147 ms（两个 bar 并发） | 每个 bar 约 100 到 150 ms，串行，但和 MOSS 重叠 |
| MOSS | 769 ms（约 101 步） | 每个 bar 约 67 步，约 0.5 s；两个 bar 串行，约 1.0 s |
| 第一小节可用 | 约 1.09 s（随整个 chunk 一起） | **约 0.7 s** |
| 整个 chunk 可用 | 约 1.09 s | 约 1.25 到 1.3 s |
| Mac 本地（同样按比例推算） | 3.3 s | 第一小节约 1.9 s，整个 chunk 约 3.4 s |

这些都是估算，第 0 阶段要实测。

## 开始前需要你决定的事

| # | 决定 | 我的建议 | 影响哪个阶段 |
|---|---|---|---|
| D1 | 我对“想法 2、想法 3”的理解（见上面“背景与目标”）是否正确 | — | 全部 |
| D2 | 选词语义：现在两个 bar 独立生成、事后配对；改成先定 bar 0，再写 bar 1（bar 1 的 prompt 带上 bar 0） | **改。** 押韵和连贯会更好，也是按 bar 流水线的前提。原来的“两 bar 独立”作为一个开关保留 | 1、4 |
| D3 | 跨 chunk 续写：chunk N+1 的人声接着 chunk N 的音频说 | **做。** 但请求的身份要带上“前缀”的标识，见第 3 阶段 | 3 |
| D4 | 真正的逐 token 文本流式（MOSS-TTS-Realtime，或者微调 v1.5） | 本计划只做探索，不进主线 | 6 |
| D5 | 代码放在哪里 | 我们仓库里新建一个包 `src/streammuse_omni/`，放流水线配置、stage 和 scheduler。改 MOSS 内部的部分沿用补丁文件的方式，和 10-01 的延迟补丁一样，固定版本、写进运行时身份 | 1 到 3 |
| D6 | 给 Mac 的协议 | 第 3 阶段之前保持 v2 不变；第 4 阶段升到 v3（按 bar 分批返回），v2 保留用于回滚 | 4、5 |

## 第 0 阶段：先验证三件事（不写生产代码）

这三项互相独立，可以并行做。

### 0a. SGLang-Omni 能力验证（H200，一张空闲卡）

- 在自己的包里定义一个最小的 `PipelineConfig`，两个 stage：一个 CPU stage 加 MOSS 的三个 stage。用 `register_config` 加 `PYTHONPATH` 启动，确认不用 fork 也能跑通。
- **`route_fn` 回环：** 下游 stage 能不能把请求送回上游 stage，用来做补救 wave。不能的话，写词和选词合并成一个 stage，补救在 stage 内部循环。
- **多个终止 stage 时的结果合并：** 文本结果加音频结果，看自定义的 `result_builder` 或新 HTTP 路由要怎么写。现在 `launcher._run_server` 不能加路由，需要照着复制大约 40 行。
- **Qwen 引擎 stage：** 照着 `model_arch_override` 和兄弟请求调度的先例，实现 n=16。测同一张卡上 Qwen 和 MOSS 两个引擎的显存怎么分，以及 Qwen 的延迟。
- 注意：协调器的准入上限按第一个引擎 stage 计算（`mp_runner.py:48-77`），Qwen 放在第一个之后，这个上限要重新核对。

### 0b. MOSS 续写的效果（关口）

- 先离线做：用 HF remote code，单卡、单进程，用 `build_assistant_message(audio_codes_list=[上一段的 codes])` 实现续写。
- 对比三种合成方式，用同一组 20 条两小节语料、同一个参考音色：

| 方式 | 说明 |
|---|---|
| A：现状 | 两小节一次合成 |
| B：按 bar 独立合成 | 每次只靠参考音色，`token_count` 为 33 |
| C：按 bar 续写 | bar 1 接着 bar 0 的音频；跨 chunk 时，bar 0 接着上一个 chunk 的 bar 1 |

- 指标：
  - Whisper WER；
  - 说话人相似度：相邻 bar、相邻 chunk 之间 ECAPA embedding 的余弦相似度；
  - 接缝处的音量和音高跳变；
  - MMS 测到的起点与目标节拍的偏差；
  - 延迟：前缀 prefill 加 delay 预热，要实测，不能只靠估算；
  - 盲听 5 对，需要你来听。
- **关口：** C 的 WER 不高于 A 的 1.1 倍，说话人相似度和接缝不比 A 差，盲听分辨不出或更好。通过就走第 3、4 阶段。不通过的话，只做跨 chunk 续写（第 3 阶段）和第 1、2 阶段，chunk 内保持两小节一次合成。

### 0c. MOSS-TTS-Realtime 小测（可选，只用来决定 D4）

- 用 `push_text` 喂说唱歌词，看能不能卡住节拍（它没有 `token_count`）、WER 怎么样，以及第一包延迟。
- 只写结论，不接进系统。

## 第 1 阶段：搭出阶段图（写词仍调外部 vLLM）

1. **新包 `src/streammuse_omni/`：**
   - `RapChunkPipelineConfig`：`lyric → moss_preprocessing → moss_tts_engine → moss_vocoder → align`。MOSS 的三个 stage 直接复用 SGLang-Omni 现成的类。
   - `lyric` 是一个 CPU stage，从现在的 planner 搬过来：
     - prompt 构造（`_build_independent_messages`、`format_flow_for_prompt`）；
     - n 选 1 的采样参数（n 为 16 或 4、`max_tokens` 32、temperature 1.0）；
     - 解析、规范化、去重、把数字转成英文单词；
     - `CmuProsodyAnalyzer`、硬门槛加评分（`rank_candidates`、ScoreWeights）；
     - wave 和补救循环，遵守“deadline 减去渲染预留”；
     - `_select_pair` 和 `chunk_render_request`。

     这些代码原样复用 `streammuse.application.rap`，stage 只是一层适配，不复制逻辑。stage 输出新的 `OmniRequest(inputs={text, references}, metadata.tts_params={token_count, seed, …})`。
   - `align` 是 MMS stage。它和 vocoder 放在同一个进程（vocoder 的流式 chunk 是 dict，只能在同一进程内传递），也可以把 vocoder 改成非终止 stage。输出就是 v2 现在的 source WAV、`source_onsets` 和 `onset_confidence`。
   - **新 HTTP 路由 `POST /v1/rap/chunk`：** 请求体就是现在的 `RemoteRapChunkRequest`，结果包含 source、起点、`selected_bars`、候选 ledger 和各阶段计时。
   - **结构化失败：** 至少有 `no_valid_candidates`（带 ledger）、`budget_exhausted` 和 `cancelled`，render server 用它们映射出现在的错误码。
2. **render server 新增后端 `--rap-pipeline-backend sglang-omni`：**
   - 去掉 planner 和 vLLM 客户端；缓存、幂等、deadline 和协议 v2 都不变。
   - **producer 指纹补上写词相关的内容：** 写词模型和 revision、prompt 版本、采样参数、ScoreWeights、候选 profile，以及流水线代码的版本。

**测试**

- 用固定的假生成器，比较流水线的 lyric stage 和现在的 planner：ledger、`source_order`、选中结果和渲染请求逐项相同。
- `/v1/rap/chunk` 的契约测试，包括每种失败码。
- 新增的指纹字段，任何一项不同，指纹就不同。

**验收**

- `client_simulation` 跑 24 个 chunk，加 `--protocol v2`、gentle_sparse，Mac 端 24 / 24 被接受。
- 服务器端合计时间不比现在的 1.09 s 差，误差在 50 ms 以内。
- 不需要再单独部署 vLLM 服务。

## 第 2 阶段：Qwen 换成流水线里的引擎 stage

1. **`lyric_engine`：** SGLang 自回归引擎，Qwen2.5-7B-Instruct，通过 `model_arch_override` 加载。
   - 在它前面加一个很薄的 CPU stage，做 chat template 和分词。
   - 照着 `MiniMaxMusic3Scheduler` 写调度：一个 payload 生成 n 个兄弟请求，各自有自己的 seed，共享 prompt 前缀，靠 radix cache 只做一次 prefill。
2. **`lyric_select`：** CPU stage，接收 n 条候选，解析、评分。需要补救时，用 `route_fn` 送回 `lyric_engine`；0a 验证不能回环的话，就用合并成一个 stage 的方案。
3. **显存：** Qwen 和 MOSS 在同一张卡上，按 0a 的结果分配，例如 Qwen 0.2、MOSS 0.5，再留一部分给 MMS 和其他开销。
4. **对比第 1 阶段：**
   - 写词的 p50 和 p90；
   - 显存；
   - 两个并发会话时的表现。

   结果更差就保留第 1 阶段的方案，第 2 阶段只作为可选项。

**验收**

- 写词的 p50 不高于外部 vLLM（147 ms）。
- 错误批次为 0。
- 同一组假生成器的测试仍然一致。

## 第 3 阶段：MOSS 续写补丁（跨 chunk 连贯）

1. **SGLang-Omni 补丁：** preprocessing 支持 `continuation`。
   - 把前缀音频 codes 和前缀文本构造成 assistant 前缀消息，引擎的 `is_continuation` 路径会接着生成。
   - 修正流式路径会重复发出前缀行的问题（`request_builders.py:596-615`）。
   - 补丁文件和 10-01 的做法一样：固定版本，写进运行时身份，preflight 检查。
2. **前缀放在服务器端：**
   - 流水线里维护一个会话存储，按 `session_id` 保存上一个 chunk 最后一个 bar 的 codes，只保留最近几个、按时间过期。
   - 请求只带 `continue_from = {request_id, source_sha256}`。
   - 前缀找不到时（比如服务重启），退回到不续写，并在 manifest 里标明。
3. **身份和缓存：**
   - 续写使 chunk 的输出依赖前一个 chunk，所以前缀的标识要进 `request_id`。
   - 重试时同一个前缀得到同一个结果；前缀变了，就是另一个请求。
4. **radix cache：** 生成出来的行不会进缓存，所以前缀每次都要重新 prefill。需要实测这部分开销，预计只有几十 ms。

**验收**

- 用 0b 的连贯性指标：相邻 chunk 之间的说话人相似度提高，接缝跳变减小。
- WER 不变差。
- MOSS 耗时增加不超过 100 ms。
- 盲听。

## 第 4 阶段：按 bar 流水线，协议 v3（需要先过 0b 的关口）

1. **流水线：**
   - `lyric` 先定 bar 0，立刻送进 MOSS（续写上一个 chunk 的结尾）。与此同时写 bar 1，prompt 带上已选定的 bar 0。
   - bar 1 选定、并且 bar 0 的音频完成之后，再续写 bar 1。
   - 每个 bar 的 `token_count` 为 33，时长靠 R3 精确对齐。
   - MMS 按 bar 对齐。
2. **协议 v3：** 一个请求，按 bar 分批返回。用分块的 HTTP 响应：每个 bar 一部分，包含这个 bar 的 manifest、source 和起点；最后再发一份整体 manifest。v2 保留。
3. **Mac 端：**
   - `RemoteMossChunkPreparationStrategy` 收到一个 bar 就拉伸、混音、提交一个 bar。
   - **按 bar 兜底：** bar 1 赶不上冻结时间时，只把 bar 1 换成 eSpeak。
   - 会话日志记录每个 bar 的状态。
4. **控制器：** 新增“前瞻 1 小节”的实验开关。默认仍是 2 小节，等数据好了再讨论要不要改。

**验收**

- H200 上第一小节可用的 p50 不超过 0.8 s，整个 chunk 不超过 1.4 s。
- 真实会话 24 小节，underrun 为 0。按 bar 统计的兜底率不比现在差。

## 第 5 阶段：Mac 端的对应实现

1. SGLang-Omni 不能在 Mac 上跑。在 Mac 上写一个轻量的进程内流水线服务，暴露同一个 `/v1/rap/chunk`、同一个协议 v3。各 stage 用：
   - 写词：mlx_chat，沿用共享 prefill；
   - MOSS：mlx-audio，加上续写。要先确认 mlx-audio 的 MOSS 移植支持 assistant 前缀，不支持就照着补；
   - 对齐：MMS（MPS 加 CPU）。
2. stage 之间的数据格式与 H200 完全一致。共用同一套 lyric 和 select 代码，只有 MOSS 和写词的后端不同。
3. **一键启动脚本：** 从三个服务变成一个流水线服务，加一个 MLX chat 服务。

**验收**

- Mac 本地 24 个 chunk，加 gentle_sparse 和 v3，24 / 24 被接受。
- 第一小节 p50 不超过 2.0 s，真实会话 underrun 为 0。
- WER 不比 Mac 现在的 1.72% 差太多，以 0b 的门槛为准。

## 第 6 阶段（探索，不承诺）：逐 token 文本流式

只有在 0c 的结论积极，并且你决定做 D4 时才开始。

- **方向一：** MOSS-TTS-Realtime 接 Qwen 的流式输出（`push_text`）。要解决没有时长控制、卡不住节拍的问题，可能需要用它的微调方案拿说唱数据微调。
- **方向二：** 把 v1.5 微调成文本和音频交错的格式。工作量大，没有先例。

## 风险

| 风险 | 应对 |
|---|---|
| SGLang-Omni 内部接口随版本变化：上游已经到 0.1.7，我们固定在 0.1.4 | 只依赖公开的类和 `register_config`；对 MOSS 的改动集中在一个补丁文件里；升级时重新做第 0 阶段 |
| 续写的效果不好：嗓音漂移、越说越快 | 0b 关口；续写可以只用在跨 chunk |
| 按 bar 合成每次都要付 delay 预热，总时长变长 | 已计入预估；如果整个 chunk 超过 1.4 s，就只把按 bar 用在 Mac 端的兜底粒度上 |
| 一张卡上放两个引擎，显存或调度互相干扰 | 0a 实测；必要时 Qwen 放另一张卡（`process` 和 `gpu` 都可配） |
| 续写打破“每个 chunk 独立、可以重放”的性质 | 前缀标识进 `request_id`；前缀缺失时显式退回，并在 manifest 里标明 |
| Mac 上 mlx-audio 不支持续写 | 第 5 阶段先验证，必要时给 mlx-audio 打补丁 |

## 测量和证据

- 沿用 10-01 的测量协议：
  - 同一张卡上 ABBA；
  - 24 个 chunk 加 2 个热身；
  - 只按自己记录的进程组启停服务。
- 新增的指标：
  - 第一小节可用时间；
  - 按 bar 的兜底率；
  - 说话人相似度和接缝跳变。
- 每个阶段结束，在本计划末尾追加实施记录。

## 实施记录

（每个阶段完成后在这里追加：日期、提交、实测数字、偏离计划之处。）
