# RAP 流程详解：旧 MOSS 后端与 SGLang-Omni 新后端

更新日期：2026-09-11。依据：当前工作区的实现，以及当天已经完成的 H200 性能实测。

> **2026-10-01 更新：R3 已经移到 Mac（协议 v2，默认）。** 现在服务器做完 MOSS 和 MMS 就返回：原始 MOSS 人声（PCM16），加上每个音节测到的发音起点。Mac 用同一份代码重建渲染请求，在本机跑 Rubber Band R3（默认 `gentle_sparse_r3`），然后加鼓、播放。输入相同时，Mac 上的 R3 和服务器上的 R3 输出逐字节一致。下文凡是写“R3 在 H200 / 服务器做”的地方，都只适用于 `--rap-protocol v1`；v1 仍然保留，用来回退。细节见 [2026-10-01 的优化计划](../plans/2026-10-01-rap-server-latency-optimization-plan.md)。同一天还做了以下改动：
>
> - 两个 bar 的 Qwen 候选改为并发生成（`--concurrent-bar-generation`），补救 wave 也是并发的；vLLM 改为 `--max-num-seqs 32`。
> - R3 的输出长度误差在 20 ms 以内时，直接补零或截断，不再整段重跑，最多跑两次。
> - MOSS 打了一个延迟补丁（`patches/`）。
> - Opus 压缩级别改为 5。
>
> 服务器到首字节的中位数约从 1.37 秒降到 1.09 秒，见 profiling 报告第 10 节。

本文讲的是项目的 **remote MOSS RAP demo**，不是仓库里所有历史音频实验。这里的“旧后端”指 `inprocess` 常驻 MOSS 后端；“新后端”指可选的 `sglang-omni` 后端。Mac 上的本地 eSpeak 是另一条兜底路径，不要把它和这里的旧后端混为一谈。

## 1. 先记住这一句话

**Qwen 写词，flow 模板规定目标节奏，MOSS 把词说出来，MMS 找到实际发音位置，R3 调整时间，Mac 按音乐时钟混音播放。**

SGLang-Omni 主要替换的是中间 **“MOSS 把词说出来”所用的推理运行方式**，不是把整套 RAP 系统推倒重来。

```text
两套后端共用的主流程：

主题 + flow 模板 + 已播放歌词
    -> Qwen 生成候选歌词
    -> 音节分析、打分、挑选两句
    -> 给音节安排目标时间
    -> MOSS 生成完整的两小节原始人声    <--- 主要替换这里
    -> MMS 找实际发音时间
    -> 打包并传给 Mac（v2：原始人声 + 音节起点）
    -> Rubber Band R3 拉伸到目标节奏与时长（v2 在 Mac 做，v1 在服务器做）
    -> Mac 加鼓、排入播放队列、按时播放
```

目前的状态也要一起记住：**新后端已接入，服务器热运行提速已实测；生产默认仍是 `inprocess`，并未自动切换。**

## 2. 先分清这些角色和单位

### 2.1 谁负责什么

| 组件 | 所在位置 | 主要职责 | 不负责什么 |
|---|---|---|---|
| 浏览器监控页 | Mac | Start/Stop/Reset、展示运行状态 | 不持有音频播放时钟，不直接生成声音 |
| RAP demo / controller | Mac Python 进程 | 场景、主题、flow、提前准备、截止时间、兜底、播放队列 | remote 模式下不在 Mac 调 Qwen 写正式歌词 |
| RAP render server | H200，示例端口 `8020` | 接请求、组织写词与渲染、校验和打包 | 不决定 Mac 应该何时移动音乐时钟 |
| Qwen / vLLM | H200，示例端口 `8001` | 生成文字候选 | 不生成最终人声音频 |
| MOSS-TTS | H200 | 根据文字、参考音色和风格生成原始人声 | 不保证每个音节严格卡到 flow 的 tick |
| MMS forced alignment | H200 | 对照已知歌词，在原始音频中定位发音 | 不是重新写词，也不是自由语音识别 |
| Rubber Band R3 | Mac（协议 v2，默认）；H200（协议 v1） | 连续、尽量保音高地调整音频时间 | 不是一个新的生成模型 |
| eSpeak + 本地鼓轨 | Mac | 提前准备兜底人声、鼓轨及混音 | 不是 SGLang-Omni 的模型后端 |

端口和 Qwen 型号是可配置项。`8001 / qwen-rap` 是当前运行文档的示例，不代表必须用某个固定 Qwen 版本；之前的 `8101 / Qwen3.6` 配置也属于“写词服务”这一层。

### 2.2 bar、tick、slot、chunk 是什么

当前 RAP 节奏按 4/4 拍组织：

- **beat**：一拍，时长为 `60 / BPM` 秒。
- **bar**：一小节，4 拍。
- **tick**：当前网格每拍分 4 个 tick，所以每小节 16 个 tick。
- **slot**：flow 模板中预定放一个音节的位置，带目标 tick、重音等信息。
- **chunk**：remote MOSS 每次准备的连续两小节，一次选两句、合成整段人声。

**16 个 tick 不等于必须写 16 个音节。** 模板可以只在部分 tick 上安排音节；之前测速使用每小节 9 个音节，是该批测试模板的选择，不是系统统一要求。

以 90 BPM 为例：

```text
1 beat  = 60 / 90             = 0.666667 秒
1 tick  = 60 / 90 / 4         = 0.166667 秒
1 bar   = 4 * 60 / 90         = 2.666667 秒
1 chunk = 2 * 4 * 60 / 90     = 5.333333 秒

24 kHz 最终人声帧数 = 24000 * 5.333333 = 128000
```

后端花 0.93 秒生成这段声音，不代表声音只有 0.93 秒；它仍然播放 5.33 秒。

## 3. 两套部署长什么样

### 3.1 旧后端：MOSS 在 render server 内部

```text
Mac
  浏览器控制页 -> RAP demo/controller -> 本地混音和音频设备
                         |
                         | SSH tunnel，只转发 RAP 接口
                         v
H200: RAP render server :8020
  |
  +-> Qwen vLLM :8001 -> 候选歌词
  |
  +-> planner -> 两句歌词 + 目标音节时间
  |
  +-> MossAlignedPhraseRenderer
         |
         +-> PersistentMossSynthesizer
         |      -> 进程内常驻 MOSS / processor / codec
         |      -> Transformers model.generate
         |      -> source.wav
         |
         +-> MMS -> R3 -> vocal.wav
  |
  +-> artifact / response package -> Mac
```

“进程内”不等于“每次请求重新加载模型”。旧实现已经会在启动时加载并预热，之后复用常驻模型；前一次测速也按这个真实热运行基线测量。

### 3.2 新后端：MOSS 交给独立 SGLang-Omni 服务

```text
Mac
  浏览器控制页 -> RAP demo/controller -> 本地混音和音频设备
                         |
                         | 同一个 SSH tunnel / 同一个 RAP 协议
                         v
H200: RAP render server :8020
  |
  +-> Qwen vLLM :8001 -> 候选歌词             [不变]
  |
  +-> planner -> 两句歌词 + 目标音节时间       [不变]
  |
  +-> MossAlignedPhraseRenderer
         |
         +-> SglangMossSynthesizer
         |      -> HTTP POST :8030/v1/audio/speech
         |      -> SGLang-Omni 独立服务进程组
         |           preprocessing -> tts_engine -> vocoder
         |      <- 完整 WAV
         |      -> 校验并保存 source.wav
         |
         +-> MMS -> R3 -> vocal.wav          [共用]
  |
  +-> artifact / response package -> Mac     [共用]
```

Mac 依然只认识 RAP render server，不需要改成直接调用 SGLang-Omni。Qwen 和 SGLang-Omni 都留在 H200 的 loopback 接口，通常不向 Mac 转发 `8001` 或 `8030`。

注意：移出的是 MOSS 推理。render server 仍然保留 MMS 对齐等工作，因此新方案不意味着 render server 从此完全不需要 GPU。

## 4. 前半段共用：从主题到两句可渲染的歌词

### 4.1 Mac 先决定“下一段要讲什么、按什么 flow 讲”

启动参数和 scenario 提供 BPM、段落主题、flow 模板、备用歌词等。当前 demo 的正常入口是这些场景配置和会话控制，不是让浏览器不断上传一段麦克风声音来生成 RAP。

controller 每次处理两个 bar。在请求远端之前，先用本地备用歌词和 eSpeak 准备好这两个 bar 的兜底音频。这样远端超时后，不需要再临时开始合成兜底。

随后发给 H200 的 `RemoteRapChunkRequest` 主要包含：

- session、request、chunk 和 bar 的标识；
- 两个 bar 的主题及 flow 模板；
- BPM、随机 seed、候选生成策略；
- 最近已提交、实际采用的歌词上下文；
- 当前还剩多少时间，即 `remaining_budget_ms`。

**Mac 此时没有提前选好远端主路径的两句歌词。写候选和挑选发生在 H200。** 请求预算是在本地兜底准备之后计算的，不能把完整 lookahead 时间都当成远端可用预算。

### 4.2 H200 接收请求并建立统一截止时间

入口是 `POST /v1/rap/chunks/render`。服务校验请求、`Idempotency-Key`、运行就绪状态及产物命名空间，再组织后续工作。

同一次被接受的执行共享一个 deadline：生成候选、渲染、产物提交等阶段都消费这份预算，不是每到一层就重新获得一整段超时时间。

重复请求可以复用同一执行或同一已完成产物；不同后端和 producer 配置通过 fingerprint 隔离，避免切到新后端后误拿旧 WAV 当成新结果。

### 4.3 Qwen 为每个 bar 生成候选

`ChunkCandidatePlanner` 调用 `IndependentChoiceCandidateGenerator`，后者通过服务器本地的 chat model client 访问 Qwen/vLLM。

Qwen 的任务是生成符合主题、目标音节数及上下文的候选句子，不是直接输出一个音频文件。当前 planner 对两个 bar 分别发起候选生成，并在必要时补候选；它不是“Qwen 一次生成整首歌”。

当前 `realtime_default` 策略是每个 bar 初始 16 条、补救批次 4 条、总上限 20 条、目标至少 3 条有效候选，并为后续渲染保留 3000 ms。预算不足时不会无限补候选。这些是当前策略参数，不是 MOSS 模型结构的要求。

### 4.4 用文本分析筛掉不适合这个 flow 的句子

候选经过清理、数字读法处理、空文本及重复过滤后，由 `CmuProsodyAnalyzer` 分析单词、音节、音素和词汇重音。

CMU 字典没有的词会走启发式后备，并记录 OOV 信息；不能理解成“所有生词都被直接拒绝”。字典分析和真实说出来的读法也不一定完全一致，这正是后面仍需检查音频的原因之一。

关键硬约束是：**候选的分析音节数必须和 flow 模板的 slot 数对应。** `align_exact` 将音节一一放到 slot 上，再计算重音匹配、词边界、主题覆盖、上下文衔接、重复程度等规则分数。

这些分数主要是确定性的文本代理指标，不是另一个大模型在判断“这句 RAP 一定好听”。

两组候选都有可选结果后，planner 比较两句的组合。当前组合选择优先看两句平均总分，再用词汇衔接、押韵等打破平局，而不是把所有条件都当成一套复杂的联合语义推理。

如果某个 bar 始终没有可选候选，服务返回失败，由 Mac 使用已准备的本地兜底；不会伪造一次成功的 MOSS 渲染。

### 4.5 生成两小节的“目标时间表”

选中两句以后，构造 `TwoBarRenderRequest`：两句用换行连接，每个音节带上所属单词、音素、重音、边界，以及绝对 tick、chunk 内 tick 和目标秒数。

```text
目标秒数 = 音节在 chunk 内的 tick * 60 / BPM / 4

例如 90 BPM，某音节放在 chunk 内 tick 6：
目标起始时间 = 6 * 0.166667 = 1.0 秒
```

到这里得到的是 **“希望它什么时候发音”**，还没有任何证据表明 MOSS 真会在这些位置发音。

## 5. 旧后端内部：两句歌词怎样变成 source.wav

入口是 `PersistentMossSynthesizer.synthesize()`。两句作为一个连贯 phrase 合成，不是逐音节单独合成后再拼接。

### 5.1 启动时加载并预热

旧后端通过 `create_runtime()` 加载 MOSS 模型、processor 和音频 codec，准备参考音频，并执行真实语音预热。运行期间保留这些模型对象。

### 5.2 每次请求构造语音生成输入

输入包括两句歌词、英文语言设置、参考音频和 RAP 风格指令。当前共用指令为 `clear, rhythmically spoken rap with restrained pitch`，意图是清晰、有节奏、音高变化克制的 RAP 式说唱。

processor 将文本和参考声音整理成模型所需的输入。旧路径会在这条请求处理链路中处理、编码参考音频。

此外还给一个粗粒度音频 token 目标：

```text
token target = round(目标秒数 * 12.5)
90 BPM 两小节：round(5.333333 * 12.5) = 67
```

这里的 token 预算不是 Qwen 的歌词 token 数，也不是 67 个音节；它用于 MOSS 的音频生成长度控制。`max_new_tokens` 是另一个生成上限，不能和 token target 混用。

**完整的逐音节目标时间表没有直接作为强制时间戳输入给 MOSS。** BPM 通过粗时长预算影响生成，flow 的细节仍主要靠后面的对齐和拉伸落实。

### 5.3 自回归生成音频编码，再解码为波形

旧实现调用 Transformers 的 `model.generate()`，逐步产生音频编码序列；随后由 processor/codec 解码为可播放波形，规范化并保存为 `source.wav`。

这份 source 是 MOSS 按自己的语速和停顿生成的原始人声。参考音频影响音色，风格指令影响表达，但它可能比两小节稍长，音节起点也可能和 flow 不一致。

在线路径是一次合成尝试，不会在这个阶段偷偷反复生成直到听起来合适。输出必须通过采样率、声道、WAV 等校验，才会交给下游。

### 5.4 旧路径的性能特点

模型已常驻，所以主要开销不是反复加载权重，而是参考处理、自回归推理、采样、codec 解码和结果写入。运行时的 attention 实现也影响耗时。

这次实测旧配置使用 BF16 主模型、SDPA 和 FP32 HF codec。它描述的是本次基线环境，不代表所有可能的旧环境都必然使用相同 attention 后端。

## 6. 新后端内部：换了执行引擎，但仍是 MOSS

入口改成 `SglangMossSynthesizer.synthesize()`，实现同一个合成接口。上层 planner 和下层 MMS/R3 仍然接收原来的数据结构。

### 6.1 先启动独立服务，再让 RAP 连接它

SGLang-Omni 使用独立、固定版本的环境加载 MOSS。RAP render server 启动时检查服务健康状态和模型，并做真实 WAV 预热；不能只凭端口已监听就认为可用。

前一次实测固定的是 SGLang-Omni `0.1.4`、commit `af3ab61a665427ff3fe3d67e32df6223524ed129`。下面对其内部组织的描述针对该版本，不承诺后续上游版本完全相同。

### 6.2 Adapter 把同一份合成意图转成 HTTP 请求

请求发往 `POST /v1/audio/speech`，包含：

- `input`：同样的两句歌词；
- `ref_audio`：SGLang 服务能读取、且在允许目录内的参考音频 URI；
- 语言、指令、token target、采样参数和 seed；
- `response_format="wav"`，以及 **`stream=false`**。

两边共享生成参数解析，尽量保持对比条件一致。但相同 seed 不保证两套采样实现生成逐字节相同的声音。

Adapter 也发送 `ref_text`。本次固定的上游版本虽然接收它，却没有把它传入 MOSS 的 `build_user_message()`，所以不能说新版因为参考文本 conditioning 更丰富而必然更好。

### 6.3 SGLang-Omni 内部的三个逻辑阶段

| 阶段 | 输入与输出 | 在这里做什么 |
|---|---|---|
| `preprocessing` | 文字、参考 WAV -> 模型输入 | 读取参考、codec 编码、构造 prompt；可缓存相同参考的编码 |
| `tts_engine` | 模型输入 -> 音频编码序列 | MOSS 专用推理引擎完成 prefill 和自回归生成 |
| `vocoder` | 音频编码 -> 波形 | 通过音频 codec 解码，生成供接口返回的音频 |

**prefill** 是先处理输入 prompt 并建立推理状态；**decode** 是在已有状态上一步步生成后续音频编码。这里的 decode 和后面“codec 把编码解码成波形”不是同一件事。

SGLang-Omni 不是又换了一个更小的 TTS 模型：本次新旧对比用的是同一份 MOSS 模型权重。变化集中在运行引擎、attention、采样、codec 实现/精度、缓存和调度组织。

另外，三个框不等于三个 GPU，也不等于三个独立模型进程。本次固定配置把这三个 stage 放在同一个 `pipeline` worker 进程和同一张 GPU 上；服务还有 HTTP/coordinator 等进程。不能从流程图推导出“新增三卡并行，所以变快”。

### 6.4 收到完整 WAV 后，回到原来的渲染链路

Adapter 对 HTTP 状态、响应类型、下载大小、截止时间及完整 WAV 做校验，确认符合 24 kHz 单声道人声契约后，原子提交为本次 `source.wav`。

之后 `MossAlignedPhraseRenderer` 按原来的方式执行 MMS 和 R3。对它而言，拿到的仍是“一份完整原始人声及其合成元数据”。

代码用 HTTP streaming API 逐块读取响应体，是为了有界下载、检查 deadline 和处理中断。**这不等于当前已经把模型生成中的音频片段实时送给 Mac 播放。**

当前要等完整 source，再对整段做 MMS/R3，最后才能返回 RAP 音频包。上游内部存在流式 stage 接口，也不能改变这条已经接入的非流式边界。

## 7. 后半段共用：为什么还要 MMS 和 R3

### 7.1 MMS 回答“实际上什么时候发音”

MMS 的输入是 `source.wav` 和已知的两句歌词。forced alignment 的意思是：假定文本已知，在音频上寻找对应的字符、单词时间位置。

系统再结合 CMU 的音节/音素信息，把对齐结果映射成每个音节的估计 onset。某些位置需要字符加权或词内时长比例后备，因此这些位置是估计值，不是绝对精确的音节真值。

于是同一个音节有两个时间：

```text
planner / flow 给的：希望它在 1.00 秒开始
MMS 从 source 找的：它实际约在 1.18 秒开始

R3 所需的映射点：source 1.18 秒 -> target 1.00 秒
```

**文本规划阶段的 `align_exact` 和音频阶段的 MMS forced alignment 是两种不同的对齐。** 前者把音节放到预定格子上，后者在已经生成的声音中寻找实际位置。

Whisper 不在这条生产链路里。之前测速后的 Whisper WER 是额外的内容诊断，没有参与每段 RAP 的生成或时间对齐。

### 7.2 R3 根据映射连续拉伸整段人声

Rubber Band R3 根据 source/target 时间锚点，对整段波形做局部时间变换：某些区间压缩、某些区间延长，并尽量保持音高。

它不是逐个词切开再硬拼，也不是简单修改播放采样率来整体变快。连续处理的目的，是在跟随节拍的同时减少断字和拼接感。

当前默认 `gentle_sparse_r3` 不强迫所有音节都精确命中原始网格。它优先保留关键锚点，例如首尾、强重音和边界，适当稀疏化、调整其他时间目标，并约束局部拉伸比例在 `0.75-1.35`。

这个比例指局部目标时长相对源时长，不是音量或音高范围。如果温和约束无法得到可行方案，代码会带 warning 回到 `all_onsets_r3`；这类输出仍可能有较大的局部拉伸和听感风险。

所以应区分两个保证：

- 最终整段时长严格匹配两个 bar，这是播放调度的硬要求。
- 每个音节都精确落在最初的 tick 上，不是温和策略的保证；它允许一定偏移换取自然度。

### 7.3 输出固定时长的最终人声

处理完毕后，renderer 生成 24 kHz、单声道 PCM16 的 `vocal.wav`，校验目标帧数，并记录 source、MMS 对齐、锚点、拉伸比例及耗时等诊断。

前一次 90 BPM 实测里，两边 source 都长 5.44 秒；经过同一 MMS/R3 路径，final 都变为 5.333333 秒。这是“模型大致控制长度，渲染器最终落实长度”的真实例子。

## 8. 从 H200 的 vocal.wav 到 Mac 听到的 RAP

### 8.1 服务器返回的是结构化产物，不只是任意 WAV

orchestrator 把选中的歌词、计划信息、音频元数据、哈希和相关诊断放进 manifest，并生成 response package。保存和发布遵循完成后再提交的边界，避免半成品被当作成功结果。

canonical 产物保持 PCM 人声。网络传输可以选择 PCM 包，也可以从同一产物派生 Opus 传输包，减少带宽；Opus 是传输编码，不是第三种 MOSS 模型。

producer fingerprint 和私有 MOSS synthesis sidecar 记录“这段声音由哪个后端、模型及配置产生”。它们用于缓存隔离、追溯和新旧对比，不改变 Mac 的公共播放协议。

### 8.2 Mac 接收、校验、混音

`RemoteMossChunkPreparationStrategy` 处理返回包，检查请求/片段身份、BPM、音频格式、预期帧数及对应完整性信息。如果是 Opus，则按传输协议解码回规定长度的 PCM。

之后把整段 24 kHz 单声道人声重采样为本地播放用的 48 kHz 双声道，分成两个 bar，加上本地鼓轨并做增益/峰值处理，得到可排入播放队列的音频。

**服务器主要交付的是 vocal，不是已经混好鼓的最终整首歌。** 本地音频设备输出、会话录音和播放监控仍由 Mac Python 进程负责，关闭浏览器页不等于停止播放进程。

接收和混音也会消耗时间。只有在 Mac 最终检查 deadline 时仍及时且有效的结果，才有资格被采用；服务器在 deadline 前返回并不自动等于客户端已经准备好了。

## 9. 连续播放靠什么：两小节 lookahead 和固定截止点

“实时”在当前系统里主要指 **生成下一段和播放当前段重叠进行**，不是当前段边生成首个音频 token 边出声。

### 9.1 开始播放之前

正常 remote demo 先要求服务器 health 达到 ready；完全没启动服务器时，不会直接把这个模式静默改成本地模式。

开始会话后，controller 准备 chunk 0 的本地兜底，并在有界 startup timeout 内尝试拿到远端第一段。首段最终确定之后，才建立连续播放的时钟起点。启动等待和后续滚动请求的等待不是同一个预算。

### 9.2 开始播放之后

```text
准备期：选定 chunk 0 的正式结果或兜底 -> 建立播放时钟

音乐时间       0 秒               5.33 秒              10.67 秒
Mac 播放       [ chunk 0: bar 0/1 ][ chunk 1: bar 2/3 ][ chunk 2 ... ]
后台准备       [ 准备 chunk 1    ][ 准备 chunk 2    ]
提交边界                         ^                    ^
                       实际在边界前的 guard tick 就要作决定
```

bar 编号在这里沿用代码的从 0 开始计数。主流程一次维护下一对待准备的 bar，不是已经实现任意深度、多段同时推理的流水线。

滚动截止时间会同时受 rolling timeout 和音乐边界前的 guard tick 约束。当前 controller 在下一段起点前一个 tick 冻结选择；90 BPM 时约提前 0.167 秒。常用 rolling timeout 还是 5.0 秒，且本地准备等操作也会占预算，所以后端不是总能独享 5.333 秒。

到了选择时刻：

1. 远端两小节完整准备好、身份正确、没有过期，就提交远端版本。
2. 远端失败、迟到、格式错误或没有可用结果，就提交提前准备的本地两小节兜底。
3. 两个 bar 的选择一起冻结并排入队列，音乐时钟不等待远端追上来。
4. 后续上下文使用真正被采用的歌词，而不是被丢弃的远端草稿。
5. 迟到结果不能回头覆盖已经提交、甚至已经开始播放的 bar。

Stop/Reset 后的旧任务也受会话 epoch 等检查约束，不能把上个会话的结果混进新会话。

因此新后端最直接的体验收益是 **更早完成下一段、增加截止时间余量、可能减少本地兜底**。实际减少多少 fallback，还需要完整请求和客户端测试，不能只靠单个模型的耗时推算。

## 10. 新旧的本质区别，以及几个不能混淆的“回退/缓存”

### 10.1 真正换了哪些东西

| 维度 | 旧 `inprocess` | 新 `sglang-omni` |
|---|---|---|
| 写词与选词 | Qwen/vLLM + 原 planner | 共用同一条链路 |
| MOSS 模型 | MOSS-TTS，进程内常驻 | 同一模型系列，独立服务常驻 |
| 调用边界 | Python 调用模型/processor | Python adapter 调 loopback HTTP |
| 推理实现 | HF/Transformers 路径 | SGLang-Omni 的 MOSS 推理与 codec 路径 |
| 完成条件 | 得到完整 source WAV | 得到并下载、校验完整 source WAV |
| 后处理 | MMS + R3 + exact frames | 共用相同处理 |
| Mac 协议与播放 | remote chunk 协议、本地混音/兜底 | 无需因这个替换而改成另一套协议 |
| 运维 | MOSS 随 RAP 服务加载 | 额外服务、独立依赖、版本固定与健康检查 |
| 当前地位 | 默认后端 | 显式开启的 experimental 后端 |

HTTP 本身不是加速来源，它反而增加少量传输和调度成本。收益来自 HTTP 后面换成了更高效的模型执行路径，并且收益超过新增成本。

本次环境观察到 FA3、CUDA graphs、fused audio heads、参考编码缓存和不同 codec 精度等因素。没有逐项消融，不能给每项优化编一个独立加速百分比。

### 10.2 三种缓存不是一回事

| 缓存 | 复用什么 | 是否仍需要生成这次的人声 |
|---|---|---|
| 参考编码缓存 | 同一参考 WAV 的 codec 编码 | 需要 |
| prefix / radix KV cache | 相同输入前缀的模型推理状态 | 仍需生成未完成部分，不是直接返回最终 WAV |
| RAP artifact / 幂等结果缓存 | 同一已完成请求的音频包 | 命中时可以不重新合成 |

前一次测速没有读取完整音频缓存，每个正式请求都真实合成；但保留了候选服务实际使用的参考编码和 prefix cache。因此它反映部署方式的热运行表现，不是“把所有缓存关闭后的裸模型比较”。

### 10.3 三种回退也不是一回事

- **Mac 播放兜底**：远端失败或迟到时使用本地 eSpeak 音频，保证当前会话不因等待远端而停拍。
- **R3 策略回退**：温和稀疏方案不可行时改用 all-onset R3，仍然处理同一份 MOSS 原始音频。
- **后端回滚**：显式把部署配置从 `sglang-omni` 改回 `inprocess` 并按运行步骤重启；不是单次 SGLang 请求失败后自动再跑一遍旧 MOSS。

取消也不能简单理解为“关掉 HTTP 就立即释放 GPU”。旧 `model.generate()` 的不可中断区间、新服务收到断连后是否停止后台生成，都需要分别处理。当前实现有 deadline、取消检查和无法确认释放时的 degraded 防护，但这次性能试验没有证明全部取消恢复场景已经生产验收通过。

当前共享 renderer 还有执行锁，本次试验也是单请求串行。SGLang 的调度能力不等于项目已经完成多用户并发吞吐优化。

## 11. 实际快了多少，应该怎样理解

2026-09-11 在同一张 H200 上完成的 ABBA 热运行试验：20 条不同的两小节歌词，各后端测 40 次，90 BPM，每段最终音频 5.33 秒。

| 测量范围 / 分位数 | 旧耗时 | 新耗时 | 旧耗时 / 新耗时 |
|---|---:|---:|---:|
| MOSS 到完整 source WAV，p50 | 2640.40 ms | 812.51 ms | 3.25x |
| renderer：MOSS + MMS + R3 + 最终 WAV，p50 | 2767.48 ms | 927.65 ms | 2.98x |
| 完整 renderer 的 p95 | 2927.91 ms | 1034.65 ms | 约 2.83x |

核心结论：**生成一段 5.33 秒的人声，渲染中位耗时约从 2.77 秒降到 0.93 秒，省下 1.84 秒。** MMS 约 20 ms、R3 约 87 ms，两边基本不变，主要收益确实来自 MOSS 路径。

但这份数据不包含 Qwen 写词、完整 RAP HTTP orchestration/打包、Mac 网络、混音及播放，也没有测多人并发。比如暂时把其余串行阶段都记为 T，那么准备耗时近似由 `T + 2.77s` 变成 `T + 0.93s`，并不是整个系统固定 3 倍。

音乐时钟更不会变成三倍速。正确理解是“同样的下一段 RAP 更早准备好”，不是“同样的 RAP 播放得更快”。

还有四个限制：

1. 两边是不同部署环境，主模型权重相同，但 torch/transformers、attention、codec 精度和缓存路径等不同；不是只切换一个 kernel 的严格消融。
2. 数字是预热后的表现。预检曾出现首次真实候选请求约 8.87 秒，不能拿 0.81 秒承诺冷启动。
3. 自动 ASR 检查不等于音质通过；最终音频 WER 为旧 2.41%、新 2.75%，尚未完成人工盲听、音色和节奏自然度验收。
4. 没有通过这次小试验自动放宽 lookahead 或重调 planner。当前 `render_reserve_ms` 仍为 3000，不会因为测到 0.93 秒就自动变成 1000。

完整数据、环境、音频样本和复现方法见 [服务器性能实测报告](../reports/2026-09-11-sglang-omni-moss-server-benchmark-report.md)。

## 12. 以后测试和启动时，应该想到哪一层

**只想比较人声渲染引擎的速度**：可以全在 H200 测，不需要 Mac。固定歌词和 flow，分别跑旧 synthesizer / 新 HTTP synthesizer，再接同一 MMS/R3；前一次测的就是这个范围。

**想知道正常 RAP 请求准备下一段需要多久**：在 H200 请求真正的 `8020/v1/rap/chunks/render`，把 Qwen 候选、选择、渲染和打包都纳入计时。这比 renderer benchmark 更接近完整后端，但仍不包含真实 Mac 链路。

**想知道听起来是否更好、兜底是否更少、能不能长期稳定播放**：需要真实 Mac client、网络、时钟与音频输出。还要测取消、错误、长时间运行和不同歌词，而不是只看一组热请求的均值。

选择后端的开关在 H200 render server：`--moss-serving-backend inprocess` 或 `--moss-serving-backend sglang-omni`。Mac 两边仍使用 `--rap-audio-renderer moss_aligned_remote`。新后端还需要固定版本、参考音频映射等参数，不是只加一个开关就能省掉独立服务的部署。

启动细节看 [RAP demo quickstart](../../docs/developer-guide/rap-demo-quickstart.md) 和 [SGLang-Omni runbook](../../docs/developer-guide/sglang-omni-moss-serving.md)。本说明不代表这些服务当前已运行；前一次测速结束时，测试服务和模型进程已停止。

后续需要下载权重时，继续使用指定缓存位置：

```bash
export HF_HOME=$HOME/mbzuai-projects/models/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
```

## 13. 想对照代码时，从这些入口看

| 想确认的问题 | 代码入口 / 关键符号 |
|---|---|
| Mac 怎样组装 remote 模式 | [rap_demo/cli.py](../../src/streammuse/presentation/rap_demo/cli.py)，`_build_audio_demo` 的 remote 分支 |
| 什么时候请求下一段、什么时候用兜底 | [chunk_realtime.py](../../src/streammuse/application/rap/chunk_realtime.py)，`RollingRapChunkController` |
| Mac 怎样处理远端人声并加鼓 | [chunk_audio.py](../../src/streammuse/application/rap/chunk_audio.py)，`RemoteMossChunkPreparationStrategy` |
| H200 HTTP 入口和后端选择 | [rap_render_server.py](../../src/streammuse/presentation/rap_render_server.py)，`create_rap_render_app`、`_compose_real_worker` |
| 候选怎么生成、两句怎么挑 | [chunk_orchestration.py](../../src/streammuse/application/rap/chunk_orchestration.py)，`ChunkCandidatePlanner`、`RapChunkOrchestrator` |
| 音节和文本分数来自哪里 | [prosody.py](../../src/streammuse/infrastructure/rap/prosody.py)、[scoring.py](../../src/streammuse/application/rap/scoring.py) |
| 请求预算和默认候选策略 | [remote_chunk.py](../../src/streammuse/domain/rap/remote_chunk.py)，`RemoteRapChunkRequest`、`RemoteCandidatePolicy` |
| 旧 MOSS 的常驻封装 | [moss_tts.py](../../src/streammuse/infrastructure/rap/moss_tts.py)，`PersistentMossSynthesizer` |
| 旧模型真正怎样 generate | [moss_backend.py](../../scripts/rap_audio_backends/moss_backend.py)，`create_runtime`、`_generate_chunk` |
| 新后端的 HTTP 请求与 WAV 校验 | [sglang_moss_tts.py](../../src/streammuse/infrastructure/rap/sglang_moss_tts.py)，`SglangMossSynthesizer` |
| 新旧共用的生成参数 | [moss_generation.py](../../src/streammuse/infrastructure/rap/moss_generation.py) |
| MMS、R3 和固定帧数如何串起来 | [moss_aligned_phrase.py](../../src/streammuse/infrastructure/rap/moss_aligned_phrase.py)，`MossAlignedPhraseRenderer` |
| 音频实际发音时间怎样定位 | [mms_forced_alignment.py](../../src/streammuse/infrastructure/rap/mms_forced_alignment.py) |
| deadline、取消和 producer 身份 | [execution.py](../../src/streammuse/application/rap/execution.py)、[producer_manifest.py](../../src/streammuse/infrastructure/rap/producer_manifest.py) |
| 之前具体实现和测试了什么 | [实现报告](../reports/2026-09-04-sglang-omni-moss-serving-implementation-report.md) |

最后再压缩成一遍：**旧版是 RAP 进程自己用 HF 跑 MOSS；新版是 RAP 通过 HTTP 让 SGLang-Omni 跑 MOSS。两边都要先拿到完整人声，再做 MMS/R3，最后让 Mac 按同一个音乐时钟播放。加速的是准备声音，不是改变音乐的速度，也不是已经实现逐 token 的端到端音频流式播放。**
