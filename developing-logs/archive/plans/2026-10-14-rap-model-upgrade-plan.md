# 写词 LLM 与 TTS 模型升级：计划（2026-10-14）

## 背景

v1.0（协议 v3，歌词流加续写）现在用的两个模型：

| 环节 | 模型 | 发布时间 | 部署 |
|---|---|---|---|
| 写词 | Qwen2.5-7B-Instruct | 2024-09 | H200 上的 vLLM，每个 bar 生成 32 条候选 |
| 合成 | MOSS-TTS v1.5（Delay 版） | 2026 年初 | H200 上的 SGLang-Omni，打了延迟补丁和续写补丁 |

**写词的问题。** trace 和离线对比里暴露的写词问题，大多像是 7B 能力不够：

- 音节数数不准，要靠 32 条候选去筛；
- 把提示词里的 “rhyme group A” 抄进歌词；
- 爱模仿前几句，出现整段同一个句式；
- 押韵率只有 75% 到 84%。

**合成的问题。** MOSS 也有短板：

- 不能控制重音和韵律；
- 在人工偏好排行榜上没有成绩（见 B 部分）；
- 续写以后，接缝处的音高仍比两句一次合成差。

**目标：** 在不破坏实时性的前提下，换上写得更好的 LLM 和说得更好的 TTS。

## 约束（来自 v1.0 的 trace）

**时间预算**

| | 90 BPM | 120 BPM |
|---|---:|---:|
| 一个 bar 的时长 | 2.67 s | 2.0 s |
| 歌词提前写的量（3 个 bar） | 8.0 s | 6.0 s |
| 音频提前合成的量（2 个 bar） | 5.3 s | 4.0 s |

- 现在写一行约 0.1 s，MOSS 合成加对齐加打包约 0.7 s（p95 约 0.8 s）。
- 在 96 个 bar 的会话里，音频提前到达客户端的时间中位数 3.8 s，最小 1.35 s。

**GPU**

- 可以用 H200 的 GPU 1 和 GPU 2，但两张卡都常有别人的进程，开跑前必须看 `nvidia-smi`。
- 在同一张卡上和 vLLM 抢资源时，MOSS 每个 bar 要 609 ms；单独测时只要 520 ms。

**每次实时测评必须报告**

- 最新版本的占比、兜底率、underrun；
- 开场等待，即从开始到第一个 bar 响起的时间；
- 音频提前到达的量；
- 修改主题后的反应时间。

**H200 上现成的环境**

- vLLM 0.24.0，transformers 5.12.1。
- Hugging Face 缓存里已有：Qwen2.5-7B-Instruct、Qwen3-8B、**Qwen3.6-27B**、**Qwen3.6-35B-A3B**、Qwen3-235B-A22B-Instruct-2507-FP8，以及 MOSS-TTS v1.5。
- 磁盘剩约 11 TB。

## A 部分：写词 LLM

### A.1 候选

| 模型 | 结构 | 权重（BF16 / FP8） | H200 缓存 | 估计写一行（32 条） | 为什么试 |
|---|---|---|---|---:|---|
| Qwen2.5-7B-Instruct（基线） | 稠密 | 15 GB | 有 | 0.1 s（实测） | 对照 |
| **Qwen3.6-35B-A3B** | MoE，每个 token 激活 3B | 70 / 35 GB | 有 | ~0.1 s | 激活参数比 7B 还少，能力强很多，可能又快又好 |
| **Qwen3.6-27B** | 稠密 | 54 / 27 GB | 有 | ~0.3–0.4 s | 上一代单卡稠密模型 |
| **Qwen3.8-27B** | 稠密 | 54 / 28 GB（官方 FP8） | 要下载 | ~0.3–0.4 s | 目前最新、最强的单卡模型（2026-08） |
| Qwen3.5-9B（可选） | 稠密 | 18 GB | 要下载 | ~0.1 s | 最新一代的小模型，作为下限；以后 Mac 本地也可能用 |

**不试的模型**

- Qwen3.5-122B-A10B：FP8 也要约 120 GB，得独占一张卡。
- Qwen3-235B（缓存里有 FP8 版，约 235 GB）：要两张卡。
- Qwen3.8-2.4T：太大。
- Qwen3.7-Max、Qwen3.8-Max：只有 API，没有开放权重。

**注意事项**

- **vLLM 版本：** Qwen3.5 以后的模型是混合注意力（大部分层是 Gated DeltaNet 线性注意力），而且是多模态的。vLLM 至少要 0.17；Qwen 官方对 3.6 建议 ≥ 0.19；Qwen3.8-27B 要求 vLLM ≥ 0.17、transformers ≥ 5.8。H200 上的 0.24 都满足。
- **只用文本：** 部署时只开文本能力。
- **思考模式：** 这几代默认会先“思考”，必须关掉（`chat_template_kwargs={"enable_thinking": false}`）。否则又慢又会把推理过程写进歌词。
- **客户端要改：** 现在的写词客户端 `LocalChatModelClient` 发的是普通的 OpenAI chat 请求，要加一个把 `chat_template_kwargs` 传进去的参数。

### A.2 做法

**A0 环境检查（约半天）**

- 在 GPU 1 上逐个启动候选模型，端口 8401 起，`--max-num-seqs 48`。
- 用 FP8（有官方 FP8 的用官方的）。显存利用率按模型调整，留出空间给以后可能同卡的 MOSS。
- 每个模型发一次和写词一样的请求（n=32），检查三件事：
  1. 输出里没有思考内容；
  2. 32 条都是单行歌词；
  3. 单次请求的耗时。

**A1 离线对比（约半天）**

- 扩展 `scripts/compare_lyric_writers.py`：可以选模型、带上 `chat_template_kwargs`，并能用多个种子。
- 每个模型写 48 个 bar，覆盖 3 个模板和多个主题，3 个种子。
- 记录的指标：

| 指标 | 说明 |
|---|---|
| 写一行的时间 | p50、p95，分别算只用首轮和包含补救轮 |
| **候选合格率** | 32 条里通过音节、押韵等检查的比例，最能直接反映模型能力 |
| 补救率、写不出合格行的 bar 数 | |
| 对句押韵率、行尾词种类数、重复开头次数 | |
| 标签泄漏、非英文内容的次数 | |

- 每个模型抽 16 行给你盲看，打乱模型顺序。

**A2 提示词调整（视结果而定）**

- 现在的过滤和补救规则是按 7B 的毛病调出来的。
- 大模型如果合格率高，可以：
  - 减少候选数，例如 32 → 16，进一步提速；
  - 放宽某些过滤；
  - 在提示词里加更多押韵、句式上的要求。

**A3 在线验证（约半天）**

- 最好的 1 到 2 个模型接入 v1.0，跑完整会话：
  - 90 BPM 跑 96 个 bar，中间改 8 次；
  - 120 BPM 跑 48 个 bar。
- 比较两种放法：写词模型放 GPU 1、MOSS 放 GPU 2；写词和 MOSS 同卡。主要看 MOSS 的时间受不受影响。
- 指标按上面“每次实时测评必须报告”的那一组。

**A 部分验收**

- 写一行的 p95（包含补救轮）≤ 1.0 s。
- 候选合格率和押韵率明显高于基线，盲看更好。
- 在线会话 100% 最新版本，音频提前量不下降。

## B 部分：TTS

### B.1 调研结论：MOSS 不是全面的 SOTA

**客观指标上，MOSS 很强**

- 按 MOSS 的技术报告，MOSS-TTS v1.5 在 Seed-TTS-eval 英文上 WER 1.93%、说话人相似度 73.3%，说是开源模型里最好的。

**人工偏好排行榜上，MOSS 没有成绩**

- Artificial Analysis 的 Controlled Voice Arena（2026-10）里没有 MOSS。
- 前几名都是闭源：HeyGen Voice、阿里 Qwen-Audio-3.1-TTS-Plus、ElevenLabs v4。
- 开源权重里排得最高的：

| 模型 | 发布 | 排名 / Elo |
|---|---|---|
| Fish Audio Drama 3 Preview | 2026-10 | 第 14，Elo 1058 |
| Voxtral TTS（Mistral） | 2026-03 | 1002 |
| Fish Audio S2 Pro | 2026-03 | 1001 |
| OpenAudio S1 | 2025-06 | 1000 |
| Higgs Audio V3（Boson） | 2026-06 | 960 |

**另一个客观榜单（Hugging Face 的 Open TTS Leaderboard，2026-09-30）**

- 英文 WER 前列：Kokoro-82M、Supertonic-3、Fish S2 Pro。
- 声音克隆表现好的：Higgs TTS 3、VoxCPM2、OmniVoice、Fun-CosyVoice3。
- 这个榜单也没有收录 MOSS。

**对 rap 来说更重要的是表现力和韵律控制，MOSS 恰好没有**

- Fish S2 Pro 支持行内韵律和情绪标记；
- Higgs V3 支持行内的情绪、风格、韵律控制 token；
- IndexTTS2 能精确指定生成的 token 数，也就是精确控制时长，情绪和音色可以分开控制。

### B.2 我们对 TTS 的要求

| 要求 | 必须 / 加分 | 原因 |
|---|---|---|
| 用我们的参考音频克隆声音 | 必须 | 和现在的声音保持一致 |
| 开源权重，能在 H200 上自己部署 | 必须 | 实时管线 |
| 每个 bar 合成 ≤ 约 0.7 s（p95 ≤ 1 s） | 必须 | 在 120 BPM 下仍有余量 |
| 时长可控（指定 token 数或时长） | 强烈加分 | 没有的话也能靠 R3 拉伸硬对齐，但拉伸比例大了音质会差 |
| 续写，或者能用上一句做前缀 | 强烈加分 | 第 6 阶段证明续写能明显改善接缝 |
| 重音和韵律控制 | 加分 | rap 的关键 |
| 有现成的推理服务（SGLang-Omni、vLLM-Omni 或自带服务） | 加分 | 接入成本 |
| 许可证允许研究使用 | 必须 | 非商用的许可证（如 CC BY-NC）做研究和 demo 可以接受，见 D2 |

### B.3 候选

| 模型 | 规模 | 许可证 | 已知能力 | 要核实的 |
|---|---|---|---|---|
| MOSS-TTS v1.5（基线） | 约 8B | 开源 | 克隆、按 token 数控制时长、续写（我们的补丁） | — |
| **MOSS-TTS-Local-Transformer v1.5** | 4B 主干，tokenizer 约 2B | 开源 | 同一家族；原生流式，48 kHz；SGLang-Omni 有现成的部署说明；支持续写；时长靠 `[pause]` 标记 | 能否按 token 数精确控制时长；单 bar 的延迟 |
| **Fish Audio S2 Pro** | 4B 慢 AR + 0.4B 快 AR | 要确认 | 开源权重里人工偏好最高；行内韵律和情绪标记 | 时长控制、前缀和续写、推理服务、许可证 |
| **Higgs Audio V3** | 4B | 要确认 | 行内的情绪、风格、韵律控制 token，克隆 | 时长控制、推理服务、延迟 |
| **IndexTTS2** | 要确认 | Apache 2.0 | 精确指定 token 数（误差 < 0.03%）；情绪和音色解耦 | 延迟、续写、推理服务 |
| **Qwen3-TTS-1.7B** | 1.7B（另有 0.6B） | Apache 2.0 | 3 秒克隆，流式首包约 100 ms | 时长控制；rap 的表现力 |
| Voxtral TTS（可选） | 4B | CC BY-NC | 开源权重里克隆排名高 | 时长控制、推理服务 |

**这一轮不试的**

- 闭源 API（HeyGen、ElevenLabs、Qwen-Audio-3.1-TTS-Plus）：没法自己部署，但可以拿来当音质参照，听一下差距有多大。
- Kokoro：不能克隆。
- XTTS v2、OpenVoice：太旧。
- 整首歌生成的模型（YuE、Freestyler）：它们是另一条路线，直接根据歌词和伴奏生成说唱人声，不走“TTS 加拉伸对齐”。值得以后单独调研，不放在这一轮。

### B.4 做法

**B0 资料核实（约半天）**

- 逐个看模型卡、代码仓库和论文，把 B.3 表里“要核实的”填满。
- 不满足“必须”的直接淘汰。
- 估算每个模型的接入成本：要写一个合成适配器，实现 `synthesize(request, out, prefix=...)` 这个接口，和现在的 `SglangMossSynthesizer` 一样。

**B1 离线对比（每个候选约半天）**

- 复用 0a 和第 6 阶段的语料：20 段两小节歌词 × 2 轮。都用同一段参考音频，逐 bar 合成；支持续写的就开续写。
- 合成结果走同一套 MMS 对齐和 R3 拉伸，再用 `score_bar_unit_audio.py` 打分：

| 指标 | 说明 |
|---|---|
| WER | 发音准不准 |
| 说话人相似度 | 和参考音频比，以及相邻 bar 之间比 |
| 接缝的音量和音高跳变 | |
| 每个 bar 的合成时间 | 热启动后的 p50 和 p95，在 GPU 1 上单独测，再在和 vLLM 同卡时测一次 |
| 自然时长和目标时长的比例 | 也就是 R3 要拉伸多少 |
| MMS 对齐置信度 | |

- 你盲听：每个模型抽 10 段，打乱顺序。

**B2 在线验证**

- 最好的 1 到 2 个模型做成 v3 的合成后端，用参数选择。
- 跑完整会话：90 BPM 跑 96 个 bar，120 BPM 跑 48 个 bar。指标同 A3，并加入 demo 页面，作为新的一列或新版本。

**B 部分验收**

- 每个 bar 的合成时间 p95 ≤ 现在的 MOSS，或者至少在预算之内。
- WER 不比 MOSS 差。
- 盲听明显更好。
- 在线会话 100% 最新版本，开场等待不变长。

## C 部分：顺序、资源和风险

**顺序**

```text
A0 → A1 → (A2) → A3                 写词：模型已缓存，成本低，先做
B0 → B1（各候选）→ B2                合成：先查资料、淘汰，再上 GPU
A3 和 B2 都通过后，组合成新版本（暂名 v1.1），加进 demo 页面，打 tag
```

**GPU 和端口**

- 候选模型都放 GPU 1：写词候选用 8401 起的端口，TTS 候选用 8501 起的端口。
- v1.0 的整套服务需要时在 GPU 2 上启动。
- 所有进程都按记录下的 PID / PGID 停止，不按名字找进程。

**风险**

| 风险 | 应对 |
|---|---|
| GPU 1 被别人占用 | 每步开始前看 `nvidia-smi`；被占时改到 GPU 2 和 v1.0 分时用 |
| 新模型需要更新的 vLLM、SGLang 或依赖 | 为每个新服务单独建 uv 环境，不动现有环境 |
| 新 TTS 没有时长控制 | 靠 R3 拉伸兜底，同时测拉伸比例分布；比例太大就淘汰 |
| 新 TTS 不支持续写 | 退回单句独立合成，接缝指标要和 MOSS 加续写比 |
| 大模型和 MOSS 同卡，拖慢 MOSS | A3 两种放法都测，必要时写词固定放 GPU 1 |
| 许可证不允许 | 只用于研究和 demo，报告里写明 |

## 开始前需要你决定的事

| 编号 | 问题 | 我的建议 |
|---|---|---|
| D1 | 写词候选 | Qwen3.6-35B-A3B、Qwen3.6-27B、Qwen3.8-27B（FP8）；Qwen3.5-9B 可选 |
| D2 | 能否用非商用许可证的模型（Voxtral 是 CC BY-NC，Fish 的许可证要确认） | 研究和 demo 可以用，报告里注明 |
| D3 | TTS 候选 | MOSS-Local-Transformer、Fish S2 Pro、Higgs V3、IndexTTS2、Qwen3-TTS-1.7B；Voxtral 可选 |
| D4 | 这一轮是否同时测 120 BPM | 测；成本只是多跑几段会话 |
| D5 | 是否拿闭源 API 生成几段当音质参照 | 可以，但要你提供 API key，且只用于对比 |

## 资料来源

- 写词模型：
  - [Qwen3.8 collection](https://huggingface.co/collections/Qwen/qwen38)
  - [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
  - [Qwen3.8-27B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
  - [Qwen3.6-35B-A3B vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B)
  - [Qwen 3.5–3.8 guide (codersera)](https://codersera.com/blog/qwen-3-5-complete-guide-2026/)
- TTS 排行榜：
  - [Artificial Analysis Controlled Voice leaderboard](https://artificialanalysis.ai/text-to-speech/leaderboard/controlled-voice)
  - [Open TTS Leaderboard（Hugging Face 博客，2026-09-30）](https://huggingface.co/blog/open-tts-leaderboard)
- MOSS：
  - [MOSS-TTS GitHub](https://github.com/OpenMOSS/MOSS-TTS)
  - [MOSS-TTS 技术报告](https://arxiv.org/pdf/2603.18090)
  - [MOSS-TTS-Local-Transformer v1.5 on SGLang-Omni（LMSYS）](https://www.lmsys.org/blog/2026-06-17-moss-tts-local-v15/)
- 其他候选：
  - [Fish Audio S2 Pro](https://dev.to/gary_yan_86eb77d35e0070f5/fish-audio-s2-pro-a-tts-model-with-emotion-in-speech-controlled-with-natural-language-1e7n)
  - [Higgs TTS 3（Boson AI）](https://www.boson.ai/blog/higgs-audio-v3-tts)
  - [IndexTTS2 论文](https://arxiv.org/abs/2506.21619)
  - [Qwen3-TTS 技术报告](https://arxiv.org/html/2601.15621v1)
  - [Voxtral TTS 论文](https://arxiv.org/html/2603.25551v1)
  - [VoxCPM2 技术报告](https://arxiv.org/html/2606.06928v1)
- 整首歌和说唱生成：
  - [Freestyler](https://arxiv.org/html/2408.15474)
  - [YuE](https://arxiv.org/html/2503.08638v1)
