# RAP 歌词流、单 bar 合成与实时交互：实施报告

日期：2026-10-12。分支：`feature/rap_optimization`，最后一个代码提交是 `3307ff87`。

依据：

- [歌词流计划（2026-10-12 修订）](../plans/2026-10-08-rap-lyric-stream-plan.md)：每个阶段的简要记录在计划末尾的“实施记录”；
- [profiling 报告](2026-09-25-rap-moss-sglang-profiling-report.md)：第 11 节是这次的 H200 整体复测。

## 1. 直接结论

**协议 v3 已在 H200 上跑通，计划中除 Mac（第 8 阶段，暂缓）和第 7 阶段（不在关键路径上，没做）以外的阶段都已完成。**

v3 的工作方式：

- **服务器写词：** Qwen 每个 bar 写 32 条候选，选出和上一句最搭的一句，一直写到播放头前 3 个 bar，并实时推给客户端。
- **服务器合成：** 持续把音频合成到播放头前 2 个 bar，每个 bar 都接着上一个 bar 的音频续写。
- **客户端取音频：** 按歌词 ID 取。
- **修改之后：** 听众改了主题或 flow，如果来不及，就播旧版本。

**96 个 bar 的完整会话**（H200 GPU 2；Mac 端 headless，走隧道；中途 8 次修改）：

| | 结果 |
|---|---|
| 最新版本 / 旧版本 / eSpeak | 92 / 4 / 0 |
| underrun | 0 |
| 音频提前合成好，中位数（最小） | 3.8 s（1.35 s） |
| 歌词提前到达，中位数（p5） | 7.4 s（3.0 s） |
| 续写比例 | 99% |
| 押韵率（同一主题内的对句） | 84% |
| GPU 2 平均利用率 / 显存 | 22% / 61.5 GB |

- 4 次播旧版本，正好对应那 4 次只提前 1 个 bar 的修改，符合设计。
- 提前 2 个 bar 的修改，都在修改的那个 bar 就听到了新内容。

**续写（第 6 阶段）通过验收。** 40 对样本，和单句独立合成相比：

| | 两句一次合成（原来的 v2） | 单句独立 | **单句续写** |
|---|---:|---:|---:|
| WER | 2.7% | 3.1% | **1.2%** |
| 相邻 bar 说话人相似度，中位数（最小） | 0.714（0.53） | 0.634（0.50） | **0.711（0.62）** |
| 接缝音高跳变，中位数 | 2.95 半音 | 4.85 半音 | 3.90 半音 |
| 接缝音量跳变，中位数 | 6.6 dB | 12.8 dB | 11.7 dB |
| 第二句的 MOSS 时间，中位数 | — | 523 ms | 518 ms |

- 说话人一致性回到了两句一次合成的水平，WER 更低。
- 接缝音量只小幅改善：主要是每个 bar 各自拉伸时首尾的静音，续写管不到，见 4.3。

**过程中发现并修掉了一个老问题：Qwen 把提示词里的 “rhyme group A” 抄进了歌词。**

- 原来的成对写词（v1、v2 用的就是它）48 行里有 43 行以 “A” 结尾，MOSS 也把这个 “A” 念了出来。
- 原因：多出的这个音节刚好补足了音节数，所以这些行反而得分高。
- 修掉以后，成对写词在 48 个 bar 里有 10 个写不出合格的行。**这会影响 v2，需要你决定怎么处理**，见 4.1。

**需要你来做的**

- 盲听：续写和单句独立的对比，以及一整段 v3 会话，见第 6 节。
- 决定 v2 写词怎么处理，见 4.1。
- 决定要不要把续写补丁提给上游。
- 用网页实际改一次主题，感受反应时间。

## 2. 做了什么

| 阶段 | 提交 | 内容 |
|---|---|---|
| 0 | `798a2872`、`2194ba3a` | 一个 bar 的渲染请求；0a 对比脚本（两句一次合成 vs 单句独立） |
| 1 到 3 | `c82dd151` | 逐句写词、会话服务、合成循环、按歌词 ID 取音频，即服务器端协议 v3 |
| 4、5 | `d1b3dbb8` | 客户端按 bar 运行的控制器；实时修改主题和 flow |
| 1（修正） | `35aa23a2`、`3a9e7335`、`a209d6ae`、`a18c79e4`、`f8b79f62` | 修正写词，见 2.1 |
| 5（测量） | `35aa23a2`、`38d7d83d`、`2a1d5c17` | `plan_edited` 事件、反应时间分析、`bar@notice` 脚本修改、不重复取旧版本 |
| 6 | `d58fd59a`、`2011f7ac`、`82dfece2`、`78172327`、`3307ff87` | MOSS 续写补丁和客户端；对比脚本加上续写这一组；记录每个 bar 是否续写；切前缀之前先检查返回的开头确实是前缀 |

对比脚本：`b4d29b99`、`0470f4e9`。全部单测通过；本机原有的 12 个 FastPitch / timing 失败与这次无关，改动前后相同。

### 2.1 写词（第 1 阶段）

**`SequentialLyricWriter`（`application/rap/lyric_writer.py`，writer v2）**

- 每个 bar 让 Qwen 独立生成 32 条，逐条检查和打分，选最好的一条。
- 对句押韵：奇数 bar 必须和前一个 bar 押韵。用 `CandidateRequest.rhyme_with` 写进提示词，打分时把上一句的韵脚作为锚点。

**这次补上的过滤**

所有过滤都写进候选记录（ledger），带拒绝原因：

| 拒绝原因 | 拒绝的行 |
|---|---|
| `rhyme_label_leak` | 以韵组标签结尾，比如 "... endless space A" |
| `non_lyric_characters` | 带方括号标注（"[phrase]"）、中文（"太空"）等非英文内容 |
| `repeats_recent_end_word` | 行尾词和最近 4 行重复。用同一个词不算押韵 |
| `repeats_recent_opening` | 开头 3 个词和最近 4 行重复，比如每行都是 "In the code where ..." |

**补救**

- 原来是 1 轮 8 条，现在最多 3 轮、每轮 16 条。
- 补救轮不带历史行，因为 Qwen 模仿历史会把 32 条写成同一句（“只收到 1 条”）。

**提示词**

- flow 提示词里的韵组改用槽位编号，不再写字母：`Rhyme slots (1-based; each list rhymes together): [[9]]`。
- 押韵要求加上“用不同的词”。

**离线对比**

48 个 bar，`scripts/compare_lyric_writers.py`：

| | 成对写词（修正提示词后） | 逐句写词 v2 |
|---|---:|---:|
| 写不出合格行的 bar | 10 | **0** |
| 对句押韵率 | 63% | 75% |
| 行尾词种类 | 18 | **41** |
| 相邻重复行尾词 | 13 | **0** |
| 用到补救的比例 | 24% | 4% |
| 每个 bar 生成时间，中位数 | 49 ms | 95 ms |

- 修正之前测的“逐句 100%、成对 79%”押韵率作废：其中大量是 “A” 对 “A”，或者同一个词对同一个词。
- 在真实会话里，逐句写词的押韵率是 84%，96 个 bar 里行尾词有 73 种。

### 2.2 会话服务、合成循环、取音频（第 2、3 阶段）

**歌词记录：`BarLyricRecord`**

- 字段：bar、版本号、主题、flow、文本、对齐结果。
- `lyric_id` 是这些内容的 sha256，不含分数和来源信息。

**会话：`LyricStreamSession`**

- 写词线程和合成线程各一个。
- 写词写到播放头前 K = 3 个 bar，合成做到前 A = 2 个 bar。
- 客户端要的 bar 超出了合成范围，就排到最前面。
- 修改之后，还没提交的 bar 重写成新版本（版本号加 1），旧版本没开始合成的就丢弃。

**取音频的结果状态**

`ready`、`older_version`、`pending`、`failed`、`superseded`、`unknown_lyric`。

**接口**

- `WS /v1/rap/sessions/{id}`：
  - 客户端发：`start`、`update`（修改）、`committed`（回报播放头）、`stop`；
  - 服务器推：`lyric`、`superseded`、`audio_ready`、`error`；
  - 断线重连后，服务器会补发当前各 bar 的最新歌词。
- `GET /v1/rap/sessions/{id}/bars/{bar}/audio?lyric_id&wait_ms&accept_older`：长轮询，返回一个 bar 的音频包（PCM 或 Opus）。

**render server 参数**

- `--lyric-stream`：开启 v3；v1、v2 的接口不变。
- `--lyric-candidates`：每个 bar 的候选数，默认 32。
- `--lyric-rhyme-scheme`：押韵方式，默认 couplet（对句押韵）。
- `--lyric-continuation`：每个 bar 续写上一个 bar，见 2.4。

### 2.3 客户端控制器和实时交互（第 4、5 阶段）

**`BarStreamController`**

- 生命周期和 v2 的 `RollingRapChunkController` 一样。
- 收到歌词后，取接下来 2 个 bar 的音频。

**提交时的选择顺序**

| 顺序 | 选择 | 什么情况用 |
|---:|---|---|
| 1 | `remote_latest` | 最新版本的音频已经到了 |
| 2 | `remote_older` | 只有旧版本的音频 |
| 3 | `espeak_stream_lyric` | 有服务器写的歌词，但没有音频，用 eSpeak 念这句歌词 |
| 4 | `espeak_local_lyric` | 连歌词都没收到，用本地写的歌词 |

**实时修改**

- 网页：`POST /api/control/plan`，内容如 `{"topic": ..., "template_id": ..., "from_bar": ...}`。
- headless：`--rap-scripted-edits "12:topic=volcano;20@1:template=..."`。`bar@notice` 表示在生效 bar 之前第 notice 个 bar 的中间修改。
- 每次修改都记一条 `plan_edited` 事件。

**分析**

- `scripts/analyze_lyric_stream_session.py` 给出各种选择的比例、提前量、续写比例。
- 对每次修改，给出第一个新内容的 bar、用的是哪种选择、从修改到开始播放的时间。

**这次的小修正**

- 已经拿到某个 bar 的旧版本时，不再用 `accept_older` 重复取一次。
- 修正前，修改后的那个 bar 会多花一次隧道往返，提前量也被记成 33 ms；修正后最小提前量是 1.35 s。

### 2.4 MOSS 续写（第 6 阶段）

**SGLang-Omni 补丁：`patches/sglang-omni-0.1.4-af3ab61-moss-continuation.patch`**

- 打在 10-01 延迟补丁之上，改 4 个文件、约 120 行。
- 语音接口新增 `prefix_audio`（`data:audio/...` 的 base64 WAV）和 `prefix_text`。
- 预处理时：
  - 用现有的参考音频编码器，把前缀编码成 codes；
  - 按 MOSS 的 continuation 模式组织输入：`user`（前缀文本 + 新文本）和 `assistant`（前缀 codes）；
  - `tokens` 设为 `token_count` 加上前缀的帧数，所以 `token_count` 仍然只表示新音频的长度。
- AR 阶段原本就支持从 assistant 前缀接着生成，不用改。
- 实测：
  - 新音频的长度和不续写时一样，都是 2.72 s；
  - 返回音频里的前缀和原音频的相关系数为 0.96 到 0.98；
  - 每个 bar 多 0 到 65 ms。

**客户端（render server 一侧）**

- `SglangMossSynthesizer.synthesize(..., prefix=SpeechPrefix(text, wav))`：
  - 把前缀补齐到整数个 codec 帧（每帧 1,920 个采样点）再发出；
  - 收到结果后，按补齐后的长度精确切掉前缀。
- `MossBarAudioRenderer`：
  - 保留最近 64 个 bar 的 MOSS 原始音频；
  - 合成 bar k 时，用 bar k−1 **最新版本**的音频做前缀（按 D5）；
  - 那个版本还没合成，就不续写；
  - 是否续写记在音频包的 `diagnostics.continuation` 里，客户端写进 `bar_frozen.continued`。

**说明**

- 0a 原计划离线用 HF 测续写，实际改为直接在 SGLang 上测，这样测的就是线上用的路径。
- 只用了非流式返回。流式返回会把前缀那几行再发一遍，这次没有改。

## 3. 结果

### 3.1 0a 与第 6 阶段：合成单位

- 设置：H200 GPU 2 上的 SGLang（带两个补丁），对齐用 GPU 1。20 条语料 × 2 轮，40 对。
- 指标在 Mac 上计算：先用 R3 拉伸，再用 Whisper 算 WER、用 ECAPA 算说话人相似度，接缝跳变取 bar 分界前后各 500 ms。
- 数字见第 1 节的表。
- 两组单句里的第一句是逐字节相同的（种子固定），只有第二句不同。所以差别完全来自续写。

### 3.2 v3 会话

| 会话 | bar 数 | 修改 | 最新 / 旧 / eSpeak | underrun | 续写比例 |
|---|---:|---|---|---:|---:|
| `demo-v3`（不续写） | 48 | 无 | 48 / 0 / 0 | 0 | — |
| `demo-v3-edits-v2` | 48 | 4 次，都提前 2 个 bar | 48 / 0 / 0 | 0 | — |
| `demo-v3-edits-notice1` | 48 | 4 次，都提前 1 个 bar | 44 / 4 / 0 | 0 | — |
| `demo-v3-continuation-2` | 48 | 2 次提前 2 个、2 次提前 1 个 | 46 / 2 / 0 | 0 | 98% |
| **`demo-v3-phase9`** | **96** | **4 次提前 2 个、4 次提前 1 个** | **92 / 4 / 0** | **0** | **99%** |

`demo-v3-phase9` 的时间：

| 中位数（p95） | |
|---|---|
| 服务器 MOSS，每个 bar | 609（684）ms，和 vLLM 写词同卡 |
| MMS | 15 ms |
| 服务器 Opus 打包 | 56 ms，单独测 |
| 客户端取音频 | 1.37（1.74）s |
| Mac 端准备：解码、拉伸、混音 | 35（37）ms |

**修改的反应时间**

- 脚本都是在 bar 中间修改，从修改到新内容开始播放都是约 4.0 s，也就是 1.5 个 bar。
- 实际能多快生效，看修改时离下一个可改的 bar 还有多久：
  - 来得及合成（服务器约 0.7 s，加客户端取回约 1.4 s），那个 bar 就是新内容；
  - 来不及，就先播一个 bar 的旧版本。
- 所以听到新内容的时间大致在 2.8 到 5.5 s 之间。

### 3.3 GPU

- 整个会话中，GPU 2 的利用率平均 22%、p95 84%，显存 61.5 GB。三个服务都在这张卡上。
- GPU 1 今天下午 ljk 的进程退出后空出来了。这次只把它用于 0a 和第 6 阶段的对齐，服务还都在 GPU 2 上。

## 4. 需要注意的地方

### 4.1 v1、v2 写词受到提示词修正的影响（需要你决定）

flow 提示词是 v1、v2、v3 共用的。改成槽位编号以后：

- **好处：** v2 不会再念出行尾的 “A”。修正之前，v2 实际合成的歌词几乎每一行都多一个 “A”。
- **代价：** 成对写词失去了这个凑音节数的办法。离线 48 个 bar 里有 10 个写不出合格的行，在 v2 里这些 bar 会退回 eSpeak。

可选的做法：

- （a）把逐句写词的过滤和补救也用到 `ChunkCandidatePlanner`；
- （b）v2 继续用原来的提示词，但在候选里拒收以标签结尾的行；
- （c）不再改 v2，以 v3 为主。

我倾向（a）或（c）。

### 4.2 部署

**v3 需要**

- render server 加 `--lyric-stream`；
- render server 的 Python 环境里装有 `websockets`。H200 上是装进 overlay 的，16.0 版。

**续写需要**

- SGLang-Omni 依次打延迟补丁和续写补丁；
- render server 加 `--lyric-continuation`；
- `--moss-runtime-patch-sha256` 填续写补丁的哈希。续写补丁只能打在延迟补丁之上，所以这一个哈希就能代表两个补丁。

**两个保护**

- `--lyric-continuation` 不能脱离 `--lyric-stream` 和 sglang-omni 后端使用，否则启动时报错。
- 服务器如果没有打续写补丁，会静默忽略前缀字段，只返回新音频。客户端在切掉前缀之前，会检查返回音频的开头和发出去的前缀是否吻合（打了补丁时相关系数为 0.96 到 0.98，低于 0.5 就报 `MossInvalidOutput`），所以不会误切正常的音频。

**H200 启动脚本：`logs/lyric_stream_20261012/stack.sh`**

- `start optimized 2`：启动整套服务；
- `restart-render optimized 2`：只重启 render server；
- `stop`：停止；
- 环境变量 `SGLANG_PATCH_ROOT`：指定打好补丁的 SGLang-Omni 目录；
- 环境变量 `LYRIC_CONTINUATION=1`：开启续写。

**Mac 本地**

- v3 暂不支持 Mac 本地运行，Mac 本地仍然走 v2（第 8 阶段）。

### 4.3 续写的局限

- **接缝音量：** 只从 12.8 降到 11.7 dB，仍比两句一次合成（6.6 dB）差。
  - 主要原因是每个 bar 单独拉伸到自己的节拍上，bar 末尾的停顿和下一个 bar 开头的起音之间，音量本来就有落差。p90 达到 33 dB 的样本，前后都有一边是静音。
  - 要继续改善，得在拉伸或混音上做，比如跨 bar 的短交叉淡化，或者给每个 bar 的首尾静音设上限。
- **前缀版本：** 续写总是接着上一个 bar 的最新版本。如果上一个 bar 最后播的是旧版本，那个接缝就不连贯。这只出现在提前 1 个 bar 的修改之后，D5 已经决定先接受。
- **前缀缺失：** 前缀在 render server 的内存里，最多 64 条。render server 重启后，第一个 bar 不续写。

### 4.4 测量

- **“取音频”的 1.37 s 基本是隧道本身的往返。** 当天 `/health` 一次就要 1.0 到 1.6 s，和 10-01 的情况一样。这个数字不能用来评估服务器。
- **服务器 MOSS 609 ms，比 0a 单独测的 520 ms 高：** 会话中 vLLM 在同一张卡上并发写词。GPU 1 可以用来把两者分开，这次没做。
- **WER 用的是 0a 和第 6 阶段的离线 40 对。** 会话里的音频没有算 WER。
- **第 7 阶段没做：** Opus 打包每个 bar 56 ms，合成线程只占约 26% 的时间，客户端准备只要 35 ms，都不在关键路径上。

## 5. 待你决定或执行

1. 盲听，文件见第 6 节。
2. v2 写词怎么处理，见 4.1。
3. 续写补丁要不要提给上游。
4. 用网页实际改一次主题。改的方法见 [quickstart](../../docs/developer-guide/rap-demo-quickstart.md) 的 “Protocol v3” 一节。
5. 第 8 阶段（Mac 本地 v3）什么时候做。

**H200 现在的状态：** 整套服务（带续写）还在 GPU 2 上运行，Mac 的隧道也开着。不用的时候停掉：在 H200 上运行 `logs/lyric_stream_20261012/stack.sh stop`。

## 6. 证据位置

**Mac：`output/lyric_stream_20261012/`**

- 盲听样本：
  - `bar_unit_6/warped/r{0,1}-NN-{phrase,bars,continued}.wav`：同一段歌词，分别用两句一次合成、单句独立、单句续写，各 40 段；
  - `bar_unit_6/warped/manifest.json`：每段的歌词；
  - `demo-v3-phase9/<session>/mixed.wav`：96 个 bar 的完整会话，带 8 次修改。
- 质量指标：`bar_unit_0a/quality.json`、`bar_unit_6/quality.json`，以及 `score_0a.log`、`score_6.log`。
- 会话：
  - `demo-v3*/<session>/events.jsonl`；
  - `demo-v3-phase9.summary.json`。

**H200：`~/mbzuai-projects/StreamMUSE-v1/logs/lyric_stream_20261012/`**

- `bar_unit_0a/`、`bar_unit_6/`：原始 MOSS 输出和对齐结果；
- `writers_phase1*.json`：写词对比。`writers_phase1.json` 是修正前的数据，已作废；最终数据是 `writers_phase1_v2d.json`；
- `continuation/`：打好两个补丁的 SGLang-Omni；
- `stack.sh`、`stack/*.log`：服务启动脚本和日志；
- `gpu_phase9.csv`：GPU 采样。
