# RAP 三个版本的对比 demo 页面：计划（2026-10-12）

## 目标

做一个 demo 页面，把 RAP 系统的三个版本放在一起听、一起比：

- 每个版本都测同样的 15 个 sample：5 个主题 × 3 个 flow 模板。
- 每个 sample 每个版本有两段录音：
  - **fake real-time**：同一套系统，但不受时间限制。每个 chunk 或 bar 都等服务器做完再往下走，没有任何兜底，代表“这个系统的生成效果上限”；
  - **real-time**：真实的端到端运行。Mac 客户端通过隧道连 H200，按节拍实时播放，来不及就兜底，代表“听众实际听到的效果”。
- 一共 3 个版本 × 15 个 sample × 2 种模式 = **90 段录音**。
- 测每个版本时用的 commit 都打上 tag，以后可以准确找回、复现。

## 三个版本

| 版本 | commit | 日期 | 合成单位 | MOSS 后端 | 协议 | 拉伸在哪 |
|---|---|---|---|---|---|---|
| v0.0 | `0a493cb8`（`b91c0a3d` 只加了文档，代码相同） | 2026-08-30 | 2 个 bar 一次 | 进程内 HF MOSS（`inprocess`） | v1 | 服务器 |
| v0.1 | `9b76415e` | 2026-10-01 | 2 个 bar 一次 | SGLang-Omni（不带补丁） | v1 | 服务器 |
| v1.0 | 测之前的 HEAD（不早于 `ac3752dc`） | 2026-10-12 | 1 个 bar，逐句续写 | SGLang-Omni（延迟补丁 + 续写补丁） | v3 | Mac |

**两版之间还有一批改动：** 10-01 的延迟优化（`83c224c4` 到 `0ea74a7f`）在 v0.1 和 v1.0 之间：协议 v2、R3 移到 Mac、写词并发、MOSS 延迟补丁。按你的划分，它不单独算一个版本，见 D1。

## 开始前需要你决定的事

| 编号 | 问题 | 我的建议 |
|---|---|---|
| D1 | v0.1 用哪个 commit | **`9b76415e`**：只换了 SGLang-Omni 后端，符合“用 SGLang-Omni 升级后端”。10-01 的优化不单独成列；以后需要时可以加一列 v0.2（`0ea74a7f`），工具都通用。 |
| D2 | 旧版本用哪种拉伸策略 | **`all_onsets_r3`**。原因见下面。 |
| D3 | 5 个主题 | space、deep sea、city nights、basketball、rainy morning。可以随便换，越不一样越好比。 |
| D4 | 每个 sample 多长 | **16 个 bar**，90 BPM 下约 43 秒。v0.x 是 8 个 chunk，v1.0 是 16 个 bar。太短听不出接缝和稳定性，太长页面太重。 |
| D5 | 传输编码 | **全部用 Opus**，各自用当时的默认级别：v0.x 是 10，v1.0 是 5。这样隧道上传的数据量差不多，比较公平。 |
| D6 | 页面放在哪 | **构建脚本和页面模板进仓库，音频不进 git。** 页面先发布成一个私有的 Claude artifact（可以分享链接）。要公开再放 GitHub Pages。 |
| D7 | tag 的名字 | `rap-v0.0`、`rap-v0.1`、`rap-v1.0`，都是 annotated tag，说明里写配置；另外给工具打 `rap-demo-page-2026-10`。都推到 origin，H200 才能拿到。 |

**D2 的原因**

- 两个旧版本的服务器默认都用 `gentle_sparse_r3`，但当时的 Mac 客户端会拒收这种 chunk。这是 10-01 报告第 6.1 节记录的问题。
- 按出厂配置跑 real-time，结果会是 100% eSpeak，没有比较价值。
- `all_onsets_r3` 是旧客户端唯一能接受的策略，服务器也支持，不需要改代码。
- 页面上会注明这一点。如果你想展示“出厂配置是坏的”，可以另加一段说明。

## 测试矩阵

**15 个 sample**

| | baseline_straight_9 | baseline_syncopated_9 | baseline_staggered_9 |
|---|---|---|---|
| 主题 1 到 5 | S01 到 S05 | S06 到 S10 | S11 到 S15 |

- 3 个模板正好是仓库内置的全部 flow 模板。

**每个 sample 一份场景文件**

- 放在 `demo_page/scenarios/S01.json` 等。
- 内容：一个 segment，从 bar 0 开始，16 个 bar，指定主题和模板；`tempo_bpm` 是 90，`loop` 为 false。
- 场景 JSON 的格式从 v0.0 到现在没有变过（`infrastructure/rap/scenarios.py`、`templates.py` 自 `0a493cb8` 起没有改动），所以三个版本都直接用 `--scenario` 读同一份文件。
- 每个 sample 的种子固定（`--seed`），三个版本、两种模式都用同一个种子。
  - **说明：** 同一个种子不保证两种模式写出完全一样的歌词。real-time 下兜底的那一行也会进入后面的上下文，所以两种模式的歌词可能从某个 bar 开始不同。页面上每段录音都显示它自己的歌词。

## 每个版本的配置

**v0.0**

- H200：
  - vLLM：Qwen2.5-7B，`--max-num-seqs 8`；
  - render server：`--moss-serving-backend inprocess --moss-warp-policy all_onsets_r3 --wire-audio-codec opus`，用 `baseline-env`。
- Mac：`--rap-audio-renderer moss_aligned_remote --rap-audio-transport opus`。

**v0.1**

- H200：
  - vLLM：同 v0.0；
  - SGLang-Omni MOSS：不带补丁，用 `env/bin/sgl-omni`；
  - render server：`--moss-serving-backend sglang-omni --moss-warp-policy all_onsets_r3 --wire-audio-codec opus`。
- Mac：同 v0.0。

**v1.0**

- H200：
  - vLLM：`--max-num-seqs 48`；
  - SGLang-Omni MOSS：带延迟补丁和续写补丁；
  - render server：`--lyric-stream --lyric-candidates 32 --lyric-continuation`，Opus 级别 5。
- Mac：`--rap-protocol v3 --rap-audio-transport opus`。

**共同的设置**

- 三个版本用同一个 Qwen 权重、同一份 MOSS 权重和参考音频、同一张 GPU（GPU 2，开跑前确认空闲）。
- 节拍 90 BPM，鼓点用各版本自带的。
- **每个版本的具体参数，以那个 commit 自己的 quickstart 为准。** 上面列的是我核对过的关键差别，第 1 阶段做冒烟测试时逐项确认，写进清单。

## 第 0 阶段：固定版本、打 tag

1. 在 Mac 上为每个版本建一个 worktree：

   ```bash
   git worktree add ../StreamMUSE-rap-v0.0 0a493cb8
   git worktree add ../StreamMUSE-rap-v0.1 9b76415e
   ```

   - 每个 worktree 里用 uv 建自己的环境，`uv sync`，缓存放在外置 SSD 上。
   - v1.0 就用主仓库。
2. 在 H200 上同样建 worktree，路径放在 `~/mbzuai-projects/rap-versions/`。
3. 打 annotated tag：
   - tag 说明里写：版本的定义、上面的关键配置、这个 tag 由哪份计划创建。
   - 推到 origin：`git push origin rap-v0.0 rap-v0.1 rap-v1.0`。

**如果旧版本跑不起来**

- 比如依赖装不上，或者和现在的环境不兼容。这时不在原 commit 上改，而是从它开一个分支 `rap-v0.x-demo`，只做让它能跑的最小修改。
- tag 打在修改后的 commit 上，说明里列出每一处修改和原因。
- 页面上也注明“在原 commit 基础上做了这些运行修复”。

## 第 1 阶段：每个版本先跑一个冒烟测试

- 每个版本跑 S01 的前 4 个 bar，fake real-time 和 real-time 各一次。
- **确认：**
  - 服务器能起来，`/health` 正常；
  - Mac 能收到远端音频，而不是全部 eSpeak；
  - 录音和事件日志都写出来了；
  - 冒烟测试中暴露的问题当场处理（第 0 阶段的修改分支），不带进正式测试。
- **H200 启动脚本：** 每个版本一个，在 `logs/rap_demo_page_2026-10/stack_v0.0.sh` 等。
  - 照着 `logs/lyric_stream_20261012/stack.sh` 写；
  - 只按记录下的 PID / PGID 停止自己的进程，用之前先看 `nvidia-smi`。

## 第 2 阶段：fake real-time 驱动

**定义：** 和 real-time 用同一个版本的同一套代码（控制器、写词、渲染、Mac 端准备），只是时间不再限制。

**做法：** 给控制器注入虚拟时钟。

- v0.0 的 `RollingRapChunkController` 和 v1.0 的 `BarStreamController` 都能注入 `monotonic` 和执行器，v0.1 和 v0.0 是同一个控制器。
- 驱动脚本一个 tick 一个 tick地往前推虚拟时间。到提交点之前，如果下一段远端音频还没好，就在真实时间里等，虚拟时间停住不动。
- 超时都设成很大，所以永远不会触发兜底。
- 输出走 WAV sink，这样录音和 real-time 的录法完全一样。

**代码放在哪**

- 驱动脚本放在 HEAD 的 `scripts/rap_fake_realtime.py`，运行时把 `PYTHONPATH` 指向对应版本的 worktree，导入那个版本自己的代码。
- 每个版本一个小适配器，负责：
  - 按那个版本的 CLI 组装运行时；
  - 判断“下一段远端音频好了没有”。
- 适配器只读控制器的状态，不改控制器的行为。

**备用方案**

- 如果某个版本的控制器内部状态拿不到，就改为不用控制器：
  - 用那个版本自己的请求构造、HTTP 客户端和 Mac 端准备代码，按顺序一个 chunk 一个 chunk 地请求，每个都等结果；
  - 上一段选中的歌词作为下一段的上下文，最后拼起来加鼓点。
- 这种做法和控制器的行为等价，但不是同一段代码，会在页面的方法说明里写明。

**验收**

- 每段 fake real-time 录音里，所有 chunk 或 bar 都来自远端，0 次兜底。
- 时长正好是 16 个 bar。
- 同一个 sample 跑两次，歌词和种子的关系符合预期：MOSS 在固定种子下可复现，Qwen 写词在固定种子下大体稳定。

## 第 3 阶段：real-time 运行

- 每段都用那个版本自己的 demo 命令：`--scenario SNN.json --max-bars 16 --audio-output wav --no-web --log-dir ...`，录下 `mixed.wav` 和 `events.jsonl`。
- **网络会影响 real-time 的结果**：当天隧道一次往返在 1.0 到 1.6 s 之间。
  - 每段开始前先测 5 次 `/health` 的往返时间，记进清单；
  - 往返中位数超过 2 s 的时候先不跑。
- **按 sample 交替各版本的顺序：** 先把 v0.0、v0.1、v1.0 三套服务同时起在 GPU 2 上，端口错开：

  | 版本 | render | vLLM | SGLang |
  |---|---|---|---|
  | v0.0 | 8120 | 8101 | — |
  | v0.1 | 8220 | 8201 | 8230 |
  | v1.0 | 8020 | 8001 | 8030 |

  - 然后每个 sample 依次跑三个版本，这样网络的起伏会平均分到各版本上。
  - **前提：** 三套服务同时放得下一张卡，开跑前先算显存：v1.0 约 61.5 GB，另外两套各约 40 到 50 GB 估计。放不下就按版本分批，但每批前后都测往返时间。
- 每个版本先空跑 1 段热身，不计入结果。
- 某段 real-time 失败（例如进程崩溃，而不是兜底）就重跑一次，并记录下来。兜底本身是结果，不重跑。

## 第 4 阶段：整理数据

每段录音生成一条记录，存进 `demo_page/data/runs.json`：

| 字段 | 内容 |
|---|---|
| 身份 | 版本、tag、commit、模式、sample、主题、模板、种子 |
| 运行条件 | 服务器参数、Mac 参数、开始时间、隧道往返中位数 |
| 每个 bar | 歌词，以及来源：远端最新 / 远端旧版本 / eSpeak。v0.x 按 chunk 记，展开到 bar |
| 汇总 | eSpeak 兜底比例、underrun 次数、服务器每个 chunk 或 bar 的时间（中位数、p95）、客户端取回的时间 |
| 音频 | 原始 WAV 的路径和哈希，页面用的压缩文件路径 |

**汇总脚本**

- 各版本的事件格式不同：
  - v0.x 用当时的 `summarize_rap_session.py`；
  - v1.0 用 `analyze_lyric_stream_session.py`。
- 在 HEAD 写一个统一的转换器，把三种格式都转成上面这张表的字段。

**页面用的音频**

- 从原始 WAV 转成 MP3 192 kbps。
- 统一做响度归一化，到 −16 LUFS，避免谁声音大谁占便宜。
- 原始 WAV 留在 `output/rap_demo_page_2026-10/`，不进 git。

**可选：自动质量指标**

- 用现成的 `score_bar_unit_audio.py` 那套 Whisper 和 ECAPA，给每段算整段 WER 和相邻 bar 说话人相似度，放进页面的汇总表。
- 费时约 20 分钟，你说要就做。

## 第 5 阶段：页面

**页面结构，从上到下**

1. **简介：** 系统是做什么的，fake real-time 和 real-time 各是什么意思。
2. **三个版本的对比表：**
   - 架构：合成单位、后端、协议；
   - 关键数字：15 段平均的 eSpeak 比例、服务器时间、underrun；
   - tag 和 commit。
3. **样本矩阵：**
   - 可以按主题或模板筛选；
   - 每个 sample 一张卡片，里面是 6 个播放器：3 个版本 × 2 种模式。
4. **每个播放器下面：**
   - 一条 16 格的 bar 条，按来源上色：远端最新、远端旧版本、eSpeak，让人一眼看出哪里兜底了；
   - 点开能看每个 bar 的歌词。播放时，当前的 bar 高亮。
5. **方法：**
   - 硬件、网络、参数；
   - D2 的拉伸策略说明；
   - 版本之间的差别，以及第 0 阶段对旧版本做过的修改（如果有）；
   - 复现方法：tag 加脚本。

**实现**

- 构建脚本 `scripts/build_rap_demo_page.py` 读 `runs.json`，生成 `output/rap_demo_page_2026-10/site/`，里面是 `index.html`、`audio/`、`data.json`。
- 纯静态页面，不依赖服务器；在手机宽度下也能看。
- 发布按 D6：先发成私有 artifact。音频大约 90 × 0.9 MB ≈ 80 MB，作为附件上传，单个文件远小于 15 MB 的上限。

## 第 6 阶段：检查

- **逐段抽听：** 每个版本每种模式至少听 3 段，确认录音完整、没有截断或静音。90 段全部检查时长和峰值。
- **核对页面数字：** 页面上的数字和 `runs.json` 一致，bar 条的颜色和事件日志一致。
- **核对 tag：** 每个 tag 都能 `git checkout` 回来，从零按“复现”一节跑出同一种结构的结果。
- **写报告：** 放在 `developing-logs/reports/`，内容是做了什么、各版本的结果、遇到的问题。

## 执行顺序和时间估计

| 阶段 | 内容 | 估计 |
|---|---|---|
| 0 | worktree、环境、tag | 1 到 2 小时，主要是旧版本装环境 |
| 1 | 冒烟测试，处理问题 | 1 到 3 小时，看旧版本出多少问题 |
| 2 | fake real-time 驱动和三个适配器 | 半天 |
| 3 | 正式运行：45 段 real-time 每段约 1 分钟，45 段 fake real-time 更快，加启动和热身 | 约 2 小时 |
| 4 | 整理数据、转码 | 1 小时 |
| 5 | 页面 | 半天 |
| 6 | 检查、报告 | 1 到 2 小时 |

## 风险

| 风险 | 应对 |
|---|---|
| 旧版本在现在的环境里装不上或跑不起来 | 第 1 阶段先冒烟测试；必要的最小修改放在单独分支并打 tag，页面注明 |
| 三套服务放不进一张卡 | 改成按版本分批跑，每批前后测网络往返时间 |
| 网络起伏让 real-time 的比较不公平 | 每段记录往返时间；交替顺序；往返太慢时暂停 |
| GPU 1 和 GPU 2 被别人占用 | 开跑前看 `nvidia-smi`，只停自己记录过 PID 的进程 |
| fake real-time 的驱动和真实控制器行为有偏差 | 优先用虚拟时钟驱动同一个控制器；用备用方案时在页面说明 |
| 两种模式写出的歌词不同，听起来不像同一首 | 这是兜底造成的真实差别；页面上每段显示自己的歌词，并在方法里说明 |
| 旧版本的歌词里有念出来的 “A”（rhyme group 标签泄漏） | 这就是当时系统的真实效果，不修；页面方法里说明，v1.0 已修复 |
