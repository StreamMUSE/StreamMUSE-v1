# RAP 服务器端延迟优化实施计划（2026-10-01）

## 目标

把远端 RAP chunk 在 H200 上的准备时间，从当前约 1.49 秒（中位数）降到约 1.1 到 1.2 秒，并去掉两类偶发长尾。前提是不改变 MOSS 的输出、不降低 chunk 契约的校验强度、不牺牲可追溯性。

依据：[profiling 报告](../reports/2026-09-25-rap-moss-sglang-profiling-report.md)，已于 2026-10-01 更正了 Qwen 部分。

## 非目标

- 不改 Mac 到 H200 的网络路由、Mac 端播放和调度逻辑。
- 不做 MOSS 的 FP8 量化、decode 重叠调度或整步 CUDA graph。这些需要单独立项，见"暂不做的更深优化"一节。
- 不改变歌词生成的设计，例如"先定 bar 0 再写 bar 1"或对句生成。两个 bar 独立生成、事后配对的现有语义保持不变。
- 不切换 production 默认后端。SGLang-Omni 仍按现有 runbook 显式开启。

## 开始前需要你决定的事

| # | 决定 | 我的建议 | 影响哪个阶段 |
|---|---|---|---|
| D1 | 工作区里已有大量未提交改动。第 0、1 阶段要改的 `chunk_orchestration.py` 和 `moss_aligned_phrase.py` 也在其中 | 先由你提交或整理现有改动，再在其上开始本计划，每阶段单独提交 | 全部 |
| D2 | 锚点问题的修法：服务器补全逐音节锚点，还是 Mac 放宽校验 | 服务器补全，见第 0 阶段 | 第 0 阶段 |
| D3 | Opus 压缩级别下调的音质门槛 | 见第 4 阶段的量化门槛，外加 5 对盲听 | 第 4 阶段 |
| D4 | 是否向上游 sgl-project/sglang-omni 提 PR | 先在本地补丁上验证，再由你决定是否对外提交 | 第 2 阶段 |
| D5 | R3 终点误差可以放宽到多少 | 先按第 3 阶段收集偏差数据，再定 | 第 3 阶段 |

## 预期收益

| 阶段 | 改动 | 中位数变化 | 尾部变化 | 证据状态 |
|---|---|---:|---|---|
| 0 | gentle_sparse 下补全逐音节锚点 | 0 | 远端 chunk 从全部被拒变为可用 | 已复现问题 |
| 1a | vLLM `--max-num-seqs` 8 改 32 | 约 −140 ms | | 已实测 |
| 1b | 两个 bar 各用一个客户端并发 | 约 −70 ms | | 已实测 |
| 2 | MOSS 两个补丁，外加 HTTP 取消等待 | 约 −77 ms | 去掉约 5% 请求的 +100 ms | 实测，尾部为推断 |
| 3 | R3 不再整段重跑 | 0 | 约 25% chunk 的 +77 到 +230 ms | 已实测重跑 |
| 4 | Opus 压缩级别 10 改 5 | 约 −64 ms | | 已实测，音质待定 |

## 通用规则

### 测量协议

所有"改了多少"的结论都按同一协议取得：

- **完整流程：** 用 `scripts/client_simulation.py`，24 个 chunk 加 2 个热身，Opus 传输，记录服务器各阶段、传输、解码和混音。第 0 阶段完成后，统一用默认的 `gentle_sparse_r3` 测，并要求 Mac 端验收 24 / 24 通过。
- **MOSS 单项：** 用 `scripts/profile_sglang_moss.py` 和 20 条固定语料。
- **比较方式：** 同一实例内配对比较。确实需要重启时，按 ABBA 顺序，每组至少重启两次，因为实例之间有约 20 ms 的稳定偏差。
- **禁止：** 在要测延迟的服务上开过 torch profiler 而不重启。停止后会残留 60 到 80 ms 的开销。
- **GPU：** 每次开始前用 `nvidia-smi` 选空闲的卡，并在记录里写明。本机是共享主机，其他用户的任务会占用任意一张卡。
- **进程管理：** 只按自己启动时记录的 PID 或进程组停止服务，并先确认属主。不要按命令行字符串匹配进程，这台机器上有其他用户的同名 vLLM 服务。

### 提交与测试

- 每个阶段一个提交，提交前运行该阶段相关的单测，以及 `uv run pytest tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation -q`。
- 所有改动都会自动进入 producer 指纹，因为指纹包含 `src/streammuse` 的 git diff。仓库外的改动，也就是第 2 阶段的 SGLang-Omni 补丁，必须显式写入运行时身份，见第 2 阶段。
- 每个阶段完成后，把实测数字追加到本计划的"实施记录"一节。

---

## 第 0 阶段：修复 Mac 拒收 gentle_sparse chunk（前置）

### 现状

- 服务器在 `gentle_sparse_r3` 下，`alignment_diagnostics.source_anchors` 和 `target_anchors` 只包含实际参与拉伸的稀疏锚点，样例中为 7 个。它们由 `moss_aligned_phrase.py` 的 `_effective_diagnostic_anchors` 生成。
- Mac 端 `RemoteMossChunkPreparationStrategy._validate_manifest` 要求锚点数等于两句的音节总数，样例中为 18。之后 `_placement_diagnostics` 还按"第 i 个音节对应第 i 个锚点"计算每个音节的放置诊断。
- 实测结果：gentle_sparse 下 48 / 48 被拒，all_onsets 下 8 / 8 接受。这段逻辑在 HEAD 中已存在。

### 方案

让服务器恢复"每个音节一对锚点"的公共契约，Mac 端代码不改：

1. `source_anchors`：使用 `_WarpPreparation.diagnostic_anchors` 中每个音节的 MMS 实测起点，这本来就是逐音节的。
2. `target_anchors`：对每个音节的实测起点，套用本次拉伸实际使用的时间映射，做分段线性插值，得到它在输出里的实际位置。参与拉伸的锚点，其插值结果就是它们的目标位置；未参与拉伸的音节，得到的是真实落点，而不是计划落点。
3. 稀疏锚点本身，即哪些音节参与了拉伸，保留在服务器端私有的 `mms_alignment.json` 里，已有记录，不进入公共 manifest。
4. `all_onsets_r3` 的输出应与现在逐字节相同，因为所有音节都参与拉伸，插值结果就是目标值本身。

### 为什么不改 Mac 端

放宽 Mac 的数量校验，会让 `_placement_diagnostics` 的逐音节索引错位，诊断会变成错的数据。而且 Mac 客户端部署在你的 Mac 上，改服务器端只需在 H200 上更新。

### 测试

- `test_moss_aligned_phrase.py`：gentle_sparse 下锚点数等于音节数；参与拉伸的锚点，其 target 与计划值一致；插值结果单调不减；all_onsets 下的诊断与改动前相同。
- 新增一个跨边界契约测试：用服务器端构造的 manifest，直接通过 Mac 端 `_validate_manifest` 和 `_prepare_bars`。两种策略各一例。
- `test_chunk_audio.py` 保持不变并通过。

### 验收

- 完整流程在 gentle_sparse 下 24 / 24 被 Mac 端接受，并记录 `mac_validation_mix_ms`。
- 服务器端各阶段耗时不变。

---

## 第 1 阶段：Qwen 写词

### 1a. vLLM 并发序列上限改为 32

**现状：** quickstart 和 SGLang-Omni runbook 中的 vLLM 启动命令都是 `--max-num-seqs 8`。一次 n=16 的请求是 16 条序列，单个请求就要分两轮解码。

**改动：** 两份文档中的启动命令改为 `--max-num-seqs 32`，并在文档里说明原因。这不涉及代码。

**风险：** KV cache 需求变大。7B 模型、`--max-model-len 2048`、显存占比 0.25 的配置下，32 条序列约 400 token 长，余量很充足。启动日志会打印实际的 KV 容量，需要确认它至少能容纳 32 条 2048 长度的序列，否则 vLLM 会排队。

**验收：** 完整流程中服务器端 `generation` 的中位数从约 300 ms 降到约 175 ms 左右。

### 1b. 两个 bar 并发生成

**现状：**
- `ChunkCandidatePlanner.plan` 用 `for state in states` 依次对两个 bar 调用 `_run_wave`，每次都阻塞到 vLLM 返回。
- 补救 wave 的循环也是逐个 bar 串行。
- `LocalChatModelClient` 同一时间只允许一个请求，第二个请求会抛出 "already has an active request"。生成器会把异常吞成空的错误批次，不会报错。这正是初版报告测错的原因。

**设计：**

1. **把 `_run_wave` 拆成两段：**
   - 生成段：调用生成器，只涉及 I/O，可以并发。
   - 入账段：解析候选、去重、分析音节、打分、写入 ledger，只涉及 CPU，按 bar 0、bar 1 的固定顺序串行执行。
   这样 ledger 的顺序、`source_order` 和最终选择都与串行时完全一致，不受哪个请求先返回的影响。
2. **每个 bar 一个生成器：** planner 接受按 bar 位置区分的生成器，形如 `(bar0_generator, bar1_generator)`。只传一个生成器时，退回现在的串行行为，所以离线工具和现有测试不受影响。
3. **执行器可注入：** 并发用可注入的执行器实现。默认是线程池；测试中可以传入确定性的假执行器，模拟"bar 1 先返回"的情况。
4. **每一轮 wave 都并发：** 初始 wave 两个 bar 同时发出。补救阶段中，每一轮需要补救的 bar 同时发出。每一轮开始前的预算检查逻辑保持不变。
5. **计时语义：** `stage_timings_ms.generation` 从"各 wave 耗时之和"改为"各轮并发阶段的墙钟时间之和"。这样它和 `total` 的关系仍然成立，并在代码注释和 manifest 文档里写明。
6. **取消与截止时间：** 两个线程都会进入同一个 `execution.uninterruptible()`。它的深度计数有锁保护，已确认可以重入。某个 bar 超时或失败时，另一个 bar 的结果照常入账，行为与现在的"生成错误后记录并继续"一致。
7. **render server 组装：** `_compose_real_worker` 为两个 bar 各创建一个 `LocalChatModelClient` 和生成器，两者都注册到 close 资源中。

**测试（`test_chunk_orchestration.py`）：**
- 现有 27 个测试在单生成器模式下全部通过，不做修改。
- 新增测试：
  - 两个生成器、假执行器按 bar 1 先完成的顺序返回时，ledger、选择结果和 `source_order` 与串行完全相同；
  - 初始 wave 和补救 wave 都是成对并发发出的，用计数和时间戳断言；
  - 一个 bar 出现生成错误时，另一个 bar 正常入账，并按现有规则进入补救；
  - `generation` 计时等于并发阶段的墙钟时间，而不是各请求之和；
  - 预算耗尽时不再发起新一轮，已完成的这一轮仍会被评估。
- `test_rap_render_server.py`：组装后有两个独立的客户端，关闭时两个都会关闭。

**验收：**
- 完整流程中服务器端 `generation` 的中位数约 105 ms。
- 所有 wave 的错误批次为 0，这一点必须检查，不能只看耗时。
- 同一组输入下，串行与并发选出的歌词一致。vLLM 采样本身有随机性，所以用固定假生成器的单测来保证这一点，不在真实服务上比较。

---

## 第 2 阶段：SGLang-Omni 的 MOSS 补丁与 HTTP 取消等待

这三处代码都在 SGLang-Omni 内部，不在本仓库。

### 补丁内容

1. **embedding 一次查表**，对应 `sglang_model.py` 的 `_prepare_multi_modal_inputs`，只在 decode 的小批次中启用：
   - 初始化时把 32 张音频 codebook 表拼成一张表，并记录每张表的起始行，额外约 0.3 GB 显存；
   - 文本通道仍单独查表，避免复制约 1.3 GB 的文本表；
   - 按原来的通道顺序逐个做 bf16 累加，先文本、后 32 个音频通道，保证逐位等价；
   - prefill 这类大批次走原路径。
2. **文本头只算 2 个控制 token**，对应 `compute_channel_logits`，只在 `is_audio` 且 fused audio heads 可用时启用：
   - 预先取出 2 个控制 token 的权重行，每步只与这 2 行相乘；
   - 保留 `use_fp32_lm_head`、`logit_scale` 和 softcap 的处理，与原路径的数值路径一致。
3. **HTTP 取消不等待**，对应 `openai_api.py` 的 `_cancel_task_bounded`：
   - 取消断开检测任务后不再最多等待 0.1 秒，改为挂完成回调，在后台吞掉结果；
   - 生成任务的取消路径保持不变，因为那里需要确认 GPU 已释放。

### 上线方式

实验时用的 `sitecustomize` 导入钩子只适合测量，不用于生产。生产方案：

1. 在仓库中保存补丁文件，例如 `patches/sglang-omni-af3ab61-moss-latency.patch`，并附一份说明，写明适用的 commit 和每处改动的等价性依据。
2. 在固定 commit 的 SGLang-Omni 源码上应用补丁，再用它构建运行环境。runbook 增加"应用补丁"一步。
3. **运行时身份：** 补丁改变了运行时代码，但 pip 的环境锁文件看不出这一点，所以指纹不会自动变化。需要：
   - render server 增加参数，例如 `--moss-runtime-patch-sha256`，写入 producer manifest 的 `runtime` 字段；
   - preflight 脚本检查补丁文件的哈希，并写入启动 manifest；
   - 先确认 `_validate_pinned_identity` 对新字段的格式要求。
4. 补丁验证稳定后，由你决定是否向上游提 PR，即 D4。

### 测试

- 本仓库新增的单测只覆盖身份相关的部分：新参数的校验、写入 producer manifest，以及不同补丁哈希产生不同指纹。
- 补丁本身的正确性，用下面的真实服务验收来保证。

### 验收

- **正确性：** 20 条语料乘 3 轮，补丁版与未打补丁版的 source WAV 逐字节相同，60 / 60。
- **延迟：** 按 ABBA 顺序重启对比，MOSS 中位数下降约 70 到 80 ms。
- **尾部：** 连续 200 次请求，用客户端和服务端时间戳对齐，统计"服务端已完成到客户端收到"超过 50 ms 的次数。未打补丁时预计约 5%，打补丁后应为 0。
- **显存：** 启动后的显存占用比未打补丁时多约 0.3 GB，不应多出 1.5 GB。

### 回滚

使用未打补丁的环境重启 SGLang-Omni，并去掉补丁哈希参数。指纹会变回原值，旧的缓存命名空间可以继续使用。

---

## 第 3 阶段：R3 不再整段重跑

### 现状

- 生产 renderer 使用实验模块 `warp.py` 中的 `RubberBandTimeMapStretcher`。rubberband 输出与目标帧数相差超过 2 帧时，它会修正时间映射的终点，然后把整段重新跑一遍，上限 6 次。
- 每次约 77 ms。20 条语料中有 5 条会触发重跑，最多 4 次。
- `_enforce_target_length` 本身也只允许 2 帧误差。
- 目前还不知道第一次输出通常偏差多少帧。

### 步骤

1. **先采集数据：**
   - 在拉伸过程中记录每次尝试的输出帧数和偏差，写入服务器端私有的 `mms_alignment.json`，不进入公共 manifest；
   - 用 20 条语料和一轮 24 chunk 的完整流程收集偏差分布。
2. **按数据决定修法（D5）：**
   - **偏差都很小时：** 如果第一次的偏差都在约 480 帧，也就是 20 ms 以内，就直接接受第一次的输出，在尾部截断或补零，并加约 5 ms 的淡出，避免爆音。截断发生在最后一个音节之后，通常是尾音衰减；采集阶段要同时记录这段尾部的能量，确认没有截到正在发声的部分。
   - **偏差更大时：** 按第一次的偏差比例一次性预测终点修正量，保证最多只跑两次。
3. **实现位置：**
   - 给 `RubberBandTimeMapStretcher` 和 `_enforce_target_length` 增加误差容限参数，默认值保持 2 帧，这样实验脚本的行为不变；
   - 由生产 renderer 显式传入新的容限。

### 测试

- 用假的 rubberband 可执行文件，按指定偏差输出帧数：
  - 偏差在容限内时只运行 1 次，输出长度精确，尾部有淡出；
  - 超出容限时，按新规则最多运行 2 次；
  - 默认参数下，行为与现在完全相同。
- 现有 `test_moss_aligned_phrase.py` 全部通过。

### 验收

- 20 条语料全部只运行 1 次 rubberband，或者按第二种修法最多 2 次。
- R3 的 p90 从约 158 ms 降到 100 ms 以内。
- 抽查被截断或补齐的样本，尾部没有可闻的爆音。

---

## 第 4 阶段：Opus 编码提速

### 现状

- 服务器在响应时用 ffmpeg 子进程编码，`-compression_level 10`，约 125 ms。其中约 39 ms 只是启动 ffmpeg 进程的开销。
- 编码结果按请求缓存为 `response.opus.zip`。
- 编码参数没有单独进入 producer 指纹，只通过代码 diff 间接体现。

### 步骤

1. **先评估音质（D3）：**
   - 对 20 条语料的最终人声，分别用级别 10 和级别 5 编码并解码回 PCM；
   - 按 8 月 Opus 实验的方法，计算相对原始 PCM 的 SNR 和波形相关系数；
   - 再做 5 对盲听。
   - 建议门槛：SNR 下降不超过 0.5 dB，相关系数不低于 0.995，盲听分辨不出。
2. **通过后再改代码：**
   - `FFmpegOpusCodec` 增加压缩级别参数，默认改为 5；
   - render server 增加对应的命令行参数，方便回退到 10；
   - 压缩级别写入 producer manifest 的 `output` 字段，并写入 Opus 缓存文件的元数据。
3. **可选的后续：** 把服务器端编码改为进程内调用 libopus，例如通过 PyAV，以省掉约 39 ms 的进程启动开销。这需要新增服务器端依赖，单独评估。Mac 端解码不受影响。

### 测试

- `test_opus_codec.py` 中对 ffmpeg 参数列表的断言，改为按参数取值，并覆盖 0、5、10 三个级别以及非法值。
- producer manifest 测试：压缩级别不同时，指纹不同。

### 验收

- 服务器端 Opus 编码的中位数约 60 ms。
- 音质门槛全部满足，评估结果保存到证据目录。

---

## 第 5 阶段：整体复测与文档

- 所有阶段完成后，按测量协议跑一轮完整流程：24 个 chunk、gentle_sparse、Opus。与第 0 阶段完成后的基线对比，并更新 profiling 报告。
- **目标：**
  - 服务器端到首字节的中位数约 1.1 到 1.2 秒；
  - p90 与中位数的差距明显缩小；
  - Mac 端验收 24 / 24 通过。
- **更新文档：**
  - quickstart 和 SGLang-Omni runbook：vLLM 参数、补丁步骤、新的命令行参数；
  - 2026-09-11 的流程说明文档中，涉及 Qwen 串行和 R3 重跑的描述。

## 执行顺序

```text
D1（整理现有改动）
  -> 第 0 阶段（锚点契约）
  -> 第 1a 阶段（只改文档，可与下面并行）
  -> 第 1b、第 3、第 4 阶段（都在本仓库，互不依赖，可以任意顺序）
  -> 第 2 阶段（仓库外补丁，工作量最大）
  -> 第 5 阶段（整体复测）
```

第 0 阶段必须最先完成。否则 Mac 会拒收每一个远端 chunk，其他改动带来的收益在真实 demo 中完全听不到。第 2 阶段放在后面，是因为它涉及环境构建和运行时身份，验证成本最高，而收益与第 1 阶段相比并不更大。

## 暂不做的更深优化

下面几项只记录，不在本计划范围内：

| 方向 | 估计收益 | 为什么暂不做 |
|---|---:|---|
| decode 的 CPU 工作与 GPU 重叠，或把采样、状态更新、embedding 放进 CUDA graph | 约 −90 ms | 需要较深地改动 SGLang-Omni 的调度器 |
| MOSS 主干改用 FP8 权重 | 最多约 −200 ms | 必须做 ASR 和盲听的音质验收 |
| GEMM kernel 调优，目前约用到峰值带宽的 66% | 不确定 | 属于上游 kernel 层面的工作 |
| delay pattern 带来的约 33 步额外 decode | 约 240 ms | 模型结构决定，服务端无法去掉 |

## 实施记录

（每个阶段完成后在这里追加：日期、提交、实测数字、偏离计划之处。）
