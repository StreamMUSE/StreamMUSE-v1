# RAP 服务器端延迟优化：实施报告

日期：2026-10-01。分支：`feature/rap_optimization`，最后一个提交是 `484584e8`。

依据：

- [2026-10-01 优化计划](../plans/2026-10-01-rap-server-latency-optimization-plan.md)：各阶段逐条的实施记录，在计划末尾的“实施记录”一节；
- [2026-09-25 profiling 报告](2026-09-25-rap-moss-sglang-profiling-report.md)：第 10 节是这次的整体复测。

## 1. 直接结论

**计划的第 0 到第 5 阶段都已实现，并在 Mac 和 H200 上验收。** 改动前，Mac 会拒收每一个 `gentle_sparse_r3` 远端 chunk。现在 R3 拉伸在 Mac 上做，远端 chunk 可以正常使用。

同一张 H200 卡上，服务器准备一个两小节 chunk 的时间变化如下：

| 中位数（p90） | 改动前 | 改动后 |
|---|---:|---:|
| Qwen 写词 | 309（412）ms | 147（196）ms |
| MOSS | 826（849）ms | 769（789）ms |
| 服务器各阶段合计 | 1236（1330）ms | **1025（1077）ms** |
| 服务器 Opus 编码 | 129 ms | 60 ms |
| 服务器到首字节（约等于合计加 Opus） | 约 1.37 s | **约 1.09 s** |

- 计划的目标是服务器约 1.07 s，加上 Mac 端拉伸整体约 1.15 s。实测服务器约 1.09 s，加 Mac 的 R3（约 0.05 s）整体约 1.13 s，基本达到。
- 只有 Qwen 这一项没到计划的估计：估计约 105 ms，实测 147 ms。
- p90 与中位数的差距从 94 ms 缩小到 52 ms。
- MOSS 和 R3 两处的偶发长尾都已去掉。

**改动没有改变 MOSS 的输出：**

- 补丁前后 source WAV 60 / 60 逐字节相同。
- 同样输入下，Mac 端拉伸与服务器端拉伸输出逐字节相同。

**还需要你来做的：**

- 5 对盲听；
- 决定是否向上游提 SGLang-Omni 补丁；
- 更新组会 slides 的架构图；
- 决定 H200 那条路下 Opus 的码率，见第 4.2 节。

## 2. 做了什么

每个阶段单独一个提交。

| 阶段 | 提交 | 内容 |
|---|---|---|
| 准备 | `f82fcf60` | 改用 Python 3.12；为 Mac 本地的几个服务各建一个 uv 项目 |
| 0 | `93c04b78`、`83c224c4`、`a9198a0e` | R3 拉伸移到 Mac 端，公共协议升到 v2 |
| 1 | `cf615941` | 两个 bar 的每一轮候选（包括补救 wave）都并发生成；vLLM 改为 `--max-num-seqs 32` |
| 3 | `e722979d` | R3 输出长度误差在 20 ms 以内时直接补齐，不再整段重跑 |
| 4 | `330ff84e` | Opus 压缩级别默认从 10 改为 5，可以用参数切回 |
| 2 | `eeb1cd6d`、`6fc2b117` | SGLang-Omni MOSS 延迟补丁，以及显式的运行时身份 |
| 记录 | `484584e8` | H200 验收数据写入计划和 profiling 报告 |

### 2.1 第 0 阶段：R3 移到 Mac，协议 v2

**原来的问题**

- 服务器做完 R3 后，只下发稀疏的锚点。
- Mac 端却要求锚点数等于音节数，而且每个目标时间与自己的节拍表一致。
- 所以在默认的 `gentle_sparse_r3` 下，每个远端 chunk 都被拒收，demo 一直退回本地 eSpeak。

**现在的做法**

- 服务器做完 MOSS 和 MMS 就返回：原始人声（PCM16，样本原样保留），以及每个音节的 `source_onsets` 和 `onset_confidence`。
- Mac 用同一份代码（新增的共享模块 `phrase_warp.py`）重建渲染请求和锚点，再在本机跑 Rubber Band R3。

**协议和兼容**

- `request_id` 包含协议版本，所以 v1 和 v2 的服务器缓存不会混用。
- 服务器 `/health` 新增 `supported_schema_versions`。v2 客户端遇到只支持 v1 的服务器时，直接报错退出，不会静默退回。
- v1 完整保留，用于回滚。

**新增参数**

- demo：`--rap-protocol v1|v2`（默认 v2）、`--rap-warp-policy`；
- `scripts/client_simulation.py`：`--protocol`、`--warp-policy`。

**Mac 启动脚本**

- 改为 v2 加 `gentle_sparse_r3`。
- 补上了 Mac 计划里写了、但脚本里漏掉的 `--rap-render-reserve-ms 3500`。

### 2.2 第 1 阶段：Qwen 写词并发

**之前的状态**

- 两个 bar 的初始 wave 已经能并发。
- 补救 wave 仍然一个 bar 一个 bar 地串行。

**这次的改动**

- 每一轮里，所有还缺合法候选的 bar 同时发出，发出前只做一次预算检查。
- 结果仍按 bar 0、bar 1 的固定顺序入账，所以 ledger、`source_order` 和最终选择与串行完全一致。
- 执行器可以注入，测试可以模拟“bar 1 先返回”。
- `generation` 计时改为各轮墙钟时间之和。
- 文档里 H200 的启动命令改为 `--max-num-seqs 32`，render server 命令加上 `--concurrent-bar-generation`。

### 2.3 第 2 阶段：SGLang-Omni MOSS 补丁

补丁文件是 `patches/sglang-omni-0.1.4-af3ab61-moss-latency.patch`，原理和每处改动的等价依据见 `patches/README.md`。共三处：

1. **decode 的 32 张音频 embedding 表一次 gather。**
   - 和现有的 fused audio heads 一样，把各张表改指向同一块拼接 buffer 的切片，所以常驻显存不增加。
   - 累加顺序与原来相同，按构造就逐位等价。
2. **audio decode 的文本头只算 2 个控制 token。**
   - 逐项对照 SGLang `LogitsProcessor` 的数值路径。
   - 遇到 softcap、LoRA、量化，或需要 TP/DP gather 时，回退到原路径。
3. **断开检测任务取消后不再等待最多 0.1 s。** 生成任务的取消路径保持不变。

**上线方式**

- 补丁打在已安装包的一份副本上，通过 `PYTHONPATH` 放在最前面。pinned 环境本身不改。
- preflight 新增 `--runtime-patch-file`、`--runtime-patch-root`：
  - 确认补丁已经打上；
  - 把补丁哈希写入启动 manifest。
- render server 新增 `--moss-runtime-patch-sha256`。打补丁的产出因此有自己的指纹和缓存命名空间；不传这个参数，指纹就和原来一样。

### 2.4 第 3 阶段：R3 不再整段重跑

**先采集数据**

- 37 个 chunk、两种策略，共 74 次拉伸。
- 第一次输出有偏差时，全部是比目标短 3 到 173 ms。
- 偏短时，输出的最后 20 ms 全部是数字静音。
- 原来的终点修正会在小偏差附近来回振荡：最多跑 5 次，最长 221 ms。

**改动**

- 偏差在 480 帧（20 ms）以内时，直接补零或截断，接缝处加 5 ms 淡出。
- 超过 480 帧时，修正一次后重跑。
- 这个容限放在共享模块里，服务器（v1）和 Mac（v2）用同一个值。
- 每个 chunk 的 R3 偏差写进 Mac 会话日志，在事件里的新字段 `local_warp` 中。

### 2.5 第 4 阶段：Opus 压缩级别 5

- `FFmpegOpusCodec` 增加压缩级别参数，默认 5。级别会写进每个 Opus 包的编码器标识。
- render server 新增 `--opus-compression-level`，回退时传 10。
- 只在 Opus 模式下，级别才进入 producer 指纹，所以 PCM 模式的缓存不受影响。

## 3. 结果

### 3.1 Mac-local 验收（第 0 阶段）

Mac 本地栈：MLX q8 MOSS 加 mlx_chat_server Qwen，PCM 传输，90 BPM。

| 项目 | 门槛 | 结果 |
|---|---|---|
| `client_simulation` 24 个 chunk，gentle_sparse | 24 / 24 | **24 / 24**，Mac 端拒收 0 |
| 端到端耗时 | — | p50 3298 ms，p95 4142 ms |
| Mac 本地 R3 | p95 ≤ 150 ms | p50 47 ms，p95 89 ms |
| 输出一致性 | 逐字节相同 | 通过，用真实 rubberband 4.0.0，两种策略都测过 |
| 真实会话（24 bar） | underrun 0 | **0**；22 / 24 bar 用了远端人声 |

真实会话里有 2 个 bar 用了 fallback，原因见第 4.4 节。

### 3.2 R3（第 3 阶段）

| 指标 | 改动前 | 改动后 |
|---|---:|---:|
| 最多运行次数 | 5 | **2** |
| gentle 的 p90 / max | 45 / 208 ms | 45 / **85** ms |
| all_onsets 的 p90 / max | 89 / 221 ms | 85 / **89** ms |

- 被补齐的部分全部接在数字静音后面，补齐前后的样本完全相同，不会产生爆音。
- 这批数据里没有出现偏长需要截断的情况。截断路径的淡出由单测覆盖。

### 3.3 Opus 级别 5（第 4 阶段）

37 段原始人声，48 kbps VBR。

| 级别 | SNR 中位数 | 相关系数中位数 | 编码耗时（H200） | 大小 |
|---|---:|---:|---:|---:|
| 10 | 21.24 dB | 0.9963 | 129 ms | 30.4 KB |
| 5 | 23.33 dB | 0.9977 | **60 ms** | 37.4 KB |

- 级别 5 在每一段上都不比级别 10 差：SNR 高 0.9 到 1.7 dB。原因是低复杂度下 VBR 花了更多比特，包大约大 23%。
- 盲听还没做，见第 5 节。

### 3.4 MOSS 补丁（第 2 阶段，H200 上 ABBA 重启对比）

| 会话 | MOSS 中位数 | 200 次请求中的尖峰 | 显存 |
|---|---:|---:|---:|
| A1 未打补丁 | 834.7 ms | 3 | 23307 MiB |
| B1 打补丁 | 762.3 ms | 0 | 23259 MiB |
| B2 打补丁 | 766.5 ms | 0 | 23259 MiB |
| A2 未打补丁 | 827.8 ms | 0 | 23307 MiB |

- 尖峰的统计口径：比同一条语料的中位数慢 50 ms 以上即算一次。
- 打补丁后快 67 ms。
- 与未打补丁时 source WAV 60 / 60 逐字节相同。
- 显存没有增加。

### 3.5 整体复测（第 5 阶段）

- 两组配置：基线是 seqs 8、两个 bar 串行、未打补丁、Opus 级别 10；优化是全部改动都开。
- 设置：同一张 H200 卡、同一份代码，客户端是 Mac 上的 `client_simulation`，协议 v2，Opus 传输。数字见第 1 节。
- 两组都没有生成错误批次。补救 wave 的数量几乎一样：基线 14、优化 13。
- Mac 端验收：基线 22 / 24，优化 23 / 24。没通过的原因见第 4.4 节。

## 4. 需要注意的地方

### 4.1 部署和兼容

- **v2 是默认协议，Mac 端必须有 `rubberband`（R3）。** demo 启动时会先用一次真实拉伸确认可用。
- **服务器要先升级：** v2 客户端连旧的、只支持 v1 的服务器时会直接报错。如果只能用旧服务器，要用 `--rap-protocol v1`，并且服务器必须用 `all_onsets_r3`，否则仍然全部被拒。
- **缓存命名空间会换。** 协议版本、补丁哈希和 Opus 级别都进入了请求 ID 或 producer 指纹。所以上线后的第一批请求没有缓存可用，旧的缓存目录也不会被复用。
- **R3 的输出和以前不完全一样。** 采用新的长度容限后，原来要重跑的 chunk 现在接受第一次的结果，所以和改动前不再逐字节相同。服务器（v1）和 Mac（v2）之间仍然逐字节相同。
- **事件契约新增 `local_warp` 字段。** 网页端会忽略这个字段；如果有别的程序严格检查字段集合，需要同步修改。

### 4.2 Opus 与 v2

- **v2 下 Opus 压缩的是原始人声，然后才做 R3。** Opus 的失真会被 R3 一起拉伸。
- R3 是 phase vocoder，在它下游，波形 SNR 和相关系数已经没有意义：只加 1 LSB 听不出来的噪声，相关系数就掉到 0.58。所以改用频谱幅度指标比较，工具是 `scripts/check_v2_opus_warp_quality.py`。
- 比较结果：
  - **48 kbps 时，v2 明显比 v1 差：** 对数谱距离 2.73 dB，v1 是 2.04 dB；
  - **96 kbps 时追平**：1.81 dB，接近 R3 本身的波动下限。
- Mac-local 默认用 PCM，不受影响。**H200 加 SSH 隧道那条路如果用 Opus，要不要提到 96 kbps，需要你决定。** 目前没改代码。

### 4.3 补丁的维护

- **补丁只适用于 SGLang-Omni 0.1.4（`af3ab61`）和 SGLang 0.5.18。** 升级 SGLang-Omni 后必须重新做补丁，并重新跑正确性验证。
- **文本头这一处不是按构造逐位等价的。** 只取 2 行做矩阵乘，GEMM 的形状变了，所以相同输出靠实测保证（60 / 60）。换 GPU、驱动或 torch 版本后，需要重跑这项验证。
- 打补丁时要用 `patch -p1 -d`。实施中发现：补丁目录在 git 仓库内部时，`git apply` 会按仓库根目录解析路径。preflight 的检查已经修正（提交 `6fc2b117`）。

### 4.4 测量和遗留问题

- **H200 的测量都在 GPU 2 一张卡上。** vLLM、MOSS 和 render server 放在同一张卡，当天只有这张卡空闲。runbook 里是分两张卡部署，实际部署时数字可能略有差别。
- **端到端墙钟时间不能用来比较改动。** 当天学校 VPN 很慢，隧道上一次 `/health` 往返就要 1.3 到 1.8 s。客户端看到的端到端时间（基线 2157 ms，优化 1883 ms）主要是网络开销，比较时只看服务器端的数字。
- **尾部尖峰比预估的少。** 未打补丁时实测约 0.75%（400 次里 3 次），计划推断的是约 5%。打补丁后为 0。
- **`client_simulation` 没有播放时钟。** fallback 和 underrun 只能靠真实 demo 观察。这次真实会话只在 Mac-local 上跑过 24 bar；H200 加隧道没有跑真实会话。
- **计划外的两个发现，还没修：**
  - **`no_valid_candidates`：** 每一次没通过验收的 chunk 都是这个原因：Mac 真实会话里的 2 个 fallback bar（同一个 chunk），以及 H200 上基线 2 个、优化 1 个。Mac-local 和 H200 上都会出现。原因是候选歌词不合 flow，与传输和本次优化无关，值得单独查。
  - **Mac-local 上的 `render_failed`：** 出现过两次，都在 MOSS 之前就失败了，重试后成功。失败原因被重试成功后的清理一起删掉了。以后应该在服务器侧把 preflight 的失败原因写进日志。
- **全量单测：** 除了 5 个依赖 H200 数据的 fastpitch 用例（环境问题，改动前就失败），其余都通过。

### 4.5 回滚

| 想回退的部分 | 做法 |
|---|---|
| 协议 v2 | Mac 用 `--rap-protocol v1`，服务器用 `--moss-warp-policy all_onsets_r3` |
| Opus 级别 | render server 用 `--opus-compression-level 10` |
| MOSS 补丁 | preflight 不传两个补丁参数，render server 不传 `--moss-runtime-patch-sha256`，指纹会恢复原值 |
| 并发写词 | render server 去掉 `--concurrent-bar-generation` |
| R3 容限 | 只能改代码：共享模块里的 `R3_STRETCHER_OPTIONS` |

## 5. 待你决定或执行

1. **盲听：** 5 对文件在 `output/rap_local_mac_v2_accept/opus-blind/`，答案在 `opus-blind-key.json`，听完再打开。
2. **D4：** 是否向上游 sgl-project/sglang-omni 提补丁。
3. **H200 那条路的 Opus 码率：** 48 还是 96 kbps，见第 4.2 节。
4. **组会 slides 第 3 页的架构图：** R3 现在在 Mac 端。
5. **是否单独排查 `no_valid_candidates`。**

## 6. 证据位置

| 内容 | 位置 |
|---|---|
| Mac-local 验收、R3 采集、Opus 评估、盲听文件 | Mac 的 `output/rap_local_mac_v2_accept/` |
| H200 整体复测（客户端） | Mac 的 `output/h200_e2e_20261001/{baseline,optimized}` |
| MOSS 补丁的 ABBA 对比 | H200 的 `logs/latency_patch_20261001/`，含 `session.sh`、`measure.py`、`analyze.py` |
| H200 整体复测（服务器） | H200 的 `logs/e2e_20261001/`；启动脚本 `logs/latency_patch_20261001/e2e_stack.sh` |

这些 H200 脚本都按自己记录的进程组启停服务，适合在共享主机上复用。

测量结束后，我启动的 H200 服务已全部停止，GPU 2 已空出来；Mac 本地服务和 SSH 隧道也已关闭。
