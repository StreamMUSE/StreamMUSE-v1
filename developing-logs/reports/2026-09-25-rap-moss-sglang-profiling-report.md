# RAP / SGLang-Omni MOSS 推理与整条流程 profiling 报告

日期：2026-09-25。性质：真实 H200 上的单请求、热运行 profiling，覆盖 MOSS 内部与服务器端完整 RAP chunk 流程。不是 production 验收。

## 1. 直接结论

**当前服务器准备一个两小节 chunk 约需 1.49 秒，MOSS 占 55%，Qwen 写词占 20%，Opus 编码和落盘占 13%。** 已实测、低风险的改动合计可省约 350 ms。

> 2026-10-01 更正：初版关于 Qwen 并发的结论有误，已按复测数据改写第 1、5.1、7 节，见第 5.1 节说明。

另有一个比性能更要紧的问题：**在默认的 `gentle_sparse_r3` 策略下，Mac 端会拒收每一个远端 chunk。** 服务器只下发稀疏锚点，Mac 校验却要求锚点数等于音节数。这里 48 / 48 次被拒，切到 `all_onsets_r3` 后 8 / 8 通过。按当前代码，真实 demo 会一直退回本地 eSpeak。详见第 6.1 节。

| 可省时间（每 chunk，中位数） | 改动 | 证据 | 风险 |
|---:|---|---|---|
| 约 210 ms | vLLM `--max-num-seqs` 从 8 改为 32，再让两个 bar 用各自的客户端并发请求 | 复测：317 ms 降到 175 ms，再降到 104 ms | 低；前一半只改启动参数 |
| 约 77 ms | MOSS decode 的 embedding 改为一次 gather，文本头只算 2 个控制 token | 实测，同一实例配对 60 次，输出 WAV 60 / 60 逐字节相同 | 需给 pinned SGLang-Omni 打补丁或提上游 |
| 约 64 ms | Opus `compression_level` 从 10 改为 5 | 实测：125 ms 降到 61 ms，体积多 11% | 需确认音质 |
| 尾部 77 到 230 ms | 修正 R3 时长重试，约 25% 的 chunk 会重跑 rubberband，观测到 2 到 4 次，上限 6 次 | 实测重试次数与耗时 | 低 |
| 尾部约 100 ms | 修正 SGLang-Omni 语音接口的取消等待，约 5% 请求命中 | 时间戳定位到 HTTP 层，与 0.1 s 常量吻合 | 低，上游代码 |

还有更大的空间，但只是估算，需要额外工作验证：
- **decode 每步剩余的 CPU 空闲：** 约 90 ms，需要重叠调度或把整步放进 CUDA graph。
- **MOSS 主干改用 FP8 权重：** 可省数量级约 200 ms，需要音质验收。

## 2. 方法与环境

- 复用 2026-09-11 benchmark 的固定环境：SGLang-Omni 0.1.4 `af3ab61`、SGLang 0.5.18、MOSS-TTS-v1.5 `cdd3b91`、相同 `moss_tts.yaml`、相同参考音频和 20 条两小节歌词语料。
- MOSS 在 GPU 0；Qwen2.5-7B vLLM 0.24.0 在 GPU 1，参数同 quickstart；实验性补丁服务在 GPU 2；MMS 与 MOSS 共用 GPU 0。
- 完整流程通过真实 `streammuse-rap-render-server`（SGLang 后端、`realtime` 候选策略、Opus 传输）和真实 Mac 端 `RemoteMossChunkPreparationStrategy` 驱动。客户端跑在 H200 本机回环上，没有 Mac、SSH 路由和音频设备。
- 使用的工具：
  - SGLang-Omni 自带的请求级事件（stage 与 hop 耗时）；
  - torch profiler（kernel 级）；
  - 长度扫描（线性拟合每帧耗时）；
  - py-spy（CPU 调用栈）；
  - 同一实例内运行时开关补丁的配对 A/B 测试；
  - render server manifest 自带的阶段计时。
- 本机是 52 个用户共享的主机。测量期间观察到一次约 30 秒、加 220 ms 的外部干扰平台期，已排除，不计入结果。

**两条测量陷阱，以后复测要注意：**

1. torch profiler 停止后，服务会残留约 60 到 80 ms 的额外开销，直到重启服务。本次长度扫描因此重做了一遍，作废数据保留在 `moss_sweep_invalid_after_torch_profiler/`。
2. 不同服务实例之间存在稳定偏差。GPU 2 上的实例始终比 GPU 0 实例慢约 19 ms。所以补丁收益必须在同一实例内配对比较，不能拿两个实例直接相减。

## 3. 整条流程的时间线

默认 `gentle_sparse_r3`、Opus 传输，24 个 chunk，取中位数。

| 阶段 | 位置 | 中位数 | 尾部 | 占服务器端 |
|---|---|---:|---:|---:|
| Qwen 写候选（两个 bar 串行，各 n=16） | H200 GPU 1 | 299 ms | p90 371 | 20% |
| 候选分析与打分 | H200 CPU | 6 ms | | <1% |
| MOSS（SGLang-Omni，含 HTTP） | H200 GPU 0 | 815 ms | max 914 | 55% |
| MMS 强制对齐 | H200 GPU 0 | 20 ms | | 1% |
| R3 时间拉伸 | H200 CPU | 87 ms | p90 158，max 237 | 6% |
| 打包与落盘（manifest 自报 packaging） | H200 磁盘 | 42 ms | | 3% |
| Opus 编码及其后的发布和 HTTP | H200 | 187 ms | max 204 | 13% |
| **服务器端到首字节** | | **约 1.49 s** | | 100% |
| 客户端 Opus 解码（ffmpeg，Linux） | 客户端 | 74 ms | | |
| Mac 校验与混鼓（仅 all_onsets 下可测） | 客户端 | 11 ms | | |
| 回环客户端总墙钟 | | 1.58 s | p90 1.65 s | |

Qwen 每个 wave 平均约 137 个生成 token、332 个 prompt token。多数 chunk 只发出 2 个初始 wave。两轮共 48 个 chunk 中，有 9 个另外触发了 1 到 2 个 4 条候选的补救 wave，这部分解释了 Qwen 的 p90 尾部。

**和 5 秒滚动预算怎么对应：** 8 月的 Opus 实验测到 Mac 到 H200 的 SSH 路由在缓存命中时也要约 1.5 秒，见 [Opus 实验](../../docs/developer-guide/rap-opus-transport-experiment-2026-08-21.md)。若路由仍是这个水平，总耗时约为 1.5 s 服务器加 1.5 s 路由再加约 50 ms Mac 处理，约 3.1 秒，低于 5.0 秒超时。路由本次未测。

## 4. MOSS 内部拆解

### 4.1 阶段拆分

基于 SGLang-Omni 请求级事件，20 次请求取均值；客户端总耗时约 808 ms。

| 阶段 | 耗时 | 占比 |
|---|---:|---:|
| HTTP、API、WAV 封装（客户端总耗时减去服务端） | 约 11 ms | 1% |
| preprocessing（prompt、参考编码缓存命中） | 13 ms | 2% |
| 请求构建与排队 | 3 ms | <1% |
| prefill | 21 ms | 3% |
| **decode（101 步自回归）** | **731 ms** | **90%** |
| vocoder（68 帧 codec 解码） | 30 ms | 4% |

客户端 adapter 自身很轻：下载约 2 ms，WAV 校验约 0.7 ms。

### 4.2 decode 为什么是 101 步

- 所有输出 source 都是 130560 帧，也就是 68 个 codec 帧（12.5 Hz）。
- MOSS 使用 32 个 codebook 的 delay pattern：最后一个 codebook 比第一个晚 31 步，所以 68 帧需要约 101 步。
- trace 里每类层内 GEMM 出现 3672 次，等于 36 层乘 102 次前向（1 次 prefill 加 101 次 decode），与此吻合。
- **约 33 步不产生新帧，约 240 ms 是模型结构固有成本。** 服务端调参去不掉，只能靠换模型或改解码方式。

长度扫描确认了这一点：每多一个 codec 帧增加 7.3 ms，拟合截距 311 ms，在 68 帧处预测 810 ms，与实测一致。

### 4.3 每个 decode step 的构成（约 7.2 ms）

基于 torch trace，kernel 时长为 GPU 实测值；空闲时间用无 profiler 的实测步长扣除得出。

| 组成 | 每步 | 说明 |
|---|---:|---|
| 36 层主干（CUDA graph） | 约 5.2 ms | 约 13.9 GB 权重，每步 GEMM 约 4.4 ms，约 3.2 TB/s，为 H200 NVL 峰值带宽的约 66% |
| 全词表文本头 | 0.31 ms | 读 155648 行共 1.27 GB 权重，随后只取其中 2 列控制 token |
| 音频头、采样、状态更新、下一步 embedding 的 GPU 部分 | 约 0.43 ms | 约 190 个小 kernel |
| **GPU 等待 CPU 的空闲** | **约 1.3 ms** | 每步约 18%，合计约 130 ms |

py-spy 显示，这段 CPU 时间集中在调度线程上串行执行的几件事：
- MOSS 专用的采样后处理与 delay 状态更新（`_collect_moss_step`、`post_decode`）；
- 为下一步逐个查 33 张 embedding 表，每次都走一遍 Python 模块调用（`_prepare_multi_modal_inputs`）；
- SGLang 的 batch 记账。

MOSS 路径没有使用重叠调度，这段 CPU 工作无法与 GPU 前向并行。

### 4.4 已实测的两项 MOSS 改动

实验服务用只作用于该服务 PYTHONPATH 的 `sitecustomize` 打补丁，pinned 环境文件未修改。每次请求前通过开关文件切换配置。20 条歌词乘 3 轮，同一实例内轮换，每种配置 60 次。

| 配置 | 中位数 | p95 | 与无补丁配对差值 | 输出与无补丁相同 |
|---|---:|---:|---:|---:|
| 无补丁 | 830.6 ms | 835.0 | 0 | 60 / 60 |
| embedding 一次 gather | 785.6 ms | 789.8 | −44.8 ms | 60 / 60 |
| 文本头只算 2 个控制 token | 799.2 ms | 801.5 | −31.5 ms | 60 / 60 |
| HTTP 取消不等待 | 830.0 ms | 833.7 | −0.8 ms | 60 / 60 |
| 全部 | 754.0 ms | 757.9 | **−76.7 ms（−9.2%）** | 60 / 60 |

- **embedding 改动：** 先用一次 gather 取出 33 个通道，再按原来的顺序做 bf16 累加，因此逐位等价。实验补丁把约 15.5 万行的文本 embedding 表也拼进了大表，会多占约 1.5 GB 显存；只拼 32 张音频表时约多 0.3 GB。
- **文本头改动：** 只与 2 个控制 token 的权重相乘，本次输出同样逐字节相同。
- **HTTP 取消改动：** 只影响尾部。这个实例在本轮恰好没有出现尖峰，所以中位数不变。

补丁代码在 `logs/profiling_20260925/exp_patches/sitecustomize.py`，只作实验用途。落地时应改上游 `sglang_model.py`，或在 pinned 版本上维护补丁。

### 4.5 MOSS 尾部尖峰

基线服务每 20 个请求左右出现一次约 +100 ms 的尖峰，恰好落在 p95。

- **位置：** 用墙钟对齐客户端与服务端事件后，尖峰发生在协调器发出 terminal response 之后、HTTP 响应送达之前。这一段平时 3.5 ms，尖峰时 104 ms。
- **原因：** 语音接口返回前会取消"客户端断开检测"任务，并最多等待 `HTTP_DISCONNECT_CANCEL_TIMEOUT_S = 0.1` 秒。偶尔取消没有及时生效，就会等满 0.1 秒。
- **修复：** 取消后不等待，改挂 done callback。实验服务上 A 组 40 次最大 794 ms，没有尖峰。

## 5. 其他环节的发现

### 5.1 Qwen：瓶颈在 vLLM 的并发序列上限

planner 先对 bar 0 发出 n=16 的请求，完成后再发 bar 1 的请求。两个请求的 prompt 互不依赖。

**更正：** 初版报告称"两个 wave 并发只需 153 ms"，这个结论是错的。当时两个并发请求共用同一个 `LocalChatModelClient`，而该客户端同一时间只允许一个请求，第二个请求会立刻失败并被生成器吞成空的错误批次。复测时共用客户端的并发组有 12 / 24 个错误批次。

2026-10-01 在 GPU 2 上复测，每个 wave 都检查了候选数量，下表中没有错误批次：

| vLLM 配置 | 单个 wave | 两个 bar 串行 | 两个 bar 并发（各用一个客户端） |
|---|---:|---:|---:|
| `--max-num-seqs 8`（quickstart 当前值） | 约 152 ms | 317 ms | 311 ms |
| `--max-num-seqs 32` | 93 ms | 175 ms | 104 ms |

- 一个 n=16 的请求就是 16 条序列。上限为 8 时，单个请求本身就要分两轮解码，两个请求并发也只是排队，所以没有收益。
- 只把上限改为 32，不改代码，就从 317 ms 降到 175 ms。
- 再让两个 bar 用各自的客户端并发，进一步降到 104 ms。补救 wave 也可以同样按 bar 并发。

### 5.2 R3：时长拟合的重跑造成长尾

- 生产 renderer 使用实验模块的 `RubberBandTimeMapStretcher`，单次约 77 ms。
- 如果输出帧数与目标差超过 2 帧，就修正终点再跑一次。代码上限是 6 次，本次观测到的最多为 4 次。
- 20 条语料里有 5 条需要重跑：4 条跑 2 次，1 条跑 4 次。重跑由歌词决定，两轮测试结果一致。
- 这就是 R3 的 p90 157 ms、max 301 ms 的来源。

可选修法：
- 允许更宽的终点误差，再做尾部补齐或裁剪（代码里已有 `_enforce_target_length`）；
- 或者按第一次的偏差一次性预测修正量。

### 5.3 Opus 与打包

- 服务端 Opus 编码是一次 ffmpeg 子进程，`compression_level 10`，约 125 ms。其中仅启动进程就约 39 ms。

| compression_level | 编码 | 体积 |
|---:|---:|---:|
| 10（当前） | 125 ms | 32.0 KB |
| 5 | 61 ms | 35.7 KB |
| 0 | 52 ms | 32.0 KB |

- 改用进程内 libopus 可以再去掉进程启动开销。这是估算，未实测。
- 客户端解码在 Linux 上约 63 到 74 ms，8 月在 Mac 上测到约 38 ms。
- packaging 约 42 ms：服务端先完整演练一遍发布流程（写 manifest、package、timing 并逐个 fsync），测出耗时，写回 manifest 后再正式发布一遍。每个文件都 fsync 文件和目录。这部分是为了可审计性，若要省时间，可以考虑异步持久化或去掉演练。

## 6. 需要优先处理的非性能问题

### 6.1 Mac 端拒收所有 gentle_sparse_r3 chunk

- `MossAlignedPhraseRenderer` 下发的 `alignment_diagnostics.source_anchors` 是实际生效的稀疏锚点，本次样例为 7 个。
- `RemoteMossChunkPreparationStrategy._validate_manifest` 要求锚点数等于两句的音节总数 18，否则报错："remote alignment anchors do not cover every selected syllable"。
- 这段逻辑在 HEAD 已提交的代码里就存在，不是本次未提交改动引入的。
- 服务器默认 `--moss-warp-policy gentle_sparse_r3`，所以真实 demo 中每个远端 chunk 都会被拒，退回 eSpeak。

| 策略 | Mac 端结果 |
|---|---|
| gentle_sparse_r3（默认） | 24 / 24 与 24 / 24 被拒 |
| all_onsets_r3 | 8 / 8 接受 |

需要决定是让服务器同时下发完整的逐音节诊断锚点，还是放宽 Mac 端校验。之后的 `_placement_diagnostics` 也按逐音节偏移读取锚点，两边要一起改。

## 7. 建议的优化顺序

1. **修复第 6.1 节的兼容问题。** 否则所有服务器端优化都不会被听到。
2. **vLLM `--max-num-seqs` 改为 32，两个 bar 的 Qwen wave 用各自的客户端并发。** 约 −210 ms。
3. **MOSS 的两个补丁。** 约 −77 ms，输出不变。同时修复 HTTP 取消等待，消除 p95 尖峰。
4. **修正 R3 重跑。** 去掉 25% chunk 的 77 到 230 ms 长尾。
5. **Opus 降到 `compression_level 5`，或改为进程内编码。** 约 −64 到 −100 ms，需确认音质。
6. **更深的 MOSS 优化，需要更多工程或验证：**
   - decode 重叠调度，或把采样、状态更新、embedding 并入 CUDA graph，估计约 −90 ms；
   - FP8 主干，估计最多约 −200 ms，需 ASR 和盲听验收；
   - GEMM 目前只用到约 66% 带宽，kernel 调优也有空间。

前五项合计，服务器端中位数可以从约 1.49 s 降到约 1.1 到 1.2 s。

## 8. 这次没有覆盖的

- Mac 到 H200 的真实路由、Mac 上的 Opus 解码、音频设备播放和长时间稳定性。
- inprocess 旧后端的同等拆解。本次只看 SGLang-Omni 路径。
- 并发多会话和冷启动。所有数字都是热运行、单请求串行。
- 补丁对音质的影响只验证了逐字节相同。Opus 压缩级别和 FP8 的音质都未评估。

## 9. 复现与证据

新增两个脚本（未提交）：

- [profile_sglang_moss.py](../../scripts/profile_sglang_moss.py)：
  - `render` 跑 MOSS、MMS、R3，带子步骤计时和服务端事件；
  - `torch` 抓 kernel trace；
  - `sweep` 做长度扫描。
- [client_simulation.py](../../scripts/client_simulation.py)：按 Mac controller 的方式连续请求 render server，记录各阶段、传输、解码、混音耗时和 vLLM token 计数。

全部原始数据在 `logs/profiling_20260925/`，该目录被 git 忽略：

| 内容 | 路径 |
|---|---|
| MOSS 渲染与服务端事件 | `moss_render/`、`moss_render2/` |
| kernel trace | `torch_traces/trace2/` |
| 长度扫描 | `moss_sweep/` |
| py-spy 采样 | `pyspy_sglang.raw`、`pyspy_worker_summary.txt` |
| 补丁配对实验 | `paired_toggle.jsonl`、`paired_toggle_summary.txt`、`exp_patches/` |
| 端到端（gentle_sparse / all_onsets） | `e2e_opus/`、`e2e_opus_allonsets/` |
| Qwen 与 Opus 微基准 | `qwen_wave_bench.txt`、`opus_microbench.txt` |
| 启动脚本 | `start_*.sh`、`env.sh` |

补丁服务、render server、两个 SGLang 服务和 vLLM 均已正常停止，8 张 GPU 已全部释放。
## 10. 2026-10-01 优化后的复测

依据：[2026-10-01 优化计划](../plans/2026-10-01-rap-server-latency-optimization-plan.md)的第 0 到第 4 阶段全部完成后，按测量协议复测。

**设置**

- H200 GPU 2：vLLM、SGLang-Omni MOSS 和 render server 都在这张卡上，当天只有这张卡空闲。
- 客户端：Mac 上运行 `scripts/client_simulation.py`，走 SSH 隧道。24 个 chunk 加 2 个热身，协议 v2，`gentle_sparse_r3`，Opus 传输。
- 两组配置先后测，用的是同一份代码。
- 原始数据：Mac 的 `output/h200_e2e_20261001/{baseline,optimized}`，以及 H200 的 `logs/e2e_20261001/`。

| 配置 | vLLM `--max-num-seqs` | 两个 bar 并发 | MOSS 补丁 | Opus 压缩级别 |
|---|---:|---|---|---:|
| 基线 | 8 | 否 | 否 | 10 |
| 优化 | 32 | 是 | 是 | 5 |

| 中位数（p90），ms | 基线 | 优化 | 差值 |
|---|---:|---:|---:|
| Qwen generation | 309（412） | 147（196） | −162 |
| MOSS | 826（849） | 769（789） | −57 |
| MMS | 20.5 | 20.5 | 0 |
| 服务器各阶段合计（`total`） | 1236（1330） | **1025（1077）** | −211（p90 −253） |
| 服务器 Opus 编码，H200 上单独测，24 段 | 129 | 60 | −69 |
| Mac 本地 R3 | 48（50） | 49（50） | — |
| Mac 端验收 | 22 / 24 | 23 / 24 | — |

**结论**

- 服务器到首字节约为 `total` 加 Opus，从约 1.37 秒降到约 1.09 秒，计划目标是约 1.07 秒。加上 Mac 端的 R3（约 0.05 秒），整体准备时间约 1.13 秒，目标约 1.15 秒。
- p90 与中位数的差距从 94 ms 缩小到 52 ms。
- 两组都没有 `generation_error` 批次：基线 26 个 chunk、优化 26 个 chunk，错误都是 0。
- 补救 wave 的数量几乎一样：基线 14、优化 13。
- 没通过验收的 chunk，基线 2 个、优化 1 个，全部是服务器返回 `no_valid_candidates`。原因是候选歌词不合 flow，与传输和本次优化无关。
- Qwen 只降到 147 ms，计划估计的是约 105 ms。剩下的主要是 n=16 本身的解码时间。
- **注意：** 当天学校 VPN 很慢，隧道上一次 `/health` 往返就要 1.3 到 1.8 秒。所以客户端看到的端到端墙钟时间（基线 2157 ms，优化 1883 ms）主要是网络开销，不能用来比较服务器的改动。
