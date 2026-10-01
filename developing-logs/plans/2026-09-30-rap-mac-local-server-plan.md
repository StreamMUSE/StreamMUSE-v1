# RAP 服务端迁移到本地 Mac Studio 的实施计划（2026-09-30）

## 目标

把 rap chunk 的服务端（Qwen 写词、MOSS 合成、MMS 对齐、R3 拉伸、打包）从 H200 搬到本地 Mac Studio，去掉 Mac 与 H200 之间的 SSH 链路。

这次迁移的目的是验证整套 rap 系统能否在 edge device 上实时运行。H200 只能通过学校 VPN 访问，不作为改进网络链路的方向。

前提：

- 不降低 chunk 契约的校验强度；
- 不牺牲 producer 指纹的可追溯性；
- 保留 H200 路径，作为并列的可选后端。

本机配置：

- Mac Studio（Mac17,14），Apple M5 Max，18 核 CPU，40 核 GPU；
- 64 GB 统一内存，内存带宽约 614 GB/s；
- macOS 27.0.1。

依据：

- [profiling 报告](../reports/2026-09-25-rap-moss-sglang-profiling-report.md)
- [SGLang-Omni 基准报告](../reports/2026-09-11-sglang-omni-moss-server-benchmark-report.md)
- [Opus 传输实验](../../docs/developer-guide/rap-opus-transport-experiment-2026-08-21.md)
- [远端 MOSS 验收](../../docs/developer-guide/realtime-remote-moss-acceptance-2026-08-21.md)

## 核心风险：省下的是通信，多出来的是 MOSS 计算

迁移能直接省下约 1.7 s：

- SSH 链路约 1.5 s。这是 8 月的数据：一个 36 KB 的缓存响应走隧道要 1.5 s，在 H200 本机上只要 5.7 ms。
- Opus 的编码、发布和解码约 0.26 s。走 loopback 可以直接传 PCM。

但 MOSS 会明显变慢：

- MOSS-TTS-v1.5 是 8B 模型。2 小节的 chunk 大约要 decode 101 步，每一步都要把整份权重读一遍，速度基本由内存带宽决定。
- H200 的带宽约 4.8 TB/s，M5 Max 约 614 GB/s，相差约 8 倍。
- mlx-audio 公布的 MOSS-TTS-v1.5 8-bit MLX 版本，在 M4 Max 上约 1.5 倍实时，bf16 约 0.8 倍实时。
- 换算到 M5 Max：一个 5.33 s 的 chunk，MOSS 大约要 3.0 到 3.6 s；H200 上是 0.82 s。

**按目前的 MLX 实现，本地端到端大概率比现在的远端更慢，而且离实时的时间上限很近。** 因此第 0 阶段是一个可行性关口：先实测，看能否满足实时约束，再决定是直接迁移，还是先做 MLX decode 优化。

上限是存在的：

- 8-bit 权重约 9 GB，614 GB/s 下每步的带宽下限约 14 ms，101 步约 1.4 s；4-bit 可以再减半。
- M4 Max 上实际每步约 35 ms，多出来的主要是每步的调度开销，不是带宽。
- 这和 H200 上的情况类似：SGLang 用 CUDA graph 等手段去掉调度开销后，MOSS 从 2.6 s 降到了 0.8 s。
- 所以本地方案要真正跑赢，大概率需要第 6 阶段的 MLX decode 优化。

## 非目标

- 不改伴奏模型（Stanley/RoFormer）的推理链路。
- 不改 Mac 端的 controller、fallback、混音和播放逻辑。只调整时间预算参数，见第 4 阶段。
- 不删除 H200 / SGLang-Omni 路径及其代码。
- 不改歌词生成的设计：两个 bar 独立生成、事后配对的语义保持不变。
- 不更换 TTS 模型。sglang-omni 在 Mac 上实测的 "MOSS-TTS Local" 是 MOSS-TTS-Local-Transformer，与我们使用的 MOSS-TTS-v1.5（MossTTSDelay 架构）不是同一个模型，见"暂不做"一节。

## 决定事项

| # | 决定 | 结论 / 建议 | 状态 | 影响哪个阶段 |
|---|---|---|---|---|
| D1 | 迁移前是否先重测 H200 链路、试 VPN 直连 | 不做。H200 只能走学校 VPN，本次目标就是 edge device | 已定（2026-09-30） | 第 0 阶段 |
| D2 | MOSS 的量化级别 | 先用 8-bit；4-bit 必须通过第 5 阶段的音质门槛才能用 | 已定（2026-09-30） | 第 2、5 阶段 |
| D3 | 是否接受 MLX 输出与 PyTorch 参考不逐位一致，改用统计方式（WER、盲听）评估音质 | 接受。原因见下方说明。Mac 产物记为新的 producer 身份，不和 H200 产物逐字节比较 | 已定（2026-09-30） | 第 2、5 阶段 |
| D4 | 每个 chunk 的**时间**不够时（不是内存，内存够用），能否用歌词质量换延迟：Qwen 候选数从 16 降到 8，或者 7B 换 3B | 不降。保持 Qwen2.5-7B 和 `realtime_default` 的候选数量。时间不够只能靠第 6 阶段优化，或者放宽 lookahead；是否达标用"无时间约束参考运行"来衡量，见 0d | 已定（2026-09-30） | 第 0、4、5 阶段 |

Mac 路径通过启动脚本显式选择 `mlx` 后端。代码里现有的默认值不改，H200 路径原样保留。

**D3 说明：为什么不一致。** 按影响从大到小：

1. **随机数生成器不同。** MOSS 是 temperature 1.7 的随机采样。PyTorch 和 MLX 的随机数算法不同，同一个 seed 会得到不同的随机数序列。只要某一步抽到的 token 不同，后面的生成就全部不同。所以即使不量化，输出也会是另一段音频。
2. **8-bit 量化**：每 64 个权重一组，用缩放系数加 8-bit 整数来近似，logits 会有轻微偏移。
3. **算子实现不同**：matmul、softmax、RMSNorm 在 Metal 和 CUDA 上的累加顺序不同，末几位的数值会不同。
4. **模型代码是社区重写的**：codec、delay pattern 和采样器都是重新实现的。

## 预期耗时对比

以 90 BPM、2 小节为一个 chunk，时长 5.33 s，MOSS 的 `token_count=67`。

| 阶段 | H200 现状（中位数） | Mac 本地（估算） | 证据状态 |
|---|---:|---:|---|
| Qwen，2 bar × 16 个候选 × 32 token | 299 ms（2026-10-01 plan 做完后约 104 ms） | 0.4 到 1.0 s | Mac 端为估算，待测 |
| 候选打分 | 6 ms | 约 6 ms | 纯 CPU |
| MOSS | 815 ms | 3.0 到 3.6 s（当前 8-bit MLX）；带宽下限约 1.4 s | 根据 M4 Max 公开数据推算，待测 |
| MMS | 20 ms | 30 到 100 ms | 待测 |
| R3 | 87 ms（p90 158 ms） | 80 到 150 ms | 待测 |
| 打包 | 42 ms | 约 42 ms | |
| Opus 编码、发布、HTTP | 187 ms | 10 到 30 ms（loopback PCM） | 估算 |
| 链路 | 约 1.5 s（8 月测量，之后未复测） | 约 0 | H200 一侧不再复测（D1） |
| 客户端解码、校验、混音 | 74 + 11 ms | 约 11 ms（PCM 不需要解码） | |
| **端到端** | **约 3.1 s** | **约 3.6 到 4.9 s** | |

两条硬约束：

- rolling timeout 是 5.0 s，deadline 还会被卡在下一个 chunk 开始前一个 tick；
- 同一时间只有一个远端请求，所以每个 chunk 的服务时间必须小于 5.33 s，否则请求会越积越多。

## 组件替换总表

| 组件 | H200 上 | Mac 上 | 改动量 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | vLLM 0.24.0（CUDA） | vllm-metal（vLLM 的 MLX 后端插件），权重用 MLX 4-bit | 只改启动方式；客户端不变，前提是 `n` 可用 |
| MOSS-TTS-v1.5 | SGLang-Omni（CUDA graph、FA3、FlashInfer） | mlx-audio，权重 8-bit MLX | 新增 `mlx` 后端 |
| MOSS Audio Tokenizer | SGLang 内置，BF16 | mlx-audio 内置 | 随 MOSS 一起 |
| MMS 对齐 | torchaudio，CUDA | torchaudio，MPS 或 CPU | 改设备默认值 |
| R3 拉伸 | `rubberband` CLI | `brew install rubberband` | 无 |
| 传输编码 | Opus | PCM | 改参数 |
| 链路 | SSH `-L 8020` | loopback `127.0.0.1:8020` | 客户端默认值本来就是这个 |

不能直接使用的部分：

- SGLang-Omni 对 Apple Silicon 的支持还是实验性的，目前只正式支持 Qwen3-ASR 和 Fun-CosyVoice3。
- [preflight_sglang_omni_moss.py](../../scripts/preflight_sglang_omni_moss.py) 要求 `nvcc`。
- 上游 vLLM 的 macOS 后端只能跑 CPU，速度不够。

## 通用规则

### 测量协议（Mac 版）

- **完整流程：** 用 `scripts/client_simulation.py`，24 个 chunk 加 2 个热身，与 2026-10-01 plan 的协议一致。区别是传输改为 PCM，并且不经过 SSH。
- **MOSS 单项：** 用 `scripts/benchmark_moss_server.py` 的 20 条固定语料，`token_count=67`，记录 p50、p95 和每步耗时。
- **机器状态：**
  - 接电源，关闭低电量模式；
  - 关闭其他占用 GPU 的应用，比如浏览器的视频和 IDE 的预览；
  - 在记录中写明 macOS、mlx、mlx-audio、vllm-metal 的版本和 commit。
- **热身：** 第一次运行会下载权重、编译 Metal kernel，这部分时间单独记录，不计入稳态。
- **热降频：** 连续跑 24 个 chunk，比较前 8 个和后 8 个的 p50，漂移超过 5% 要在记录里注明。
- **比较方式：** 同一进程内配对比较；确实需要重启时按 ABBA 顺序。
- **进程管理：** 只按自己启动时记录的 PID 停止服务。

### 提交与测试

- 每个阶段一个提交。提交前运行该阶段相关的单测，以及：
  `uv run pytest tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation -q`
- 仓库外的运行时，也就是 mlx-audio、vllm-metal 和模型权重，必须显式写入 producer 运行时身份，做法与 sglang-omni 的固定身份参数一致，见第 2 阶段。
- 每个阶段完成后，把实测数字追加到本计划的"实施记录"一节。

---

## 第 0 阶段：前置工作与决策关口

### 0a. 锚点契约修复（依赖）

先完成 [2026-10-01 plan](2026-10-01-rap-server-latency-optimization-plan.md) 的第 0 阶段：

- 在 `gentle_sparse_r3` 下，服务端只发稀疏锚点；
- 但 Mac 端 [chunk_audio.py:409](../../src/streammuse/application/rap/chunk_audio.py#L409) 要求锚点数等于音节数，所以远端 chunk 全部被拒（48 / 48）；
- 不修复这一点，后面所有端到端测量都没有意义。

### 0b. Mac 环境

0. **存储位置（先于任何下载）**：模型权重和所有大的缓存都放在外接 SSD `/Volumes/ZBW-SSD1` 上，不占内置硬盘。
   - 这块 SSD 是 APFS 格式、USB 接口，2026-09-30 时剩余约 282 GB。卷名已经去掉了空格，路径可以直接使用。
   - 缓存根目录：`/Volumes/ZBW-SSD1/streammuse-cache/`。
   - 在 `~/.zshenv` 里统一设置下面这些环境变量，并写进一键启动脚本：

     | 变量 | 指向 | 用途 |
     |---|---|---|
     | `HF_HOME` | `/Volumes/ZBW-SSD1/streammuse-cache/huggingface` | MOSS、Qwen 等 HF 权重，mlx-audio 和 vllm-metal 都通过它下载 |
     | `HF_HUB_CACHE` | `$HF_HOME/hub` | 同上 |
     | `TORCH_HOME` | `/Volumes/ZBW-SSD1/streammuse-cache/torch` | torchaudio 的 MMS_FA；渲染服务的 `--aligner-cache` 也指到这里 |
     | `XDG_CACHE_HOME` | `/Volumes/ZBW-SSD1/streammuse-cache/xdg` | 其他按 XDG 规范存缓存的库，比如 whisper |
     | `UV_CACHE_DIR` | `/Volumes/ZBW-SSD1/streammuse-cache/uv` | uv 下载的 wheel，torch 之类的包很大 |
     | `UV_PYTHON_INSTALL_DIR` | `/Volumes/ZBW-SSD1/streammuse-cache/uv-python` | uv 管理的 Python 解释器 |
     | `PIP_CACHE_DIR` | `/Volumes/ZBW-SSD1/streammuse-cache/pip` | pip 缓存 |
     | `VLLM_CACHE_ROOT` | `/Volumes/ZBW-SSD1/streammuse-cache/vllm` | vllm-metal 缓存 |

   - 启动脚本要先检查 SSD 是否已挂载，没挂载就直接报错退出，不能静默写到内置硬盘。
   - USB 的速度只影响冷启动时加载权重，不影响稳态推理。测冷启动时间时要注明权重是从 USB SSD 读的。
1. 系统依赖：`brew install espeak-ng portaudio ffmpeg rubberband`
2. 主项目环境：`uv sync`。确认它在这台机器上能通过；`deepspeed`、`tensorflow`、`nvitop` 都在主依赖里，如果装不上，就移到 extras。
3. 模型服务单独建环境，不进主项目的 `pyproject.toml`，和 H200 上 SGLang 单独一个环境的做法一致：
   - `rap-mlx-moss`：`mlx`、`mlx-audio`。固定 commit，因为 MOSS 的 MLX 移植还在快速变化。
   - vllm-metal：`brew tap vllm-project/vllm-metal && brew install vllm-project/vllm-metal/vllm-metal`。
   - 原因：主项目使用 vendored 的 `transformers` fork，和 mlx-audio、vLLM 依赖的 transformers 很可能冲突。
4. 权重：
   - MOSS-TTS-v1.5 的 8-bit MLX 权重，例如 `luisarn/MOSS-TTS-v1.5-MLX-8bit`，要核对它转换自哪个上游 revision；
   - 需要时自行从固定的上游 revision `cdd3b911` 转换 4-bit；
   - Qwen2.5-7B-Instruct 的 MLX 4-bit 权重。
   - 所有权重都固定 revision 并记录 sha256。

### 0c. 微基准

| 项目 | 候选 | 记录 |
|---|---|---|
| MOSS | MLX 8-bit / 4-bit / bf16；PyTorch MPS bf16 作为对照，需先做第 3 阶段的 3b | p50、p95、每步耗时、峰值内存 |
| Qwen | vllm-metal 加 4-bit 权重：先确认 `n=16` 确实返回 16 个 choices；再测两 bar 串行、两 bar 并发、`max-num-seqs` 32 | p50、p90 |
| MMS | MPS 和 CPU | p50 |
| R3 | Mac CPU | p50、p90、整段重跑的比例 |

### 0d. 无时间约束参考运行

目的：知道同一条流程在没有时间压力时会产出什么。以后可以衡量实时运行因为时间限制损失了多少，这也是 D4 不降质量的衡量基准。

现成的工具：[client_simulation.py](../../scripts/client_simulation.py)。它是一个客户端模拟器：
- 服务端和客户端的核心逻辑都是真实的，只模拟了 Mac controller 发请求的方式；
- 省略了播放时钟、音频设备和 fallback 逻辑。

所以它测不出 fallback 比例和 underrun，这两项要用真实的 demo 来测。

- 它和 Mac controller 的发请求方式一致：连续请求 2-bar chunk，使用 `realtime_default` 策略，把已选中歌词的最后四行作为下一个请求的 context，形成一条滚动的 context 轨迹。
- 它走真实的客户端校验路径，也就是 `RemoteMossChunkPreparationStrategy`。
- seed 固定为 `--seed + index`。
- `--budget-ms` 同时决定请求里的 `remaining_budget_ms` 和客户端的 deadline。服务端对这个值没有上限，只要求是正整数（[remote_chunk.py:529](../../src/streammuse/domain/rap/remote_chunk.py#L529)）。
- 把 `--budget-ms` 设成 60000 时，每个 chunk 都能完整跑完 16 个初始候选，需要时再加 4 个 rescue 候选，以及 MOSS、MMS、R3，不会被 `render_reserve_ms` 截断。

运行方式：

```bash
uv run python scripts/client_simulation.py \
  --render-url http://127.0.0.1:8020 --vllm-url http://127.0.0.1:8001/v1 \
  --transport pcm --budget-ms 60000 --chunks 24 --seed 20260925 \
  --out output/rap_mac_unconstrained_<date>
```

需要的小改动：

1. 增加 `--save-audio`：把每个 chunk 解码后的人声，以及和鼓混音后的 WAV 写进 `--out`，方便试听和跑 ASR。现在脚本只解码校验，不保存音频，只在 `records.jsonl` 里记录选中的歌词。
2. `vllm_counters` 读的是 vLLM 的 metrics。如果 vllm-metal 不提供这个接口，脚本要能跳过，不能报错。

和实时运行比较时要注意：

- 只要某个 chunk 选中的歌词不同，后面的 context 轨迹就会不同。2026-08-16 的候选池实验已经记录过这个限制。
- 所以要分两种比较：
  - **整体比较**：两边各自滚动 context，比较每个 chunk 被接受的比例、选中候选的分数分布、WER 和盲听结果；
  - **逐 chunk 比较**：实时运行时，强制使用参考运行记录的 context 和 seed，逐个 chunk 对比候选数量、最高分和选中的歌词。这需要给脚本加一个 `--replay-context <records.jsonl>` 选项。

这个参考运行同时也是 Mac 上最完整的端到端基准：它记录了各阶段在不被截断时的真实耗时，可以直接作为关口的依据。

### 关口：可行性

关口看的是能否满足实时约束，不要求比 H200 快。H200 的已有数据只作参考。

根据 0c 和 0d 的结果，得到 Mac 本地的端到端耗时，也就是各阶段 p95 之和。满足下面两条，就进入第 1 阶段：

1. 端到端 p95 低于 4.0 s，给 5.0 s 超时留出余量；
2. 单个 chunk 的服务时间稳定低于 5.33 s 的 chunk 周期，连续 24 个 chunk 的 p50 不出现漂移。

不满足时，先做第 6 阶段的 MLX decode 优化，达标后再回到第 1 阶段。第 1 阶段（Qwen）不依赖 MOSS，可以同时进行。

---

## 第 1 阶段：Qwen 改用 vllm-metal

### 现状

- 渲染服务通过 [local_chat_client.py](../../src/streammuse/infrastructure/inference/local_chat_client.py) 调用 OpenAI 兼容的 `/chat/completions`，靠 `n=16` 一次拿到 16 个候选（:153）。
- 同一个客户端同时只允许一个请求。
- 启动时会探测 `GET {vllm}/models`。

### 步骤

1. 用 vllm-metal 在 `127.0.0.1:8001` 上启动 `qwen-rap`，参数尽量与 H200 一致：`--max-model-len 2048`；`--max-num-seqs` 按 0c 的结果决定。
2. 渲染服务的 `--vllm-url` 改为 `http://127.0.0.1:8001/v1`，其余代码不改。
3. 如果 0c 发现 `n` 不可用，或者被静默截成 1 个 choice，有两个备选：
   - 新增 `scripts/mlx_chat_server.py`：一个很薄的 OpenAI 兼容服务，用 `mlx_lm` 的批量生成来实现 `n`，共享的 prompt 只做一次 prefill；
   - 或者在生成器里把一个 `n=16` 的请求拆成 16 个并发请求。这需要放开"一个客户端同时只有一个请求"的限制，改动比前一个方案大。
   - 注意：`mlx_lm.server` 会忽略 `n` 参数，不能直接替换。
4. 与 2026-10-01 plan 第 1b 阶段的关系：
   - 那里是"两个 bar 各用一个客户端并发"；
   - 在 Mac 上，只有一块 GPU，两个并发请求会抢带宽，所以要实测并发和合成一个 batch 哪个更快，再定方案。

### 测试

- 新增一个契约测试：针对 vllm-metal 的响应，检查 `n=16` 时返回的 choices 数量和结构。实际调用的版本标成 integration，CI 里跳过。
- `test_local_chat_client.py` 保持不变并通过。

### 验收

- 一次 `n=16`、`max_tokens=32` 的请求，返回 16 个合法 choice。
- 两 bar 写词的 p90 小于 1.0 s。

---

## 第 2 阶段：MOSS 新增 `mlx` 后端

### 现状

- [rap_render_server.py:109](../../src/streammuse/presentation/rap_render_server.py#L109) 的 `_MOSS_SERVING_BACKENDS` 只有 `inprocess` 和 `sglang-omni` 两种。
- `sglang-omni` 后端通过 [sglang_moss_tts.py](../../src/streammuse/infrastructure/rap/sglang_moss_tts.py) 调用 loopback 的 `POST /v1/audio/speech`：
  - 参考音频以 `file://` 形式传递；
  - 带 `token_count` 和采样参数；
  - 返回 24 kHz 单声道 WAV；
  - 自带 deadline、取消、大小上限和格式校验。

### 步骤

1. **MOSS MLX 服务**：新增 `scripts/mlx_moss_server.py`，在 `rap-mlx-moss` 环境里运行，监听 `127.0.0.1:8030`。
   - 接口：`/health`、`/v1/models`、`/v1/audio/speech`，请求和响应格式与 SglangMossSynthesizer 使用的子集一致。
   - 如果 mlx-audio 自带的 server 能支持全部所需字段，就直接用它，只补缺少的部分。
2. **参数一致**：以下参数都要和 [moss_generation.py](../../src/streammuse/infrastructure/rap/moss_generation.py) 一致：
   - `MAX_NEW_TOKENS=256`；
   - audio temperature 1.7、top_p 0.8、top_k 25、repetition penalty 1.0；
   - `token_count` 控制时长（12.5 token/s）；
   - 用参考 WAV 做音色克隆；
   - seed 规则 `20260816 + chunk_index*1000 + (attempt-1)` 要真正传到 MLX 的采样器里。
3. **渲染服务**：
   - `_MOSS_SERVING_BACKENDS` 加上 `"mlx"`，`moss_device="external"`；
   - HTTP 客户端复用 `SglangMossSynthesizer`；把它的通用部分提出来，改成后端无关的名字，或者派生一个子类；
   - 沿用 loopback 地址检查和启动时的 warmup，并检查 warmup 输出是 24 kHz WAV。
4. **运行时身份**：仿照 sglang-omni 的固定身份参数，新增这些参数并写入 producer manifest 的指纹：
   - `--mlx-moss-version`、`--mlx-audio-commit`；
   - `--mlx-moss-model-revision`、`--mlx-moss-quantization`、`--mlx-moss-weights-sha256`。
   - 这样 H200 和 Mac 的产物不会混在一起（D3）。
5. **取消语义**：
   - MLX 生成同样是不可中断的；
   - 取消时关闭 HTTP 连接，与 sglang 后端的 `transport_closed_abort_unconfirmed` 行为一致；
   - 服务端要能在连接断开后尽快停止生成。

### 测试

- 参照 `test_sglang_moss_tts.py` 新增 `test_mlx_moss_tts.py`，覆盖：请求字段、seed 透传、deadline 截断、响应校验、取消。
- `test_rap_render_server.py`：
  - `mlx` 后端的参数解析和互斥规则；
  - 缺少任一固定身份参数时拒绝启动；
  - 非 loopback 地址时拒绝启动。
- `test_producer_manifest.py`：`mlx` 后端的身份信息进入指纹，并且和 sglang 的身份信息互不相同。

### 验收

- 20 条语料都能生成，输出是 24 kHz 单声道，帧数符合 `token_count`。
- MOSS 的 p95 达到第 0 阶段关口所需的值。
- 同一个 seed 运行两次，输出一致，或者把不一致的原因记录下来。

---

## 第 3 阶段：去掉 CUDA 硬编码、改传输方式、一键启动

### 3a. 设备默认值

- [rap_render_server.py:1695](../../src/streammuse/presentation/rap_render_server.py#L1695) 的 `--aligner-device` 和 [:1780](../../src/streammuse/presentation/rap_render_server.py#L1780) 的 MOSS 设备，默认值都是 `cuda`。
- 改为 `auto`，按 cuda → mps → cpu 的顺序选择，并把实际选中的设备写进 manifest。

### 3b. PyTorch inprocess 后端在 MPS 上的修正

这一步只在 0c 需要 MPS 对照，或者想保留 PyTorch 后端作为备选时才做：

- [moss_backend.py:82](../../scripts/rap_audio_backends/moss_backend.py#L82)：现在只要不是 CUDA 就用 fp32，MPS 上应该改为 bf16。
- [moss_backend.py:606](../../scripts/rap_audio_backends/moss_backend.py#L606)：现在只要不是 CUDA 就用 `eager`，MPS 上应该改为 `sdpa`。
- seed 设置加上 MPS 分支。
- `_configure_torch_backends` 只在 CUDA 上调用。

### 3c. MMS

- 声学模型的 emission 在 MPS 上算，`forced_align` 放在 CPU 上，如果 MPS 不支持的话。
- 锁定 torchaudio 版本，因为新版本在逐步移除 `forced_align`。
- torchaudio 不在主依赖里，要显式加上。

### 3d. 传输

- 本地配置使用 `--wire-audio-codec pcm`，跳过 Opus 编码和 Mac 端的 ffmpeg 解码。
- Opus 路径保留给 H200。

### 3e. 一键启动

新增 `scripts/run_rap_local_mac.sh`：

- 依次启动 vllm-metal、MOSS MLX 服务和渲染服务，每一步等健康检查通过再继续；
- 把 PID 写进文件，退出时只停掉自己启动的进程；
- 最后启动 demo；
- 不再需要 H200 quickstart 里的 `LD_LIBRARY_PATH`、`CUDA_VISIBLE_DEVICES` 和 SSH 隧道。

客户端默认就连 `http://127.0.0.1:8020`（[cli.py:137](../../src/streammuse/presentation/rap_demo/cli.py#L137)），不需要改。

### 测试

- 设备自动选择的单测：模拟 cuda、mps、cpu 三种情况。
- `test_moss_backend` 或对应测试：MPS 下的 dtype 和 attention 选择。

### 验收

- 一条命令就能在这台 Mac 上把整套服务拉起来，`/health` 正常，demo 能跑完 20 小节。

---

## 第 4 阶段：按 Mac 的实测数字重新调时间预算

- **`render_reserve_ms`**（[remote_chunk.py:198](../../src/streammuse/domain/rap/remote_chunk.py#L198)，现在是 3000）：
  - 改为 Mac 上 MOSS + MMS + R3 + 打包的 p95，再加 300 ms 余量；
  - 预留调大以后，留给 Qwen 的时间会变少，要确认 16 个初始候选还能在时间内生成完。
- **候选数量和 Qwen 模型保持不变**（D4）。时间不够时，只能用不降低质量的办法：第 6 阶段的优化，或者放宽 lookahead。
- **lookahead**：
  - [cli.py:584](../../src/streammuse/presentation/rap_demo/cli.py#L584) 把 `moss_aligned_remote` 的 lookahead 写死成 2；
  - 放宽到 3 可以多一些延迟余量，但处理速度仍然必须快于 5.33 s 一个 chunk；
  - 所以只有在"延迟超标、但处理速度够"的时候才改。
- **资源争抢**：
  - 推理与 PortAudio 播放、Web UI 在同一台机器上，要确认播放线程不会被饿死，即 underrun 为 0；
  - 内存估计：Qwen 4-bit 约 4.5 GB，MOSS 8-bit 约 9 GB，MMS 约 1.2 GB，加上 KV cache，64 GB 足够；
  - 需要时用 MLX 的 cache limit 限制内存占用，避免挤压系统内存。

### 验收

- 默认策略下，24 个 chunk 全部在 deadline 之前返回，并且被 Mac 端接受。

---

## 第 5 阶段：端到端验收

- **完整流程**：
  - 90 BPM 跑 20 小节以上，记录三项：远端 chunk 被接受的比例（8 月是 2 / 20）、underrun 次数、各阶段的 p50 和 p95；
  - 把 H200 的已有数据列在旁边作参考。
- **与无时间约束参考运行对比**（D4 的衡量方式）：使用同一组 seed 和同一个 scenario，做 0d 里的两种比较：
  - 整体比较：两边各自滚动 context；
  - 逐 chunk 比较：用 `--replay-context` 固定 context。
  - 实时运行选中候选的分数、可用候选的数量、WER，都不应该明显低于参考运行。门槛在跑完第一批后再定。
- **音质**：MLX 量化版本与 PyTorch 参考不逐位一致，需要检查：
  - 可懂度：用 [check_moss_benchmark_audio.py](../../scripts/check_moss_benchmark_audio.py) 跑 Whisper 转写，WER 和 H200 输出相比不能明显变差，具体门槛在跑完第一批后再定；
  - 时间精度：R3 拉伸后，节拍对齐的误差分布；
  - 盲听：5 对，Mac 和 H200 各一条，同一句歌词。
- **文档**：在 [rap-demo-quickstart.md](../../docs/developer-guide/rap-demo-quickstart.md) 里新增一节 "Mac 本地运行"。

---

## 第 6 阶段（可选）：深度优化

这部分在第 5 阶段之后做；如果第 0 阶段关口没过，就提前做。

| 方向 | 估计收益 | 说明 |
|---|---:|---|
| MLX 的 MOSS decode 步骤用 `mx.compile` 编译，32 个 codebook head 和采样融合成一次 | MOSS 从约 3.3 s 降到 1.5 到 2.0 s | 收益最大的一项。目标是每步接近 14 ms 的带宽下限。可以考虑把改动提交给 mlx-audio 上游 |
| MOSS 4-bit | 每步的带宽下限再减半 | 必须通过第 5 阶段的音质门槛 |
| 全部放进一个进程：Mac demo 直接调用 `RapChunkOrchestrator`，跳过 HTTP、ZIP 打包、fsync 预演和重复校验 | 约 −0.1 到 −0.25 s | 要另外保证 producer 指纹和契约校验的强度不变 |
| 下一个 chunk 的 Qwen 与当前 chunk 的 MOSS 流水线并行 | 视带宽争用而定 | 两者共用同一块 GPU 的带宽，收益要实测 |

## 执行顺序

```text
第 0a 阶段（锚点契约，依赖 2026-10-01 plan）
  -> 第 0b、0c、0d 阶段（环境、微基准、无时间约束参考运行）
  -> 关口：可行就继续 / 不可行就先做第 6 阶段
  -> 第 1、第 2 阶段（Qwen 和 MOSS，互不依赖，可以并行）
  -> 第 3 阶段（设备默认值、传输、一键启动）
  -> 第 4 阶段（预算重调）
  -> 第 5 阶段（端到端与音质验收）
  -> 第 6 阶段（按需）
```

## 风险

| 风险 | 影响 | 应对 |
|---|---|---|
| MLX 版本的 MOSS 达不到时间预算 | 本地方案不成立 | 第 0 阶段关口；第 6 阶段优化（4-bit 也在其中，但要过音质门槛） |
| vllm-metal 不支持 `n` 或行为不一致 | 候选数量不够，每个 chunk 都会失败 | 第 1 阶段第 3 步的两个备选 |
| MLX 移植的数值或采样与参考实现有差异 | 音质、时长控制不同 | 当作新的 producer 身份；第 5 阶段的音质门槛 |
| mlx-audio 的 MOSS 移植变化快，或有 bug | 结果不可复现 | 固定 commit 和权重 sha256，写入指纹 |
| 推理与播放共用一台机器 | 播放出现 underrun | 第 4 阶段检查；必要时限制 MLX 内存，或调整线程优先级 |
| torchaudio 的 `forced_align` 被移除 | MMS 不能用 | 锁定 torchaudio 版本 |

## 暂不做

| 方向 | 为什么暂不做 |
|---|---|
| 换成 sglang-omni 的 "MOSS-TTS Local"（Mac 上 MLX 吞吐约为 MPS 的 1.79 倍，还没有合并） | 它是 MOSS-TTS-Local-Transformer，和 MOSS-TTS-v1.5 不是同一个模型，换了要重新做质量评估和契约验证。等它合并到主线后再单独评估 |
| SGLang 主仓库的 MLX 后端（`SGLANG_USE_MLX=1`） | 还在早期阶段，Qwen 用 vllm-metal 已经够了 |
| 伴奏模型迁移到本地 | 不在本计划范围内 |

## 参考资料

- [luisarn/MOSS-TTS-v1.5-MLX-8bit](https://huggingface.co/luisarn/MOSS-TTS-v1.5-MLX-8bit)：M4 Max 上 8-bit 约 1.5 倍实时，bf16 约 0.8 倍
- [mlx-community/MOSS-TTS-8B-8bit](https://huggingface.co/mlx-community/MOSS-TTS-8B-8bit)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)，[文档](https://docs.vllm.ai/projects/vllm-metal/en/latest/)
- [jevflow PR #4](https://github.com/limiteinductive/jevflow/pull/4)：`mlx_lm.server` 会忽略 `n`
- [sglang-omni Apple Silicon roadmap (#1967)](https://github.com/sgl-project/sglang-omni/issues/1967)，[support matrix PR (#1992)](https://github.com/sgl-project/sglang-omni/pull/1992)
- [SGLang Apple Device Support Roadmap (#19137)](https://github.com/sgl-project/sglang/issues/19137)
- [MacRumors：M5 Max vs M5 Ultra](https://www.macrumors.com/guide/m5-max-vs-m5-ultra/)：40 核 GPU 版 M5 Max 的带宽是 614 GB/s

## 实施记录

（每个阶段完成后在这里追加：日期、提交、实测数字、偏离计划之处。）
