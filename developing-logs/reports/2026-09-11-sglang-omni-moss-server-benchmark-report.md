# RAP / SGLang-Omni 服务器性能实测报告

日期：2026-09-11。性质：真实 H200 上的小规模、单请求、热运行性能试验，不是 production promotion 验收。

## 1. 直接结论

**能加速。在本次固定配置下，MOSS 完整语音合成约快 3.25 倍；包含 MMS 对齐和 Rubber Band R3 拉伸的服务器渲染约快 2.98 倍。**

生成一段最终长度 **5.33 秒** 的两小节人声，服务器渲染中位耗时从 **2.77 秒降到 0.93 秒**，约省 **1.84 秒 / 段**。

| 指标 | 旧后端 inprocess | SGLang-Omni | 变化 |
|---|---:|---:|---:|
| MOSS 完整 source WAV，p50 | 2640.40 ms | 812.51 ms | 3.25x；耗时降低 69.23% |
| MOSS 完整 source WAV，p95 | 2662.92 ms | 833.03 ms | 耗时降低 68.72% |
| 完整 renderer，p50 | 2767.48 ms | 927.65 ms | 2.98x；耗时降低 66.48% |
| 完整 renderer，p95 | 2927.91 ms | 1034.65 ms | 耗时降低 64.66% |
| 正式请求成功数 | 40 / 40 | 40 / 40 | 无失败、超时或 OOM |

这里的倍数是旧耗时除以新耗时，不是“降低 325%”。p95 是全部 40 次热测量的经验分位数，不是各轮 p95 的平均数。

**这不等于整个 RAP 用户端体验快 3 倍。** 本次没有测 Qwen 写词、完整 RAP HTTP orchestration、artifact packaging、服务器到 Mac 的网络、客户端混音和播放。也没有验证多人并发吞吐。

结果文件：[summary.json](../../logs/sglang_moss_benchmark_20260911/summary.json)。

## 2. 实际测了什么

没有运行 Mac client。测试驱动和 SGLang-Omni 服务都在服务器上，候选后端仍经过项目真实的 `SglangMossSynthesizer` HTTP adapter，访问 `127.0.0.1:8030`。

```text
旧：固定歌词 -> PersistentMossSynthesizer -> source.wav -> MMS -> R3 -> vocal.wav
新：固定歌词 -> SglangMossSynthesizer -> localhost SGLang-Omni
                                       -> source.wav -> MMS -> R3 -> vocal.wav
```

- 使用同一张物理 GPU 0，NVIDIA H200 NVL，单请求串行执行。
- 20 条不同的两小节歌词，每条 18 音节、90 BPM，目标长度 5.333333 秒。
- 语料含 2 条既有 fixture bar 和 38 条为本次试验编写的 bar，不是生产流量抽样。
- 顺序为 **A1 -> B1 -> B2 -> A2**，每轮 20 次正式请求、3 次额外预热；共 80 次正式测量。
- 每轮旧模型重新加载；B1 和 B2 之间重启候选服务，避免保留上一轮完整歌词的 prefix cache。
- 每个后端的前 3 条正式歌词也用于预热；候选保留实际的 reference cache 和 radix prefix cache。
- 两边都是常驻模型的热运行，不把旧后端每条重新加载权重作为 baseline。
- 不读取已合成的音频缓存，每次都真实生成，再执行相同 MMS 和 R3 实现。
- ASR 内容诊断在全部测速结束后运行，不计入耗时，也不与测速抢 GPU。
- GPU 0 对本次测试独占；机器 CPU 和其他 GPU 并非全机独占，GPU 4-7 上原有任务没有被修改或停止。

MOSS 计时包含参考处理、生成、解码、source WAV 写入/校验，以及候选的 loopback HTTP 请求。完整 renderer 计时到最终 WAV 生成完毕，不是首包或首个音频 token 延迟。

## 3. 配对一致性与环境

已验证：每条歌词在四轮中的 request SHA-256 和 resolved generation settings 完全一致；模型文件视图中的全部文件与原 HF snapshot 为相同 inode，权重字节未修改。

| 项目 | 固定值 |
|---|---|
| RAP HEAD | `b91c0a3dda03067598d8681a747344d2422aac17`，包含已有未提交实现 |
| SGLang-Omni | release `0.1.4`，commit `af3ab61a665427ff3fe3d67e32df6223524ed129` |
| SGLang | `0.5.18` |
| MOSS-TTS-v1.5 revision | `cdd3b911b1585e3f2dbc7775ef10f9926f58850a` |
| MOSS-Audio-Tokenizer revision | `3cd226ba2947efa357ef453bcad111b6eafba782` |
| 参考音频 | 同一 SeedTTS 英文参考，SHA-256 见 manifest |
| 语言 / 指令 | English / `clear, rhythmically spoken rap with restrained pitch` |
| token target / max_new_tokens | `67` / `256` |
| audio temperature / top-p / top-k / repetition penalty | `1.7` / `0.8` / `25` / `1.0` |
| seed | `20260816 + chunk_index * 1000`，每条固定 |
| 旧运行环境 | Python 3.12.3，torch 2.9.1+cu128，transformers 5.0.0 |
| 新运行环境 | Python 3.12.3，torch 2.13.0+cu130，transformers 5.12.1 |
| MMS / R3 | 两边驱动均用 baseline-env；相同 MMS_FA、Rubber Band 3.3.0 R3 |

**这是部署方案对比，不是只切换一个 kernel 的严格消融。** 旧代码按其现有逻辑选择 SDPA，主模型 BF16、HF codec FP32；候选使用 FA3、CUDA graphs、优化后的本地 BF16 codec。候选缓存参考音频编码，旧代码每次重新编码参考音频。依赖版本和实现也不同，因此不能把全部收益归因于 HTTP adapter 或某一项优化，也不能保证已装 FlashAttention 2 的另一套旧环境得到相同倍数。

同一个 seed 不保证不同采样实现产生相同音频。实际观测到各后端内部两轮的 20 条 source/final WAV 均逐字节复现，但新旧后端的音频不是同一份。

本次 pinned 上游接受 `ref_text` 并存入状态，却没有把它传给 `processor.build_user_message`；参考文本不构成这一版本中的额外文本 conditioning。该结论只适用于这个 commit，见 [上游 request_builders.py](https://github.com/sgl-project/sglang-omni/blob/af3ab61a665427ff3fe3d67e32df6223524ed129/sglang_omni/models/moss_tts/request_builders.py)。

## 4. 分轮结果与耗时去向

| 轮次 | MOSS p50 | renderer p50 | 成功数 |
|---|---:|---:|---:|
| A1 旧 | 2657.89 ms | 2770.40 ms | 20 / 20 |
| B1 新 | 811.90 ms | 927.02 ms | 20 / 20 |
| B2 新，重启后 | 813.10 ms | 928.91 ms | 20 / 20 |
| A2 旧 | 2615.46 ms | 2729.55 ms | 20 / 20 |

先对每条歌词的两次重复取平均，再做新旧配对，**20 / 20 条歌词都变快**。配对速度比的中位数为 MOSS 3.24x、renderer 2.97x，与合并样本的 p50 比值接近。

- MMS 中位耗时：旧 19.96 ms，新 20.18 ms。
- R3 中位耗时：旧 87.47 ms，新 87.69 ms。
- 因此主要改善发生在 MOSS 路径，MMS/R3 没有发生实质性提速。
- 0.5 秒间隔 GPU 采样观察到的最大显存：旧 25793 MiB，新 25865 MiB，约 25.2 / 25.3 GiB。采样覆盖预热和测量，不代表捕获了初始化或瞬时分配的绝对峰值。
- 首次真实候选请求在预检中曾耗时 8.87 秒，包含首次 kernel 准备等影响；不能用热运行 0.81 秒承诺冷启动首请求。

候选日志确认启用了 decode CUDA graph、sampling CUDA graph、fused audio heads 和参考编码缓存。推测这些优化共同减少了逐 token 推理和 codec 开销，但没有做逐项消融或完整 profiler；API 未返回内部阶段耗时，报告不虚构 prefill/decode/vocoder 的独立加速比例。

## 5. 音频与内容检查

两边所有正式 source WAV 都是 **24 kHz、130560 frames、5.44 秒**；经过同一对齐拉伸后，final WAV 都是 **128000 frames、5.333333 秒**。所有输出均可读取、数值有限、非静音，没有用较短音频制造速度优势。

用同一个 Whisper `small.en`，固定英文、temperature=0、beam_size=5，在测速后对 source 和 final 分别转写。用同一个 EnglishTextNormalizer 和 jiwer 计算词错误率；160 条检查记录对应 80 份不同音频，重复 WAV 按哈希复用转写结果。

| 自动识别诊断 | 旧后端 | 新后端 |
|---|---:|---:|
| source WAV corpus WER | 2.06% | 0.69% |
| final WAV corpus WER | 2.41% | 2.75% |
| final 零识别错误记录 | 30 / 40 | 30 / 40 |
| MMS alignment confidence 中位数 | 0.9669 | 0.9602 |
| 含低 MMS 字符分数警告的记录 | 26 / 40 | 30 / 40 |

**这只是自动识别诊断，不是人工质量通过。** final WER 新版高约 0.34 个百分点；部分差异来自 `liftoff / lift off`、`bass line / baseline`、`drum beats / drumbeats` 等分词或拼写，不必然等于真实发音错误。仍有 `the / this` 等差异，不能全部归为识别器问题。

没有进行人工盲听、音色相似度或节奏自然度评分。MMS forced alignment 的高分也不能替代歌词完整性或听感验收。20 条不同歌词、每条重复两次，不能当作 40 条独立质量样本。

质量证据：[汇总](../../logs/sglang_moss_benchmark_20260911/audio-quality/summary.json)、[逐条转写](../../logs/sglang_moss_benchmark_20260911/audio-quality/samples.jsonl)。示例配对：[旧 final](../../logs/sglang_moss_benchmark_20260911/A1/sample-00/vocal.wav)、[新 final](../../logs/sglang_moss_benchmark_20260911/B1/sample-00/vocal.wav)。

## 6. 本次实际补了什么

- 新增 [benchmark_moss_server.py](../../scripts/benchmark_moss_server.py)：固定语料、真实 backend 调用、预热排除、逐条耗时/失败/音频哈希记录、GPU 监控和配对汇总。
- 新增 [check_moss_benchmark_audio.py](../../scripts/check_moss_benchmark_audio.py)：测速后的自动转写检查，显式指定权重下载目录，保存参数、权重哈希和逐条结果。
- 新增 [benchmark 单测](../../tests/unit/scripts/test_benchmark_moss_server.py)：语料合法性、不覆盖不同实验数据、预热和失败统计边界。
- 在忽略目录 `logs/sglang_moss_benchmark_20260911/` 内准备独立 baseline / candidate 环境、配置、FFmpeg / Rubber Band 本地工具和全部原始结果。
- 未修改本轮之前已有的生产实现、项目依赖配置或系统软件；没有切换 production 默认后端。

验证：benchmark 和相关 MOSS/adapter/renderer targeted tests **80 passed**；两份新增脚本通过 `py_compile`；真实正式渲染 **80 / 80 成功**。本次没有重新运行仓库全量测试，也没有运行正式 promotion evaluator。

## 7. 环境问题与处理

1. 旧后端的音频加载需要匹配的 TorchCodec 和 FFmpeg 共享库。使用独立 baseline-env、CPU TorchCodec 0.9.1 和本地解包的 FFmpeg 6.1.1 依赖解决，没有改系统安装。
2. 新 Transformers 从 HF snapshot 符号链接加载远程 Python 文件时，可能把相对 import 定位到 `blobs/`。本次使用解引用后的硬链接模型视图，文件与 HF cache 为相同 inode，不修改模型代码、不额外复制大权重。
3. 参考 WAV 若是 HF snapshot 符号链接，真实路径会落到 allowlist 外，HTTP 请求返回 400。将同一 WAV 字节复制为 `media/reference.wav` 普通文件，并仅 allowlist 该 media 目录。
4. 候选环境显式提供 ninja 和 CUDA toolkit，首次编译留在预热阶段。没有因为测速去修改用户之前的 vLLM 环境。
5. 上述失败保留在 `preflight-*` 日志中，不进入正式统计。`preflight-A3` 是被移出的早期失败尝试，内部曾使用 A1 标签；最终汇总只读根目录下的 `A1/B1/B2/A2`，不递归收集预检数据。

模型缓存始终使用用户指定路径：

```bash
export HF_HOME=$HOME/mbzuai-projects/models/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$HF_HOME/torch
```

Whisper 检查模型放在 `$HF_HOME/whisper`。MOSS 模型硬链接视图位于实验目录，但底层权重仍是指定 HF cache 中的同一文件。

## 8. 证据与复现

实验根目录：`logs/sglang_moss_benchmark_20260911/`。

- [manifest.json](../../logs/sglang_moss_benchmark_20260911/manifest.json)：完整运行命令、绝对路径、实际 CLI override、版本、预检排除规则和证据 SHA-256。
- `protocol.json` / `corpus.json` / `moss_tts.yaml`：语料和配置；协议保留原始参考路径，manifest 记录实际普通文件路径，两者字节哈希相同。
- `A1/B1/B2/A2/records.jsonl`：原始计时、生成参数、音频信息、MMS 诊断和 warnings。
- 各轮 `sample-XX/source.wav` / `vocal.wav` / `mms_alignment.json`：真实输出和对齐结果。
- `server-B1.log` / `server-B2.log`、各轮 `gpu.csv`：运行路径与显存采样证据。
- `baseline-environment.lock` / `sglang-environment.lock`：独立环境包版本。
- `rap-source-snapshot.tar.gz` / `tracked-dirty.patch`：被测 RAP 代码快照及已有 tracked 变更，避免只记录 HEAD 漏掉未提交实现。

重新读取已有结果：

```bash
R="$HOME/mbzuai-projects/StreamMUSE-v1/logs/sglang_moss_benchmark_20260911"
"$R/baseline-env/bin/python" scripts/benchmark_moss_server.py summarize --root "$R"
```

重新测量时使用新的实验 root 并执行 `prepare`，参照 manifest 中的环境变量和完整命令按 ABBA 顺序运行。每次先确认 GPU 可用，B1/B2 用独立服务进程；不要覆盖已有 block。模型加载、预热、ASR 检查不纳入正式耗时。

## 9. 还不能下的结论

- 不能承诺整条 RAP E2E 快 3 倍；若其他串行阶段耗时为 T，总耗时近似从 `T + 2.77s` 变成 `T + 0.93s`，实际还取决于流水线重叠。
- 不能宣称音质完全无回退，也不能把自动 ASR 指标当成盲听或音色验收。
- 不能推广为所有歌词长度、语言、参考音色、GPU 和并发负载下固定 3.25x。
- 没有做真实取消/排队恢复/故障重启/长时间 canary，本次 40 / 40 不代表生产可靠性结论。
- 本试验规模小于 plan 要求的正式 100-pair 验收，也缺少 Mac 和其他 hard-gate 证据。

**当前决策：服务器热运行提速已实测证明，值得继续集成验证；仍保持 experimental，不自动切换 production 默认值。** 后续优先做新旧音频盲听及更多真实歌词，再做完整服务器请求和 Mac 端到端测试。

本次测试和 ASR 进程均已退出，GPU 0 已释放，8030 测试服务已停止；独立环境和全部证据保留供复现。
