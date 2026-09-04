# StreamMUSE MOSS-TTS 迁移到 SGLang-Omni Serving 计划（2026-09-03）

## 1. 目标

把 realtime rap 的 MOSS-TTS-v1.5 推理从 `streammuse-rap-render-server` 进程内的
Transformers `model.generate() + processor.decode()`，迁移为独立的
SGLang-Omni TTS 服务，同时保留 StreamMUSE 已经验证过的完整 rap 语义：

- H200 上的 Qwen lyric candidate generation 和 selection；
- 两个 bar、固定 `token_count` 的 connected-phrase MOSS synthesis；
- MMS forced alignment、syllable onset mapping 和 Rubber Band R3 warp；
- 精确时长的 vocal artifact、request idempotency 和 artifact cache；
- H200 到 Mac 的 Opus transport；
- Mac 本地 drums、mix、playback、fallback 和 Web UI；
- 现有 trace、latency breakdown 和可复现实验 metadata。

迁移不是简单地把 HTTP URL 换掉。实施结果必须回答三个问题：

1. SGLang-Omni 是否在真实 H200、单请求 realtime workload 下比当前 backend 更快；
2. 通过 `/v1/audio/speech` 得到的 MOSS 输出是否保持当前 voice、文本正确性和对齐质量；
3. SGLang-Omni 的 streaming 能否在不破坏 MMS + R3 精确落拍的前提下继续降低延迟。

## 2. 当前基线

当前分支：`feature/rap_optimization`。

当前 H200 production path：

```text
Mac StreamMUSE
    |
    | SSH-forwarded HTTP, final PCM/Opus package
    v
streammuse-rap-render-server :8020
    |-- Qwen candidate client ----------> vLLM :8001
    |-- PersistentMossSynthesizer
    |      |-- AutoProcessor
    |      |-- AutoModel.generate()
    |      `-- processor.decode() -> source.wav
    |-- resident MMS forced aligner
    |-- Rubber Band R3 warp
    `-- artifact/package cache
```

主要实现位置：

- `scripts/rap_audio_backends/moss_backend.py`：当前 Transformers MOSS runtime；
- `src/streammuse/infrastructure/rap/moss_tts.py`：production persistent synthesizer；
- `src/streammuse/infrastructure/rap/moss_aligned_phrase.py`：MOSS -> MMS -> R3；
- `src/streammuse/presentation/rap_render_server.py`：H200 composition、health 和 HTTP API；
- `src/streammuse/infrastructure/rap/remote_chunk_client.py`：Mac 到 H200 client；
- `src/streammuse/application/rap/chunk_audio.py`：Mac Opus decode、drums 和 mix。

已测 warm H200 stage baseline 大约为：

| Stage | 当前量级 |
|---|---:|
| Candidate generation | 约 0.46 s |
| MOSS synthesis bucket | 约 2.70 s |
| MMS alignment | 约 0.02 s |
| R3 warp | 约 0.09 s |
| Packaging | 约 0.05 s |

这里的 `MOSS synthesis bucket` 目前把 reference processing、prefill、AR generation、
codec decode、device-to-host copy 和 WAV 写入合在一起。迁移前必须保存当前 backend 的
可复现 A/B baseline，不能把 SGLang 官方的并发 benchmark 直接当作本项目结果。

## 3. 已确定的设计决策

### 3.1 SGLang-Omni 是独立服务，不加入 StreamMUSE 主环境

SGLang-Omni 使用独立的 H200 container 或 Python 3.12 virtual environment。不要把它
加入 StreamMUSE `pyproject.toml` 的主 dependencies，也不要让它改写当前 vendored
Transformers、PyTorch、torchaudio 或 MMS 环境。

原因：

- SGLang-Omni、SGLang、CUDA、FlashAttention 和 Transformers 有紧密的版本约束；
- StreamMUSE render server 仍需稳定运行 MMS、SciPy、Rubber Band 和 artifact logic；
- 独立进程允许分别升级、回滚、采样 GPU memory，并避免 dependency resolution 污染；
- SGLang 官方当前推荐 NVIDIA CUDA 使用 Docker，并建议 pin image digest，而不是依赖
  会移动的 `dev` tag。

初始端口约定：

| 服务 | H200 bind | Mac 是否 forward |
|---|---|---|
| Qwen vLLM | `127.0.0.1:8001` | 否 |
| SGLang-Omni MOSS | `127.0.0.1:8030` | 否 |
| StreamMUSE rap render server | `127.0.0.1:8020` | 是 |

Mac 仍只访问 `:8020`。SGLang-Omni 是 H200 内部依赖，不直接暴露到公网或 Mac。

### 3.2 第一版使用 non-streaming WAV

第一版调用：

```text
POST http://127.0.0.1:8030/v1/audio/speech
stream=false
response_format=wav
```

拿到完整 `source.wav` 后继续执行现有 MMS + R3。第一版不把 raw acoustic tokens 传到
Mac，也不直接播放 SGLang 的 PCM stream。

原因：当前 MMS 需要完整 waveform 才能获得 word/syllable timing，R3 需要完整
waveform 和完整 time map 才能输出精确长度。直接播放 streaming chunk 会绕过已经验证的
beat alignment contract。

### 3.3 保留旧 in-process backend，但禁止隐式 fallback

增加显式 backend selector：

```text
--moss-serving-backend inprocess|sglang-omni
```

- rollout 初期默认保持 `inprocess`；
- H200 acceptance 通过后，将 quickstart 推荐值切换为 `sglang-omni`；
- 启动或请求失败时不得自动加载旧 MOSS model，因为这会隐藏部署错误、突然占用额外显存，
  并使 latency/identity metadata 失真；
- fallback 必须由操作者显式选择 backend，游戏运行时仍使用既有 Mac audio fallback。

### 3.4 保持 StreamMUSE 的 synthesis contract

SGLang client 继续返回现有 `MossPhraseResult`，使
`MossAlignedPhraseRenderer` 不需要知道底层是本地 model 还是 HTTP service。

统一的内部 contract 至少包含：

- `synthesize(request, output_wav) -> MossPhraseResult`；
- `warmup()`；
- `close()`；
- model/backend identity；
- reference hash；
- generation settings 和 latency metadata。

不要在 domain 或 application layer 中引入 SGLang-specific request schema。

### 3.5 Voice reference 使用 H200 本地固定文件

`rap_render_server` 和 SGLang-Omni 位于同一台 H200。初始实现将固定 reference WAV 的
server-local path 传给 `ref_audio`，不在每个请求里重复 base64 编码和传输。

如果 SGLang 运行在 container 内，reference WAV 必须 read-only mount 到 container，并在
两个进程中使用一致、不可变的绝对路径。

增加：

```text
--moss-reference-text-file /path/to/reference.txt
```

在 `sglang-omni` 模式下要求文件存在、UTF-8、非空。启动时读取一次，计算 SHA-256，
请求中作为 `ref_text` 使用。health 和 manifest 只记录 hash，不输出完整 reference transcript。

SGLang 官方说明 reference transcript 会明显改善 voice cloning quality，因此不能把它作为
未记录的可选行为。

### 3.6 Duration control 只负责 source guidance

每次请求继续使用：

```text
token_count = moss_token_target(two_bar_request)
```

SGLang 的 `token_count` 是 target duration hint，不保证 waveform 精确长度。最终精确的
两个 bar 长度仍由 MMS + R3 和现有 frame-count validation 保证。

### 3.7 首次迁移不提高并发

`MossAlignedPhraseRenderer` 当前用 lock 串行保护 synthesis、alignment 和 warp。第一版保留
这个边界，不因为 SGLang 支持 continuous batching 就删除 lock。

本项目首先优化的是单用户 realtime p50/p95，而不是最大 QPS。只有在单请求正确性通过、
MMS/R3 thread safety 被单独证明后，才评估 concurrency 2/4/8。

## 4. 目标架构

```text
H200 GPU 1 (example)
  Qwen vLLM :8001
          ^
          |
H200 rap render server :8020
  |-- candidate generation/selection
  |-- SglangMossSynthesizer HTTP client ----> SGLang-Omni :8030
  |                                             |-- preprocessing/reference encode
  |                                             |-- MOSS AR tts_engine
  |                                             `-- vocoder -> 24 kHz WAV
  |-- MMS forced alignment
  |-- Rubber Band R3
  |-- canonical artifact cache
  `-- Opus package
          |
          | one SSH-forwarded private endpoint
          v
Mac StreamMUSE
  |-- validate/decode Opus
  |-- local drums + mix
  `-- playback/Web UI/trace
```

GPU placement 不写死在代码中。部署前运行 `nvidia-smi`，记录每个 process、physical GPU、
显存上限和 CUDA-visible mapping。推荐先沿用已验证布局：Qwen 在一张独立 GPU，
SGLang MOSS 与轻量 MMS 在另一张 GPU；如果共卡，必须单独验证 peak memory 和 tail latency。

## 5. SGLang 请求映射

`TwoBarRenderRequest` 到 `/v1/audio/speech` 的映射必须集中在一个 builder 中，并写 contract
tests。建议 payload：

```json
{
  "model": "OpenMOSS-Team/MOSS-TTS-v1.5",
  "voice": "streammuse-rap-reference",
  "input": "selected two-bar lyric text",
  "ref_audio": "/models/streammuse/reference.wav",
  "ref_text": "exact transcript of reference.wav",
  "response_format": "wav",
  "stream": false,
  "language": "English",
  "instructions": "clear, rhythmically spoken rap with restrained pitch",
  "token_count": 67,
  "max_new_tokens": 256,
  "audio_temperature": 1.7,
  "audio_top_p": 0.8,
  "audio_top_k": 25,
  "audio_repetition_penalty": 1.0,
  "seed": 20260816
}
```

具体 `token_count` 和 `seed` 按 request 计算，示例值不是常量结果。

迁移时保留当前 generation settings：

- language：`English`；
- instruction：`clear, rhythmically spoken rap with restrained pitch`；
- `max_new_tokens=256`；
- `audio_temperature=1.7`；
- `audio_top_p=0.8`；
- `audio_top_k=25`；
- `audio_repetition_penalty=1.0`；
- stable per-request seed。

SGLang 支持显式 per-request seed，并声明固定 server configuration/hardware 下不受 batch
neighbour 影响。仍不能要求旧 Transformers backend 与 SGLang waveform bitwise identical；
验收比较语义、voice、alignment 和最终 exact-duration artifact。

## 6. 代码调整

### 6.1 稳定 synthesizer contract

修改 `src/streammuse/infrastructure/rap/moss_tts.py`：

- 定义公开或模块内稳定的 `MossSynthesizer` Protocol；
- 保留 `MossPhraseResult` 作为 backend-neutral result；
- 抽出 generation defaults 和 per-request seed resolver，避免 production SGLang client
  继续 import `scripts.rap_audio_backends.moss_backend`；
- 保持现有 `PersistentMossSynthesizer` 行为，作为 in-process adapter；
- 为 `close()` 提供幂等语义。

离线 protocol-comparison script 继续可直接运行；如果共享 settings 被迁入 `src`，script
改为 import 正式 helper，不能维护两份会漂移的常量。

### 6.2 新增 SGLang HTTP synthesizer

新增 `src/streammuse/infrastructure/rap/sglang_moss_tts.py`：

- `SglangMossConfig`：base URL、model、reference path/text、timeout、response byte limit；
- 一个进程内复用的 `httpx.Client`，禁止每个 phrase 新建 TCP connection；
- `GET /health` 和 `GET /v1/models` probe；
- request builder 和禁止 payload key collision 的 validation；
- non-streaming `/v1/audio/speech` request；
- bounded streaming read，即使响应是 non-streaming 也不能无上限读入内存；
- 只接受成功 HTTP status 和受支持的 WAV media type；
- 将 response 写到 `.source.partial.wav`，验证完成后用 `os.replace()` 原子提交；
- 复用或提取现有 24 kHz、mono、finite、nonempty、nonsilent validation；
- 网络错误、HTTP 400/429/500/503、invalid WAV、timeout 使用可区分的异常信息；
- `close()` 幂等，关闭 persistent client；
- 不把 reference bytes、reference transcript、绝对私有路径写入普通日志或 public health。

初始版本不做隐藏 retry。realtime 请求有严格 deadline，自动 retry 会制造重复 GPU work，
并可能让旧请求在客户端放弃后继续占用 SGLang。错误交给既有 render failure/fallback 路径。

### 6.3 Render server wiring 和 CLI

修改 `src/streammuse/presentation/rap_render_server.py`：

- `RapRenderServerConfig` 增加 backend、SGLang URL、request timeout、reference text file；
- parser 增加：
  - `--moss-serving-backend`；
  - `--moss-sglang-url`；
  - `--moss-request-timeout-s`；
  - `--moss-reference-text-file`；
- `inprocess` 模式继续使用 `--moss-device`；
- `sglang-omni` 模式不加载 MOSS weights，不 import heavy MOSS backend；
- 根据 backend factory 构造统一 synthesizer；
- startup 顺序改为：probe Qwen -> probe SGLang -> real MOSS warmup -> MMS warmup -> ready；
- 任一依赖未 ready 或 warmup artifact invalid 时 fail closed，不启动一个假健康服务；
- shutdown 只关闭 HTTP client 和本进程资源，不负责终止外部 SGLang process。

health 中的 `moss` 增加：

```text
backend=sglang-omni
identity=SGLang-Omni/MOSS-TTS
model
model_revision（能够可靠获得时）
server_version（能够可靠获得时）
reference_audio_sha256
reference_text_sha256
warmup_time_ms
endpoint_host/port（去除 credentials 和私有路径）
```

不得伪造 SGLang 没有返回的 revision/version；未知时记录 `unknown` 和明确 warning。

### 6.4 Artifact 和 monitoring metadata

更新 MOSS result、render manifest 和 `server_timing.json`：

- `moss_backend`：`inprocess` 或 `sglang-omni`；
- `moss_service_request_ms`；
- `moss_response_first_byte_ms`；
- `moss_response_download_ms`；
- `moss_response_bytes`；
- SGLang model/server identity；
- resolved generation parameters；
- reference audio/text hashes；
- streaming flag，初始固定为 `false`。

保留现有顶层 `stage_timings_ms["moss"]`，以免 UI、summary 和历史分析工具失效；新增
breakdown 作为向后兼容的扩展字段。schema 是否需要 bump 由现有 manifest strictness tests
决定，不能直接修改历史 artifact。

### 6.5 部署和文档

新增或更新：

- `docs/developer-guide/rap-demo-quickstart.md`；
- `docs/developer-guide/sglang-omni-moss-serving.md`；
- 必要时增加一个只负责 preflight/launch 的 H200 shell script；
- 不在脚本中硬编码个人 home、GPU id、Hugging Face cache 或 reference voice path。

文档必须给出三个 H200 terminal：

1. Qwen vLLM；
2. SGLang-Omni MOSS；
3. StreamMUSE rap render server。

SGLang 环境 pin：

- 优先使用 exact image digest；或
- pin `sglang-omni==0.1.4`、Python 3.12 和匹配的 SGLang/CUDA stack；
- 保存 upstream `moss_tts.yaml` 对应的 SGLang-Omni commit/digest；
- 记录 model snapshot revision，不只记录 floating Hugging Face name。

示意启动命令：

```bash
CUDA_VISIBLE_DEVICES=<moss-gpu> sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config examples/configs/moss_tts.yaml \
  --host 127.0.0.1 \
  --port 8030
```

实际 CLI 必须在 pinned release 上运行 `--help` 核对；文档不能仅依据 main branch 写入未经
验证的 flag。

## 7. Startup、failure 和安全语义

### Startup gate

render server 只有在以下条件全部成立后才返回 `ready=true`：

- Qwen `/v1/models` 返回预期 model；
- SGLang `/health` 为 200；
- SGLang `/v1/models` 包含预期 MOSS model；
- reference WAV 和 transcript hash 已计算；
- 一次真实、非静音、24 kHz mono MOSS warmup 成功；
- MMS 对该 warmup WAV 的 known transcript alignment 成功；
- Rubber Band probe 成功；
- artifact root 可写。

### Runtime failure

- HTTP 400：configuration/request contract error，不 retry；
- HTTP 429/500/503、connect error、timeout：记录 bounded diagnostics，当前 request 失败；
- invalid/truncated WAV：删除 partial file，request 失败；
- client disconnect：关闭当前 response body；另测 SGLang 是否真正 abort GPU generation；
- request 失败不得覆盖已有 successful canonical artifact；
- 不在同一次运行中自动切换 backend。

### Security

- SGLang 和 render server 都 bind loopback；
- Mac 只 forward render server port；
- reference path 来自 server startup config，不接受 Mac request 覆盖；
- health、trace 和错误信息进行现有 private-path/credential sanitization；
- container 只 read-only mount model/reference，artifact root 只给 render server 写权限。

## 8. 测试策略

### Unit tests

新增 `tests/unit/infrastructure/rap/test_sglang_moss_tts.py`：

- exact request payload mapping；
- `token_count` 和 per-request seed；
- reference text/path handling；
- persistent connection/client reuse；
- health/model probe；
- successful WAV atomic commit；
- response byte upper bound；
- wrong status/content type；
- truncated、stereo、wrong-rate、silent、NaN WAV；
- timeout/connect failure；
- partial/stale file cleanup；
- close idempotency；
- error messages do not leak transcript or private credentials。

更新 `tests/unit/infrastructure/rap/test_moss_tts.py`：

- backend-neutral contract；
- current in-process behavior unchanged；
- shared generation settings do not drift。

更新 `tests/unit/presentation/test_rap_render_server.py`：

- both backend CLI modes；
- incompatible/missing flags fail before heavy imports；
- SGLang mode never constructs `PersistentMossSynthesizer`；
- in-process mode never constructs SGLang client；
- startup probe/warmup order；
- dependency failure prevents ready；
- health sanitization and new metadata；
- shutdown closes the selected synthesizer exactly once。

更新 manifest、monitoring、summary tests，确保旧 artifact 仍能读取。

### Hermetic integration tests

使用本地 fake SGLang FastAPI server：

- `/health`、`/v1/models`、`/v1/audio/speech` 完整 round trip；
- fake server 返回 fixture WAV，真实 `MossAlignedPhraseRenderer` 后续 validation 继续工作；
- concurrent same request id 仍由 render server single-flight 合并；
- timeout、disconnect、503、malformed audio 后下一次请求可恢复；
- complete artifact cache hit 不再次调用 SGLang；
- render server 对 Mac 的 public rap package contract 不变。

真实 MOSS checkpoint 和 H200 不进入默认 pytest gate。

## 9. H200 A/B 实验

### 固定实验条件

- 同一 H200 host、同一 physical GPU 型号；
- 独占或记录所有 competing processes；
- 同一个 MOSS snapshot、reference WAV、reference transcript；
- 同一组至少 30 个 two-bar requests；
- 相同 text、flow、`token_count`、generation parameters 和 seed；
- cold startup 单独测，warm requests 才进入主要 latency distribution；
- current in-process backend 和 SGLang-Omni 分开启动，不能同时抢同一 GPU；
- 每种 backend 至少重复两轮，交换运行顺序以减小时段偏差。

### 收集指标

性能：

- startup/model-load/warmup time；
- reference preprocessing；
- request to first response byte；
- complete MOSS WAV latency p50/p95/max；
- MMS、R3、package 和 server total；
- GPU memory peak、utilization、power；
- error、timeout、runaway 和 cancellation recovery；
- Mac end-to-end first usable two-bar package latency。

质量：

- WAV rate/channel/finite/non-silent；
- generated duration distribution；
- MMS coverage/confidence；
- ASR WER/deletion/substitution；
- R3 local stretch ratio 和 fallback frequency；
- final vocal exact frame count；
- speaker similarity（若现有工具可复用）；
- 固定盲听样本的 intelligibility、voice identity、rhythm/artifact 检查。

### Acceptance gate

SGLang-Omni 成为推荐 backend 必须满足：

- 30/30 warm requests 完成，无 crash/OOM/hung request；
- final vocal 全部满足现有 exact-duration 和 package validation；
- 没有新增 MMS coverage failure 或 Mac playback underrun；
- WER、alignment confidence、stretch/fallback 指标不劣于当前 backend 的预设容差；
- warm MOSS p50 优于当前约 2.7 s baseline；
- warm MOSS p95 不比 current baseline 高 10% 以上；
- complete H200 server p95 和真实 Mac deadline miss rate不退化；
- health/trace 能明确证明请求使用了 SGLang，而不是旧 backend 或 cache-only 假成功。

如果 latency 没有提升但质量和稳定性通过，保留为实验 backend，不把“迁移成功”写成
“性能优化成功”。

## 10. Streaming 第二阶段

只有 non-streaming acceptance 完成后才开始。SGLang MOSS streaming 路径可以在 AR 尚未
结束时输出 raw PCM，但 StreamMUSE 不能直接播放未对齐 PCM。

需要分别验证三种方案：

1. **仅测 TTFA，不接入游戏**：记录 SGLang first PCM chunk，量化 serving 本身收益；
2. **一 bar buffer**：收齐一个 bar 的 source PCM 后在 H200 做 MMS/R3，再提前返回第一个
   已对齐 bar；需要重新定义当前 two-bar alignment/context contract；
3. **server-internal overlap only**：利用 SGLang AR -> vocoder streaming/overlap，但
   StreamMUSE 仍等待完整 source，再做 MMS/R3；这只能减少 synthesis wall time，不能直接
   产生可播放的 early chunk。

不得在完成以下条件前把 streaming 设为默认：

- chunk boundary 没有 audible discontinuity；
- delay-pattern 32-codebook 的初始 chunk 完整；
- cancellation 不留下永久占用 GPU 的 request；
- source stream 聚合后的 WAV 与 non-streaming 质量等价；
- 明确定义一 bar/两 bar 的 alignment 和 artifact schema；
- Mac 不会播放尚未通过 validation 的 audio。

## 11. 分阶段 TODO List

### Phase 0：冻结现状和依赖版本

- [ ] 记录当前 branch、commit、dirty state、H200 checkout 和 model snapshot revision。
- [ ] 保存 current in-process backend 的 30-request warm baseline 和原始 artifacts。
- [ ] 把当前 `moss` bucket 拆出至少 HTTP 可对比的阶段边界，确认计时定义。
- [ ] 记录 Qwen、MOSS、MMS 的 physical GPU 和 peak memory。
- [ ] 核对 reference WAV 的真实 transcript，保存 UTF-8 text 和两个 SHA-256。
- [ ] 确认现有 affected pytest suite 全绿。

### Phase 1：独立部署并资格验证 SGLang-Omni

- [ ] 在 H200 创建独立 pinned SGLang-Omni runtime，不修改 StreamMUSE 主环境。
- [ ] pin SGLang-Omni release/image digest、upstream config revision 和 MOSS snapshot。
- [ ] 在 loopback `:8030` 启动 `OpenMOSS-Team/MOSS-TTS-v1.5`。
- [ ] 验证 `/health` 和 `/v1/models`。
- [ ] 用 curl 执行 reference-less、voice-clone、fixed-token non-streaming smoke。
- [ ] 验证返回 WAV 为 24 kHz mono、非静音、可被 MMS 读取。
- [ ] 执行一次 streaming smoke，只记录能力，不接入游戏。
- [ ] 记录 startup、VRAM、warm latency、server logs 和 exact launch command。

### Phase 2：建立 backend-neutral synthesis contract

- [ ] 定义 `MossSynthesizer` Protocol 和幂等 lifecycle。
- [ ] 抽出共享 generation settings 和 seed resolver。
- [ ] 保持 `MossPhraseResult` 的已有消费者兼容。
- [ ] 让旧 `PersistentMossSynthesizer` 实现统一 contract。
- [ ] 更新 offline backend 使用同一份 settings，删除语义重复。
- [ ] 补 contract 和 in-process regression tests。

### Phase 3：实现 SGLang MOSS client

- [ ] 新建 `SglangMossConfig` 和 `SglangMossSynthesizer`。
- [ ] 实现 health/model probe。
- [ ] 实现 exact request payload builder。
- [ ] 使用 persistent `httpx.Client` 和 bounded response read。
- [ ] 实现 partial WAV、validation 和 atomic rename。
- [ ] 实现 timeout/status/protocol/audio error classification。
- [ ] 实现 timings、identity、hash 和 generation metadata。
- [ ] 实现幂等 close 和无 secret/path 泄漏的 diagnostics。
- [ ] 完成全部 unit tests。

### Phase 4：接入 rap render server

- [ ] 增加 backend selector 和 SGLang CLI/config fields。
- [ ] 校验 backend-specific required/forbidden combinations。
- [ ] 根据 selector 构造且只构造一个 backend。
- [ ] 实现 Qwen/SGLang/MOSS/MMS/R3 的严格 startup gate。
- [ ] 扩展 health，保持 public response bounded/sanitized。
- [ ] 扩展 server timing 和 manifest metadata，保持旧 artifact readable。
- [ ] 验证 artifact cache/single-flight/failure persistence 语义不变。
- [ ] 更新 composition、CLI、health、shutdown tests。

### Phase 5：Hermetic integration 和回归

- [ ] 建立 fake SGLang HTTP fixture。
- [ ] 跑完整 fake SGLang -> source WAV -> MMS/R3 boundary integration。
- [ ] 覆盖 cache hit、same-ID concurrency、timeout、503、truncated WAV 和 recovery。
- [ ] 验证 Mac remote package schema、Opus path 和 fallback 不变。
- [ ] 运行全部 rap unit/integration tests。
- [ ] 运行完整默认 pytest suite并记录结果。

### Phase 6：部署文档和操作流程

- [ ] 编写独立 SGLang install/pin/launch/health/stop 文档。
- [ ] 更新 rap quickstart 为 Qwen、SGLang、render server 三进程布局。
- [ ] 记录 container mount、reference transcript、cache 和 GPU mapping。
- [ ] 给出只 forward `:8020` 的 SSH 命令。
- [ ] 给出 `inprocess` rollback 命令和 artifact 清理边界。
- [ ] 给出进程定位、日志查看和只终止自己进程的操作说明。

### Phase 7：真实 H200 A/B acceptance

- [ ] 运行固定 30-request current baseline。
- [ ] 运行固定 30-request SGLang non-streaming candidate。
- [ ] 生成 p50/p95/max、VRAM、quality 和 failure 对比报告。
- [ ] 盲听固定样本并保存评分/备注。
- [ ] 运行一次真实 Mac tunnel + Web UI 完整游戏。
- [ ] 验证 trace 中的 backend identity 和新增 breakdown。
- [ ] 按 acceptance gate 决定推荐、保留实验或回滚。

### Phase 8：受控 rollout

- [ ] acceptance 通过后把 quickstart 推荐 backend 改为 `sglang-omni`。
- [ ] 继续保留显式 `inprocess` rollback 至少一个稳定周期。
- [ ] 保存 pinned deployment manifest 和实际 H200 acceptance artifacts。
- [ ] 观察多场游戏的 timeout、fallback、deadline miss 和音频质量。
- [ ] 稳定后再决定是否删除 production 对 scripts backend 的依赖。

### Phase 9：Streaming 研究

- [ ] 单独测 SGLang raw PCM TTFA、chunk cadence 和 complete latency。
- [ ] 比较 non-streaming、stream-and-aggregate 和 server-internal overlap。
- [ ] 验证 cancellation/recovery 和 chunk boundary quality。
- [ ] 设计一 bar finalized-audio protocol，不直接播放 unaligned source PCM。
- [ ] 只有端到端 first-usable-audio 明显改善时才进入生产设计。

## 12. 明确不在本次迁移内

- 不把 MOSS acoustic decoder 移到 Mac；
- 不把 raw MOSS tokens 暴露给 Mac；
- 不删除 MMS forced alignment 或 R3；
- 不改变两个 bar 的 public request/response contract；
- 不改变 Mac drums、mix、playback clock 或 Web UI ownership；
- 不把 SGLang port 暴露到公网；
- 不用官方并发 benchmark 代替本项目单请求 acceptance；
- 不因 SGLang 支持 batching 就提前移除 renderer lock；
- 不在失败时静默启动第二份 MOSS model。

## 13. 主要风险及控制

| 风险 | 后果 | 控制 |
|---|---|---|
| SGLang 与当前输出采样语义不同 | voice/文本/节奏变化 | 显式传全量参数、固定 seed、A/B quality gate |
| `token_count` 不是精确时长 | source duration drift | 保留 MMS + R3 和 exact frame validation |
| reference transcript 缺失或错误 | cloning quality 下降 | SGLang 模式启动时强制校验 text file 和 hash |
| SGLang dependency 污染主环境 | MMS/Transformers 失效 | 独立 pinned container/venv |
| HTTP 增加进程边界开销 | 单请求反而变慢 | loopback persistent client、30-request A/B gate |
| renderer 串行导致 batching 无收益 | 吞吐优化没有作用 | 首先评估 c=1；并发作为后续独立任务 |
| streaming 绕过 alignment | syllable 不落 beat | 第一版 non-streaming；只播放 finalized audio |
| disconnect 后 GPU request 未取消 | tail latency/资源泄漏 | cancellation/recovery experiment、bounded timeout |
| health 报告假 ready | 游戏开始后才失败 | real synthesis + MMS warmup 后才 ready |
| upstream `dev` 漂移 | 无法复现 | pin release、image digest、config 和 model revision |

## 14. 参考资料

- [SGLang-Omni MOSS-TTS cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html)
- [SGLang-Omni installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)
- [SGLang-Omni architecture](https://sgl-project.github.io/sglang-omni/developer_reference/main.html)
- [SGLang-Omni TTS usage](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)
- [SGLang-Omni API server design](https://sgl-project.github.io/sglang-omni/developer_reference/apiserver_design.html)
- [MOSS-TTS optimization tracking](https://github.com/sgl-project/sglang-omni/issues/1233)
- `docs/developer-guide/realtime-remote-moss-acceptance-2026-08-21.md`
- `docs/developer-guide/rap-opus-transport-experiment-2026-08-21.md`
- `docs/developer-guide/rap-demo-quickstart.md`

