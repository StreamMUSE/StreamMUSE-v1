# StreamMUSE MOSS-TTS 迁移到 SGLang-Omni Serving 计划（2026-09-03，2026-09-04 审查修订）

> 本次修订补齐了首次方案中会影响实验可信度和线上稳定性的边界：producer-aware
> artifact namespace、端到端 deadline/cancellation、v1 strict schema 冻结、可复现依赖 pin、
> reference transcript 的双轨 A/B，以及带数值门槛的最终验收。

## 0. 实施状态（2026-09-04）

当前工作树已完成可在当前仓库中交付和 hermetic 验证的 Phase 2-6 实现，并交付 Phase 0/1/7/8
所需的 preflight、实验冻结、证据校验和 runbook 工具。这里的“已实现”只表示代码、测试和操作入口
已经存在，不代表真实 H200、Mac 或人工盲听 gate 已通过。

| 范围 | 状态 | 证据/边界 |
|---|---|---|
| Phase 2 contract/deadline | 已实现 | `execution.py`、共享 generation settings、in-process checkpoints 和 unit tests |
| Phase 3 producer/cache/schema | 已实现（真实历史 artifact 待验） | canonical fingerprint、namespace、private sidecar、single-flight isolation 和 tests |
| Phase 4 SGLang client | 已实现（真实服务 fixture 待采集） | exact payload、persistent `httpx`、bounded probes/WAV、no retry 和 tests |
| Phase 5 render integration | 已实现 | backend CLI/lazy composition、startup gates、waiter cancellation、degraded state 和 tests |
| Phase 6 hermetic/docs | 已实现；仓库全量 gate 未绿 | targeted/cross-layer tests 全绿；默认 suite 只剩下述外部 fixture blocker |
| Phase 0/1/7 evidence | 工具已实现，真实执行待完成 | preflight、immutable launch/root manifest、freeze/evaluator 已交付；无 H200 证据不得勾 gate |
| Phase 8 Mac/canary | 待执行 | 默认 backend 继续为 `inprocess`，quickstart 未提升 candidate |
| Phase 9 streaming | 待独立实验 | 本次只实现 non-streaming WAV，不改变 public v1 |

当前验证记录：

- affected targeted suite：215 passed（包含 FastAPI ASGI round-trip 和 SGLang -> WAV -> MMS/R3
  -> final artifact 跨层 case）；
- `uv run pytest tests -q`：当前只剩 5 个失败，均缺少仓库外
  `output/rap_album_10x50_90bpm_20260816_v4/.../chosen_lyrics.jsonl`；其余结果为 1934 passed、
  8 skipped；显式 deselect 这 5 个外部数据节点时 suite 全绿；
- fake audio callback 已显式注入 termination factory，`test_audio_output.py` 为 28 passed，测试机
  不再因缺少 PortAudio 产生假失败；
- 从仓库根直接运行 `uv run pytest -q` 会额外收集 vendored `transformers/`，因 Keras 3 环境缺少
  `tf_keras` 在 collection 阶段失败；
- 仓库未配置 formatter/linter/type checker；`git diff --check` 已通过。

因此 Phase 6 的“默认全量 suite 全绿”和所有真实硬件/人工 gate 保持未完成。promotion evaluator 在
没有可信证据时 fail closed，production 默认值也不会由本地实现自动切换。

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

### 3.4 保持 StreamMUSE 的 synthesis contract，并显式传递执行预算

SGLang client 继续返回现有 `MossPhraseResult`，使
`MossAlignedPhraseRenderer` 不需要知道底层是本地 model 还是 HTTP service。

统一的内部 contract 至少包含：

- `synthesize(request, output_wav, *, execution: SynthesisExecutionContext) -> MossPhraseResult`；
- `warmup()`；
- `close()`；
- model/backend identity；
- reference hash；
- generation settings 和 latency metadata。

`SynthesisExecutionContext` 至少包含：

- API 接受请求时基于 `time.monotonic()` 计算的一次性 absolute deadline；
- cancellation signal/predicate；
- bounded correlation id，用于 render server、SGLang request 和内部 sidecar 对齐；
- 读取剩余预算并在预算耗尽时抛出统一 deadline exception 的 helper。

deadline 必须从 `remaining_budget_ms` 一路穿过 candidate planning、renderer、synthesizer、MMS、
R3 和 packaging，不能只限制 candidate planning。SGLang HTTP connect/read/write/pool timeout 每次
调用都取 `min(configured_timeout, remaining_deadline)`；写入成功 artifact 前再次检查 deadline 和
cancellation。旧 in-process backend 如果无法中断正在执行的 `model.generate()`，必须明确记录为
`cancel_requested_but_not_interruptible`，但仍需在进入生成前、返回后和各后处理阶段检查预算。

不要在 domain 或 application layer 中引入 SGLang-specific request schema。

### 3.5 Voice reference 使用不可变文件，并区分 host path 与 service URI

`rap_render_server` 和 SGLang-Omni 位于同一台 H200。初始实现将固定 reference WAV 的
service-visible URI 传给 `ref_audio`，不在每个请求里重复 base64 编码和传输。保留现有
`--moss-reference-wav` 作为 render server 可读的 host path，并新增：

```text
--moss-sglang-reference-uri file:///models/streammuse/reference.wav
```

如果 SGLang 运行在 container 内，reference WAV 必须 read-only mount 到固定 container path；
host path 与 container path 可以不同，但两边读取到的 bytes 必须具有同一个 SHA-256。只有在
pinned SGLang-Omni release 上核实对应 local-media allowlist flag 后，才允许使用 `file://` URI；
launch manifest 必须保存核实后的 flag 和 mount 映射。

增加：

```text
--moss-reference-text-file /path/to/reference.txt
```

在 `sglang-omni` 模式下要求文件存在、UTF-8、非空。启动时读取一次，计算 SHA-256，
请求中作为 `ref_text` 使用。health、producer manifest 和 private sidecar 只记录 hash，不输出完整
reference transcript。

SGLang 官方说明 reference transcript 会明显改善 voice cloning quality，因此不能把它作为
未记录的可选行为。

初始请求使用 API 已定义的 `voice="default"`。命名 voice 的注册/上传接口不进入本次迁移，
避免发送一个服务端从未注册的 `streammuse-rap-reference` 名称。

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

single-flight 必须感知 waiter，而不是把一个客户端断开等价为取消共享 owner：单个 waiter
断开时只移除该 waiter；absolute deadline 到期或所有 waiter 都离开后，才向 owner 发出取消。
如果上游 SGLang 无法在 cancellation grace period 内确认释放请求，则 render server 将依赖标记为
`degraded`，拒绝继续接受需要 MOSS 的新请求，并按 runbook 重启 SGLang；不得用隐藏 retry 叠加
第二份 GPU work。

### 3.8 Artifact cache 必须按 producer/deployment 隔离

当前 public `request_id` 只描述 canonical rap request，不包含后端和部署身份；它继续保持不变，
以免破坏 Mac/request schema。render server 另行计算不可变的 `producer_fingerprint`，artifact
路径改为：

```text
<artifact_root>/<producer_fingerprint>/<request_id>/
```

fingerprint 由 canonical JSON 计算 SHA-256，至少覆盖：

- backend（`inprocess` / `sglang-omni`）；
- MOSS exact model snapshot revision；
- SGLang-Omni/SGLang package version、commit 或 image digest；
- vendored `moss_tts.yaml` 内容 hash 和所有影响输出的 launch options；
- resolved generation settings；
- reference audio/text SHA-256；
- aligner identity、warp policy 和 public output contract/schema version。

canonical JSON 使用稳定 key order、明确类型和版本化 fingerprint schema；不得包含 credentials、
完整 transcript 或个人绝对路径。每个 namespace 根保存一份 immutable producer manifest，cache hit
前同时校验 fingerprint 和 canonical request。single-flight key 使用
`(producer_fingerprint, request_id)`。A/B baseline 和 candidate 还应使用不同 artifact roots，形成
第二层防误用；旧 cache 不自动迁移到新 namespace。

### 3.9 首次迁移冻结 public artifact schema v1

现有 `RemoteRapChunkManifest`、diagnostics 和 ZIP member parser 都是 exact-key/strict schema。
首次迁移因此不向 public v1 manifest、diagnostics、`REMOTE_CHUNK_ARTIFACT_IDS` 或 ZIP members
增加字段，也不重命名 `stage_timings_ms["moss"]`。

新增 serving 细节使用 backend-neutral 的内部 `MossServingMetadata`，并为
`MossPhraseResult` 提供兼容旧构造器的默认值。详细内容写入 H200 artifact 目录下的 private
`internal/moss_synthesis.v1.json` sidecar；该文件不进入发给 Mac 的 ZIP，也不加入稳定 artifact id。
已有 `model_tool_versions["moss"]` 可写 backend-qualified、bounded identity，但 key 集合保持不变。
health 只暴露经过 sanitization 的有界摘要。任何需要把详细 serving 字段公开给 Mac 的需求，另开
public schema v2 和 dual-reader 迁移，不夹带在本次 backend 切换中。

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
  "voice": "default",
  "input": "selected two-bar lyric text",
  "ref_audio": "file:///models/streammuse/reference.wav",
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
`ref_audio` 只能来自启动时配置并通过 allowlist 校验，不能由 public rap request 覆盖。

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
- 定义 backend-neutral `SynthesisExecutionContext` 和统一 deadline/cancellation exception；
- 保留 `MossPhraseResult` 作为 backend-neutral result；
- 给 `MossPhraseResult` 增加带默认值的内部 `MossServingMetadata`，保持现有测试和 adapter
  构造方式兼容；
- 抽出 generation defaults 和 per-request seed resolver，避免 production SGLang client
  继续 import `scripts.rap_audio_backends.moss_backend`；
- 保持现有 `PersistentMossSynthesizer` 行为，作为 in-process adapter；
- 让 in-process adapter 在生成前后检查 execution context，并如实记录其不可中断区间；
- 为 `close()` 提供幂等语义。

离线 protocol-comparison script 继续可直接运行；如果共享 settings 被迁入 `src`，script
改为 import 正式 helper，不能维护两份会漂移的常量。

### 6.2 新增 SGLang HTTP synthesizer

新增 `src/streammuse/infrastructure/rap/sglang_moss_tts.py`：

- `SglangMossConfig`：base URL、model、service-visible reference URI、reference text、各阶段
  timeout 上限、cancellation grace period、response byte limit；
- 一个进程内复用的 `httpx.Client`，禁止每个 phrase 新建 TCP connection；
- `GET /health` 和 `GET /v1/models` probe；
- request builder 和禁止 payload key collision 的 validation；
- non-streaming `/v1/audio/speech` request；
- 每次请求从 `SynthesisExecutionContext` 计算实际 connect/read/write/pool timeout，不能在
  `remaining_budget_ms` 耗尽后继续使用静态 timeout；
- bounded streaming read，即使响应是 non-streaming 也不能无上限读入内存；
- 只接受成功 HTTP status 和受支持的 WAV media type；
- 将 response 写到 `.source.partial.wav`，验证完成后用 `os.replace()` 原子提交；
- 复用或提取现有 24 kHz、mono、finite、nonempty、nonsilent validation；
- 网络错误、HTTP 400/429/500/503、invalid WAV、timeout 使用可区分的异常信息；
- 对 cancellation 关闭本请求的 response/connection，并在真实 SGLang qualification 中证明
  GPU work 是否随 client disconnect 被释放；不能把本地 Future cancel 当作服务端已 abort；
- `close()` 幂等，关闭 persistent client；
- 不把 reference bytes、reference transcript、绝对私有路径写入普通日志或 public health。

初始版本不做隐藏 retry。realtime 请求有严格 deadline，自动 retry 会制造重复 GPU work，
并可能让旧请求在客户端放弃后继续占用 SGLang。错误交给既有 render failure/fallback 路径。

### 6.3 Render server wiring 和 CLI

修改 `src/streammuse/presentation/rap_render_server.py`：

- `RapRenderServerConfig` 增加 backend、SGLang URL、request timeout、cancellation grace、
  reference text file、service-visible reference URI 和 producer identity；
- parser 增加：
  - `--moss-serving-backend`；
  - `--moss-sglang-url`；
  - `--moss-request-timeout-s`；
  - `--moss-cancellation-grace-s`；
  - `--moss-reference-text-file`；
  - `--moss-sglang-reference-uri`；
- `inprocess` 模式继续使用 `--moss-device`；
- `sglang-omni` 模式不加载 MOSS weights，不 import heavy MOSS backend；
- 根据 backend factory 构造统一 synthesizer；
- 在 API 接受请求时生成 absolute monotonic deadline，传入 planner、renderer 和 synthesizer；
- 监听 client disconnect 并维护 single-flight waiter count；只有 deadline 到期或最后一个 waiter
  离开时才取消共享 owner；
- `--moss-request-timeout-s` 只是 server-side 上限，实际请求上限取它与剩余预算的最小值；
- startup 顺序改为：probe Qwen -> probe SGLang -> real MOSS warmup -> MMS warmup -> ready；
- 任一依赖未 ready 或 warmup artifact invalid 时 fail closed，不启动一个假健康服务；
- 根据 producer manifest 初始化 namespaced `_ArtifactStore`，cache mismatch 必须 fail closed；
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
producer_fingerprint
state=ready|degraded
endpoint_host/port（去除 credentials 和私有路径）
```

不得伪造 SGLang 没有返回的 revision/version；未知时记录 `unknown` 和明确 warning。

### 6.4 Artifact 和 monitoring metadata

public v1 render manifest、diagnostics、artifact id 和 ZIP members 保持字节级 schema 兼容。
不要在这些 strict object 中直接加入 `moss_backend` 等新 key。新增 private
`internal/moss_synthesis.v1.json`，由 render server 原子写入 H200 artifact 目录，内容包括：

- `moss_backend`：`inprocess` 或 `sglang-omni`；
- `moss_service_request_ms`；
- `moss_http_response_headers_ms` 和 `moss_http_first_body_byte_ms`；
- `moss_response_download_ms`；
- `moss_response_bytes`；
- `cache_hit`（benchmark 样本必须可证明为 `false`）；
- SGLang model/server identity；
- resolved generation parameters；
- reference audio/text hashes；
- producer fingerprint 和 config hash；
- cancellation/deadline outcome；
- streaming flag，初始固定为 `false`。

保留现有顶层 `stage_timings_ms["moss"]`，以免 UI、summary 和历史分析工具失效。non-streaming
HTTP 的 response-header/first-body-byte 时间只能描述 HTTP service response onset，不能命名为
streaming TTFA。SGLang 内部 reference encode、prefill、AR 和 vocoder breakdown 只有在 pinned
launcher 的 profiler/trace 能可靠按 correlation id 对齐时才记录；否则写 `unavailable` 及原因，
不得用客户端 wall time 猜测内部阶段。

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

- 优先使用 exact image digest；或 pin exact SGLang-Omni commit/custom wheel，而不是 floating
  branch/tag；
- `sglang-omni==0.1.4` 只能作为待资格验证的候选 pin，不能在 capability audit 前直接视为最终 pin；
- 保存 Python、SGLang-Omni、SGLang、PyTorch、CUDA、FlashAttention/FlashInfer 的完整版本；
- 把已验证的 upstream `moss_tts.yaml` vendor 到部署目录，使用绝对路径并记录内容 SHA-256；
- 记录 model snapshot revision，不只记录 floating Hugging Face name。

Phase 0 必须填写 capability matrix，不能把 upstream optimization tracking 中“已提出/待合并”的
项目当作当前 release 已包含的能力：

| 能力 | 首版是否必需 | pinned build 证据 | 决策 |
|---|---|---|---|
| non-streaming `/v1/audio/speech` WAV | 是 | smoke + API contract | 缺失则阻塞 |
| `ref_audio` local URI + `ref_text` | 是 | exact payload smoke | 缺失则阻塞 |
| `token_count`、seed 和 generation knobs | 是 | repeatability test | 缺失则阻塞 |
| client disconnect / deadline 后有可验证的安全恢复 | 是 | queue recovery test | 直接 abort 或 degraded + restart，二者均需验证 |
| request-level profiler/correlation | 否 | profiler probe | 缺失则内部 breakdown 标 `unavailable` |
| raw PCM streaming | 第二阶段 | streaming smoke | 不影响首版 |

若目标优化所在 PR 不在候选 release 中，选择包含该 PR 的 exact commit/custom image，并重新跑完整
qualification；不能把不同 commit、不同硬件和不同测量点的上游数字相加，推导本项目收益。

示意启动命令：

```bash
CUDA_VISIBLE_DEVICES=<moss-gpu> sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-TTS-v1.5 \
  --config /opt/streammuse/sglang-omni/moss_tts.yaml \
  --allowed-local-media-path /models/streammuse \
  --host 127.0.0.1 \
  --port 8030
```

`--allowed-local-media-path` 及其语义必须在 pinned release 上用 `--help` 和真实请求核对；若该
release 的 flag 名称不同，文档和 launch manifest 使用实际名称。文档不能仅依据 main branch
写入未经验证的 flag，也不能依赖当前 shell working directory 才能找到 config。

## 7. Startup、failure 和安全语义

### Startup gate

render server 只有在以下条件全部成立后才返回 `ready=true`：

- Qwen `/v1/models` 返回预期 model；
- SGLang `/health` 为 200；
- SGLang `/v1/models` 包含预期 MOSS model；
- host reference WAV、service-visible reference WAV 和 transcript hash 已计算且 audio hash 一致；
- pinned runtime/config/model identity 已解析，producer manifest 与 namespace 一致；
- 一次真实、非静音、24 kHz mono MOSS warmup 成功；
- MMS 对该 warmup WAV 的 known transcript alignment 成功；
- Rubber Band probe 成功；
- artifact root 可写。

### Runtime failure

- HTTP 400：configuration/request contract error，不 retry；
- HTTP 429/500/503、connect error、timeout：记录 bounded diagnostics，当前 request 失败；
- invalid/truncated WAV：删除 partial file，request 失败；
- request deadline 从 API acceptance 开始计算；超时后不得继续进入下一个 stage 或提交 artifact；
- client disconnect：移除当前 waiter；最后一个 waiter 离开时取消 owner 并关闭当前 response body；
- 只有实际观察到 SGLang request 消失且后续短请求及时完成，才能把 cancellation 记为成功；
- cancellation grace 超时后将依赖标为 `degraded` 并执行显式 restart runbook；
- request 失败不得覆盖已有 successful canonical artifact；
- 不在同一次运行中自动切换 backend。

### Security

- SGLang 和 render server 都 bind loopback；
- Mac 只 forward render server port；
- reference path 来自 server startup config，不接受 Mac request 覆盖；
- SGLang local-media allowlist 只覆盖 reference mount，不允许任意 host filesystem；
- health、trace 和错误信息进行现有 private-path/credential sanitization；
- container 只 read-only mount model/reference，artifact root 只给 render server 写权限。

## 8. 测试策略

### Unit tests

新增 `tests/unit/infrastructure/rap/test_sglang_moss_tts.py`：

- exact request payload mapping；
- `token_count` 和 per-request seed；
- reference text/path handling；
- `voice="default"` 和 service-visible `file://` URI mapping；
- persistent connection/client reuse；
- health/model probe；
- successful WAV atomic commit；
- response byte upper bound；
- wrong status/content type；
- truncated、stereo、wrong-rate、silent、NaN WAV；
- timeout/connect failure；
- deadline 已经过期、在 response header 前到期和下载中到期；
- cancellation signal、response close 和 cancellation outcome metadata；
- partial/stale file cleanup；
- close idempotency；
- error messages do not leak transcript or private credentials。

更新 `tests/unit/infrastructure/rap/test_moss_tts.py`：

- backend-neutral contract；
- current in-process behavior unchanged；
- `MossServingMetadata` 默认值兼容现有构造器；
- monotonic deadline helper 和 in-process 不可中断状态；
- shared generation settings do not drift。

更新 `tests/unit/presentation/test_rap_render_server.py`：

- both backend CLI modes；
- incompatible/missing flags fail before heavy imports；
- SGLang mode never constructs `PersistentMossSynthesizer`；
- in-process mode never constructs SGLang client；
- startup probe/warmup order；
- dependency failure prevents ready；
- health sanitization and new metadata；
- producer fingerprint canonicalization、namespace、cache mismatch 和 legacy cache isolation；
- single-flight key 包含 producer fingerprint；
- 一个 waiter disconnect 不取消共享 owner，最后一个 waiter disconnect 才取消；
- deadline 到期后不提交 successful artifact；
- shutdown closes the selected synthesizer exactly once。

更新 manifest、monitoring、summary tests，使用 golden fixtures 证明 public v1 exact keys、artifact ids
和 ZIP members 不变，旧 artifact 仍能读取，private sidecar 不进入 public package。

### Hermetic integration tests

使用本地 fake SGLang FastAPI server：

- `/health`、`/v1/models`、`/v1/audio/speech` 完整 round trip；
- fake server 返回 fixture WAV，真实 `MossAlignedPhraseRenderer` 后续 validation 继续工作；
- concurrent same request id 仍由 render server single-flight 合并；
- timeout、最后一个 waiter disconnect、503、malformed audio 后下一次请求可恢复；
- 两个 backend 对同一 request id 使用不同 namespace，candidate 不会命中 baseline cache；
- complete artifact cache hit 不再次调用 SGLang；
- deadline/cancellation 不产生 successful manifest、残留 partial WAV 或永久占用的 single-flight entry；
- render server 对 Mac 的 public rap package contract 不变。

真实 MOSS checkpoint 和 H200 不进入默认 pytest gate。

## 9. H200 A/B 实验

### 9.1 先冻结实验设计

在生成任何 candidate 结果前提交 experiment manifest，固定：commit、producer fingerprints、
physical GPU、runtime/image digest、MOSS snapshot、config hash、reference hashes、prompt corpus、
generation settings、seed resolver、计时定义、统计脚本版本和下述数值门槛。candidate 跑完后不得
根据结果修改阈值；如确需修改，旧实验作废并更换 experiment id 全量重跑。

- 30 条 warm uncached request 只作 qualification smoke，不用于最终 promotion 结论；
- 最终 acceptance 每个 backend 至少 100 条 warm uncached paired requests；
- baseline/candidate 使用不同 artifact roots 和 producer namespaces；每个 block 使用空 root；
- 每个样本必须有 sidecar 证明 `cache_hit=false`、实际 backend 和 producer fingerprint；
- current in-process backend 和 SGLang-Omni 分开启动，不能同时抢同一 GPU；
- 至少四个 counter-balanced blocks，顺序由预先保存的随机 seed 生成；
- cold startup/model-load/warmup 单独测，不能混入 warm distribution；
- 独占 GPU；无法独占时记录所有 competing processes，并使该 block 无效后重跑。

### 9.2 分成两个明确命名的 cohort

**Cohort A：serving-only parity。** baseline 和 candidate 使用同一个 MOSS snapshot、reference WAV、
text、flow、`token_count`、generation parameters 和 seed；两边都不传 reference transcript。只有
pinned SGLang build 明确支持缺省 `ref_text` 且 smoke 通过时才运行。该 cohort 用于估计 serving
边界变化，不能代表最终 voice-clone 质量。

**Cohort B：production-candidate。** baseline 是现有 in-process production 行为（reference WAV，
无 transcript），candidate 是计划上线的 SGLang 行为（同一 WAV 加准确 `ref_text`）。这是部署决策
的主要 cohort，但报告必须明确标注 conditioning 不完全相同，不能把全部质量差异归因于 serving
框架。若 Cohort A 不可运行，则不发布“纯 serving 加速”结论，只报告 production path 的端到端
结果。

### 9.3 收集指标和计时定义

性能：

- startup/model-load/warmup time；
- render server 排队时间、MOSS stage wall time、HTTP response headers/first body byte/download；
- complete MOSS WAV latency p50/p95/max；
- MMS、R3、package 和 complete H200 server latency；
- GPU memory peak、utilization、power；
- error、timeout、runaway、deadline 和 cancellation queue-recovery；
- Mac end-to-end first usable finalized two-bar package latency。

non-streaming `response_format=wav` 的 HTTP first body byte 不称为 TTFA。reference preprocessing、
prefill、AR generation 和 vocoder 仅在 request-level profiler 能可靠关联时单列，否则明确标记
`unavailable`。

质量：

- WAV rate/channel/finite/non-silent；
- generated duration distribution；
- MMS coverage/confidence；
- ASR corpus WER、deletion 和 substitution；
- R3 local stretch ratio 和 fallback frequency；
- final vocal exact frame count；
- speaker similarity（现有工具可复用时作为硬门槛，否则在 experiment manifest 中预先降级为
  只报告指标）；
- 固定盲听样本的 intelligibility、voice identity、rhythm/artifact 检查。

报告同时给出原始样本、p50/p95/max、paired delta 和 bootstrap 95% confidence interval；不能只给
最优轮次或把多个 upstream benchmark 数字相加。

### 9.4 数值化 Acceptance gate

以下是默认门槛；Phase 0 可基于现有指标量纲收紧，但必须在 candidate 运行前冻结：

- qualification：两边各 30/30 完成，无 crash、OOM、hung request、schema violation 或错误 cache hit；
- final reliability：production-candidate 两边各至少 100/100 完成，且 timeout/cancellation probe 后
  第一个短请求在其正常 warm p95 上限内完成；
- artifact：100% final vocal 满足现有 exact frame count 和 package validation，public v1 golden
  schema/ZIP members 不变，private sidecar 100% 可关联；
- cache：candidate 的 SGLang 调用数等于 uncached 样本数，`wrong_producer_cache_hit=0`；
- ASR：candidate corpus WER 不高于 baseline `+1.0` 个百分点，且 deletion rate 不高于 baseline
  `+0.5` 个百分点；
- alignment：candidate median MMS coverage 不低于 baseline `-0.01`，median confidence 不低于
  baseline `-0.02`，且不新增低于现有 hard floor 的样本；
- warp：candidate R3 local-stretch p95 不高于 baseline 的 `1.10x`，fallback 数不超过 baseline
  `+max(1, 1% * N)`；
- voice：若 speaker-similarity 工具进入硬门槛，candidate median 不低于 baseline `-0.02`；
- 盲听：至少 20 个预先固定的 paired 样本、至少 2 位评审；5 分制 intelligibility、voice identity
  和 rhythm 的 candidate median 均不低于 baseline `-0.5`，且没有新增一致判定的 critical artifact；
- primary latency：production-candidate 的 paired MOSS latency median delta `< 0`，且 paired bootstrap
  95% CI 上界 `< 0`；
- tail latency：candidate warm MOSS p95 不高于 baseline `1.10x`，complete H200 server p95 不高于
  baseline `1.05x`；
- Phase 8 realtime：固定 Mac E2E 场景的 deadline miss 和 playback underrun 均不高于 baseline，且
  不得出现 cancellation 后持续占用 GPU 的 request；
- provenance：health、sidecar 和 experiment manifest 三处 identity 一致，证明使用的是目标
  SGLang build，而不是旧 backend、错误 model/config 或 cache-only 假成功。

任一 reliability、artifact、cache、schema 或 provenance 门槛失败都直接阻塞 rollout。latency 没有
显著提升但质量和稳定性通过时，只保留为实验 backend，不把“迁移成功”写成“性能优化成功”。
Phase 7 的 H200 evaluator 只计算 qualification、reliability、artifact、cache、quality、latency 和
provenance gates；Phase 8 realtime evidence 在此时必须报告为 `pending_phase_8`，不能反向阻塞
H200 `promotion-candidate`。`promotion-candidate` 仅允许进入 Mac gate；加入真实 Mac evidence
重新评估并完成 canary/rollback 演练后，才允许 rollout。

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

## 11. 分阶段 Gate 总览

本节只描述 phase 顺序和退出条件；可直接执行的逐项 checklist 见文末第 15 节。

### Phase 0：Design freeze

- [ ] 冻结 public v1 schema golden、producer fingerprint schema 和 cache namespace 规则。
- [ ] 冻结 experiment corpus、双轨 cohort、统计方法和数值 acceptance gate。
- [ ] 记录当前 branch/commit/dirty state、H200 stack、model/reference hashes 和 GPU mapping。
- [ ] 填完 SGLang capability matrix，选出 exact candidate build；未知能力不得进入实现假设。
- [ ] affected tests 全绿后才进入 Phase 1。

### Phase 1：SGLang runtime qualification

- [ ] 建立独立 pinned runtime/image、绝对 config 路径和只读 reference mount。
- [ ] 验证 loopback API、完整 request knobs、`voice="default"`、local URI 和 transcript。
- [ ] 验证 WAV、MMS compatibility、deadline/disconnect 后 GPU queue recovery。
- [ ] 保存 exact launch manifest、package lock、logs、VRAM 和 capability evidence。
- [ ] 任一首版 required capability 缺失则阻塞，不能靠 StreamMUSE client 猜测兼容。

### Phase 2：Backend-neutral contract

- [x] 实现 `MossSynthesizer`、`SynthesisExecutionContext` 和统一 lifecycle/error contract。
- [x] 增加默认兼容的 `MossServingMetadata`，抽出 generation settings/seed resolver。
- [x] 让 in-process adapter 实现 deadline checkpoints 并保持现有输出行为。
- [x] contract 和 in-process regression tests 全绿。

### Phase 3：Cache namespace 和 schema protection

- [x] 实现 canonical producer manifest/fingerprint 和 namespaced `_ArtifactStore`。
- [x] single-flight key 纳入 producer fingerprint，legacy artifacts 保持隔离。
- [x] 新增 private sidecar，public v1 manifest/diagnostics/ZIP exact schema 不变。
- [ ] cache contamination、golden package 和 old artifact tests 全绿（代码 fixture 已覆盖，真实旧 artifact 待验）。

### Phase 4：SGLang client

- [x] 实现 config、probe、exact payload、persistent HTTP 和 bounded WAV download。
- [x] 实现 monotonic deadline、best-effort cancellation、错误分类和 partial cleanup。
- [x] 写入 backend-neutral result metadata 和 private sidecar 输入。
- [x] SGLang client unit tests 全绿。

### Phase 5：Render server integration

- [x] 增加 backend-specific CLI/config validation，并确保只 import/构造选中的 backend。
- [x] deadline 从 API acceptance 贯穿 planner、synthesis、MMS、R3 和 package commit。
- [x] 实现 waiter-aware single-flight cancellation、degraded state 和 restart runbook trigger。
- [x] 实现严格 startup gate、bounded health 和 clean shutdown。
- [x] presentation tests 全绿。

### Phase 6：Hermetic integration、回归和文档

- [x] fake SGLang/MockTransport 覆盖 success/error/timeout/disconnect/recovery/cache namespace。
- [x] public Mac package、Opus、fallback 和代码内历史 artifact compatibility 回归通过。
- [ ] rap targeted tests 与默认 pytest suite 全绿（targeted 已绿；默认 suite 有已记录的外部 blocker）。
- [ ] install/launch/health/stop/rollback 文档可由另一位开发者照做成功（文档已交付，第二人待验）。

### Phase 7：H200 qualification 和 final A/B

- [ ] 两个 backend 先各跑 30 条 warm uncached qualification。
- [ ] 按可用能力运行 Cohort A，并始终运行 Cohort B。
- [ ] final 每个 backend 至少 100 条 paired warm uncached requests，四个 counter-balanced blocks。
- [ ] 生成含原始样本、置信区间、quality、failure 和 provenance 的不可变报告。
- [ ] 所有第 9.4 节 hard gates 通过后才进入 Mac E2E。

### Phase 8：Mac E2E 和受控 rollout

- [ ] 通过真实 tunnel、Web UI、Opus、deadline、disconnect 和多场游戏验证。
- [ ] acceptance 通过后才把 quickstart 推荐 backend 改为 `sglang-omni`。
- [ ] 保留显式 `inprocess` rollback 至少一个稳定观察周期。
- [ ] 观察期内任何 hard gate 回归立即 rollback，并保留失败 artifacts。

### Phase 9：Streaming 独立研究

- [ ] 单独测 raw PCM TTFA、chunk cadence、aggregate quality 和 cancellation。
- [ ] 比较 non-streaming、stream-and-aggregate、server-internal overlap 和 one-bar finalize。
- [ ] 另行设计 public schema v2/dual-reader；不修改本次冻结的 v1 contract。
- [ ] 只有 first-usable finalized audio 明显改善且所有质量门槛通过才考虑生产化。

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
| baseline 无 transcript、candidate 有 transcript | 把 conditioning 收益误报为 serving 收益 | Cohort A/B 分开命名；不可运行 A 时不发布纯 serving 结论 |
| SGLang dependency 污染主环境 | MMS/Transformers 失效 | 独立 pinned container/venv |
| HTTP 增加进程边界开销 | 单请求反而变慢 | loopback persistent client、30 条 qualification + 100 条 paired acceptance |
| 相同 request id 命中旧 backend cache | A/B 假成功、线上 provenance 错误 | producer fingerprint namespace、独立 roots、uncached call-count gate |
| 给 strict v1 object 新增字段 | Mac/parser 直接拒绝 package | 冻结 exact keys；详细数据只写 private sidecar；未来另开 v2 |
| renderer 串行导致 batching 无收益 | 吞吐优化没有作用 | 首先评估 c=1；并发作为后续独立任务 |
| streaming 绕过 alignment | syllable 不落 beat | 第一版 non-streaming；只播放 finalized audio |
| threadpool/Future cancel 不会停止 GPU request | tail latency/资源泄漏 | end-to-end deadline、真实 queue-recovery probe、degraded + restart runbook |
| 一个 waiter 断开取消共享 owner | 其他同 request 客户端无故失败 | waiter-aware single-flight，最后一个 waiter 才取消 owner |
| health 报告假 ready | 游戏开始后才失败 | real synthesis + MMS warmup 后才 ready |
| local-media allowlist 过宽 | SGLang 可读取非预期 host 文件 | 只读单目录 mount、固定 URI、启动 hash 比对 |
| upstream `dev` 或未合并优化漂移 | 无法复现或宣称不存在的优化 | capability matrix，pin exact commit/image/config/model revision |
| profiler 不支持 request correlation | 内部阶段计时被误归因 | 标记 `unavailable`，只用可验证的 client/server wall time 做 gate |

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

## 15. 详尽实施 TODO List（执行版）

### 15.0 执行规则和证据约定

- [ ] 为本计划创建唯一 implementation/experiment id，并在代码、部署 manifest、benchmark
  输出目录和报告中保持一致。
- [ ] 每个 checklist 完成时记录对应 commit、命令、测试输出或 artifact 路径；不能只勾选无证据。
- [ ] 每个 Phase 的 exit gate 单独 review；上一个 gate 未通过，不开始依赖它的后续实现。
- [ ] benchmark threshold、corpus、block 顺序和统计脚本在 candidate 运行前提交到 git。
- [ ] 所有失败样本保留 request payload 的 sanitized form、producer fingerprint、日志和 artifacts。
- [ ] 不把 token、credential、reference transcript、个人 home path 写进 tracked evidence。
- [ ] 若执行中改变 public contract、模型 snapshot、config 或 generation settings，递增 experiment id，
  重新生成 producer fingerprint 并使此前 benchmark 失效。

### 15.1 Phase 0：冻结代码、协议和实验设计

**Repository 与基线证据**

- [ ] 记录 `git branch --show-current`、`git rev-parse HEAD` 和 `git status --short`。
- [ ] 若 worktree 非 clean，保存 patch digest；正式 H200 acceptance 使用 clean commit 或明确纳入
  producer manifest 的 patch digest。
- [ ] 记录 Mac checkout、H200 checkout、Python 和 StreamMUSE package/version identity。
- [ ] 运行现有 rap targeted unit/integration tests并保存完整结果。
- [ ] 运行默认 pytest suite，记录 pass/skip/fail 数和已知非本任务失败。
- [ ] 保存当前 public request payload、PCM package 和 Opus package 的 golden fixtures。
- [ ] 保存 `RemoteRapChunkManifest`、diagnostics、artifact ids 和 ZIP member exact-key 快照。
- [ ] 盘点 `_ArtifactStore` 当前目录、complete marker、failure marker、single-flight 和 cache-hit 语义。
- [ ] 记录当前 `remaining_budget_ms` 实际约束到哪些阶段，建立 deadline gap 清单。
- [ ] 保存 current in-process cold startup、warmup 和 30 条 warm uncached qualification baseline。

**Reference、环境与 producer identity**

- [ ] 人工核对 reference WAV 与 transcript 逐字一致，包括标点、缩写、数字读法和静音边界。
- [ ] 确认 reference WAV sample rate、channel、frame count、duration、finite 和 non-silent。
- [ ] 计算 reference WAV SHA-256 和 UTF-8 transcript SHA-256。
- [ ] 定义 host reference path 与 SGLang service URI/mount mapping，确认二者 bytes hash 相同。
- [ ] 记录 Qwen、MOSS、MMS 对应 physical GPU、CUDA-visible index 和 baseline peak VRAM。
- [ ] 定义 `producer_manifest_schema_version=1` 的字段、类型、canonical JSON 规则和示例。
- [ ] 明确 backend implementation revision：clean commit；dirty development 额外带 patch digest。
- [ ] 明确 producer fingerprint 不包含 credential、transcript 原文、hostname 或个人绝对路径。
- [ ] 为两个不同 field order 的等价 manifest 编写相同 fingerprint test vector。
- [ ] 为任一输出相关字段变化编写 fingerprint 必须变化的 test vectors。

**SGLang capability freeze**

- [ ] 列出候选 release、exact commit/custom image 和对应 optimization PR 集合。
- [ ] 对每项 optimization 标记 `merged`、`included_in_pin`、`enabled_by_config`、`verified_on_H200`。
- [ ] 核对 candidate build 的 Python、CUDA、PyTorch、SGLang、SGLang-Omni、FlashAttention 和
  FlashInfer compatibility。
- [ ] 核对 `/v1/audio/speech` 的 exact keys、`voice` 语义、local media flag 和 response MIME。
- [ ] 核对 `ref_text` 是否可省略，决定 Cohort A 是 runnable 还是 `not_supported`。
- [ ] 核对 request-level seed、`token_count` 和全部 audio sampling knobs 是否生效。
- [ ] 核对 health/models/profiler/cancellation/streaming 能力，完成第 6.5 节 capability matrix。
- [ ] 选定唯一 candidate pin；若更换 pin，重新执行本节全部 capability 项。

**Experiment freeze**

- [ ] 建立至少 100 条唯一 two-bar paired corpus，覆盖常见长度、押韵、标点和困难发音。
- [ ] 从 final corpus 固定选出 30 条 qualification subset，不允许 qualification 后替换难样本。
- [ ] 固定每条 request 的 text、flow、tempo、`token_count`、seed 和 expected frame count。
- [ ] 固定 Cohort A/B 的 reference/transcript 差异并写入 machine-readable experiment manifest。
- [ ] 固定至少四个 counter-balanced block 的运行顺序和随机化 seed。
- [ ] 固定每个 block 的独立空 artifact root 命名规则。
- [ ] 固定 cold/warm 边界、warmup 次数和不计入分布的请求编号。
- [ ] 固定 GPU telemetry 采样频率、clock/power policy 和 competing-process invalidation 规则。
- [ ] 固定第 9.4 节全部数值 gate、hard/soft 分类和缺失 speaker metric 的处理方式。
- [ ] 固定 paired bootstrap 实现、resample 次数和 confidence interval 输出格式。
- [ ] 固定至少 20 条 blind-listening subset、随机播放顺序、评分表和 critical artifact 定义。
- [ ] 提交 experiment manifest、corpus、thresholds 和 analysis script 后打 design-freeze tag。

**Phase 0 exit gate**

- [ ] public v1 golden、producer fingerprint test vectors、capability matrix 和 experiment manifest
  均已 review。
- [ ] existing affected tests 全绿，baseline evidence 可读取，所有未决项都有明确 blocker owner。

### 15.2 Phase 1：部署并资格验证 pinned SGLang runtime

**Runtime 构建**

- [ ] 在 StreamMUSE 主环境之外选择 Docker 或独立 Python 3.12 virtual environment。
- [ ] Docker 路径记录 image repository、tag 和 immutable digest；venv 路径记录 exact wheel hashes。
- [ ] 保存完整 package lock、GPU driver、CUDA runtime 和 `nvidia-smi` 输出。
- [ ] 下载并 pin exact MOSS snapshot revision，验证所有 model files 完整。
- [ ] 从选定 commit vendor `moss_tts.yaml` 到稳定部署目录。
- [ ] 使用绝对 config path，计算 config SHA-256，并确认 launch 不依赖 shell current directory。
- [ ] 将 reference WAV read-only mount 到 `/models/streammuse/reference.wav`。
- [ ] 将 local-media allowlist 收窄到 `/models/streammuse`，验证不能读取 allowlist 外文件。
- [ ] 在 pinned binary 上保存 `sgl-omni serve --help`，逐项核对计划中的 launch flags。
- [ ] 建立 machine-readable launch manifest，包含 argv、env allowlist、versions、hashes 和 GPU mapping。

**API qualification**

- [ ] 只 bind `127.0.0.1:8030`，确认远端网卡无法直接访问。
- [ ] 验证 `/health` 返回 200，记录 exact response shape。
- [ ] 验证 `/v1/models` 包含预期 MOSS model identity，记录 revision 是否可可靠获得。
- [ ] 发送 reference-less non-streaming smoke，确认服务基础生成路径可用。
- [ ] 发送 `ref_audio` only smoke；若失败，将 Cohort A 标记 `not_supported` 并保存证据。
- [ ] 发送 `ref_audio + ref_text` production payload，确认 `voice="default"` 可用。
- [ ] 逐项变更 `token_count`、seed、temperature、top-p、top-k 和 repetition penalty，确认 API 接受且
  server log/trace 显示参数被消费。
- [ ] 用相同 seed/config/hardware 重复请求，记录 determinism 范围，不要求跨 backend bitwise 相同。
- [ ] 验证错误 model、错误 local URI、空 transcript 和未知 payload key 的失败语义。

**Audio 与恢复能力**

- [ ] 验证响应 status、MIME、Content-Length/streaming behavior 和最大合理 response bytes。
- [ ] 验证 WAV 为 24 kHz mono、finite、nonempty、nonsilent，duration 与 `token_count` 关系合理。
- [ ] 用真实 MMS 读取并对齐至少一个 known transcript sample。
- [ ] 用现有 R3 path 处理 sample 并验证 exact final frame count。
- [ ] 记录 model load、warmup、首个请求和连续 warm 请求 latency。
- [ ] 记录 idle、warm 和 peak VRAM，不与 Qwen 并发占同一目标 GPU。
- [ ] 发起长请求后主动断开 client，观察 server request/queue/GPU utilization 是否释放。
- [ ] 在 disconnect 后立即发送短请求，确认其 completion 未被 abandoned work 长期阻塞。
- [ ] 发起会超过 deadline 的请求并重复 queue-recovery probe。
- [ ] 若 abort 无法证明，测出 cancellation grace、记录 degraded/restart 所需动作和最坏恢复时间。
- [ ] 探测 request profiler/correlation；不可靠时把内部 breakdown 能力标记 `unavailable`。
- [ ] 做一次 raw PCM streaming smoke，只记录 format、TTFA、chunk cadence 和完整性，不接 StreamMUSE。

**Phase 1 exit gate**

- [ ] required capability、WAV/MMS/R3、loopback/security 和 recovery probes 全部有证据。
- [ ] exact pin、config、model、reference 和 launch manifest 已归档；缺少首版 required capability 时
  停止实施并更新设计，不进入 Phase 2。

### 15.3 Phase 2：实现 backend-neutral synthesis contract

**Execution context**

- [x] 在 `src/streammuse/application/rap/execution.py` 定义 `SynthesisExecutionContext`，避免 infrastructure
  类型反向渗入 application orchestration。
- [x] context 使用 absolute monotonic deadline，不保存可漂移的 wall-clock deadline。
- [x] context 提供 `remaining_seconds()`、`raise_if_expired()` 和 `raise_if_cancelled()`。
- [x] context 携带 thread-safe cancellation signal 和 bounded correlation id。
- [x] 校验 NaN/inf/过去 deadline、空/超长 correlation id 和错误 signal 类型。
- [x] 定义 deadline、cancelled、backend unavailable 和 invalid output 的稳定 exception hierarchy。
- [x] 使用 fake clock 测试预算递减、恰好到期、取消优先级和错误消息 sanitization。
- [x] 为 startup warmup 创建独立、显式上限的 execution context，不复用用户 request budget。

**Protocol、result 和旧 backend**

- [x] 定义 `MossSynthesizer.synthesize(request, output_wav, *, execution)` Protocol。
- [x] 定义 backend-neutral `MossServingMetadata`，所有新增字段有兼容默认值。
- [x] 保持现有 `MossPhraseResult` positional/keyword consumers 可用，只追加 default field。
- [x] 抽出 generation defaults、payload-independent validation 和 per-request seed resolver。
- [x] 让 offline comparison script 和 production adapter import 同一份 settings。
- [x] 让 `PersistentMossSynthesizer` 在 load/generate/decode/write 前后执行 deadline/cancel checkpoints。
- [x] 将不可中断 `model.generate()` 区间明确记录为 `cancel_requested_but_not_interruptible`。
- [x] 确保过期/取消结果不原子提交为成功 WAV。
- [x] 保持 `warmup()`/`close()` 幂等，并测试 repeated close 和 partial initialization close。
- [x] 更新所有 fake synthesizer/test doubles 以实现新 contract。
- [x] 运行 `test_moss_tts.py`、`test_moss_aligned_phrase.py` 和共享 settings regression tests。

**Phase 2 exit gate**

- [x] in-process backend 的现有成功输出和 public metadata 无回归。
- [x] contract、deadline、cancel、lifecycle 和 backward-construction tests 全绿。

### 15.4 Phase 3：实现 producer-aware cache 和 v1 schema protection

**Producer manifest 与 fingerprint**

- [x] 定义 immutable `ProducerManifestV1`，字段类型禁止依赖任意 `repr()`。
- [x] 使用 UTF-8、sorted keys、固定 separators 和禁止 NaN 的 canonical JSON encoder。
- [x] 将 backend name、backend implementation revision 和 StreamMUSE clean commit/patch digest 纳入。
- [x] 将 exact MOSS snapshot revision 纳入，禁止只使用 floating model name。
- [x] SGLang backend 纳入 SGLang-Omni/SGLang versions、revision、environment 和 config SHA-256。
- [x] in-process backend 纳入相关 Transformers/PyTorch identity 和 generation implementation revision。
- [x] 纳入全部 resolved generation settings 和 seed policy version。
- [x] 纳入 reference audio/text SHA-256，不纳入 transcript、host path 或 service URI 原文。
- [x] 纳入 aligner model/tool identity、warp policy、output sample rate、codec contract 和 public schema version。
- [x] 明确并排除 bind port、日志级别和 artifact root 等 operational 字段。
- [x] 计算 `sha256(canonical_manifest_bytes)` 并使用完整 64-character hex。
- [x] 保存 canonical manifest bytes/hash，启动和 cache load 时自检一致性。

**Artifact store 与 single-flight**

- [x] 将 `_ArtifactStore` 根改为 `<artifact_root>/<producer_fingerprint>/`。
- [x] namespace 初始化时原子创建 producer manifest；已存在但内容不一致时 fail closed。
- [x] request artifact 保持 `<namespace>/<request_id>/`，public request id 算法不变。
- [x] cache load 同时校验 producer namespace、canonical request、complete marker 和 artifact validation。
- [x] single-flight map key 改为 `(producer_fingerprint, request_id)`。
- [x] legacy flat artifact root 默认不可见，不自动复制、hardlink 或标记为当前 producer cache。
- [x] 为 rollback backend 使用它自己的 producer namespace，不删除 candidate artifacts。
- [x] 为 benchmark 强制每个 block 使用独立空 root，并拒绝 root 中存在意外 complete artifact；
  root manifest hash 同时进入每条 row provenance 和 evaluator hard gate。
- [x] namespace/manifest race 使用 lock + atomic link/create；并发初始化 regression 已覆盖。
- [x] failed/cancelled/deadline artifact 不能留下 complete marker 或污染已有 successful artifact。
- [x] stale partial 文件只在对应 namespace/request 内清理，不跨 producer 删除。

**Private sidecar 与 public v1**

- [x] 定义 `internal/moss_synthesis.v1.json` schema、大小上限和 sanitization 规则。
- [x] sidecar 使用 partial file + fsync/replace 的项目原子写 helper。
- [x] sidecar 记录 backend、producer fingerprint、correlation id、resolved settings 和 reference hashes。
- [x] sidecar 记录 `cache_hit`，uncached benchmark 可与实际 backend call count 交叉验证。
- [x] sidecar 记录 queue/service/header/first-body/download/validation timings 的明确定义。
- [x] sidecar 记录 deadline、cancel request、abort confirmation、grace timeout 和 recovery outcome。
- [x] profiler 不可关联时写结构化 `unavailable_reason`，不伪造 internal stage durations。
- [x] packaging 使用显式 public member allowlist，保证 `internal/` 永不进入 PCM/Opus ZIP。
- [x] 保持 `stage_timings_ms["moss"]`、`model_tool_versions["moss"]` key 和现有 value constraints。
- [x] 当前 PCM/Opus package 与代码内 legacy/public fixtures 通过 round-trip。

**Tests**

- [x] 测试同 manifest 不同 key order 得到相同 fingerprint。
- [x] 参数、reference hash、model revision、config hash、backend、warp policy 任一变化都会换 namespace。
- [x] port、artifact root、日志级别不属于 manifest，不会无意义地换 namespace。
- [x] 同 request id 的 in-process 与 SGLang artifacts 互不命中。
- [x] producer manifest mismatch、缺失、截断、非法 JSON 均 fail closed。
- [x] legacy flat cache 不被新 store 命中，代码内旧 public reader fixtures 保持兼容。
- [x] private sidecar 不改变 strict manifest/diagnostics keys、artifact ids 或 ZIP member set。
- [x] concurrent same-producer same-request 仍只执行一次 owner work。
- [x] concurrent different-producer same-request 各执行一次，不共享 owner。

**Phase 3 exit gate**

- [x] cache contamination tests、namespace race tests 和 public v1 contract tests 全绿。
- [ ] 用一个真实旧 artifact 验证兼容读取，并证明 candidate 不会把它当 SGLang 成功结果。

### 15.5 Phase 4：实现 `SglangMossSynthesizer`

**配置和启动 probe**

- [x] 新建 `src/streammuse/infrastructure/rap/sglang_moss_tts.py`。
- [x] 定义 immutable `SglangMossConfig`，包含 base URL、model、reference URI/text、timeout caps、
  cancellation grace 和 response byte limit。
- [x] 对 URL scheme/host/port/path 做严格 validation，health/log 中移除 credentials 和 query secrets。
- [x] 只允许启动时配置的 reference URI；public request 不提供覆盖入口。
- [x] 启动时读取 transcript 一次，验证 UTF-8/非空/大小上限并计算 SHA-256。
- [x] composition 与 preflight 比对 host WAV 和 service-visible WAV hash。
- [x] 实现 bounded `GET /health`，区分 connect、timeout、status、JSON/schema 错误。
- [x] 实现 bounded `GET /v1/models`，要求目标 model 存在，不伪造缺失 revision。
- [x] probe response 和错误消息遵守长度上限及 private-path sanitization。

**Exact request mapping**

- [x] 建立唯一 request builder，禁止 call site 手写第二份 payload。
- [x] 固定 `voice="default"`、`response_format="wav"`、`stream=false`。
- [x] 映射 selected lyric、language、instructions、`token_count`、`max_new_tokens` 和 audio sampling knobs。
- [x] 使用共享 seed resolver，记录最终 resolved integer seed。
- [x] 映射固定 `ref_audio` service URI 和 startup-loaded `ref_text`。
- [x] request builder 不接受 overrides，caller 无法覆盖 model/reference/stream contract。
- [x] 编写 exact JSON contract test。
- [ ] 从真实 H200 qualification 保存并回放 sanitized request/response fixture。

**HTTP、deadline 和 cancellation**

- [x] 复用一个进程级 HTTP connection pool，不为每个 phrase 新建 client/TCP connection。
- [x] 明确 thread-safety 和 lifecycle；renderer lock 下只允许一个 active MOSS request。
- [x] 每次调用前检查 execution deadline/cancellation。
- [x] 将 connect/read/write/pool timeout 分别限制为当前 remaining deadline 与配置 cap 的最小值。
- [x] 以 streaming response API 有界读取 body，即使业务请求是 `stream=false`。
- [x] 在读取每个 body chunk 前后检查 deadline/cancellation 和累计 bytes。
- [x] 对过大 `Content-Length` 提前拒绝；无/错误长度时仍靠累计 byte limit 防护。
- [x] cancellation 时关闭本 request handle/response；不得通过关闭共享 client 误伤其他 waiter。
- [ ] 用 Phase 1 结论验证 transport 是否能终止 blocked request；不能时实现 degraded/restart 路径。
- [x] 不实现自动 retry，包括 429/500/503、connect reset 和 read timeout。
- [x] 记录本地 cancel requested 与上游 abort confirmed 为两个不同状态。

**WAV commit 与 metadata**

- [x] response 先写同目录唯一 `.source.partial.wav`，成功前不暴露 final path。
- [x] 只接受资格验证过的 WAV media type；兼容参数化 MIME 时显式 parse，不做 substring 猜测。
- [x] 验证 RIFF/WAV 可完整读取、24 kHz、mono、finite、nonempty 和 nonsilent。
- [x] 验证 frame/duration 在配置的安全上限内，防止异常音频拖垮 MMS/R3。
- [x] 在 validation 后再次检查 deadline/cancellation，再用 `os.replace()` 提交 final source WAV。
- [x] 任一异常删除本次 partial，不删除已有成功 final artifact。
- [x] 填充 `MossServingMetadata`，使用 `http_response_headers_ms` 等无歧义名称。
- [x] backend/model/server revision 未知时显式 `unknown`，不得从 URL 或 model name 推断。
- [x] 所有 metadata string/value 有长度、范围和 finite validation。
- [x] `close()` 幂等；partial init、probe failure 和 repeated close 都不泄漏 connection。

**Unit tests**

- [x] 覆盖 exact payload、seed、token count、reference、voice 和不可覆盖 contract。
- [x] 覆盖 client reuse、probe success/model missing 和 close idempotency。
- [x] 覆盖 200 WAV、chunked body、missing Content-Length 和 atomic commit。
- [x] 覆盖 400/429/500/503、connect error、headers timeout、mid-body timeout 和 oversized body。
- [x] 覆盖 malformed/truncated/stereo/wrong-rate/empty/silent/NaN/overlong WAV。
- [x] 覆盖 pre-expired、mid-download deadline 和 cancellation，确认无 final/partial 泄漏。
- [x] 覆盖 transcript、credential、absolute path 不出现在 exception/log/health。

**Phase 4 exit gate**

- [x] `test_sglang_moss_tts.py` 全绿，MockTransport + hermetic ASGI service cases 可重复。
- [ ] 从 exact payload 到 valid source WAV 的 contract 与 Phase 1 真实服务 fixture 一致。

### 15.6 Phase 5：接入 render server、deadline 和 waiter-aware cancellation

**CLI 与 composition**

- [x] 扩展 `RapRenderServerConfig`，加入 backend、SGLang URL、service reference URI、transcript file、
  request timeout cap、cancellation grace 和 producer identity inputs。
- [x] parser 增加第 6.3 节列出的 flags，并为每个 flag 编写 help/default test。
- [x] `inprocess` 模式拒绝只属于 SGLang 的冲突配置，继续要求当前 MOSS device/reference 参数。
- [x] `sglang-omni` 模式要求 URL、reference URI/text 和 pinned producer identity，忽略或拒绝
  `--moss-device` 的歧义用法。
- [x] backend-specific validation 在 heavy import/model load 前完成。
- [x] 使用 lazy factory，只 import/构造被选中的 synthesizer；测试另一 backend 永不构造。
- [x] composition 计算 producer manifest/fingerprint 后再初始化 `_ArtifactStore`。
- [x] shutdown 只关闭所选 synthesizer 一次，不擅自 kill 外部 SGLang process。

**Deadline propagation**

- [x] FastAPI route 一接受 request 就用 monotonic clock 记录 acceptance time，并据此计算 waiter absolute deadline。
- [x] request body 有界读取；JSON validation 后立即按 acceptance time 建 context，并在 cache lookup/
  single-flight join 前检查 budget 是否已耗尽。
- [x] 修改 orchestration/renderer API，把同一个 execution context 传到 candidate planning 和 MOSS path。
- [x] candidate planning 继续保留 render reserve，但不能成为唯一使用 budget 的阶段。
- [x] 在 cache load 后、进入 synth 前、MMS 前、R3 前、package 前和 response 前设置 checkpoints。
- [x] SGLang timeout 取 remaining deadline 与 server cap 的最小值。
- [x] 超时后禁止写 complete marker、成功 manifest 或成功 sidecar outcome。
- [x] cache hit 也必须在发送 response 前检查 waiter deadline。
- [x] 使用 fake clock 覆盖 stage 边界到期，避免依赖 sleep 的脆弱 deadline tests。

**Single-flight waiter semantics**

- [x] 为每个 `(producer_fingerprint, request_id)` 建立 owner state、waiter set 和 cancellation signal。
- [x] owner deadline 固定为首个 owner request 的 deadline；后来的 waiter 不得延长或复活它。
- [x] 每个 waiter 保留自己的 deadline；较短 waiter 到期只 detach 自己。
- [x] 一个 waiter disconnect 只 detach，不取消仍有 active waiter 的 owner。
- [x] 最后一个 waiter disconnect 或 owner deadline 到期时设置 owner cancellation signal。
- [x] route 监听 `request.is_disconnected()`，同时保证 watcher 在正常完成时被回收。
- [x] 本地 Future/task cancel 只记录为 local outcome，不等价于上游 GPU abort confirmed。
- [x] owner success 后只向仍未过期的 waiter 返回 artifact。
- [x] owner failure/cancel 后清除 single-flight entry，后续显式新请求可以重新执行。
- [x] 禁止同一次 owner 内做 hidden backend fallback 或 retry。

**Startup、health 和 degraded state**

- [x] startup 顺序实现为 config/input validation -> Qwen probe -> SGLang probe -> real MOSS warmup
  -> MMS warmup -> R3 probe -> artifact write probe -> ready。
- [x] warmup 使用真实 reference/text 和 production payload shape，但独立 artifact/request id。
- [x] 任一 startup gate 失败则进程 fail closed，不返回 `ready=true`。
- [x] health 暴露 backend、bounded identity、reference hashes、producer fingerprint、warmup time 和 state。
- [x] health 不暴露 transcript、credential、host/container absolute path 或未验证的 revision。
- [x] cancellation grace 超时将 state 原子切到 `degraded`，新 MOSS request 快速失败。
- [x] degraded health 给出 bounded reason/code 和 restart-required flag，不输出内部 traceback。
- [x] 外部 SGLang restart 后必须重启 render server 以重新 probe/warmup；进程内不会未经 gate 自动回 ready。

**Presentation tests**

- [x] 覆盖两种 backend 的 required/forbidden CLI combinations 和 lazy import。
- [x] 覆盖 startup gate 顺序、dependency/probe failure、partial cleanup 和 artifact write failure。
- [x] 覆盖 deadline/cancellation 后无成功 commit。
- [x] 覆盖两个 waiter 中一个 disconnect、最后 waiter 离开、不同 deadline 和 owner failure。
- [x] 覆盖 degraded transition、拒绝新请求、restart-only recovery gate 和 shutdown。
- [x] 覆盖 health bounds/sanitization、producer identity 一致性和 unknown version。

**Phase 5 exit gate**

- [x] `test_rap_render_server.py`、orchestration、renderer 和 artifact-store targeted tests 全绿。
- [x] 无任何代码路径在 SGLang 失败时隐式加载 in-process MOSS 或提交成功 artifact。

### 15.7 Phase 6：Hermetic integration、全量回归和操作文档

**Fake SGLang integration harness**

- [x] 建立测试内 fake HTTP service，不依赖真实 checkpoint、GPU 或公网。
- [x] fake `/health` 支持 healthy、non-200、slow/timeout、malformed 和 oversized response。
- [x] fake `/v1/models` 支持 expected model、missing model、unknown revision 和 malformed payload。
- [x] fake `/v1/audio/speech` 记录 call count/exact payload，并返回 deterministic fixture WAV。
- [x] 支持 delayed headers/timeout、slow chunks、disconnect、429/500/503、truncated 和 oversized body。
- [x] 支持观察 client response close，并把 local cancel、transport close 与 upstream abort confirmation 分开。
- [x] fixture 使用 MockTransport/TestClient、显式 close 和 `tmp_path`，teardown 后无端口或临时文件残留。

**End-to-end hermetic cases**

- [x] 跑 fake SGLang -> source WAV -> real boundary validation -> MMS/R3 adapter -> final artifact。
- [x] 验证成功路径只调用一次 speech endpoint，sidecar 与 producer identity 可关联。
- [x] 验证同 producer/same request 并发只调用一次 SGLang。
- [x] 验证不同 producer/same request 分别调用对应 backend，绝不复用 cache/single-flight。
- [x] 验证 complete cache hit 不调用 SGLang，且返回 public package 与首次结果一致。
- [x] 验证 400/429/500/503/connect timeout 不 retry、不 fallback、不提交 successful marker。
- [x] 验证 malformed/truncated/oversized audio 清理 partial，下一次新请求可恢复。
- [x] 验证一个 waiter disconnect 时另一个 waiter 仍能成功。
- [x] 验证最后一个 waiter disconnect 触发 cancel，single-flight entry 最终释放。
- [x] 验证 owner deadline 到期后 late response 不会被提交为成功 artifact。
- [x] 验证 cancellation grace 超时切 degraded，恢复前新请求快速失败。
- [x] 验证 private sidecar 不进入 PCM/Opus package，public exact-key parsers 全部通过。
- [x] 验证 Mac-side decode、drum mix、fallback 和 playback-facing contract 无需 SGLang-specific 改动。

**Regression suite**

- [x] 运行所有 `tests/unit/infrastructure/rap/` tests。
- [x] 运行所有 `tests/unit/application/rap/` tests。
- [x] 运行 `tests/unit/presentation/test_rap_render_server.py`。
- [x] 运行 remote client、chunk package、Opus transport 和 Web UI 相关 tests。
- [ ] 运行默认完整 pytest suite 并保存结果。
- [x] 项目未配置 formatter/linter/type checker；已运行 `compileall` 和 `git diff --check`。
- [x] 新增测试使用 `tmp_path`、MockTransport/TestClient，无真实 home path、固定监听端口或公网依赖。

**部署文档与 runbook**

- [x] 新增 `docs/developer-guide/sglang-omni-moss-serving.md`。
- [x] 文档记录 exact image digest/commit、package lock、MOSS snapshot 和 config hash 获取方式。
- [x] 文档列出 JIT/build preflight，包括 pinned runtime 实际需要的 `ninja`、compiler/CUDA 工具。
- [x] 文档给出 reference read-only mount、local-media allowlist 和 audio/text hash 核对步骤。
- [x] 文档给出 Qwen、SGLang、render server 三个独立 terminal 的 exact commands。
- [x] 所有 config/reference/model path 通过参数或环境配置，不硬编码个人 home 和 GPU id。
- [x] 更新 `docs/developer-guide/rap-demo-quickstart.md`，Mac 只 SSH-forward `:8020`。
- [x] 文档给出 health/preflight、正常 stop、日志定位和只终止本次进程的方法。
- [x] 文档给出 `degraded` 判定、cancellation grace 超时和 SGLang restart/re-warmup 流程。
- [x] 文档给出显式 `inprocess` rollback 命令、独立 artifact namespace 和禁止删库的边界。
- [x] launch/preflight/root-preparation script 已覆盖参数、missing binary、unsafe bind 和非空 root tests。
- [ ] 由未参与实现的开发者按文档从空 shell 完成一次 dry run/qualification，并记录修正。

**Phase 6 exit gate**

- [ ] hermetic integration、targeted regressions、default suite 和 configured static checks 全绿。
- [ ] 文档 dry run 成功，rollback/recovery 操作不依赖作者记忆或未记录 shell state。

### 15.8 Phase 7：执行真实 H200 qualification 和 final A/B

**实验启动前检查**

- [ ] checkout design-freeze 指定的 clean commit，确认没有未记录 patch。
- [ ] 校验 experiment manifest、analysis script 和 thresholds hash 未变化。
- [ ] 校验 candidate image/commit、config、MOSS snapshot、reference 和 producer fingerprints。
- [ ] 确认 baseline 与 candidate 都使用计划指定的同一 physical GPU，另一 backend 已停止。
- [ ] 清点 competing processes；发现未允许 GPU process 时暂停，不带污染继续跑。
- [ ] 固定 GPU clocks/power policy 或记录实际 policy，启动 telemetry sampler。
- [ ] 为每个 backend/block 创建全新的空 artifact root，并保存 root manifest。
- [ ] 启动 backend 后完成不计入分布的固定 warmup，确认 health identity 与 experiment manifest 一致。

**30-request qualification**

- [ ] in-process backend 跑固定 30 条 qualification subset，全部为 uncached。
- [ ] SGLang production payload 跑同一 30 条 subset，全部为 uncached。
- [ ] 核对两边 30/30 complete、无 crash/OOM/hang/schema error 和 wrong cache hit。
- [ ] 核对 candidate SGLang speech endpoint call count 恰好等于 30。
- [ ] 核对每条 final frame count、MMS/R3 hard validation 和 private sidecar。
- [ ] 运行 timeout/disconnect/cancellation recovery probes，确认服务回到 ready 或按 runbook 恢复。
- [ ] qualification 任一 hard failure 立即停止，不进入 100-request final run。

**Cohort A：serving-only parity**

- [ ] 仅当 capability matrix 证明 SGLang 支持 `ref_audio` without `ref_text` 时启用 Cohort A。
- [ ] baseline/candidate 使用同一 model/reference/text/flow/token count/settings/seed，不传 transcript。
- [ ] 四个 paired blocks 各覆盖 25 条 corpus；每个 block 内 backend 顺序按冻结 schedule counter-balance。
- [ ] 每个 backend 每条样本恰好一次 uncached execution，总数至少 100。
- [ ] 若 Cohort A 不支持，报告写 `not_supported` 和证据，不用 Cohort B 冒充 serving-only parity。

**Cohort B：production-candidate**

- [ ] baseline 使用当前 in-process reference-WAV-only 行为。
- [ ] candidate 使用 SGLang reference WAV + 已核对 transcript 的 production payload。
- [ ] 四个 paired blocks 各覆盖 25 条 corpus；每个 block 内 backend 顺序按冻结 schedule counter-balance。
- [ ] 每个 backend 每条样本恰好一次 uncached execution，总数至少 100。
- [ ] 每次 backend 切换后重新核对 process list、GPU、health、producer fingerprint 和空 root。
- [ ] block 中出现 competing process、thermal/power anomaly、server restart 或 cache hit 时整 block 作废重跑。

**数据完整性与统计**

- [ ] 校验 sample ids、request ids、seed、expected frames 在 paired sides 一一对应。
- [ ] 校验每个 sidecar 的 backend、producer/config/model/reference hashes 与 experiment manifest 一致。
- [ ] 校验 candidate service call count 等于 uncached candidate 样本数。
- [ ] 保存 raw latency rows，不只保存 percentile summary。
- [ ] 计算 startup/warmup、queue、MOSS、HTTP、MMS、R3、package 和 H200 total 指标。
- [ ] 只在 profiler correlation 可靠时报告 reference encode/prefill/AR/vocoder；否则输出 unavailable rate。
- [ ] 汇总 GPU peak memory、utilization、power、OOM、timeout、cancel 和 queue-recovery。
- [ ] 计算 source duration、MMS coverage/confidence、ASR WER/deletion/substitution 和 R3 stretch/fallback。
- [ ] 计算 100% final exact-frame/package/schema pass rate。
- [ ] 若 Phase 0 启用 speaker similarity hard gate，运行固定工具/model version 并保存 raw scores。
- [ ] 对固定 20 条 blind subset 隐藏 backend、随机顺序，由至少 2 位评审独立评分。
- [ ] 计算 p50/p95/max、paired deltas 和预先定义的 bootstrap 95% confidence intervals。
- [ ] 报告所有失败/离群样本；不得静默删除，只能按预先冻结 invalidation rule 标记。
- [ ] 分开报告 Cohort A 与 B，不跨 cohort 合并 quality 或 latency 分布。

**Acceptance decision**

- [ ] 用机器可读 gate evaluator 逐项检查第 9.4 节，不手工挑选通过项。
- [ ] reliability、artifact、cache、schema、provenance 任一 hard gate 失败，结论为 blocked/rollback。
- [ ] quality 通过但 primary latency CI 不通过，结论为 experimental，不推荐为优化 backend。
- [ ] 仅在全部 hard gates 和 primary latency gate 通过时，结论才可为 promotion candidate。
- [ ] 报告写明 candidate transcript conditioning 差异和 Cohort A 是否可运行。
- [ ] 保存 experiment manifest、launch manifests、raw rows、telemetry、artifacts、blind scores 和报告 hash。
- [ ] 由至少一位未参与实现者 review provenance、统计和 gate evaluator 结果。

**Phase 7 exit gate**

- [ ] 产出唯一 signed-off decision：`blocked`、`experimental` 或 `promotion-candidate`。
- [ ] 只有 `promotion-candidate` 可以进入 Phase 8 的推荐值切换步骤。

### 15.9 Phase 8：真实 Mac E2E、canary 和 rollback

**Mac E2E acceptance**

- [ ] 只建立 Mac -> H200 render server `:8020` tunnel，不 forward SGLang `:8030`。
- [ ] Mac 启动前验证 render health 为 ready，backend/producer fingerprint 与 H200 report 一致。
- [ ] 用 production Opus path 跑完整游戏，验证下载、decode、drums、mix 和 playback clock。
- [ ] 用 PCM debug path 做一次对照，确认问题定位不被 codec 混淆。
- [ ] 验证每个 chunk 的 exact frame count、chunk ordering 和 session/request identity。
- [ ] 记录 first usable finalized two-bar latency、deadline miss、fallback 和 playback underrun。
- [ ] 覆盖正常 session、连续快速 session、浏览器刷新和重复 same-request 场景。
- [ ] 主动断开 tunnel/client，确认 waiter/cancellation/degraded/recovery 语义符合 runbook。
- [ ] 模拟 SGLang 503/timeout，确认不隐式加载旧 MOSS，Mac 既有 fallback 正常工作。
- [ ] 验证 public package/trace 中没有 private sidecar、transcript、credential 或 H200 private path。
- [ ] 将 E2E 指标与 baseline 的相同固定场景比较，逐项检查第 9.4 节 realtime gate。

**Canary 与观察窗口**

- [ ] 在 quickstart 改默认前，先以显式 `--moss-serving-backend sglang-omni` 运行 canary。
- [ ] rollout 前冻结观察窗口和 rollback thresholds；默认至少 20 场完整游戏并跨 2 个工作日。
- [ ] 持续记录 request count、p50/p95、deadline miss、fallback、underrun、degraded 和 restart count。
- [ ] 每次启动核对 producer fingerprint，禁止 floating image/model/config 悄然变化。
- [ ] 任一 schema/cache/provenance 错误、OOM/hang、重复 degraded 或质量 hard gate 回归立即 rollback。
- [ ] rollback 只切显式 backend/launch config，保留 baseline/candidate 各自 artifact namespace。
- [ ] rollback 后重新验证 in-process health、warmup 和一场完整 Mac session。
- [ ] 保留触发 rollback 的 logs/artifacts，不以清 cache 代替 root-cause analysis。

**完成 rollout**

- [ ] canary 通过后更新 quickstart 推荐 backend 为 `sglang-omni`，继续文档化 `inprocess` rollback。
- [ ] 归档实际 production image digest、config/model/reference hashes 和 acceptance report link。
- [ ] 至少一个稳定观察周期后再单独评审是否删除 production 对 scripts backend 的依赖。
- [ ] 删除旧 backend 前另开任务，不能作为本迁移收尾时的顺手清理。

**Phase 8 exit gate**

- [ ] Mac E2E realtime gates 和 canary observation gates 全部通过，且 rollback 已演练。
- [ ] quickstart、runbook 和 production deployment manifest 与实际运行状态一致。

### 15.10 Phase 9：Streaming follow-up（独立实验，不阻塞首版）

- [ ] 创建新的 design/experiment id，不复用 non-streaming acceptance 数据集目录。
- [ ] 冻结 pinned build、stream payload、PCM format、chunk protocol 和 non-streaming control。
- [ ] 测量真实 first PCM byte TTFA、chunk cadence、complete generation 和 aggregate duration。
- [ ] 验证 delay-pattern 32-codebook 的初始化和首 chunk 完整性。
- [ ] 验证所有 chunk 顺序、sample format/rate/channel 和 stream termination marker。
- [ ] 将 stream 完整聚合为 WAV，与 non-streaming 做 ASR/MMS/speaker/盲听质量对比。
- [ ] 测试首 chunk 前、mid-stream 和末尾 cancellation 及 queue recovery。
- [ ] 测试 slow Mac/downstream backpressure，不允许无限 buffer。
- [ ] 测试 server-internal overlap 是否降低 complete source latency，即使 public response 仍 non-streaming。
- [ ] 原型 one-bar buffer 时继续在 H200 完成该 bar 的 alignment/R3/final validation 后才能发送。
- [ ] 定义 cross-bar context、边界连续性、第二 bar failure 和 partial-session fallback 语义。
- [ ] 对 chunk boundary 做点击、断裂、能量跳变和节奏偏移检查。
- [ ] 以 first usable **finalized** audio 为 primary metric，不用 raw source first byte 冒充用户收益。
- [ ] 若需要改变 Mac package，设计 public schema v2、版本协商、dual-reader 和 rollback fixtures。
- [ ] v2 reader 先上线并兼容 v1，再考虑发送 v2；不得原地扩展 strict v1 keys/members。
- [ ] 只有 E2E latency 显著改善、所有 quality/reliability gates 通过且 rollback 可用，才提出默认切换。

### 15.11 最终 Definition of Done

- [ ] exact SGLang runtime/config/model/reference 可由 launch manifest 完整复现。
- [ ] producer fingerprint 对所有输出相关差异敏感，baseline/candidate/rollback cache 永不串用。
- [ ] public v1 request、manifest、diagnostics、artifact ids 和 PCM/Opus ZIP members 保持兼容。
- [ ] private sidecar 完整、bounded、sanitized、可关联且不会发送到 Mac。
- [ ] deadline 从 API acceptance 贯穿所有阶段，超时/取消不会提交成功 artifact。
- [ ] waiter-aware single-flight 和 cancellation/degraded/recovery 语义有 unit + integration + H200 证据。
- [ ] SGLang failure 不会触发 hidden retry、hidden fallback 或第二份 in-process MOSS load。
- [ ] startup health 只有在真实 MOSS + MMS + R3 + artifact probe 通过后才 ready。
- [ ] targeted tests、default pytest 和项目既有 static checks 全绿或有书面、非本任务 waiver。
- [ ] H200 qualification 与至少 100-pair final report 通过所有 hard gates。
- [ ] Cohort A/B、transcript confound、不可用 profiler 指标和所有异常样本在报告中如实披露。
- [ ] Mac E2E 和 canary 通过，deadline miss/playback underrun 不劣于 baseline。
- [ ] install、launch、health、degraded recovery、stop 和 rollback 文档已由第二人验证。
- [ ] production 推荐值只在 `promotion-candidate` 决策后修改，并保留可演练的 in-process rollback。
- [ ] 若 primary latency gate 未通过，最终状态明确写为 `experimental`，不宣称 RAP 优化完成。

### 15.12 剩余执行 TODO（按阻塞顺序）

以下项目是代码与 hermetic 验证完成后的真实剩余工作。完成它们需要外部语料、H200 服务、Mac 客户端或人工评审，因此不能由当前本地实现结果代替。

**P0：恢复仓库验收基线并冻结实现**

- [ ] **Owner：仓库维护者。** 恢复 `output/rap_album_10x50_90bpm_20260816_v4/` 下完整 campaign corpus，尤其是当前缺失的 `chosen_lyrics.jsonl`；若该语料已正式退役，需提交书面 waiver 并同步修改对应测试。随后重跑 5 个失败节点和 `uv run pytest tests -q`，归档完整日志。
- [ ] **Owner：仓库维护者。** 决定仓库根目录 `pytest` 对 vendored `transformers/` 的收集策略：要么在 pytest 配置中明确排除，要么安装与当前 Keras 兼容的 `tf_keras`。把决定、命令和结果写入验收记录。
- [ ] **Owner：实现负责人。** 整理独立提交，记录 branch、HEAD、`git status`、patch digest、实现 ID 与实验 ID，并创建 design-freeze tag。证据中不得混入后续实验产生的 artifact。

**P1：冻结语料、模型与实验输入**

- [ ] **Owner：实验负责人。** 准备不少于 100 条的冻结语料，固定 corpus hash、代码 commit、依赖锁、阈值、随机种子、block schedule 和 cohort 定义；运行 freeze 工具并由第二人复核不可变 manifest。
- [ ] **Owner：语音质量负责人。** 人工核对 reference transcript 与 reference WAV 的配对关系，确认 host 与 service 端计算出的内容 hash 完全一致，并归档抽检记录。
- [ ] **Owner：实验负责人。** 固定 candidate image digest、SGLang Omni/MOSS-TTS package 版本、启动参数、模型 revision、reference bundle revision，以及每个实验 coordinate 的 GPU 映射。

**P2：H200 资格验证**

- [ ] **Owner：Serving 负责人。** 在独立 H200 runtime 中安装并确认 `ninja`、编译器和 CUDA toolchain；运行 preflight，归档 JSON 输出、驱动信息、显存信息和启动日志。
- [ ] **Owner：Serving 负责人。** 仅绑定 loopback 启动 SGLang Omni，逐项验证 health、models、全部采样参数、两种 reference 模式、错误响应，以及本地 reference allowlist。
- [ ] **Owner：RAP 负责人。** 用真实 WAV、MMS、R3 路径完成端到端测试，覆盖 sanitized fixture 和至少一个真实旧 artifact 的兼容性。
- [ ] **Owner：可靠性负责人。** 验证 disconnect、deadline 和 cancel 后 GPU 队列能够恢复；实测 upstream 是否确认 abort，据此确定 cancellation grace 和必要的 service restart 策略。
- [ ] **Owner：第二位开发者。** 完全按照 runbook 从空环境执行一次 dry run，记录所有偏差并在实验开始前修正文档。

**P3：资格赛与正式 A/B**

- [ ] **Owner：实验负责人。** 冻结 30 条 qualification subset；使用 root 准备脚本为每个 coordinate 创建独立空目录和 manifest。仅 cohort B 时共需 10 个 root；若同时启用 cohort A，再增加 8 个 final root。
- [ ] **Owner：实验负责人。** 完成 baseline/candidate 的 30 条 qualification，并执行所有 recovery probes；任何 hard gate 失败都不得进入正式实验。
- [ ] **Owner：实验负责人。** 按 counterbalanced schedule 运行 4 个正式 block，确保每个 backend/cohort 至少 100 条有效样本；发生 cache、root 或版本污染时整块作废并重跑。
- [ ] **Owner：观测负责人。** 收集 raw rows、private sidecars、service call counts、延迟与显存 telemetry、ASR、MMS、R3、speaker-similarity 和盲评结果，保证每条记录可追溯到 coordinate root manifest。
- [ ] **Owner：独立评审者。** 使用全部 artifact-root manifests 运行 evaluator，归档 acceptance report 及其 hash，并完成书面 sign-off。

**P4：Mac 链路验证**

- [ ] **Owner：客户端负责人。** Mac 侧只通过 `8020` 隧道访问 RAP，比较 PCM、Opus、完整游戏、fallback、deadline 和 underrun 行为。
- [ ] **Owner：可靠性负责人。** 注入断连、503 和 timeout，确认客户端不会偷偷加载 in-process 模型，也不会泄露 private metadata。
- [ ] **Owner：客户端负责人。** 产出 Mac 验收证据并重跑 evaluator；只有所有 hard gates 通过后才允许 canary。

**P5：灰度、回滚与默认值切换**

- [ ] **Owner：发布负责人。** 显式选择 `sglang-omni` 做 canary，至少运行 20 局并跨越 2 个工作日，持续监控成功率、超时率、降级率、队列恢复和质量阈值。
- [ ] **Owner：发布负责人。** 演练回滚到 `inprocess`，确认 producer namespaces 与失败证据均被保留，且旧 artifact 仍可读取。
- [ ] **Owner：仓库维护者。** 仅在 canary 与人工 sign-off 通过后更新默认 quickstart/deployment 配置，并归档实际部署 pins；在此之前默认 backend 保持 `inprocess`。
- [ ] **Owner：架构负责人。** Phase 9 streaming 继续作为独立项目，不纳入本次 backend 替换的验收范围。
