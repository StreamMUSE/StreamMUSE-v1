# SGLang-Omni MOSS Serving 实现报告（2026-09-04）

## 1. 报告结论

本轮已经完成 StreamMUSE RAP 将 MOSS-TTS 从 render server 进程内推理切换为可选
SGLang-Omni 独立服务所需的仓库内实现。交付内容覆盖执行 deadline/cancellation、backend
抽象、SGLang HTTP adapter、producer-aware artifact namespace、render server 集成、启动探针、
实验冻结与验收工具、操作文档和自动测试。

当前最准确的结论是：

- **代码和 hermetic 验证已完成。** 受影响的 targeted/cross-layer suite 为 `215 passed`。
- **仓库 `tests/` 全量测试的失败栈中未观察到本次实现相关失败。** 最终结果为
  `1934 passed, 8 skipped, 5 failed`；5 个失败均来自缺失的外部 campaign corpus。
- **真实优化效果尚未证明。** 当前没有真实 H200 qualification、正式 A/B、Mac E2E、人工盲听
  或 canary 数据，因此不能声称 SGLang-Omni 已经降低 RAP 延迟或达到 production 条件。
- **production 默认值没有改变。** `--moss-serving-backend` 仍默认为 `inprocess`，候选 backend
  必须显式选择 `sglang-omni`。
- **promotion 是 fail-closed 的。** 缺少 artifact-root、恢复探针、质量指标、Mac evidence 或其他
  hard-gate 证据时，evaluator 不会给出 `promotion-candidate`。

本报告是 implementation/hand-off report，不是性能实验结果报告。

## 2. 工作树信息与范围

| 项目 | 当前值 |
|---|---|
| 分支 | `feature/rap_optimization` |
| 当前 HEAD（变更未提交） | `b91c0a3dda03` |
| 状态 | 实现仍在未提交工作树中 |
| RAP 对外端口 | `8020`，默认仅 loopback |
| MOSS backend 默认值 | `inprocess` |
| 新候选 backend | `sglang-omni` |
| 本轮范围 | non-streaming WAV serving；保留 MMS + R3 + final artifact 流程 |
| 明确不在本轮范围 | SGLang streaming、public schema v2、真实部署 promotion |

工作树中预先存在的未跟踪目录 `streammuse.egg-info/` 未被本轮修改或清理。报告只描述本轮
SGLang-Omni/MOSS 相关实现，不把该目录算作交付内容。

## 3. 原有路径与目标路径

原有路径：

```text
RAP render server
    -> PersistentMossSynthesizer
    -> Transformers model.generate + processor.decode
    -> source WAV
    -> MMS forced alignment
    -> Rubber Band R3 warp
    -> final vocal artifact
```

新增候选路径：

```text
Mac client
    -> RAP render server :8020
    -> SglangMossSynthesizer
    -> loopback SGLang-Omni /v1/audio/speech
    -> validated mono WAV
    -> existing MMS forced alignment
    -> existing Rubber Band R3 warp
    -> producer-namespaced final artifact
```

这次替换的边界仅为 MOSS phrase synthesis。歌词候选、两 bar connected phrase、alignment、R3、
PCM/Opus package、Mac drums/mix/playback 和 public RAP request contract 均继续沿用原有语义。

## 4. 已完成实现

### 4.1 统一 deadline 与 cancellation

新增 `src/streammuse/application/rap/execution.py`：

- `SynthesisExecutionContext` 使用绝对 monotonic deadline，避免各阶段重新获得完整 timeout。
- 同一 deadline 从 HTTP request acceptance 贯穿 orchestration、MOSS、alignment、R3 和 artifact
  commit 前检查点。
- 支持线程安全 cancellation event、cancel callback、correlation ID 和剩余时间计算。
- 区分 `ExecutionDeadlineExceeded` 与 `ExecutionCancelled`。
- 记录 cancellation outcome、grace exceeded、upstream abort confirmation 和 recovery outcome。
- 为不可中断调用提供 `uninterruptible()` 标记，避免把“已收到取消”误报成“调用已终止”。
- single-flight owner 与 waiter 使用独立 cancellation signal，但共享固定 deadline；单个 waiter
  断开不会错误取消仍被其他 waiter 需要的 owner 工作。

RAP API 会从 `remaining_budget_ms` 计算一次固定 deadline。请求断开、budget 到期或服务 degraded
时返回明确错误，不提交伪成功 artifact。

### 4.2 Backend-neutral MOSS contract

扩展 `src/streammuse/infrastructure/rap/moss_tts.py`：

- 定义统一 `MossSynthesizer` protocol，使 renderer 不依赖具体推理实现。
- 增加稳定的错误类型：backend unavailable、request rejected、invalid output 和 synthesis failed。
- `MossPhraseResult` 同时携带音频结果与受限的 serving metadata。
- in-process backend 增加 deadline/cancellation checkpoints，并明确不可中断的 model call 边界。
- 提供共用的 mono WAV 文件/bytes 校验，供 in-process、SGLang adapter 和 preflight 使用。

新增 `src/streammuse/infrastructure/rap/moss_generation.py`，统一两种 backend 的 generation
settings、seed policy、language、instruction 和已知 caveat，减少 baseline/candidate 参数漂移。

### 4.3 SGLang-Omni HTTP adapter

新增 `src/streammuse/infrastructure/rap/sglang_moss_tts.py`：

- 复用持久化 `httpx.Client` connection pool。
- 按 MOSS-TTS cookbook 调用 `/v1/audio/speech`，发送固定模型、文本、reference audio URI、
  reference text 和 generation settings。
- 每个 synthesis attempt 只发送一次 HTTP 请求；adapter 内不做隐藏 retry 或 fallback。
- request timeout 取 adapter 上限与 execution 剩余 budget 的较小值。
- cancellation 时关闭 response/transport，并将结果记录为 abort confirmed 或 unconfirmed。
- 映射 transport error、HTTP status、媒体类型错误和 WAV 错误为 backend-neutral 异常。
- 对 `Content-Length` 与流式累计 bytes 同时设限，防止无界响应占用内存。
- 校验 media type、RIFF/WAV、mono、sample rate、有限样本、非空和最大音频时长。
- 提供 bounded `/health` 与 `/v1/models` probe，覆盖非 2xx、超时、畸形 JSON 和超大响应。

安全约束：

- SGLang base URL 只接受 `127.0.0.1`、`::1` 或 `localhost` origin。
- 拒绝 URL credentials、query、fragment 和额外 path。
- reference URI 只允许绝对本地 `file://` URI。
- reference WAV/text/config 必须有固定 hash 和有界大小。

### 4.4 Producer identity 与 artifact 隔离

新增 `src/streammuse/infrastructure/rap/producer_manifest.py`：

- `ProducerManifestV1` 覆盖 backend、实现 revision、StreamMUSE commit/patch、模型、runtime、
  generation、reference、alignment、warp policy 和输出格式等 output-affecting 输入。
- 对 manifest 做 canonical JSON + SHA-256，生成稳定 `producer_fingerprint`。
- artifact 根目录按 fingerprint 分 namespace，baseline/candidate/rollback 不再共享同一 cache key。
- namespace 初始化使用文件锁、原子发布、`fsync` 和 byte-exact verification。
- 已存在 artifact 但缺少可信 manifest，或 manifest/fingerprint 不匹配时 fail closed。

带来的行为变化是：同一个 RAP request 在不同 producer 下会产生不同 artifact namespace。旧缓存
不会被候选 backend 静默复用，这是预期的隔离行为，但也会增加磁盘占用。

### 4.5 Render server 集成与可靠性状态

扩展 `src/streammuse/presentation/rap_render_server.py`：

- 新增 `--moss-serving-backend inprocess|sglang-omni`，默认 `inprocess`。
- 两种 backend 的参数互斥且 fail-fast；SGLang 模式要求 URL、reference、runtime、revision 和
  hash pins 齐全。
- SGLang 模式采用 lazy import/composition，不加载进程内 MOSS weights，也不接受
  `--moss-device`。
- RAP server 和 SGLang upstream 均默认限制为 loopback。
- startup readiness 绑定 MOSS probe、MMS/R3 health 和 artifact write/read probe；未完成 warmup
  不对外宣称 ready。
- public `/health` 只返回 bounded/sanitized 状态，不泄露本地绝对路径或 private metadata。
- 增加 waiter-aware single-flight，重复 request 共享 owner 结果，但仍分别遵守各自 deadline 和
  disconnect。
- owner 成功后通过原子 workspace commit 发布 artifact；timeout/cancel/failure 不发布成功结果。
- private sidecar 保存 backend、producer、correlation、queue/generation timing、response bytes、
  cancellation/deadline/recovery 等证据，并做 exact schema、大小和内容校验。
- private sidecar 不进入 Mac public package。
- cancellation 后若在 grace 内无法确认 upstream release，runtime 进入 `degraded`，新请求返回
  503，要求明确恢复或重启，避免继续向可能堵塞的 GPU queue 加压。

### 4.6 Orchestration 与 renderer 传播

扩展 `chunk_orchestration.py`、`moss_aligned_phrase.py` 和 scripts backend adapter：

- execution context 贯穿歌词候选、phrase synthesis、WAV 读取、MMS alignment、R3 render 和
  final validation。
- 在昂贵阶段前后和 artifact publish 前设置 checkpoint。
- 保留现有 connected-phrase、token count、syllable onset 和精确目标时长语义。
- SGLang 返回的 source WAV 进入现有 MMS + R3 路径，不另建简化质量路径。
- serving metadata 通过内部结果和 private sidecar 传递，不扩展 strict public v1 schema。

### 4.7 Preflight 与可复现启动证据

新增 `scripts/preflight_sglang_omni_moss.py`：

- 检查 Python/runtime executable、`ninja`、C++ compiler、`nvcc` 和 GPU identity。
- 强制只选择一个 `CUDA_VISIBLE_DEVICES`。
- 验证 host reference WAV 与 service reference WAV byte-identical。
- 验证 reference WAV 合法性、时长、reference transcript UTF-8 和无外围空白。
- 校验 service reference 位于本地 allowlist，且 `file://` URI 精确指向该文件。
- 固定 runtime/SGLang/model revision、config/env/reference hash 和 launch argv。
- 输出 self-hashed、write-once launch manifest；不同内容不能覆盖已有 manifest。
- 支持 dry-run evidence，但 dry run 会明确标记工具未执行，不能充当 H200 qualification。

该工具会在启动前发现用户之前日志中的 `FileNotFoundError: ninja` 类问题，但本轮没有替用户在
真实 serving 环境安装 `ninja` 或 CUDA toolchain。

### 4.8 A/B 冻结、独立 roots 与 acceptance evaluator

新增 `src/streammuse/experiments/sglang_moss_acceptance.py` 以及四个 CLI：

- `freeze_sglang_moss_experiment.py`
- `prepare_sglang_moss_ab_root.py`
- `evaluate_sglang_moss_ab.py`
- `preflight_sglang_omni_moss.py`

实验工具实现：

- 冻结至少 100 条 corpus、pins、qualification/blind subset、counterbalanced blocks、随机种子和
  acceptance thresholds。
- 区分 cohort A `serving-only-parity` 与 cohort B `production-candidate`，显式披露 reference
  transcript conditioning confound。
- 每个 cohort/phase/block/backend coordinate 必须使用独立、空、非 symlink artifact root。
- root marker 绑定 experiment manifest 与 coordinate；被占用或错绑的 root 直接拒绝。
- evaluator 验证 qualification、完整 block、artifact、cache miss、producer provenance、service
  call count、恢复探针、ASR、MMS、R3、speaker similarity、盲听、latency 和 Mac evidence。
- primary latency 使用 paired median candidate-minus-baseline 和 paired bootstrap CI。
- 最终决策只有 `blocked`、`experimental` 或 `promotion-candidate`；hard gate 缺失或失败时
  必定 `blocked`。

## 5. 兼容性与刻意保持不变的内容

- public RAP request schema v1 未增加字段。
- public diagnostics、artifact ID 和 PCM/Opus package member contract 保持兼容。
- in-process backend 仍可使用，且仍为默认和 rollback 路径。
- MOSS source audio 之后继续使用同一 MMS + R3 质量链路。
- 不进行自动 backend fallback，避免一次请求产生两份 MOSS work 或污染 A/B。
- 不把 upstream 服务路径、reference text、绝对路径或 private timing 发送给 Mac。
- 本轮只接受完整 WAV response，没有把 raw first byte 当成最终用户可播放音频 latency。

## 6. 自动验证结果

### 6.1 Targeted 与跨层验证

结果：`215 passed`。

覆盖重点包括：

- monotonic deadline、cancel callback、outcome 和 owner/waiter context。
- producer manifest canonicalization、namespace 初始化、并发与 tamper detection。
- SGLang exact request payload、seed、timeout、HTTP status、invalid/oversized WAV 和 no retry。
- health/models probe 的 non-200、malformed JSON、timeout、Content-Length 和 streamed oversize。
- cancellation 会关闭正在读取的 response，并记录 abort-unconfirmed 语义。
- server CLI 参数矩阵、lazy backend composition、startup gate 和 private/public metadata 边界。
- idempotency、cache isolation、single-flight、waiter disconnect 和 owner cancellation。
- degraded/recovery state 与 cancellation grace。
- FastAPI ASGI round-trip。
- MockTransport SGLang -> source WAV -> 实际 renderer validation -> fake MMS -> R3 ->
  `vocal.wav` 的跨层路径，并确认只发出一次 speech request。
- 实验 freeze/root/evaluator 的正常和 fail-closed cases。

### 6.2 `tests/` 全量 suite

命令：

```bash
uv run pytest tests -q
```

结果：

```text
5 failed, 1934 passed, 8 skipped, 1 warning
```

5 个失败均读取不到：

```text
output/rap_album_10x50_90bpm_20260816_v4/
  01_space_exploration/chosen_lyrics.jsonl
```

其中 4 个节点位于 `tests/unit/experiments/rap_audio_protocols/test_timing.py`，1 个位于
`tests/unit/scripts/test_rap_audio_fastpitch_backend.py`。它们在进入本轮 SGLang 代码前即因
`FileNotFoundError` 退出。显式 deselect 这些外部数据节点时，其余 suite 全绿。

### 6.3 其他检查

- `tests/unit/infrastructure/rap/test_audio_output.py`：`28 passed`；fake callback 已显式注入，
  不再因测试机缺少 PortAudio 产生假失败。
- Python `compileall`：通过。
- `git diff --check`：通过。
- 仓库当前没有统一 formatter、linter 或 type-checker 配置，因此没有可报告的全仓 lint/type gate。

从仓库根直接执行不限定目录的 `uv run pytest -q` 会额外收集 vendored `transformers/`，当前
环境因 Keras 3 缺少兼容 `tf_keras` 在 collection 阶段失败。该问题与 `uv run pytest tests -q`
的 5 个 corpus failures 是两个独立问题。

## 7. 已知问题与残余风险

### 7.1 P0：缺失外部 corpus，默认 `tests/` gate 尚未全绿

**现象：** 5 个测试缺少 `chosen_lyrics.jsonl`。

**影响：** 无法宣称默认 `tests/` suite 完全通过；也无法重新验证这些 real-corpus FastPitch
timing cases。

**处理建议：** 恢复完整 campaign output，或由仓库维护者明确退役数据并提交书面 waiver/fixture
替代方案，之后重跑 5 个节点和全量 suite。

### 7.2 P0：根目录 pytest 收集边界不稳定

**现象：** 根目录 pytest 收集 vendored `transformers/`，被 Keras 3/`tf_keras` 阻断。

**影响：** 开发者运行相似命令可能得到不同结论，CI 命令边界不清楚。

**处理建议：** 明确 pytest `testpaths`/ignore 策略，或固定兼容 TensorFlow/Keras 依赖；不要把
collection error 当成本次 RAP 实现失败，也不要长期依赖口头约定。

### 7.3 P1：尚未在真实 H200 + SGLang-Omni 上验证

当前 HTTP adapter 使用 MockTransport/hermetic ASGI service 验证，尚未证明指定 SGLang-Omni
commit、MOSS model revision、CUDA driver 和 H200 组合可以成功启动并返回相同格式。

实际环境仍可能遇到：

- 缺少 `ninja`、compiler、`nvcc` 或不兼容 CUDA architecture。
- 首次 JIT/warmup 时间过长或 autotuner 峰值显存 OOM。
- cookbook 与具体 pinned revision 的字段、endpoint 或 health response 有差异。
- 服务端 `file://` reference 路径不可见、allowlist 不一致或 hash 不一致。
- 单 GPU 同时残留 in-process MOSS 与 SGLang weights 导致显存不足。

必须以 preflight manifest 和真实 probe 关闭这些风险。

### 7.4 P1：HTTP transport close 不等于服务端 generation 已停止

客户端断开时 adapter 可以关闭 response/transport，但除非 upstream 提供可验证的 abort
acknowledgement，否则无法证明 GPU kernel 或队列任务已释放。本实现刻意记录
`transport_closed_abort_unconfirmed`，等待 cancellation grace 后将服务置为 degraded，而不是
假装取消成功。

真实 qualification 需要测量：

- disconnect 后 GPU memory/queue 是否恢复；
- SGLang 是否有可用的 request ID/cancel/abort confirmation；
- 合理的 grace 时间；
- 何时必须重启独立服务。

### 7.5 P1：deadline 是协作式检查，不是任意 GPU 调用的硬抢占

MMS、R3 或 in-process model call 如果进入不可中断 native/GPU 调用，只能在调用返回后观察
deadline/cancellation。因此 API 可以停止等待和拒绝发布结果，但不能保证所有底层计算在 deadline
瞬间停止。真实长尾与 queue recovery 必须在 H200 压测中验证。

### 7.6 P1：性能与质量结论仍为空

当前没有真实数据证明 candidate：

- MOSS p50/p95 或完整 server latency 更低；
- ASR WER/deletion 不退化；
- MMS coverage/confidence 不退化；
- R3 stretch/fallback 不恶化；
- speaker identity 和人工 intelligibility/rhythm 不退化；
- Mac playback deadline miss/underrun 不增加。

因此当前状态最多是“可实验候选”，不是“优化完成”。即使服务端 MOSS 变快，如果 final playable
audio 没有变快，也不能通过 primary optimization gate。

### 7.7 P1：真实历史 artifact 兼容性尚未抽检

自动测试覆盖了 synthetic/fixture artifact 和 strict schema，但还没有从真实历史 RAP session 中
选择 artifact 完成 byte/member/reader compatibility 验证。producer namespace 也意味着新实现不会
命中旧 namespace；需验证旧 reader 仍可读旧 artifact，同时新旧 cache 不串用。

### 7.8 P2：更严格的启动配置会增加部署操作成本

SGLang backend 要求绝对路径、revision 和 SHA-256 pins 全部存在，配置不完整会直接拒绝启动。
这是为了实验可信度和安全性，但也意味着临时手工启动命令更长。应使用 runbook、环境文件或受控
service manifest，不建议逐次手敲并跳过 pin。

### 7.9 P2：artifact namespace 与 sidecar 会增加存储

每个 producer fingerprint 使用独立 cache，正式 A/B 又要求每个 coordinate 使用独立 root。
好处是不会串 cache；代价是重复 WAV/ZIP/private sidecar 和更高 inode/容量需求。正式实验前应做
容量估算和只针对已归档 experiment ID 的清理策略，不能用清 cache 掩盖失败证据。

### 7.10 P2：文件锁实现依赖 POSIX 语义

producer namespace 使用 `fcntl`、hard link 和 directory `fsync`，适合当前 Linux/H200 目标环境，
但未声明支持 Windows，也未在所有 NFS/Lustre 配置上验证 locking/durability。artifact root 若迁移到
网络文件系统，应单独做并发与 crash consistency 测试。

### 7.11 P2：private sidecar 是新的敏感运行证据面

sidecar 已做大小、schema、路径字符串和 public package 隔离，但它仍包含 backend/runtime timing 与
恢复信息。部署时应保持 artifact root 权限、日志保留周期和访问边界，不应直接通过 Web/Mac API
暴露整个 namespace。

### 7.12 P2：正式实验流程严格，错误操作会被整块拒绝

evaluator 要求独立空 root、完整 provenance、cache miss、service call count 和 counterbalanced block。
若某 block 混入竞争 GPU process、复用 root、缺行或版本漂移，整个 block 应作废重跑。这会增加
实验时间，但不能为了便利放宽，否则 A/B 延迟结论不可解释。

## 8. 尚未完成且不能在本地冒充完成的 gate

- 恢复/豁免缺失 campaign corpus 并取得默认全量测试绿灯。
- 固定真实 SGLang-Omni image/commit/package/model/config/reference/GPU pins。
- 在独立 H200 runtime 运行 preflight、启动、health/models/speech 和错误注入。
- 用真实 MOSS WAV 跑 MMS + R3 + final artifact，包括至少一个历史 artifact compatibility case。
- 完成 timeout/disconnect/cancellation 的 GPU queue recovery 验证。
- 冻结不少于 100 条 corpus，并完成 30 条 qualification。
- 完成每 backend/cohort 不少于 100 条的正式 counterbalanced A/B。
- 收集 ASR、MMS、R3、speaker similarity、盲听和 profiler/latency evidence。
- 完成 Mac PCM/Opus/full-game/fallback/deadline/underrun 验证。
- 显式 `sglang-omni` canary 至少 20 局、跨 2 个工作日。
- 演练 rollback，并在通过全部 hard gates 后才讨论更改默认 backend。

完整 owner、证据要求和阻塞顺序见 serving plan 的 `15.12` 节。

## 9. 推荐下一步执行顺序

1. 恢复缺失 corpus，固定全量测试命令并让仓库基线可复现。
2. 整理当前未提交工作树，记录 patch digest，创建 implementation/design-freeze ID。
3. 在 H200 独立环境安装 `ninja`/compiler/CUDA，执行 preflight 并归档 launch manifest。
4. 以 loopback 启动真实 SGLang-Omni，先完成单条 speech、WAV、MMS、R3 和 artifact smoke。
5. 做 timeout/disconnect/cancel/restart qualification，确定真实恢复策略。
6. 冻结 corpus、pins、schedule 和独立 artifact roots，运行 30 条 qualification。
7. qualification 全绿后运行正式 A/B 和盲听，不在同一 GPU 上混入其他任务。
8. evaluator 得到 `promotion-candidate` 后完成 Mac E2E 和显式 canary。
9. canary 与 rollback 均通过后，才更新 quickstart/deployment 默认值。

## 10. 主要交付文件

| 类型 | 文件 |
|---|---|
| 执行控制 | `src/streammuse/application/rap/execution.py` |
| orchestration | `src/streammuse/application/rap/chunk_orchestration.py` |
| backend contract/in-process | `src/streammuse/infrastructure/rap/moss_tts.py` |
| shared generation settings | `src/streammuse/infrastructure/rap/moss_generation.py` |
| SGLang adapter | `src/streammuse/infrastructure/rap/sglang_moss_tts.py` |
| producer identity | `src/streammuse/infrastructure/rap/producer_manifest.py` |
| aligned renderer | `src/streammuse/infrastructure/rap/moss_aligned_phrase.py` |
| server composition/API | `src/streammuse/presentation/rap_render_server.py` |
| experiment contract | `src/streammuse/experiments/sglang_moss_acceptance.py` |
| preflight | `scripts/preflight_sglang_omni_moss.py` |
| freeze | `scripts/freeze_sglang_moss_experiment.py` |
| A/B root preparation | `scripts/prepare_sglang_moss_ab_root.py` |
| evaluator CLI | `scripts/evaluate_sglang_moss_ab.py` |
| runbook | `docs/developer-guide/sglang-omni-moss-serving.md` |
| quickstart | `docs/developer-guide/rap-demo-quickstart.md` |
| implementation plan/TODO | `developing-logs/plans/2026-09-03-sglang-omni-moss-serving-plan.md` |

主要新增/扩展测试位于：

- `tests/unit/application/rap/test_execution.py`
- `tests/unit/infrastructure/rap/test_sglang_moss_tts.py`
- `tests/unit/infrastructure/rap/test_producer_manifest.py`
- `tests/unit/infrastructure/rap/test_moss_tts.py`
- `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py`
- `tests/unit/presentation/test_rap_render_server.py`
- `tests/unit/experiments/test_sglang_moss_acceptance.py`
- `tests/unit/scripts/test_preflight_sglang_omni_moss.py`

## 11. 参考文档

- [SGLang-Omni MOSS-TTS cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts.md)
- [SGLang-Omni local MOSS-TTS cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_tts_local.md)
- `docs/developer-guide/sglang-omni-moss-serving.md`
- `developing-logs/plans/2026-09-03-sglang-omni-moss-serving-plan.md`

## 12. 最终判断

本轮已经把 SGLang-Omni MOSS serving 从一个迁移想法推进到可配置、可回滚、可观测、可验证且
不会污染现有 RAP public contract 的实验候选实现。代码层的主要边界已有自动测试，已知失败也已
定位并与本次实现区分。

剩余工作的核心不再是继续堆 adapter 代码，而是恢复仓库数据基线，并用固定版本在真实 H200 与
Mac 链路上产生可信证据。在这些证据通过 evaluator、人工评审和 canary 之前，保持
`inprocess` 默认值是正确状态。
