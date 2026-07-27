# StreamMUSE 交互式语音输入实现报告

日期：2026-07-16  
分支：`feature/voice`  
首个支持任务：`zip_zap_zop`  
语音模型：`faster-whisper tiny.en`，CPU，int8

## 1. 结论

本次已经完成交互式游戏语音输入 MVP 的生产实现。现在 `streammuse-task play` 可以在轮到
人类玩家时打开麦克风，采集一段有界发言，执行实时端点检测，使用常驻的
`faster-whisper tiny.en` 转成文本，再由 `ZipZapZopTask` 做确定性规范化，最后继续使用原有
裁判逻辑判断答案是否正确。

完整数据流如下：

```text
人类回合开始
  -> 显示游戏提示
  -> 打开所选麦克风
  -> PortAudio 回调复制 PCM 到有界队列
  -> WebRTC VAD 检测语音起始和尾部静音
  -> 必要时重采样到 16 kHz float32
  -> 常驻 faster-whisper 转录
  -> ZipZapZopTask 解析口语答案
  -> 原有 validate_response() 裁判
  -> response_trace.jsonl / turn artifact / summary
  -> 关闭本回合麦克风流
```

这里的语音层只负责“音频到原始文本”。它不会判断答案，也不会读取当前正确答案去修复 ASR
结果。游戏语义仍然由任务层拥有。

代码级功能已经完成并通过聚焦测试。尚未完成的是依赖真实用户、真实麦克风和正式语料库的
产品验收，因此目前应称为“可测试的语音输入 MVP”，不能称为“所有准确率和延迟门槛已经通过”。

## 2. 完成的功能

### 2.1 独立的人类输入层

新增了独立于音乐输入栈的 `HumanResponseSource`：

- `TerminalHumanResponseSource`：封装原来的终端输入行为；
- `VoiceHumanResponseSource`：编排麦克风采集、端点检测、ASR 和可选音频保存；
- `HumanResponseRequest`：传递 turn id、提示文本、剩余截止时间和静态语音上下文；
- `HumanResponse`：返回原始文本、无语音/空转录状态、是否耗尽游戏截止时间、权威延迟和诊断元数据；
- `HumanInputConfig` / `VoiceInputConfig`：冻结、强类型、构造时验证的应用层配置。

这套接口没有复用音乐侧的 `InputSource`。音乐输入输出、MIDI 实时循环和
`RealTimeMusicService` 没有被语音功能修改。

终端仍然是默认模式：

```text
--human-input terminal
```

只有显式选择 `--human-input voice` 时，工厂才会导入语音基础设施。默认终端导入检查确认不会
加载 `faster_whisper`、`sounddevice` 或 `webrtcvad`。

### 2.2 Zip-Zap-Zop 语音语义

`ZipZapZopTask` 新增了两个语音能力：

- `build_speech_context()`：提供静态的 Zip/Zap/Zop 初始提示和热词；
- `parse_spoken_response()`：将 ASR 文本转换为裁判可接受的规范文本。

当前解析器支持：

- 大小写、末尾标点和重复空白规范化；
- `zip`、`zap`、`zop`；
- `zip zap` -> `ZipZap`；
- `zip zop` -> `ZipZop`；
- `zap zop` -> `ZapZop`；
- `zip zap zop` -> `ZipZapZop`；
- 数字字符串，例如 `21`、`-21`；
- 有界英语数字，例如 `twenty one`、`negative twenty-one`、
  `nine hundred ninety nine million ...`。

支持的口语整数范围为：

```text
-999999999 <= value <= 999999999
```

语音模式会在游戏启动前检查 `start_number` 到本局最大 turn 覆盖的范围，超出范围直接拒绝。

解析器刻意不支持模糊猜测。例如不会把 `sap` 自动改成 `zap`，也不会利用当前期望答案修复
错误转录。无法确定的文本会变成结构化的 `unrecognized`，并被原裁判判为错误答案。

### 2.3 常驻 faster-whisper 识别器

新增 `FasterWhisperRecognizer`，其行为包括：

- 每局只构造一次模型；
- 显式默认 `model=tiny.en`、`device=cpu`、`compute_type=int8`；
- 游戏计时开始前执行一次真实的 1 秒静音预热转录；
- 直接接收 16 kHz 单声道 `numpy.float32`，不创建临时 WAV；
- 完整消费 faster-whisper 返回的 segment generator；
- 每一轮独立解码，不继承上一轮文本条件；
- 记录实际初始提示、热词和模型诊断；
- 支持缓存根目录、固定 revision 和 `local_files_only`；
- 对远程模型标识符先解析到本地 snapshot，再把本地目录交给 `WhisperModel`；
- 记录请求 revision、解析 revision、snapshot 路径、模型解析/加载/预热耗时。

固定解码参数为：

```text
language="en"
beam_size=1
condition_on_previous_text=False
task="transcribe"
temperature=0
vad_filter=False
word_timestamps=False
without_timestamps=True
```

`vad_filter=False` 是有意的：实时停止录音由 WebRTC VAD 状态机负责，不再让 Whisper 对完整
缓冲区执行第二次隐藏 VAD。

缺包、缓存未命中、模型加载失败、CTranslate2 不兼容和转录失败都会转换成带上下文的语音基础设施
异常，不会被记成玩家答错。

### 2.4 麦克风采集与端点检测

新增的 `MicrophoneCapture` 使用 `sounddevice.RawInputStream`：

- 单声道、有符号 16-bit PCM；
- 优先尝试 16 kHz，然后依次尝试 48/32/8 kHz；
- 明确拒绝只能以 44.1 kHz 打开的设备；
- PortAudio 回调只复制输入缓冲区和状态，不执行 VAD、重采样、文件 I/O 或 ASR；
- 回调与消费线程之间使用有界队列；
- 队列溢出和 PortAudio callback status 都上报主线程；
- 任意 callback chunk 会累积成精确 20 ms WebRTC VAD frame；
- 非 16 kHz 的完整发言使用 SciPy `resample_poly` 转到 16 kHz；
- `close()` 幂等，并能解除正在等待音频的采集调用。

端点状态机覆盖：

- 发言前预卷，默认 100 ms；
- 等待开口安全上限，默认 5000 ms；
- 尾部静音，默认 300 ms；
- 单次最大发言，默认 5000 ms；
- 游戏 hard/challenge 绝对截止时间；
- 无语音、尾部静音、最大发言和截止时间等明确 endpoint reason。

采集时间不再使用 callback 被 Python 执行的时间来猜测。实现会读取 PortAudio 的
`inputBufferAdcTime` / `currentTime`，并使用 `stream.time` 校准到本地 `perf_counter`。
即使 callback 因调度延迟较晚到达，也能区分音频究竟是在截止时间之前还是之后被 ADC 采集。

### 2.5 截止时间语义

人类语音回合的计分延迟保持为：

```text
提示显示 + 等待开口 + 发言 + 端点静音 + ASR
```

关键规则：

- `soft`：麦克风仍有等待开口和最大发言安全上限，但安全上限本身不会伪装成游戏强制超时；
- `hard` / `challenge`：把当前 turn 的剩余绝对预算交给麦克风；
- 到达采集截止时间后停止接受截止时间之后采集的音频；
- 已经开始的同步 CTranslate2 ASR 会完成，然后按完整耗时判定是否超时；
- 不从玩家延迟中减去 ASR 时间，也不自动延长 deadline；
- 只有显式保存 WAV 时，经过验证的音频文件写入耗时才从计分管线中排除；
- 同一答案同时超时和无效时，deadline loss 优先，但两个事实都会保留在 trace。

由于 CTranslate2 在当前进程内无法安全强杀，hard/challenge 是“计分截止时间”，不是保证函数在
同一毫秒返回的执行截止时间。

### 2.6 运行时与资源生命周期

交互运行时现在接收 `HumanResponseSource`，但保留同一个 `TerminalIO` 做菜单、输出和命令显示。

资源规则包括：

- 构造函数无麦克风、模型下载或模型加载副作用；
- `start()` 和 `close()` 幂等；
- 模型加载、下载和预热都发生在第一个计时 turn 之前；
- unsupported speech task 在麦克风/模型启动之前失败；
- source 构造失败、client 构造失败、source 启动失败、运行时失败、正常退出和 Ctrl-C 都清理资源；
- 清理异常不会替换更早发生的主异常；
- `KeyboardInterrupt` 写入 `stop_reason=user_interrupt`，CLI 返回 130；
- 语音采集期间不并发读取终端冒号命令，紧急退出使用 Ctrl-C。

终端模式的 `:help`、`:hint`、`:expected`、`:summary`、`:quit` 仍在任务语音解析之前分派，且不
消耗 turn。hard 模式下，已经成功读到的终端命令也不会因为显示帮助花费了时间而被误判为玩家
deadline loss；重新提示会启动新的输入计时器。

### 2.7 运行产物和隐私

交互式 manifest 升级到 schema v2，并为终端和语音两种模式记录 `human_input`。

典型目录：

```text
task_runs/zip_zap_zop_interactive_<timestamp>_<id>/
  manifest.json
  response_trace.jsonl
  run_summary.json
  artifacts/turn/
    0001_turn_0000.json
    ...
```

语音 turn 的 `metadata.human_input` 包含：

- `mode` / `status`；
- `raw_transcript`；
- `canonical_response`；
- `parse_status` / `parse_reason`；
- `wait_for_speech_ms`；
- `utterance_ms`；
- `endpoint_silence_ms`；
- `last_voiced_offset_ms`；
- `endpoint_detected_offset_ms`；
- `asr_start_offset_ms` / `asr_end_offset_ms`；
- `asr_latency_ms` / `total_latency_ms`；
- `endpoint_reason`；
- `capture_sample_rate_hz`；
- `audio_overflow`；
- `audio_artifact` / `artifact_persistence_ms`；
- ASR segment 诊断。

默认不保存原始麦克风音频。只有显式传入：

```text
--voice-save-audio
```

才会在 `artifacts/turn/` 下生成 `*_human.wav`。

回合持久化也进行了事务性加固：只有 turn artifact 和 JSONL trace 都成功后，才提交 transcript、
统计和游戏状态。JSONL 即使在写入后才发生 flush/close 异常，也会 truncate 回追加前长度；失败的
新 turn artifact 会删除，避免 manifest 显示 0 turn 但 trace 已经多出一行。

### 2.8 CLI 和依赖

`play` 新增：

```text
--human-input {terminal,voice}
--voice-model
--voice-device
--voice-compute-type
--microphone-device
--voice-model-cache
--voice-model-revision
--voice-local-files-only
--voice-save-audio
```

新增轻量子命令：

```bash
streammuse-task voice-devices
```

它不要求 `--task`，不构造 LLM client，也不加载 Whisper 模型。

`pyproject.toml` 新增 `voice` 可选依赖组：

- `faster-whisper>=1.2,<1.3`；
- Python 3.10 使用 `onnxruntime<1.24`；
- `sounddevice>=0.5,<0.6`；
- `webrtcvad-wheels>=2.0,<2.1`。

`uv.lock` 已重新生成并通过 `uv lock --check`。

### 2.9 离线资格验证工具

新增：

```text
scripts/qualify_voice_input.py
```

该工具重放经过同意采集的本地语料，不打开麦克风。正式 profile 会验证：

- acceptance 与 dev split 都存在；
- acceptance/dev 说话人和 session 不重叠；
- 每条样本都有 speaker、session、distance、environment；
- distance/environment 只能使用安全类别 token；
- 音频路径和 SHA-256 不重复；
- Zip/Zap/Zop 各至少 100 条；
- 组合词至少 100 条；
- 数字至少 150 条；
- 四类负样本各至少 50 条，总计至少 200 条；
- 正确率、错误命令率、空转录率和 ASR p95 达到计划门槛；
- `tiny.en`、CPU、int8、task context、`local_files_only=true`；
- 官方 `Systran/faster-whisper-tiny.en` snapshot 和固定 40 位 commit；
- 当前安装的 10 个语音相关包逐项匹配 `uv.lock`；
- qualification 脚本、Zip-Zap-Zop parser、recognizer 源码在运行中没有变化；
- HF/local 模型关键文件树在运行中没有变化；
- manifest、音频和 lock 在运行中没有变化。

任何自定义门槛都会把运行标为 exploratory，不能通过正式 `passed` gate。这样不能通过把样本量
降到 0 或把允许错误率提高到 100% 来伪造产品资格。

输出采用同父目录 staging，再原子重命名整个 artifact set。`summary.json` 保存
`samples.jsonl` SHA-256；已有 output directory 不会被覆盖，输出也不能包含或覆盖 manifest、
lock、音频、源码或模型目录。

## 3. 主要代码位置

### 领域层

- `src/streammuse/domain/tasks/models.py`
  - 人类响应、语音上下文和 speech-aware task 协议。
- `src/streammuse/domain/tasks/zip_zap_zop.py`
  - 静态语音上下文和有界口语解析器。

### 应用层

- `src/streammuse/application/tasks/human_input.py`
  - 配置、终端协议、终端响应源。
- `src/streammuse/application/factories/human_input_factory.py`
  - 延迟选择 terminal/voice source。
- `src/streammuse/application/tasks/interactive_runtime.py`
  - 人类 source 接线、命令、deadline、元数据、生命周期和持久化事务。

### 基础设施层

- `src/streammuse/infrastructure/voice/__init__.py`
  - typed errors、JSON-safe 工具和公开导出。
- `src/streammuse/infrastructure/voice/microphone.py`
  - 设备、PortAudio、ADC 时间映射、VAD、端点和重采样。
- `src/streammuse/infrastructure/voice/faster_whisper.py`
  - snapshot 解析、模型加载/预热和转录。
- `src/streammuse/infrastructure/voice/response_source.py`
  - 每回合采集/ASR 编排和可选 WAV。

### 表现层和工具

- `src/streammuse/presentation/task/cli.py`
  - CLI 参数、设备枚举、工厂和资源清理。
- `scripts/qualify_voice_input.py`
  - 冻结语料库重放和正式资格 gate。

### 文档

- `docs/user-guide/voice-input.md`
- `docs/developer-guide/voice-input-qualification.md`
- `developing-logs/plans/2026-07-16-interactive-voice-plan.md`

## 4. 当前验证结果

### 4.1 固定环境

本次实际验证环境：

| 项目 | 值 |
|---|---|
| OS | macOS 15.1（24B83），arm64 |
| CPU / 内存 | Apple M1 Pro 10 核 / 16 GB |
| Python | 3.10.18 |
| 模型 | `tiny.en` / CPU / int8 |
| HF repo | `Systran/faster-whisper-tiny.en` |
| snapshot commit | `0d3d19a32d3338f10357c0889762bd8d64bbdeba` |
| `uv.lock` SHA-256 | `08155ed0a91102efd9506fb6031c42f2772c417f53581bd9f333a490f326c2e4` |

实际语音包版本：

| 包 | 版本 |
|---|---:|
| faster-whisper | 1.2.1 |
| ctranslate2 | 4.8.1 |
| av | 17.1.0 |
| onnxruntime | 1.23.2 |
| tokenizers | 0.21.1 |
| huggingface-hub | 0.33.0 |
| sounddevice | 0.5.5 |
| webrtcvad-wheels | 2.0.14 |
| numpy | 2.1.3 |
| scipy | 1.15.3 |

一次已有缓存的启动烟雾样本：

| 阶段 | 耗时 |
|---|---:|
| snapshot 解析 | 1309.0 ms |
| 模型加载 | 180.4 ms |
| 静音预热 | 180.7 ms |

这是单次启动烟雾数据，不是正式 p50/p95。

### 4.2 自动化测试

语音聚焦套件：

```text
214 passed, 2 skipped
```

固定本地模型和 commit 后的真实模型集成测试：

```text
1 passed, 1 skipped
```

跳过的是需要经过同意采集的真实预录语音文件的用例。

静态检查：

```text
uv lock --check     passed
compileall          passed
git diff --check    passed
```

全仓单元测试：

```text
582 passed, 14 failed
```

14 个失败都在此次功能没有修改的 melody robustness / perturbation campaign 代码，主要是旧测试仍
按旧 candidate/freeze 契约调用。

全仓集成测试：

```text
4 passed, 1 failed, 2 skipped
```

唯一失败是未修改的 Lekai session contract。它不经过交互式任务或语音代码。

### 4.3 用户主机真实麦克风烟雾测试

2026-07-17 在用户 macOS 主机上使用设备 `ZBW-iPhone Microphone` 完成了一次 human-first、
单回合、3000 ms soft deadline 烟雾测试。结果如下：

| 指标 | 结果 |
|---|---:|
| 原始 ASR | `"1."` |
| 规范响应 | `"1"` |
| 裁判结果 | `OK` |
| 等待检测到语音 | 2202.7 ms |
| 发言时长 | 1300.0 ms |
| 配置的尾部静音 | 300.0 ms |
| ASR 推理 | 160.0 ms |
| 权威总延迟 | 4227.0 ms |
| 3000 ms deadline | missed |

本次运行 `stop_reason=completed`、`valid_count=1`、`deadline_miss_count=1`。这证明主机上的
麦克风采集、VAD、端点检测、固定模型离线加载、ASR、任务解析、裁判和 artifact 发布链路已经
端到端工作；它不证明 3000 ms 延迟门槛通过。最大单项是提示出现后等待开口的 2202.7 ms，
而不是 ASR 推理。模型 snapshot 解析、加载和预热发生在人类回合开始前，不计入 4227.0 ms。

### 4.4 独立最终审查

最终只读审查复验了：

- 输入/输出路径碰撞不会覆盖冻结音频；
- 转录期间修改模型文件不会发布资格结果；
- 修改 parser/recognizer/qualification 源码会失败；
- 非默认 threshold 不能伪造 formal pass；
- summary 写入失败不会留下新 samples 配旧 summary；
- JSONL 第一行和后续行写后失败都能回滚；
- terminal 默认导入路径不加载语音可选依赖。

审查结论是当前未发现仍会阻止合并的代码问题。

## 5. 当前欠缺和限制

### 5.1 尚无正式同意语料库

当前仓库没有满足协议的真实语音 acceptance/dev corpus。因此尚未完成：

- 5 名以上验收说话人；
- 每个基础词 100 条；
- 100 条组合词；
- 150 条数字；
- 200 条四分类负样本；
- 原始 ASR 与规范化准确率；
- 混淆矩阵和 Wilson 95% 置信区间；
- 静态热词/初始提示/基线的正式留出集比较。

资格工具已经实现，但没有数据就不能产生可信产品结论。

### 5.2 自动化执行沙箱没有麦克风设备

本次执行：

```text
streammuse-task voice-devices
No microphone input devices found.
```

这说明自动化执行沙箱没有暴露主机设备。用户 macOS 主机已经使用 iPhone 麦克风完成上述单回合
端到端烟雾测试，但仍未完成：

- 目标 Mac 真实麦克风 20 turn soft 游戏；
- hard/challenge 麦克风烟雾测试；
- 正样本 false-no-speech rate；
- 起始语音截断率；
- 真实采集队列溢出率；
- 最后语音帧到文本 p50/p95；
- 3000 ms 脚本化成功率；
- 1000 ms challenge 资格。

### 5.3 1 秒 challenge 尚未证明可用

当前计分包含等待开口、发言、300 ms 尾部静音和 ASR。历史微基准只有平均 ASR 数据，不能证明
一秒端到端 p95。当前建议先用 3000 ms soft；在正式麦克风重复测试通过前，不应宣称 1000 ms
challenge 可用。

### 5.4 44.1 kHz-only 设备不支持

WebRTC VAD 只接受 8/16/32/48 kHz。MVP 会尝试以这些速率打开设备。如果设备只能提供
44.1 kHz，启动会明确失败。支持这种设备需要在 VAD 之前增加有状态流式重采样器。

### 5.5 语音模式不并发接收终端命令

麦克风采集期间不会同时监听 `:quit`、`:help` 等终端命令。语音模式紧急退出使用 Ctrl-C。
未来可以增加 multiplexed source，但不需要修改任务或裁判协议。

### 5.6 ASR 不可强制取消

一旦进入进程内 CTranslate2 转录，当前实现会等待它同步结束。hard/challenge 可以准确记为超时，
但不能保证函数恰好在 deadline 时返回。严格执行上限需要可终止的 worker process。

### 5.7 仅支持英语 tiny.en 和 Zip-Zap-Zop

当前 production speech context 和数字语法都是英语。其他任务必须显式实现
`SpeechAwareInteractiveTask`，否则语音模式会在启动阶段拒绝。

### 5.8 不做模糊纠错

历史基准里出现过 `zip zap` -> `Zip sap.`。当前不会自动把 `sap` 改成 `zap`，因为这种规则可能
把真实错误答案改成正确答案。只有正式 dev/acceptance 数据证明某个静态混淆规则不会增加错误命令
后，才应考虑加入。

### 5.9 CPU 竞争尚未测试

本功能与音乐实时栈没有接线，也没有在同一主机同时运行 ASR、LLM server 和音乐生成。若将来需要
共机运行，应重新测量 CPU contention 和调度延迟。

### 5.10 没有 TTS 和回声消除

本次只实现语音输入。LLM 回答仍显示为文本，没有：

- TTS backend；
- 扬声器播放；
- acoustic echo cancellation；
- 播放完成到麦克风重开的 guard interval。

计划中的阶段 7 要先独立选择 TTS 后端和目标平台，不能把这一项视为当前 STT MVP 的遗漏 bug。

## 6. 如何测试

建议严格按下面顺序测试。前一步失败时先解决该层，不要直接进入完整 20 turn 游戏。

### 6.1 确认分支和安装依赖

```bash
git branch --show-current
# 预期：feature/voice

uv sync --frozen --extra voice
uv lock --check
```

查看 CLI：

```bash
uv run --frozen --extra voice streammuse-task --help
uv run --frozen --extra voice streammuse-task play --help
```

### 6.2 运行语音聚焦自动测试

```bash
uv run --frozen --extra voice pytest \
  tests/unit/infrastructure/voice \
  tests/unit/scripts/test_qualify_voice_input.py \
  tests/unit/application/tasks/test_interactive_runtime.py \
  tests/unit/presentation/task/test_task_cli.py \
  tests/unit/domain/tasks/test_zip_zap_zop_task.py \
  tests/unit/application/tasks/test_human_input.py \
  tests/unit/application/factories/test_human_input_factory.py \
  tests/integration/test_voice_model.py \
  -q
```

没有设置真实模型环境变量时，预期：

```text
214 passed, 2 skipped
```

### 6.3 测试终端模式没有回归

这一步不需要可用的 LLM server，因为 `max-turns=1` 且 human first，不会执行 LLM turn：

```bash
uv run --frozen streammuse-task play \
  --task zip_zap_zop \
  --human-input terminal \
  --human-first \
  --max-turns 1 \
  --start-number 1 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --output-dir /private/tmp/streammuse-terminal-smoke
```

看到提示后可以依次输入：

```text
:help
1
```

预期：

- `:help` 显示命令但不消耗 turn；
- `1` 被判为 `OK`；
- summary 为 1 turn、1 valid；
- manifest 的 `human_input.mode` 为 `terminal`；
- 不需要任何麦克风或 Whisper 模型。

### 6.4 枚举麦克风

```bash
uv run --frozen --extra voice streammuse-task voice-devices
```

正常设备输出类似：

```text
Microphone input devices:
[0] MacBook Pro Microphone (channels=1, default_rate=48000 Hz, hostapi=0)
```

记录要使用的索引。若没有设备：

1. 检查“系统设置 -> 隐私与安全性 -> 麦克风”；
2. 确认运行命令的 Terminal、IDE 或 Python 获得权限；
3. 不要在未透传音频设备的容器/SSH 会话中测试；
4. 重新运行 `voice-devices`。

### 6.5 准备固定模型缓存

正式离线测试应固定本次已经验证的 commit：

```text
0d3d19a32d3338f10357c0889762bd8d64bbdeba
```

如果缓存还没有模型，需要有意执行一次联网下载。下面的命令可直接从项目根目录执行；
`.cache/` 已被本项目的 `.gitignore` 忽略：

```bash
mkdir -p .cache/voice-models

uv run --frozen --extra voice python -c \
  "from faster_whisper.utils import download_model; print(download_model('tiny.en', cache_dir='.cache/voice-models', revision='0d3d19a32d3338f10357c0889762bd8d64bbdeba'))"
```

之后的游戏和正式测试都使用 `--voice-local-files-only`，避免首轮意外联网。

### 6.6 运行固定真实模型集成测试

```bash
STREAMMUSE_TEST_VOICE_MODEL=tiny.en \
STREAMMUSE_TEST_VOICE_MODEL_CACHE="$PWD/.cache/voice-models" \
STREAMMUSE_TEST_VOICE_MODEL_REVISION=0d3d19a32d3338f10357c0889762bd8d64bbdeba \
uv run --frozen --extra voice pytest tests/integration/test_voice_model.py -q
```

没有预录音频时预期：

```text
1 passed, 1 skipped
```

这证明真实模型能从固定本地 snapshot 加载、预热并完成一次转录调用。

如果有一段经过同意采集的测试音频，再添加：

```bash
STREAMMUSE_TEST_VOICE_AUDIO=/absolute/path/to/zip-zap.wav \
STREAMMUSE_TEST_VOICE_EXPECTED=ZipZap
```

连同上面的 model/cache/revision 环境变量运行，预期为 `2 passed`。`EXPECTED` 必须是规范答案，
例如 `Zip`、`ZipZap`、`21`。

### 6.7 不连接 LLM，测试一轮真实麦克风完整链路

这是最有用的第一条人工语音测试。`max-turns=1` 保证不会请求 LLM server：

```bash
uv run --frozen --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --human-first \
  --max-turns 1 \
  --start-number 1 \
  --microphone-device 0 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --output-dir /private/tmp/streammuse-voice-smoke
```

把 `--microphone-device 0` 换成枚举得到的索引。看到 `1:` 后清楚地说 `one`。

预期：

- 启动阶段先完成麦克风 preflight 和模型预热；
- 人类回合才打开采集流；
- 原始转录接近 `one`；
- 规范响应为 `1`；
- 裁判显示 `OK`；
- 游戏结束后麦克风输入流关闭；
- 默认没有生成 `*_human.wav`。

### 6.8 检查运行产物

进入刚生成的 `zip_zap_zop_interactive_*` 目录，然后运行：

```bash
python -m json.tool manifest.json
python -m json.tool run_summary.json
sed -n '1,3p' response_trace.jsonl
find artifacts -type f -print
```

重点检查：

1. `manifest.json` 的 `schema_version` 是 2；
2. `status` 是 `completed`；
3. `human_input.mode` 是 `voice`；
4. recognizer 是 `tiny.en` / CPU / int8；
5. requested/resolved revision 是固定 commit；
6. `response_trace.jsonl` 的规范响应是 `1`；
7. `metadata.human_input.raw_transcript` 保留原始 ASR；
8. `latency_ms` 与 `metadata.human_input.total_latency_ms` 一致；
9. `audio_artifact` 是 null；
10. 没有 `*_human.wav`。

### 6.9 测试显式音频保存

重复 6.7 的命令并增加：

```text
--voice-save-audio
```

预期会出现：

```text
artifacts/turn/0001_turn_0000_human.wav
```

并且 trace 中：

- `audio_artifact` 是相对路径；
- `artifact_persistence_ms >= 0`；
- 文件写入耗时不计入玩家 deadline 分数。

测试结束后应按敏感录音处理该 WAV，不要默认提交或共享。

### 6.10 连接 LLM server，运行 20 turn soft 游戏

先确认 OpenAI-compatible 服务可访问：

```bash
curl http://127.0.0.1:8000/v1/models
```

然后运行：

```bash
uv run --frozen --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --human-first \
  --max-turns 20 \
  --start-number 1 \
  --microphone-device 0 \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output-dir task_runs
```

根据屏幕当前数字回答：

- 3 的倍数说 Zip；
- 4 的倍数说 Zap；
- 5 的倍数说 Zop；
- 同时满足时按 Zip、Zap、Zop 顺序组合；
- 其他数字直接读数字。

`max-turns=20` 是 human 和 LLM 合计 20 turn；human first 时通常有 10 个麦克风回合。

验收时记录：

- 是否发生首音节截断；
- 是否经常把 Zip/Zap/Zop 混淆；
- 每轮 endpoint reason；
- 每轮 ASR latency；
- deadline miss；
- 是否出现 queue overflow；
- 游戏结束后系统麦克风指示是否关闭。

### 6.11 hard 和 challenge 烟雾测试

先使用 3000 ms hard，不要直接从 1000 ms 开始：

```bash
uv run --frozen --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --max-turns 6 \
  --microphone-device 0 \
  --voice-model-cache .cache/voice-models \
  --voice-model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --voice-local-files-only \
  --deadline-mode hard \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

预期第一个超时或错误答案立即结束，并记录 winner、loser 和 stop reason。
游戏规则导致的输赢仍然是一次正常完成的 CLI 运行，进程可以返回 0；是否失败应读取
`manifest.json` 的 `stop_reason`、`winner` 和 `loser`。只有麦克风、模型或其他基础设施故障
才按错误退出。

较宽松的 challenge 烟雾测试可以使用：

```text
--deadline-mode challenge
--challenge-stage-turns 4
--challenge-deadline-ms-list 5000,3000
```

在正式 p95 和重复成功率数据出现之前，不要把 1000 ms challenge 标记为通过。

### 6.12 运行正式离线语料资格验证

manifest 示例：

```json
{
  "audio_path": "audio/acceptance-0001.wav",
  "expected": "ZipZap",
  "split": "acceptance",
  "category": "combination",
  "speaker": "consented-speaker-id",
  "session": "session-id",
  "distance": "near",
  "environment": "quiet_room"
}
```

正式命令必须使用新的、尚不存在的 output directory：

```bash
uv run --frozen --extra voice python scripts/qualify_voice_input.py \
  --manifest /path/to/frozen-corpus/manifest.jsonl \
  --output-dir /path/to/new-results/task-context \
  --model tiny.en \
  --device cpu \
  --compute-type int8 \
  --model-cache .cache/voice-models \
  --model-revision 0d3d19a32d3338f10357c0889762bd8d64bbdeba \
  --local-files-only \
  --context-profile task
```

退出码：

- 0：formal offline corpus gate 全部通过；
- 1：运行完成，但一个或多个产品门槛失败；
- 2：manifest、冻结证据、模型、依赖或配置无效。

输出：

```text
task-context/
  samples.jsonl
  summary.json
```

检查 `summary.json`：

- `qualification_profile == "formal"`；
- `passed == true` 才代表离线 corpus gate 通过；
- `full_feature_qualification == false` 仍然是正常值，因为离线重放不能验证麦克风端点；
- `artifact_set.samples.sha256` 应与实际 `samples.jsonl` 一致；
- `acceptance.not_measured` 列出仍需麦克风测试的指标。

基线和自定义 context 用于开发比较，必须写到不同新目录。它们会标为 exploratory，不能返回
formal pass。

### 6.13 运行全仓测试

```bash
uv run --frozen --extra voice pytest tests/unit -q
uv run --frozen --extra voice pytest tests/integration -q
```

当前已知基线不是全绿：单元测试有 14 个与语音无关的 robustness/perturbation 失败，集成测试有
1 个与语音无关的 Lekai 失败。判断语音回归时应同时看 6.2 的聚焦套件，不能把这些既有失败误归因
于本功能；但在合并整个仓库前，团队仍应单独决定是否修复或更新这些旧测试契约。

## 7. 常见故障

| 现象 | 含义和处理 |
|---|---|
| `No microphone input devices found` | 当前进程看不到输入设备；检查系统权限、容器/SSH 和 PortAudio |
| 缺少 `sounddevice` / `webrtcvad` / `faster-whisper` | 运行 `uv sync --extra voice` |
| `local_files_only` cache miss | revision/cache 不匹配；先有意下载固定 snapshot 或修正路径 |
| 设备只支持 44.1 kHz | MVP 不支持；换输入配置/设备，或后续实现 VAD 前流式重采样 |
| `Zip sap` 等错误转录 | 查看 raw transcript；当前不会用期望答案模糊修正 |
| 经常 start timeout | 检查所选设备、权限、输入电平和是否在提示后才开口 |
| 经常 3000 ms deadline miss | 查看 wait/utterance/endpoint/asr 分阶段时间，先不要使用 hard/challenge |
| queue overflow | CPU 消费跟不上 callback；关闭竞争负载并检查队列/调度指标 |
| `voice input error` 但没有 turn | 这是正确的基础设施失败策略；检查 startup manifest 的 provenance/error |
| 资格工具拒绝 output directory | output 必须不存在，且不能与任何冻结输入、源码或模型路径重叠 |
| 资格运行始终 exploratory | 必须使用默认门槛、`context-profile task`、tiny.en/cpu/int8/offline 和固定官方 snapshot |
| 采集期间无法输入 `:quit` | 语音 v1 不并发读取终端命令；使用 Ctrl-C |

## 8. 建议的验收顺序

建议由开发者/测试者按下面顺序签字：

1. 聚焦自动测试 `214 passed, 2 skipped`；
2. 终端一轮测试通过；
3. `voice-devices` 能看到目标麦克风；
4. 固定本地 snapshot 模型测试通过；
5. 不连接 LLM 的一轮 `one -> 1 -> OK` 通过；
6. 默认无 WAV，显式保存才有 WAV；
7. 20 turn soft 游戏完成且无流泄漏；
8. 3000 ms hard/challenge 烟雾测试符合输赢语义；
9. 同意语料 formal qualification 产出混淆矩阵和 ASR p95；
10. 重复麦克风会话补齐端点截断、queue overflow、last-voice-to-text 和 deadline success rate；
11. 完成以上数据后再决定是否支持 1000 ms challenge；
12. TTS 作为独立阶段选择后端后再实现。

## 9. 最终状态

实现计划中的阶段 1-4 已完成。阶段 0 已完成依赖锁、固定模型烟雾测试、正式语料协议和资格工具，
但没有真实 corpus 结果。阶段 5 已完成代码测试和文档，物理麦克风/预录语音/20 turn/混淆矩阵和
p95 仍待执行。阶段 6 的长期稳定化和阶段 7 的 TTS 尚未开始，符合计划中的前置条件。

因此当前最准确的状态是：

```text
语音输入代码 MVP：完成
自动化契约和错误路径：完成
固定真实模型加载/预热：完成
正式语料准确率资格：未完成
目标麦克风产品验收：未完成
TTS/双向语音：不在本次交付范围
```
