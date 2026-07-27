# StreamMUSE 交互式语音功能计划

日期：2026-07-16

目标分支：`feature/voice`

状态：阶段 1-4 的生产实现与自动化测试已完成；阶段 0/5 的真实语料和目标麦克风验收待外部条件补齐，阶段 6/7 尚未开始。

## 1. 目标与范围

为 `streammuse-task play` 添加可选启用的语音输入，首个支持的任务为
`zip_zap_zop`，在 CPU 上使用常驻的 `faster-whisper` `tiny.en` 模型并采用
`int8` 计算。

首个交付内容为：

```text
人类语音 -> 麦克风采集 -> 端点检测 -> faster-whisper
         -> 任务专属规范文本 -> 现有裁判/运行时
```

首个交付内容**不会**合成 LLM 的响应。Faster-whisper 仅提供 STT。完整的双向语音
需要另行选定 TTS 后端，并将在阶段 7 中实现。

兼容性要求：

- 终端输入仍为默认方式，其行为必须与当前完全一致。
- 语音只能在 `play` 子命令上选择启用。
- `streammuse-task run` 以及音乐/MIDI 实时栈保持不变。
- 未安装语音可选依赖时，应用仍必须能够导入并运行终端模式。
- 除非用户明确启用调试选项，否则不持久化任何麦克风音频。

## 2. 当前状态调研结果

### 2.1 现有游戏路径

交互式游戏的调用路径为：

```text
pyproject.toml: streammuse-task
  -> presentation/task/cli.py:main()
  -> presentation/task/cli.py:play_task()
  -> InteractiveTaskRuntime.play()
  -> InteractiveTaskRuntime._run_human_turn()
  -> TerminalIO.prompt() / prompt_with_timeout()
  -> ZipZapZopTask.validate_response()
  -> response_trace.jsonl + 回合产物 + 运行摘要
```

这是一个同步、逐回合执行的运行时。它不使用 `RuntimeSessionBuilder`、
`RealTimeMusicService` 或音乐输入工厂。

`domain/interfaces/InputSource` 专门用于提供 `MusicalEvent`。语音答案是交互式任务的
文本响应，不得被加入该接口，也不得加入 `InputConfig`/`InputSourceFactory`。

### 2.2 现有语音工作

语音分支目前只添加了一个独立的微基准测试：

- `scripts/voice_microbench.py`
- `tests/unit/scripts/test_voice_microbench.py`
- `docs/developer-guide/voice-microbenchmark-results.md`

它未修改任务运行时、CLI、依赖锁文件、麦克风输入或日志结构。生产代码不得导入该
基准测试脚本；运行时适配器应当是 `src/streammuse` 下一个小型且具有类型标注的实现。

该基准测试得出了 `tiny.en`、CPU、`int8` 配置下的以下结果：

| 环境 | 加载 | 首次推理 | 稳态平均值 | 精确匹配 |
|---|---:|---:|---:|---:|
| Mac CPU | 1676.6 ms | 202.7 ms | 159.9 ms | 9/10 |
| H200 主机 CPU | 510.0 ms | 215.4 ms | 189.1 ms | 9/10 |

这些数字仅涵盖完整 WAV 文件，不包括麦克风采集、语音端点检测、游戏运行时和播放。
两次 CPU 运行都出现了相同的领域关键错误：`zip zap` 被识别成了 `Zip sap.`

当前项目环境不包含 `faster-whisper`、麦克风采集包或流式 VAD 包。该基准测试使用了
独立环境，且没有记录可复现的依赖锁定信息。

### 2.3 基线验证

编写本计划前，针对相关功能的基线测试已通过：

```text
38 passed
```

测试涵盖了交互式运行时、Zip-Zap-Zop 任务、任务 CLI 和语音微基准测试辅助代码。

## 3. 设计决策

### 3.1 引入任务专用的人类响应源

为交互式任务添加 `HumanResponseSource` 协议。不要把 `TerminalIO` 变成麦克风抽象，
也不要复用音乐侧的 `InputSource` 协议。

将不依赖额外包的数据记录和协议放在 `domain/tasks/models.py` 中，与
`InteractiveTask` 和 `InteractiveTurnRecord` 并列：

```python
@dataclass(frozen=True)
class SpeechContext:
    initial_prompt: str | None = None
    hotwords: tuple[str, ...] = ()


@dataclass(frozen=True)
class HumanResponseRequest:
    turn_id: int
    prompt: str
    timeout_s: float | None
    speech_context: SpeechContext | None = None


@dataclass(frozen=True)
class HumanResponse:
    text: str
    status: Literal["ok", "no_speech", "empty_transcript"] = "ok"
    deadline_expired: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanResponseSource(Protocol):
    mode: Literal["terminal", "voice"]

    def start(self) -> None: ...
    @property
    def provenance(self) -> dict[str, Any]: ...  # 不可变、可安全序列化为 JSON 的快照
    def read_response(self, request: HumanResponseRequest) -> HumanResponse: ...
    def close(self) -> None: ...
```

`deadline_expired=True` 表示请求中非 `None` 的游戏时间预算已耗尽。语音源内部用于
`soft` 模式的安全超时绝不能设置该字段。采集、重采样或模型失败时应抛出带类型的基础设施
异常，而不是返回 `HumanResponse`。
溯源属性 `provenance` 在启动前、启动期间及启动后均可调用；它返回配置值，以及当时已经解析出的
设备/模型值。这样，无需通过基础设施属性内省，启动错误仍然可诊断。

`TerminalHumanResponseSource` 封装现有的 `TerminalIO` 行为。运行时继续使用
`TerminalIO` 显示横幅、截止时间模式菜单和屏幕输出。现有的冒号命令分派器仍保留在
运行时中；由于终端源返回输入的文本行，终端模式会以与当前完全相同的方式到达该分派器。
语音模式 v1 只接受口述的游戏答案，不承诺在麦克风启用时并发接收键入的冒号命令；
`Ctrl-C` 仍是紧急退出方式。未来的混合源可以多路复用终端命令和麦克风帧，而无需修改
运行时契约。

构造函数不得产生副作用。`start()` 和 `close()` 必须是幂等的，且 `start()` 部分失败后
仍必须可以关闭。运行时用 `try/finally` 包裹 `source.start()` 和整个游戏；同时，
`play_task()` 使用外层 `ExitStack`，从获取第一个可关闭资源起就进行注册，以确保后续构造
失败时，响应源和聊天客户端均受到保护。启动失败时写入 v2 运行清单，其中
`status="startup_error"`，并包含已配置或部分解析出的溯源信息，但绝不创建玩家回合。

测试必须覆盖客户端构造失败、响应源构造失败、响应源启动/预热失败、运行时失败以及
`KeyboardInterrupt`；同时必须保留原始异常，并按逻辑所有权将每个已获取资源关闭一次
（允许执行幂等的备用关闭）。

当运行时自行构造默认终端源时，必须向它传入实际解析得到的同一个终端实例：
`TerminalHumanResponseSource(self.terminal)`。这样可保持现有直接运行时测试的行为，
这些测试注入同一个 `FakeTerminal` 同时用于输入和输出。

如果为不满足 `SpeechAwareInteractiveTask` 的任务请求语音模式，应在启动阶段、构造
麦克风/模型之前，以 unsupported-task 错误失败，而不是回退到无边界的通用解析器。

### 3.2 将语音语义保留在任务中

音频采集和 Whisper 与具体任务无关。游戏词汇和响应规范化属于任务本身。

添加独立的 `SpeechAwareInteractiveTask` 协议，使纯文本任务不必被迫实现语音方法：

```python
def build_speech_context(state, transcript) -> SpeechContext: ...
def parse_spoken_response(
    state,
    transcript,
    raw_text: str,
) -> SpokenResponseParseResult: ...
```

`SpeechAwareInteractiveTask` 应使用 `@runtime_checkable`（或通过一个明确的注册表能力
进行检查），并且 `SpokenResponseParseResult` 包含 `canonical_text: str | None` 以及
机器可读的解析状态/原因。这样可以明确区分未检测到语音、ASR 输出为空和已说话但文本
无法解析这三种情况。

对于 `ZipZapZopTask`，`SpeechContext` 包含 `Zip`、`Zap` 和 `Zop` 等静态词汇；
其中不得包含当前预期答案。

规范化应当：

- 移除末尾标点，并统一空白和大小写；
- 合并精确匹配的 `zip`/`zap`/`zop` 词元，使 `zip zap` 变成 `ZipZap`；
- 将有效的英语数字短语转换为其数字表示形式；
- 拒绝无法解析的文本，而不是进行猜测；
- 规范化后仍保留现有裁判的严格比较。

初期不要添加 `sap -> zap` 之类的模糊映射，也不要利用预期答案修复转录文本。任一做法
都可能在无提示的情况下将错误的人类答案变为正确答案。应先尝试静态热词（`hotwords`）和
初始提示词（`initial_prompt`），之后仅添加由留出的真实语音数据集证明合理的混淆映射。

解析成功时，规范化后的值仍写入 `InteractiveTurnRecord.response`，因为裁判和后续 LLM
历史都使用该字段。无法解析的口述响应使用空的规范值，在现有裁判下仍判定为无效，并在
人类输入元数据中记录具体的解析失败信息。未经修改的 ASR 输出保留在
`record.metadata["human_input"]["raw_transcript"]` 中。

人类回合的处理顺序固定为：

```text
读取响应
  -> 若终端文本以 ":" 开头：分派命令并重新提示
  -> 若为语音：解析口述响应
  -> 验证规范响应
  -> 结束回合
```

因此，命令分派发生在任务语音解析之前。终端命令不会消耗回合，并保持当前行为：运行时
重新提示输入时启动一个新的输入计时器。

### 3.3 使用常驻识别器

实现一个专用的 `FasterWhisperRecognizer`，它应：

- 仅加载一次 `WhisperModel("tiny.en", device="cpu", compute_type="int8")`；
- 在游戏计时器启动前执行一次真实的预热转录；
- 直接接收 16 kHz 单声道 `numpy.float32` 波形，无需临时 WAV 文件；
- 对彼此独立的短轮次设置 `language="en"`、`beam_size=1` 和
  `condition_on_previous_text=False`；
- 在这些解码选项通过阶段 0 验证后，显式设置 `task="transcribe"`、
  `temperature=0`、`vad_filter=False`、`word_timestamps=False` 和
  `without_timestamps=True`，从而避免隐藏默认值和第二次完整缓冲区 VAD 处理；
- 在选择最终解码配置之前，使用语音测试语料库评估静态 `hotwords` 和
  `initial_prompt`；
- 使用稳定且经过验证的空格连接规则，将任务的热词元组渲染为 API 所需的单个字符串，
  并记录实际渲染出的初始提示词/热词值；
- 在计量的 ASR 时间区间内完整消费返回的分段结果生成器；
- 返回原始文本和模型级诊断数据，不执行任务特定的改写。

CPU/int8 必须是显式默认值，而不是 `device="auto"`。当前的 GPU 基准测试使用的是
`float16`，并非选定配置，而且该仓库已记录 CUDA/CTranslate2 兼容性方面的阻碍。

模型加载、下载和预热均在第一轮之前进行。除了 Python 包之外，还要固定并记录解析后的
模型仓库的版本修订号和快照。支持显式下载根目录和仅使用本地文件的行为；模型缺失、
离线缓存未命中或 CTranslate2 安装不兼容都属于启动错误，而不是玩家失败。

### 3.4 按轮次启用麦克风采集并进行流式端点检测

麦克风仅在人类玩家的轮次中打开。这样可以避免持续监听，也能防止之后的大部分 LLM/TTS
播放内容被采集。

建议的首版实现：

- 使用 `sounddevice.RawInputStream` 进行跨平台 PortAudio 采集；
- 单声道、有符号 16-bit PCM；
- 通过预检选择 VAD 支持的设备采样率，优先选择 16 kHz，其次为 48/32/8 kHz；
- 使用帧累积器将大小不定的回调数据块转换为精确的 20 ms 帧，然后由
  `webrtcvad-wheels` 判断有声/无声；
- 在 PortAudio 回调与主线程之间使用有界队列；
- 使用较短的预滚动环形缓冲区，以免语音起始部分被截断；
- 在达到可配置的尾部静音时长、最大发言时长或游戏截止时间后停止；
- 在调用 Whisper 前，使用现有 SciPy 依赖将完整发言重采样至 16 kHz。

WebRTC VAD 无法直接处理 44.1 kHz。对于 MVP，如果 PortAudio 无法以任何受支持的采样率
打开设备，则预检应失败并提供可操作的错误消息，而不是悄然传入无效帧。若要支持真正仅限
44.1 kHz 的设备，需要在 VAD 之前添加有状态的流式重采样器，这属于后续工作。

PortAudio 回调只能将帧和状态复制到有界队列中。它不得执行 VAD、重采样、文件 I/O
或 ASR。队列溢出应作为基础设施错误暴露，而不是悄然丢弃音频。

Faster-whisper 内置的 Silero VAD 可以过滤已有的音频缓冲区，但无法决定实时录音器应在何时
停止。因此，它不能替代流式端点状态机。

仅当自动端点检测未通过真实麦克风验收门槛时，才提供按键说话/固定时间窗口作为回退方案。
对于一秒挑战阶段，它不应成为唯一的用户交互方式。

### 3.5 保持当前的截止时间语义

现有运行时测量从呈现一个轮次到响应就绪的完整耗时。语音模式应保持这一定义：

```text
人类回合延迟 = 等待语音 + 发言时长 + 端点静音 + ASR
```

- `hard` 和 `challenge` 将剩余的绝对时间预算传入语音响应源。达到截止时间时停止采集。
- 允许 ASR 同步完成；如果总耗时超过截止时间，则随后将其标记为超时。不要在后台线程中留下
  无法取消的陈旧 ASR 任务。
- `soft` 仍然需要针对等待和发言时长设置独立的安全上限，否则失效的麦克风可能会永久阻塞。
- 在 `hard`/`challenge` 游戏截止时间到达时仍未检测到语音，应返回
  `deadline_expired=True`，并视为强制错过截止时间。在 `soft` 模式安全上限到达时仍未检测到
  语音，应返回 `deadline_expired=False`；这是无效的空答案，而 `deadline_missed` 仍只由实测
  耗时与所配置的 `soft` 截止时间的比较结果决定。
- 语音产生空白或无法解析的转录文本时，视为无效答案。
- 设备、音频流、重采样或模型故障属于运行错误，绝不能进入 `_handle_turn_outcome()`，也不能
  归因于人类玩家。

如果响应既超时又无效，应保持当前处理顺序：截止时间导致的失败优先，同时在跟踪日志中保留
这两个事实。

这是一个**计分截止时间**，而不是有保证的挂钟时间返回上限。进程内的 CTranslate2 调用无法
被安全终止；`hard`/`challenge` 模式可能要等 ASR 完成后才返回，此时已经超过名义截止时间。真正有界的
执行截止时间需要可终止的进程边界，不在 MVP 范围内。

由于语音 v1 在采集期间无法输入 `:quit`，需要显式处理 `KeyboardInterrupt`：停止输入源，
写入 `stop_reason="user_interrupt"` 且不指定赢家/输家，并在产物最终写入完成后，让 CLI 返回
惯例中的中断状态。

不要从人类玩家的结果中减去实测 ASR 延迟，也不要暗中延长截止时间。以该轮次的单调时钟
起点为基准，持久化 `last_voiced_frame`、`endpoint_detected`、`asr_start` 和 `asr_end` 的偏移量。
根据这些精确时间点推导端点延迟、ASR 延迟和最后语音帧到文本的耗时，以便未来研究能够显式
引入不同的截止时间基准。

当前 `challenge` 时间表结束于 1000 ms。语音回答、端点静音以及实测约 160 ms 的 ASR 时间之和
可能无法稳定满足这一限制。保持时间表可配置，并记录一个更宽松的语音时间表，直到目标
硬件上的 p95 结果足以支持一秒阶段。

### 3.6 保持运行时同步

任务运行时是串行的，并且不与音乐节拍循环共享。PortAudio 回调线程仅用于采集；端点
检测和 ASR 保留在当前人类玩家轮次中。MVP 不要引入 asyncio 或通用工作线程池。

如果语音功能之后与 `RealTimeMusicService` 同时使用，应先对 CPU 竞争进行基准测试，届时再将
ASR 移至有界工作线程/进程边界之后。这是一个独立的集成问题。

## 4. CLI 和依赖接口

仅为 `streammuse-task play` 添加以下选项：

```text
--human-input {terminal,voice}       # 默认值：terminal
--voice-model tiny.en
--voice-device cpu
--voice-compute-type int8
--microphone-device <name-or-index>
--voice-model-cache <path>
--voice-model-revision <commit-or-tag>
--voice-local-files-only
--voice-save-audio                   # 默认值：false
```

添加 `streammuse-task voice-devices` 作为轻量级设备枚举子命令。它不得要求 `--task`、构造 LLM
客户端或加载 Whisper 模型。`main()` 应在访问 `args.task` 或执行现有 `run`/`play` 任务验证之前分派
该命令。其余所有语音配置仍放在 `play` 上。

将这些参数解析为冻结的 `HumanInputConfig`/`VoiceInputConfig` 应用层数据类。工厂接受强类型
配置，绝不接受 `argparse.Namespace`；配置本身持有隐藏的端点默认值，并在选择 `terminal` 模式时
拒绝仅适用于语音的选项。

最初应将端点调优参数保留在稳定的默认值背后。只有当真实设备测试表明用户必须调节它们时，
才暴露高级参数：

```text
--voice-start-timeout-ms
--voice-end-silence-ms
--voice-max-utterance-ms
--voice-vad-aggressiveness
```

在 `pyproject.toml` 中添加 `voice` 可选依赖组，然后重新生成 `uv.lock`。候选包系列包括：

- `faster-whisper` 1.2.x;
- `sounddevice` 0.5.x;
- `webrtcvad-wheels` 2.0.x.

只有在锁定环境中重新运行专项基准测试和麦克风冒烟测试后，才能由 `uv.lock` 固定确切版本。这三个包
都必须采用延迟导入；若未安装该可选依赖组却选择语音模式，应抛出一条可操作的安装提示。

启动预检应报告：

- 选定的输入设备及其原生采样率；
- 麦克风权限/设备错误；
- 模型标识符或本地模型路径；
- 解析后的设备/计算类型；
- 模型加载和预热时长。

支持本地模型目录/缓存设置，使离线运行不依赖第一轮中隐式进行的 Hugging Face 下载。
`local_files_only` 必须显式透传；运行成功后，应记录请求的版本修订号，并在可用时记录解析后的
快照/模型路径。

## 5. 跟踪日志和产物变更

保留现有的顶层 `response_trace.jsonl` 字段，以确保下游读取方继续正常工作。在人类玩家的轮次中
添加以下嵌套结构：

```json
{
  "metadata": {
    "human_input": {
      "mode": "voice",
      "status": "ok",
      "raw_transcript": "Zip zap.",
      "canonical_response": "ZipZap",
      "wait_for_speech_ms": 120.0,
      "utterance_ms": 510.0,
      "endpoint_silence_ms": 300.0,
      "last_voiced_offset_ms": 630.0,
      "endpoint_detected_offset_ms": 930.0,
      "asr_start_offset_ms": 930.0,
      "asr_end_offset_ms": 1095.0,
      "asr_latency_ms": 165.0,
      "total_latency_ms": 1095.0,
      "endpoint_reason": "trailing_silence",
      "capture_sample_rate_hz": 48000,
      "audio_overflow": false,
      "audio_artifact": null
    }
  }
}
```

为 `_finish_turn()` 添加一个显式的 `human_input_metadata` 参数。运行时在唯一允许的键
`metadata["human_input"]` 下嵌入其防御性副本；响应源无法覆盖 `failure_reason`、词元字段、
`deadline_mode`、`deadline_ms`、`challenge` 字段或 `model_error`。
`InteractiveTurnRecord.latency_ms` 仍是用于截止时间计分的唯一权威值。嵌套的
`total_latency_ms` 用于诊断，并且必须在文档规定的时钟/舍入容差内与其一致。

运行清单新增内容：

- 人类输入模式；
- 麦克风设备/采样率；
- STT 模型/设备/计算类型；
- 解码和端点配置；
- 加载/预热计时；
- 是否启用原始音频持久化。

在 `source.start()` 之后，运行时应在写入初始运行清单之前读取其不可变、可安全序列化为 JSON 的
`provenance` 快照。对 `terminal` 和 `voice` 两类运行都将交互式运行清单结构升级到 v2，
保留所有现有字段并添加 `human_input` 对象；回合跟踪日志的变更仍保持增量式。

若启用 `--voice-save-audio`，仅将有界发言保存在运行目录下，并存储相对路径。
`HumanResponseRequest.turn_id` 和响应源中配置的运行产物根目录决定稳定的名称，即使 LLM 先行
也是如此。默认运行保留转录文本和计时信息，但不保存原始音频。

## 6. 计划中的代码改动

### 新增生产模块

- `src/streammuse/application/tasks/human_input.py`
  - 将 `TimedPromptResult`、`TerminalIO` 和 `StdTerminalIO` 从
    `interactive_runtime.py` 中移出，同时保留其公共重新导出；
  - 定义并验证 `HumanInputConfig`/`VoiceInputConfig`；
  - 添加 `TerminalHumanResponseSource`；无依赖契约放在
    `domain/tasks/models.py` 中。将终端类型保留在这里，可以避免运行时与其响应源适配器之间的循环导入。
- `src/streammuse/application/factories/human_input_factory.py`
  - 延迟选择并组装终端/语音输入源。
- `src/streammuse/infrastructure/voice/__init__.py`
- `src/streammuse/infrastructure/voice/microphone.py`
  - 设备预检、回调队列、VAD 端点状态机和重采样。
- `src/streammuse/infrastructure/voice/faster_whisper.py`
  - 常驻识别器和预热。
- `src/streammuse/infrastructure/voice/response_source.py`
  - 从采集到 ASR 的编排以及结构化计时元数据。

### 需要修改的现有模块

- `src/streammuse/domain/tasks/models.py`
  - 人类响应契约和可选的语音感知任务契约。
- `src/streammuse/domain/tasks/zip_zap_zop.py`
  - 静态语音上下文和确定性的口语响应规范化。
- `src/streammuse/domain/tasks/__init__.py`
  - 导出新的领域契约。
- `src/streammuse/application/tasks/interactive_runtime.py`
  - 注入/使用 `HumanResponseSource`，传递元数据，并保持命令和截止时间行为。
- `src/streammuse/application/tasks/__init__.py`
  - 导出应用层适配器/配置。
- `src/streammuse/presentation/task/cli.py`
  - 语音参数、工厂接线、启动和清理。
- `pyproject.toml` 和 `uv.lock`
  - 可选的语音依赖组。
- 安装/用户文档
  - 语音可选依赖组、麦克风权限/设备、离线模型缓存、示例命令、故障诊断和隐私行为。

此功能不要修改 `application/config/models.py`、`InputSourceFactory`、
`RealTimeMusicService` 或音乐 CLI。

## 7. 实施阶段

### 阶段 0：可复现的语音资格验证

- [x] 在当前功能分支中创建并校验 `voice` 可选依赖锁。
- [x] 使用锁定版本完成 `tiny.en`/CPU/int8 本地模型加载、预热和转录烟雾测试，并记录准确的软件包、
  操作系统、CPU、模型仓库版本修订号/快照以及缓存来源。
- [ ] 构建经过同意采集的真实语音评估集，覆盖单独及组合形式的
  `Zip`/`Zap`/`Zop`、游戏中使用的数字、静音、噪声和背景扬声器播放。
- [x] 在资格工具中冻结并强制执行语料库协议：开发集与验收集按说话人隔离；验收集至少包含
  5 名说话人、每个基础词 100 个留出样本、100 个组合词、150 个数字答案，
  以及 200 个负样本片段，分别来自静音/噪声/播放/非命令语音。记录距离、
  环境以及说话人/会话分层，但不要在运行跟踪日志中存储身份信息。
- [ ] 比较基线解码、静态热词和初始提示词。选择能够提高留出集精确准确率、
  且纠正程度最小的配置。
- [ ] 分别报告原始 ASR 和规范化后的精确准确率，并附上混淆计数和置信区间；
  分别报告错误命令、无语音误拒、起始语音截断和采集溢出。
- [ ] 确认目标机器能够以 VAD 支持的采样率打开麦克风。

退出标准：所选软件包锁、解码设置和端点默认值均已记录且可复现。如果所选模型
无法达到准确率门槛，不要用基于预期答案的纠错来掩盖问题。

### 阶段 1：契约和终端兼容性

- [x] 添加 `HumanResponseSource`、请求/结果记录和终端适配器。
- [x] 添加冻结且经过验证的 `HumanInputConfig`/`VoiceInputConfig`，以及可安全序列化为 JSON
  的输入源来源信息。
- [x] 将其注入 `InteractiveTaskRuntime`，并以终端行为作为默认值。
- [x] 完整保留截止时间菜单、终端模式下的冒号命令、虚假终端测试、跟踪日志和文本模式计时。
- [x] 在成功、用户退出、运行时异常和启动失败时执行生命周期清理；添加干净的
  `KeyboardInterrupt`/`user_interrupt` 路径。

退出标准：不安装语音可选依赖组时，所有现有测试均通过。

### 阶段 2：常驻 faster-whisper 适配器

- [x] 实现延迟导入、配置验证、加载、预热、转录、生成器消费和关闭行为。
- [x] 固定/传入模型版本修订号、缓存/下载根目录和仅使用本地文件策略；
  如果可以获得解析后的快照，则记录它。
- [x] 直接传递 NumPy 音频；不要为每个回合创建临时文件。
- [x] 添加静态语音上下文支持，但不传递预期答案。
- [x] 分别测量模型解析、加载、预热和每次转录。
- [x] 对缺失可选依赖、缓存未命中和不支持的设备/计算类型给出可操作的错误信息。

退出标准：虚假模型测试证明每局游戏只加载一次模型、预热发生在回合计时器之外、
参数具有确定性且异常映射正确。

### 阶段 3：麦克风和端点处理流水线

- [x] 实现设备枚举和采集预检。
- [x] 实现从回调到有界队列的采集。
- [x] 将任意大小的回调数据块累积成精确的 VAD 帧；如果无法用任何支持的流采样率
  打开某个设备，则拒绝该设备。
- [x] 实现包含预卷、语音起始、尾部静音、最大话语长度和截止时间端点原因的 VAD 状态机。
- [x] 实现采样率转换和规范化的 float32 音频输出。
- [x] 将溢出和 PortAudio 回调错误上报到主线程。
- [x] 使关闭操作具有幂等性，并证明它能解除正在进行的采集阻塞。

退出标准：确定性的音频帧测试覆盖每一种状态转换；在目标 Mac 麦克风上的手动冒烟测试成功，
且没有流泄漏。

### 阶段 4：游戏集成和规范化

- [x] 添加 CLI/工厂接线，并在游戏开始前执行模型/麦克风预检。
- [x] 实现 Zip-Zap-Zop 语音上下文和有界语法解析器。
- [x] 在任何语音解析之前分派终端冒号命令；以结构化方式表示解析失败，
  且不会把命令作为一个回合消耗掉。
- [x] 只把规范化后的响应文本传给现有裁判/历史记录。
- [x] 在回合元数据中保留原始转录文本和所有阶段的计时。
- [x] 仅在 `metadata.human_input` 下合并输入源元数据，并对照权威回合延迟
  验证其中嵌套的总延迟。
- [x] 实现上文定义的无语音、语音无法解析、截止时间和基础设施故障策略。
- [x] 音频持久化保持为显式启用。

退出标准：虚假语音输入源能够完成 `terminal`、`soft`、`hard` 和 `challenge` 场景，
且胜负结果和跟踪日志数据正确。

### 阶段 5：验证和文档

- [x] 运行领域层、基础设施层、应用层、CLI 和集成测试套件，并分别记录本功能结果与未修改模块中的既有失败。
- [ ] 使用预录制测试音频运行显式启用的离线真实模型测试；默认情况下，CI
  绝不能下载模型或要求使用麦克风。
- [ ] 在目标机器上至少完成一局 `soft` 模式下、使用真实麦克风的 20 回合游戏，
  以及一次 `hard`/`challenge` 冒烟测试。
- [ ] 发布受限词汇表的混淆矩阵和阶段级 p50/p95 延迟。
- [x] 记录安装、设备选择、权限、离线缓存、隐私、截止时间行为和 TTS 范围。

退出标准：第 10 节中的所有验收门槛均通过。

上述真实游戏只是硬件/用户体验冒烟测试。准确率和延迟资格应由语料库重放和重复的脚本化
麦克风会话决定，而不是由一局成功的游戏决定。

### 阶段 6：扩大范围前的稳定化

- [ ] 在重复游戏中观察错误触发、起始语音截断、静音超时和队列溢出。
- [ ] 使用留出录音而非验收集，调整 VAD/端点默认值。
- [ ] 如果 LLM 服务器或音乐栈在同一主机上运行，重新检查 CPU 争用情况。
- [ ] 完成上述工作后，才考虑其他交互式任务或持续监听。

### 阶段 7：可选的双向语音/TTS

此阶段仅在选定输出语音后开始。当前基准测试并未确定一个可移植的选择：Piper 在 H200
上速度很快，但在测试的 Mac 环境中失败；`espeak-ng` 速度很快但质量较低；Kokoro 更重。

- [ ] 通过独立的准确率/质量、首段音频、稳态和播放基准测试，选择 TTS 后端和目标平台。
- [ ] 添加面向任务的 `SpeechOutput` 协议；不要复用音乐的 `OutputSink`。
- [ ] 播放 LLM 的规范化响应，等待播放完成，并在打开人类麦克风前增加一小段保护间隔。
- [ ] 记录合成开始、首段音频、播放开始和播放结束。
- [ ] 定义 LLM 截止时间是在文本就绪、首个可听样本还是播放完成时结束。
- [ ] 测试扬声器到麦克风的泄漏；如果未实现声学回声消除，则要求使用耳机，
  或制定明确的回声策略。

## 8. 测试计划

### 领域测试

- 每个游戏词的标点/大小写/空格；
- 游戏中使用的每一种有效组合顺序；
- 数字及英文数字短语；
- 错误的数字/词语仍应判错；
- 不泄漏预期答案，也不进行宽泛的模糊纠错。

### 基础设施测试

- 设备选择和不支持采样率的诊断；
- 使用确定性 PCM 帧测试语音起始/尾部静音的端点检测；
- 可变回调数据块的累积，以及明确拒绝仅支持 44.1-kHz 的设备；
- 无语音、最大话语长度、游戏截止时间、溢出和回调失败；
- 重采样长度/采样率的正确性；
- 识别器只加载一次、只预热一次、消费所有分段结果并返回原始文本；
- 仅在选择语音模式时才出现可选依赖错误。

### 应用层/运行时测试

- 终端输入源的行为保持完全一致；
- `hard` 模式下的终端命令在解析前分派，不消耗回合，并保持现有计时器行为重新提示；
- 语音元数据写入 `response_trace.jsonl` 和回合产物；
- 原始转录文本与规范化响应不同，但不会破坏历史记录；
- `soft` 模式未命中后继续；`hard`/`challenge` 模式超时则判负；基础设施错误不指定胜者；
- 在经过时间超过配置的游戏截止时间前，`soft` 模式安全超时绝不能强制判定错过截止时间；
- 输入源来源信息出现在 v2 运行清单中，且不覆盖保留的运行时元数据；
- 在完成、退出和错误退出时关闭语音输入源；
- 模型加载/预热不计入回合延迟。

### 表现层测试

- 解析器默认使用终端输入；
- 所有语音参数均传递到工厂；
- `voice-devices` 不要求提供任务，也不启动游戏/模型服务器；
- 缺失语音可选依赖时仅生成一条可操作的信息；
- 任意一侧失败时，聊天客户端和语音输入源都会关闭。

### 集成/硬件测试

- 默认集成测试使用虚假麦克风和虚假模型，不使用网络或物理设备；
- 显式启用的真实模型集成测试使用本地模型/缓存和预录音频；
- 手动硬件矩阵首先覆盖目标 macOS；只有在声明支持 Linux/Windows 时，
  才覆盖这些平台。

## 9. 风险和缓解措施

| 优先级 | 风险 | 缓解措施 |
|---|---|---|
| P0 | Tiny.en 混淆游戏核心词 | 真实语音语料库、静态热词、有界解析器；使用失败门槛而非基于预期答案进行修复 |
| P0 | 端点和截止时间语义使 1 秒游戏无法实现 | 阶段计时、可配置时间表、明确的端到端截止时间定义 |
| P0 | 冷模型加载污染第一回合 | 常驻加载，并在游戏开始前完整消费预热结果 |
| P1 | 语音实现破坏终端菜单/命令 | 使用独立的人类响应源；`TerminalIO` 仍负责界面 |
| P1 | 音频回调阻塞或静默丢帧 | 有界队列、最小化回调、将溢出作为明确错误 |
| P1 | 首次运行意外下载模型 | 在游戏开始前执行预检/缓存选项和离线失败处理 |
| P1 | 设备采样率/权限因平台而异 | 设备枚举、采样率预检、可操作错误、分阶段声明支持范围 |
| P1 | 音频/转录隐私 | 默认不保存原始音频；显式启用并使用相对运行产物路径 |
| P2 | ASR 与其他实时工作负载争用资源 | CPU/int8 首版、阶段指标、有界线程；测量确认有需要后再进行进程隔离 |
| P2 | 扬声器播放被重新采集 | 当前采用回合门控；在 TTS 阶段决定播放保护/耳机/AEC 策略 |

## 10. 验收门槛

只有在以下所有条件均满足时，STT 功能才算完成：

1. 文本模式没有行为或测试回归，且不需要语音软件包。
2. 语音模式在游戏开始前加载并预热一个 `tiny.en` CPU/int8 模型。
3. 麦克风人类回合通过确定性的规范化响应进入现有 Zip-Zap-Zop 裁判逻辑，
   同时原始转录文本仍可审计。
4. 无语音、无效语音、`hard`/`challenge` 截止时间和基础设施故障遵循第 3.5 节中的策略。
5. 任意退出后均不残留输入流、回调、工作线程或模型资源。
6. 默认运行目录中不包含原始音频。
7. 冻结的验收语料库满足阶段 0 中的样本/划分协议。在目标 Mac 上，规范化后的精确准确率
   总体至少达到 95%，Zip、Zap 和 Zop 各自至少达到 95%，组合词与数字响应合计至少达到
   95%。分别报告原始 ASR 和规范化结果、混淆计数以及 95% 置信区间。
8. 负样本片段产生的错误游戏命令不超过 1%，正样本片段的无语音误判不超过 5%，
   端点检测造成的起始语音截断不超过 1%，并且验收采集集中的队列溢出为零。
9. 在同一目标机器上，热启动 ASR p95 不超过 300 ms，
   最后语音帧到文本的 p95（包括端点延迟）不超过 700 ms。
   这些是拟议的产品门槛，并非当前仅报告平均值的 10 文件基准测试已经证实的数值。
10. 在截止时间为 3000 ms 的重复脚本化麦克风回合中，至少 95% 正确说出的验收命令
    能在截止时间前变为规范化响应。1000 ms 的 `challenge` 阶段在独立满足相同的成功率/超时
    门槛之前，不得声明为通过资格验证。
11. 聚焦的单元/集成测试套件以及仓库中更广泛的非硬件测试套件均通过。
12. 已记录安装、权限、缓存、设备选择、隐私、截止时间语义，以及阶段 1 不包含 TTS。

## 11. 实施后建议运行的第一个命令

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --voice-model tiny.en \
  --voice-device cpu \
  --voice-compute-type int8 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

从 `soft` 模式开始。只有在了解目标麦克风的端点和 ASR p95 数值后，
才应在真实游戏中启用 `hard`/`challenge` 模式。

## 12. 外部实现参考资料

- faster-whisper 官方仓库和 API：
  `https://github.com/SYSTRAN/faster-whisper`
- python-sounddevice 文档：
  `https://python-sounddevice.readthedocs.io/`
- WebRTC VAD 帧约束/参考实现：
  `https://github.com/wiseman/py-webrtcvad`
