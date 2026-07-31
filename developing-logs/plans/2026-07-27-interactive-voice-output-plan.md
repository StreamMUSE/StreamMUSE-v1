# StreamMUSE 交互式语音输出（TTS）计划

日期：2026-07-27

目标分支：`feature/voice`

状态：修订版（rev 9），**软件实施完成，真人/硬件资格验收待完成**。这是 `2026-07-16-interactive-voice-plan.md` 第 7 阶段（可选双向语音）的展开计划。决策记录见 §12。

修订记录：

- rev 9（2026-07-30）：完成阶段 1-5 的软件实现、测试、CLI、锁文件与用户文档。本机验证 `say` 可合成实际游戏短语，`speaker-devices` 可枚举设备，`null` 后端通过真实 PortAudio callback 播放取得 DAC 时间戳并正常 drain。聚焦与既有 voice/runtime 回归 149 项通过；仓库全量为 654 passed、4 skipped、15 failed，15 项均位于未修改的 Lekai/robustness 模块，已作为既有失败单独记录。阶段 0 剩余 espeak/Kokoro 基准与真人盲测，阶段 6 剩余真实双向对局和回声统计。
- rev 8（2026-07-30）：修正 rev 7 引入的一处竞态与三处一致性问题。竞态：`termination_kind` 被 callback 线程与主线程共写，而 §3.5.1 规定音频 callback 不取锁，因此"主线程置 `aborted` → callback 又写回 `normal`"的交错会让被截断的播放拿到 `completed_normally=true`，进而在 `audio_end` 下错误计分（超时 + `warn` 路径下对局还会继续）。改为单向闩锁 `abort_requested` + 两条件相与（§3.5.2），并补竞争序列专项测试。另补：§3.4.2 新增的"kokoro 缺 model/revision 即拒绝"未进 §4.1 拒绝清单，各处"三组"计数同步改为四组；风险表按优先级重排（rev 7 把新增行堆在表首，P0 被 P1/P2 隔断）；§12.1 标题日期补上末次更新。
- rev 7（2026-07-30）：收敛最后一轮 review。统一 §3.1、§3.5、§3.11 与测试中的异常契约；runtime 从 LLM 成功返回后开始保护整个后处理区，并原样重抛 `BaseException`。将 PortAudio 的公共完成信号改名为 `inactive_event`，只有正常 `CallbackStop` 才产生 `playback_drained_offset_ms`；abort/部分播放失败只记录 `stream_inactive_offset_ms` 并回退到 `text`。新增 `artifact_failed` / `internal_error` 状态与保存 WAV 失败语义。盲测规模改为由实际数字集合计算，并清理旧章节引用、过期验收文字。
- rev 6（2026-07-30）：**产品简化——不实现任何音频 cue**。人类回合只保留现有终端文字 prompt；删除 `--speech-cue`、`build_spoken_cue()`、cue 预合成、cue trace 字段及对应测试。回声边界简化为“机器答案完整播放并排空 → 一次 guard → 打印文字提示并开麦”。人类先手且此前没有机器音频时不支付 guard。
- rev 5（2026-07-30）：修正 6 处。两处 P1：`audio_end` 口径在 `empty_text` / `cache_miss_skipped` / `synthesis_failed` / LLM 超时等无音频分支下无定义，且纯墙钟差值可能小于 `text` 口径（§3.6.1 改为两段未舍入实测量相加，§3.6.2 定义回退与 `effective_deadline_basis`）；rev 3/rev 4 的"`speak()` 永不抛异常"契约会把 `SystemExit(2)` 降级成 `KeyboardInterrupt`、把程序缺陷伪装成播放失败（§3.1 改为把"记录先落盘"的保证从输出汇上移到运行时，异常原样重抛）。另修正四处：`say --version` 本机实测不支持，版本溯源改用 `sw_vers`（§3.4.1）；补 `--speech-model` 并要求钉死默认模型与 revision（§3.4.2）；cue 场景固定为一次 drain + 一次 guard（§3.8）；盲测数字条目改为按具体值判分并覆盖 `17/70` 等易混对（§7.1）。
- rev 4（2026-07-30）：修正 PortAudio 播放层的两处会造成实际故障的错误（§3.5.2）。其一，sounddevice 的 `_wrap_callback`（`sounddevice.py:2773`）**丢弃 callback 的返回值**，只识别抛出的控制异常，因此 rev 3 写的"返回 `CallbackAbort`"等价于 `paContinue`，流根本不会停——必须 `raise`。其二，rev 3 让 callback 在送完最后一帧时置位完成事件，那一刻音频还在设备缓冲区里没出声；完成信号必须来自 `finished_callback`（文档保证它在全部音频播完后才触发）。后者会让 `playback_drained_offset_ms` 偏早，进而使 §3.8 的 guard 提前起算、尾音仍在响时就开麦，直接击穿 P0 回声防护，故按 P0 处理。
- rev 3（2026-07-27）：**产品决策——TTS 只朗读 LLM 生成的答案，不播报裁判结果与胜负**，结果仍然打印到终端并写入 log。`--speech-announce` 整个删除。另修正 8 处 review 问题，其中两处是事实错误：rev 2 用 `CallbackStop` 实现"立刻停流"是错的（它会播完缓冲区，应用 `CallbackAbort`/`stream.abort()`）；rev 2 依赖 callback 抛异常回传主线程也是错的（sounddevice 文档明确保证不传播）。
- rev 2（2026-07-27）：根据 review 修正 10 处。manifest 保持 v2 不升版；播放失败不得让已生成的机器回合从 trace 消失；保护间隔由运行时在开麦前统一执行。
- rev 1（2026-07-27）：初稿。

## 1. 目标与范围

在现有语音输入闭环之后补上机器侧的音频输出，使 `streammuse-task play` 形成完整的双向语音对局：

```text
轮到人类 -> 麦克风 -> VAD -> faster-whisper -> Zip-Zap-Zop 严格解析 -> 裁判 -> 关闭采集
轮到机器 -> LLM 文本 -> 朗读文本规范化 -> TTS 合成 -> 扬声器播放 -> 排空 -> 保护间隔
```

范围约束（与输入侧保持一致的原则）：

- 语音输出**默认关闭**。关闭时，`play` 的行为、计时、trace 与 manifest 字段与今天完全一致。
- **TTS 只朗读 LLM 生成的答案**。裁判结果（`OK`/`MISS`）、胜负、横幅、截止时间菜单、`:help` 一律只走终端打印和既有 log，不合成语音。
- 只作用于 `play` 子命令。`run` 子命令与音乐/MIDI 实时栈不受影响。
- 未安装可选 TTS 依赖时，应用仍必须能导入并以文字输出运行。
- 默认不持久化任何合成音频。
- 语音输出与语音输入正交：允许"人打字 + 机器说话"，也允许"人说话 + 机器只打字"。**这条正交性必须体现在依赖分组上**（见 §4.2），只想要机器说话的用户不应被迫安装 STT 依赖。
- 人类回合的提示只使用现有终端文字 prompt；不提供提示音或数字朗读等音频 cue。

明确**不在**本次范围内：

- 结果/胜负播报。这是 rev 3 的产品决策，不是延期项。它同时消除了一整类事务性难题：裁判结果只有在 `_finish_turn()` 完成后才可知，而那时 trace 与 artifact 已经落盘，任何事后播报都会面临"元数据写在哪、播报失败怎么记、要不要回改已落盘记录"的问题。不做，就没有这些问题。
- 声学回声消除（AEC）。用轮次门控 + 保护间隔 + 建议耳机来替代。
- 打断（barge-in）：人类不能在机器播放中途抢答。
- 流式合成（边生成边播）。首版按整句合成后播放。

## 2. 当前状态调研结果

### 2.1 机器回合的现有路径

```text
InteractiveTaskRuntime._run_llm_turn()
  -> terminal.write("[n] LLM thinking... (deadline …)")
  -> task.build_llm_messages()
  -> model_client.generate(...)
  -> terminal.write("    LLM > {text}")
  -> _finish_turn(...)  -> 裁判 -> 状态推进 -> trace -> 产物
```

见 `src/streammuse/application/tasks/interactive_runtime.py:543`。机器侧目前只有 `terminal.write`，没有任何音频出口。

`_finish_turn()`（`interactive_runtime.py:607`）已经有 `human_input_metadata` 这一受控的元数据注入参数，机器侧可以对称地加一个 `speech_output_metadata`。注意它同时承担裁判、状态推进、trace 追加和 artifact 写入，并且带有精细的回滚逻辑（`_append_response_trace` 失败时会 unlink 已写的 artifact，见 `interactive_runtime.py:677-684`）——**不要重构这个函数的职责边界**，改动应当限制在参数通道上。

### 2.2 输入侧已建立的模式（要复用的）

阶段 1-4 已落地并可直接对称照搬：

- 领域契约放 `domain/tasks/models.py`，不依赖任何可选包。
- 应用层配置放 `application/tasks/human_input.py`，冻结 dataclass + `__post_init__` 校验。
- CLI 参数一律先解析为 `None`，再由强类型配置填默认值，从而能区分"用户没传"和"用户显式传了一个恰好等于默认值的值"（`presentation/task/cli.py:373`）。
- 工厂 `application/factories/human_input_factory.py` 延迟导入基础设施。
- 基础设施放 `infrastructure/voice/`，带类型化异常层级和 `_json_safe()`。
- **PortAudio callback 的错误回传模式**：`microphone.py` 用 `callback_errors: queue.Queue[BaseException]` + `_set_callback_error()` / `_raise_callback_error()`，callback 内捕获一切并写队列，主线程负责取出并转换。输出侧必须照搬（原因见 §3.5.1）。
- **产物持久化耗时的排除模式**：`_response_latency_ms()`（`interactive_runtime.py:485-519`）用 `artifact_persistence_ms` + `allow_artifact_exclusion` 把"存 WAV"的耗时从人类延迟里扣掉，并对该值做有限性/上界校验。输出侧保存合成音频要照搬（§5.3）。
- 运行时用 `_shift_voice_stage_offsets()`（`interactive_runtime.py:531`）把输入源的相对计时统一平移到回合原点。
- `play_task()` 用 `ExitStack` + `_preserving_close` 保证获取到的资源一定被关闭一次。

### 2.3 manifest 版本现状

交互式 manifest 当前是 `schema_version: 2`，写入点两处：`interactive_runtime.py:1003` 与 `presentation/task/cli.py:495`。仓库内**存在 4 处硬断言**：

```text
tests/unit/application/tasks/test_interactive_runtime.py:245 / :574 / :980
tests/unit/presentation/task/test_task_cli.py:420
```

rev 1 声称"没有断言 `schema_version == 2` 的消费者"是错误结论，来自一次被 `head` 截断的检索。因此**不升版**，理由见 §5.1。

### 2.4 启动失败与错误文案的现状

- `_best_effort_startup_manifest()`（`presentation/task/cli.py:469`）只接收 `HumanInputConfig`，没有承载语音输出配置的位置。
- CLI 对基础设施错误**统一打印 `voice input error`**（`cli.py:178` 与 `cli.py:551`）。
- `VoiceInfrastructureError` 的 docstring 写的是 "actionable voice-input infrastructure failures"。

三处都需要接线，见 §3.12。

### 2.5 已有的 TTS 基准数据

来自 `docs/developer-guide/voice-microbenchmark-results.md`（10 个 1-3 词短语，不含播放）：

| 环境 | 后端 | setup ms | 首次 ms | 稳态均值 ms | 峰值 RSS MB | 状态 |
|---|---|---:|---:|---:|---:|---|
| Mac 内置 | 系统 `say` | X | 含在 CLI 调用中 | X | 39.8 | OK（全程均值 577.7） |
| Mac Homebrew | `espeak-ng` | X | 80.9 | 27.0 | 5.5 | OK |
| Mac venv | Piper `en_US-amy-low` | — | — | — | — | **失败**：wheel 内 `espeak-ng-data/phontab` 路径写死 |
| Mac venv + Homebrew espeak | Kokoro | 4884.8 | 793.5 | 309.7 | 1631.4 | OK |
| H200 venv 常驻 | Piper `en_US-amy-low` | 1265.2 | 23.4 | 27.0 | 137.3 | OK |
| H200 venv | Kokoro | 2616.8 | 997.4 | 108.8 | 2374.3 | OK |

结论：目标 Mac 上唯一"快且零依赖"的是 `espeak-ng`/`say`，质量最好的 Kokoro 加载要 4.9 秒、单句 310 ms、占 1.6 GB。**Piper 在 Mac 上不可用**，不要设为默认。

`system` 后端的可行性已在目标 Mac 上实测确认：

```bash
printf 'Zip Zap' | say -f - --data-format=LEI16@22050 -o out.wav
# exit=0；mono / 16-bit / 22050 Hz / 18816 frames (0.853 s)；标准库 wave 可直接读取
```

这同时坐实了三件事：文本可走 stdin（不必进命令行）、`say` 可直出标准 WAV、`speech` extra 不需要 `soundfile`。

## 3. 设计决策

### 3.1 新增独立的 `SpeechOutputSink` 协议

不要复用音乐侧的 `OutputSink`，也不要把播放塞进 `TerminalIO`。在 `domain/tasks/models.py` 中与 `HumanResponseSource` 并列新增无依赖契约：

```python
SpeechOutputMode = Literal["silent", "audio"]
SpeechPlaybackStatus = Literal[
    "ok",
    "disabled",                 # 静默实现
    "not_attempted",            # 上游 LLM 未产生可交给 TTS 的文本
    "empty_text",               # 任务判定该回答不应朗读
    "not_renderable",           # 任务不支持朗读（启动期已拦截，这里是兜底）
    "cache_miss_synthesized",   # 词表未命中，回合内实时合成成功
    "cache_miss_skipped",       # 词表未命中，按配置跳过
    "synthesis_failed",
    "playback_failed",
    "artifact_failed",          # 音频已播完，但可选 WAV 持久化失败
    "interrupted",              # KeyboardInterrupt / SystemExit，见 §3.11
    "internal_error",           # 意外程序异常；记录后必须原样重抛
]


@dataclass(frozen=True)
class SpeechRequest:
    turn_id: int
    actor: InteractiveActor
    text: str                      # 已规范化的朗读文本
    source_text: str               # 未加工的原始回答，仅用于溯源


@dataclass(frozen=True)
class SpeechPlayback:
    status: SpeechPlaybackStatus = "ok"
    spoken_text: str = ""
    cached: bool = False
    synthesis_ms: float = 0.0
    audio_duration_ms: float = 0.0
    completed_normally: bool = False
    # 以下偏移量的原点一律是 speak() 入口。运行时负责平移到回合原点，见 §3.10。
    playback_start_offset_ms: float | None = None
    first_dac_sample_offset_ms: float | None = None
    playback_drained_offset_ms: float | None = None
    stream_inactive_offset_ms: float | None = None
    audio_artifact: str | None = None
    artifact_persistence_ms: float = 0.0
    error: dict[str, str] | None = None      # {"type","message"}，失败时必填且 JSON 安全
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeechOutputSink(Protocol):
    mode: SpeechOutputMode

    def start(self) -> None: ...
    @property
    def provenance(self) -> dict[str, Any]: ...
    def prepare(self, phrases: tuple[str, ...]) -> None: ...   # 游戏计时器之前的预合成
    def speak(self, request: SpeechRequest) -> SpeechPlayback: ...
    def drain(self) -> None: ...                               # 确保所有已提交音频播完
    def close(self) -> None: ...
```

**`speak()` 的异常契约（rev 5 重写）：**

rev 3 / rev 4 把契约写成"`speak()` 永远返回、绝不抛异常"，本意是保证"失败回合不得消失"（§3.11）。但那样会连带吃掉两类不该被吃掉的异常：`SystemExit(2)` 被降级成 `KeyboardInterrupt`（退出码与异常身份全丢），`TypeError` / `AssertionError` 这类程序缺陷被伪装成普通的 `playback_failed`，在 `warn` 模式下甚至静默继续。

正确做法是把保证从**输出汇**上移到**运行时**：不变式是"记录一定先落盘"，而不是"异常一定不发生"。

- `speak()` 只把**预期内的**失败转成 `SpeechPlayback` 状态：合成失败、设备失败、播放失败、子进程超时、可选 WAV 持久化失败。
- 其余一切原样向上传播，**但 `speak()` 必须在 `finally` 里中止音频流**，绝不能让异常逃逸时还有音频在放。
- **运行时**从 `model_client.generate()` 成功返回的下一条语句开始，用 `try / except BaseException` 包住终端打印、`build_spoken_text()`、`speak()` 和元数据整理。无论后处理正常返回还是抛出：先构造 `speech_output_metadata` 并调用 `_finish_turn()` 把回合记录写完，再按下表分派。这样 `build_spoken_text()` 自身的程序缺陷或这一小段中的用户中断也不会让已发生的模型调用消失。

| `speak()` 的退出方式 | 回合记录 | 之后 |
|---|---|---|
| 返回 `ok` / `cache_miss_*` / `empty_text` | 正常写入 | 继续 |
| 返回 `synthesis_failed` / `playback_failed` / `artifact_failed` | 写入，`error` 非空 | `fail` 抛 `SpeechOutputError`；`warn` 继续 |
| 抛 `KeyboardInterrupt` | 写入，`status="interrupted"` | **重新抛出原异常对象** |
| 抛 `SystemExit` | 写入，`status="interrupted"` | **重新抛出原异常对象，退出码保持不变** |
| 抛其他任何异常（程序缺陷） | 写入，`status="internal_error"`，`error` 记原类型 | **重新抛出原异常对象，不受 `warn` 影响** |

要点：异常分支在同一个 `except` 中完成 `_finish_turn()` 后使用裸 `raise`。**重新抛出的必须是原异常对象本身**，不是同类型的新实例，不包装成新的 `SpeechOutputError`，更不是统一换成 `KeyboardInterrupt`。`SystemExit` 的退出码、程序缺陷的类型与 traceback 必须原样保留。只有已经被 `speak()` 明确识别为预期基础设施失败并返回状态的分支，才受 `--speech-on-error` 控制。

`start()` / `prepare()` 的启动期失败一律直接抛（那时还没有任何回合，无记录可写）。

`SpeechRequest` **不含 `deadline_s`**：播放一旦开始就不因截止时间中断（§3.6），带一个永不生效的字段只会误导实现者。

默认实现 `SilentSpeechOutput`（`mode="silent"`）：`speak()` 返回 `status="disabled"`，且运行时在该模式下**完全不写** `speech_output` 元数据键（§5）。

构造函数无副作用；`start()`/`close()` 幂等；`start()` 部分失败后仍可 `close()`。

### 3.2 朗读文本属于任务，不属于基础设施

`"ZipZap"` 直接丢给 TTS 会被读成一个不存在的单词。拆词、数字读法、以及"什么答案不该朗读"都是任务语义。对称 `SpeechAwareInteractiveTask`，新增：

```python
@runtime_checkable
class SpeechRenderableTask(Protocol):
    def build_spoken_text(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
        response_text: str,
        *,
        actor: InteractiveActor,
    ) -> str | None: ...

    def speech_vocabulary(
        self,
        state: TaskState,
        *,
        max_turns: int,
    ) -> tuple[str, ...]: ...
```

`ZipZapZopTask` 的实现：

- `build_spoken_text("ZipZap")` → `"Zip Zap"`；`"ZipZapZop"` → `"Zip Zap Zop"`；`"17"` → `"17"`。
- 空字符串、超时空答案、无法解析的答案 → 返回 `None`，运行时记 `status="empty_text"` 并跳过播放。不要为错误答案编造朗读内容。
- **不做任何纠正**：LLM 答错了就照着错的念。朗读层绝不能把 `"16"` 修成 `"Zap"`，否则人类听到的和裁判判的不是同一个东西。
- `speech_vocabulary()` 返回本局可能出现的有界词表：7 个游戏词组合 + `[start_number, start_number + max_turns)` 区间内所有非游戏词数字。

不满足 `SpeechRenderableTask` 的任务请求 `--speech-output audio` 时，在启动阶段就以 unsupported-task 报错，与 `_validate_human_input_task()` 的做法一致。

### 3.3 预合成缓存

Zip-Zap-Zop 的机器输出词表**有界且可提前枚举**。因此在 `start()`/`prepare()` 阶段（与 whisper 预热同一时刻，游戏计时器启动之前）把整张词表一次性合成为内存 PCM。

带来的直接后果：

- 回合内合成延迟降为 0，只剩播放时间，`synthesis_ms == 0.0`。
- Kokoro 的 4.9 秒加载和 310 ms/句全部落在游戏之外，后端选择不再是 P0 阻塞项——可以按**音质**而不是按延迟来选后端。
- 未命中时的策略可配置：`synthesize`（默认，回合内实时合成，记 `cache_miss_synthesized`）或 `skip`（不播放，记 `cache_miss_skipped`）。

缓存**双重有界**，两个上限都能从 CLI 配：`--speech-cache-max-entries`（默认 512）与 `--speech-cache-max-bytes`（默认 64 MiB）。任一上限触发即停止预合成，剩余词条退化为运行时实时合成。词表按上限截断的事实与实际预合成条数一并写入 manifest。

预合成时长单独计量并写进 manifest（`prewarm_ms`、`prewarm_entry_count`、`prewarm_truncated`），绝不计入任何回合延迟。

### 3.4 后端选择：分层适配器，默认 `system`

```python
class SpeechSynthesizer(Protocol):
    def start(self) -> None: ...
    @property
    def provenance(self) -> dict[str, Any]: ...
    def synthesize(self, text: str) -> SynthesizedAudio: ...   # float32 单声道 + sample_rate_hz
    def close(self) -> None: ...
```

| 后端 | 实现 | 建议 |
|---|---|---|
| `system` | macOS `say`，Linux 回退 `espeak-ng` | **v1 默认**，已在目标 Mac 实测（§2.5） |
| `espeak_ng` | 直接调 `espeak-ng` | 速度地板/跨平台一致基线 |
| `kokoro` | Python API，需 Homebrew `espeak-ng` | 可选高音质，独立 extra |
| `null` | 生成静音 PCM | 测试与 CI |

Piper 保留在协议后面但 v1 不实现（Mac wheel 路径缺陷）。所有后端延迟导入，缺失时抛可操作的安装提示。

#### 3.4.1 子进程后端的硬性要求

朗读文本来自 LLM，是**不可信输入**。`system` / `espeak_ng` 后端必须：

- `subprocess.run(..., shell=False)`，参数用列表传递；
- **文本通过 stdin 传入**（`say -f -`、`espeak-ng --stdin`），不作为命令行参数——否则以 `-` 开头的模型输出会被解释成选项；
- 强制 `--speech-synthesis-timeout-s`（默认 10 s），超时记 `synthesis_failed`；
- `KeyboardInterrupt` 时终止子进程并回收，不留孤儿；
- 临时音频文件在**所有**异常路径上删除（`try/finally` + `missing_ok=True`）；
- 显式解析可执行文件绝对路径（`shutil.which`）并写进 provenance。**版本溯源按后端分别处理，不要统一依赖 `--version`**——本机实测 `say --version` 返回 ``say: unrecognized option `--version'``：

| 后端 | provenance 记录 |
|---|---|
| macOS `say` | 可执行文件绝对路径；`sw_vers` 的 `ProductVersion` + `BuildVersion`（本机实测 15.1 / 24B83）；选定音色名（`say -v '?'` 可枚举，用于校验音色存在） |
| `espeak-ng` | 可执行文件绝对路径；`espeak-ng --version` 输出 |

`say` 没有独立版本号，它的行为由系统版本决定，所以 `sw_vers` 才是这个后端真正的版本溯源锚点。

`say` 用 `--data-format=LEI16@22050 -o out.wav` 直出 WAV，标准库 `wave` 读取，与 `response_source.py:25` 写 WAV 的方式对称。因此 `system` 后端**不需要 `soundfile`**。

#### 3.4.2 Kokoro 的可复现性要求

只锁 Python 包版本锁不住模型资源。对称照搬 STT 侧已有的开关，**包括模型标识本身**：

```text
--speech-model <repo-or-path>          # 缺失会导致模型不可复现，见下
--speech-model-cache <path>
--speech-model-revision <commit-or-tag>
--speech-local-files-only
```

rev 3/rev 4 只写了"provenance 记录模型 ID"，却没有任何地方**指定**模型 ID，也没有钉死默认值——等于要求记录一个无人决定的值。STT 侧的对应物是 `--voice-model`，默认 `tiny.en`（`cli.py:391`），必须对称补上。

- 阶段 0 尚未选定默认快照之前，`SpeechOutputConfig` 中 Kokoro 的 model/revision 默认都为 `None`；选择 `--speech-backend kokoro` 时若没有同时显式提供 model 与 revision，就在启动校验中拒绝。**绝不临时回退到上游 latest。**
- 阶段 0 选定 Kokoro 后，再把模型 ID 与**默认 revision**一起钉进 `SpeechOutputConfig` 并写回本计划。这样阶段 1 可以先实现契约，又不会为了“阶段互不阻塞”引入不可复现默认值。
- provenance 记录解析后的模型 ID、revision、快照路径、voice asset 名称与其内容 hash。
- 模型缺失、离线缓存未命中都是**启动错误**。
- 非 `kokoro` 后端传 `--speech-model` 一律拒绝（§4.1）。

#### 3.4.3 `--speech-rate` 的统一语义

`say -r` 与 `espeak-ng -s` 的单位是词/分钟，Kokoro 用的是速度倍率，同一个参数三种含义会让配置不可移植。

约定：**`--speech-rate` 是相对该后端默认语速的倍率，`1.0` 表示后端默认**。各适配器负责映射到自己的原生单位（例如 `say` 的默认约 175 wpm，`1.2` → `-r 210`），并在 provenance 中**同时记录倍率与解析出的原生值及其单位**。校验在 `SpeechOutputConfig` 中做（有限、`> 0`、合理上下界）。

### 3.5 播放：callback 模式 + 真实 DAC 时间戳

`sounddevice` 用于播放，但**不使用阻塞 `write()`**。

原因：阻塞 `OutputStream.write()` 返回只表示数据被底层缓冲区消费，不表示扬声器已发声。用它做时间戳，"首个可听样本"这个指标就是假的。sounddevice 只在 **callback 模式**下通过 `time_info.outputBufferDacTime` 提供真实的 DAC 播出时刻。

这套机制仓库里已经有了：`microphone.py:108` 的 `_PortAudioClockMapper` 正在用输入侧的 `inputBufferAdcTime` 做"PortAudio 流时钟 → 本地单调时钟"的映射。`outputBufferDacTime` 完全对称，把这个类泛化后输出侧直接复用（§6）。

- `sd.OutputStream(callback=...)` 单声道 float32，采样率跟随合成结果；必要时重采样到设备支持率。
- callback 只从预填好的缓冲区拷贝数据并记录首帧的 `outputBufferDacTime`，不做合成、不做文件 I/O——与麦克风 callback 的约束一致。
- `first_dac_sample_offset_ms` 由 DAC 时间戳经时钟映射得到，字段名如实反映它是**估计的**首个可听样本时刻。
- `playback_drained_offset_ms` 在流真正播完后记录，不是提交完成的时刻。
- 设备通过 `--speaker-device <name-or-index>` 选择，启动预检解析并写入 provenance。

**保护间隔不在这里。** `speak()` 只负责"播完并排空"，`speech_guard_ms` 由运行时在开麦前统一执行，见 §3.8。

#### 3.5.1 callback 的错误回传（rev 3 修正）

rev 2 计划让 callback 抛异常来报告失败，这是错的。sounddevice 0.5.5 文档明确：

```text
If another exception is raised, its traceback is printed to `sys.stderr`.
Exceptions are *not* propagated to the main thread, i.e. the main Python
program keeps running as if nothing had happened.
                            —— sounddevice.py:1774-1784
```

也就是说，依赖普通异常的话主线程既拿不到失败信息，也无法返回 `playback_failed`。必须照搬麦克风侧已经验证过的模式：

```text
callback_errors: queue.Queue[BaseException]   # 复用 _set_callback_error / _raise_callback_error
frame_cursor: int                              # 只由音频 callback 线程推进
```

- callback 内 `try/except Exception`，捕获后把**原异常对象**写入 `callback_errors`，然后 `raise CallbackAbort`。普通异常不能直接逃逸出 callback，因为 sounddevice 不会把它传播到主线程。
- PortAudio status、设备断开等预期错误先包装成 `SpeakerPlaybackError` 再入队；`TypeError` / `AssertionError` 等未知异常保持原类型入队。
- 主线程醒来后先调用 `_raise_callback_error()`：预期的 `SpeakerPlaybackError` 由 `speak()` 转成 `playback_failed`，未知异常继续向 runtime 抛出并按 §3.1 记为 `internal_error`。
- 音频 callback 不获取 `threading.Lock`、不做文件 I/O、不调用 PortAudio 控制 API。帧游标由 callback 独占；主线程只在 `inactive_event` 建立 happens-before 边界后读取最终状态。

#### 3.5.2 停止必须用 `raise`，完成信号必须来自 `finished_callback`（rev 4 修正）

rev 3 在这里写错了两处，都会造成实际故障。

**其一：sounddevice 丢弃 callback 的返回值。**

```python
# sounddevice.py:2773-2782
def _wrap_callback(callback, *args):
    args = args[:-1] + (CallbackFlags(args[-1]),)
    try:
        callback(*args)          # ← 返回值没有被接收
    except CallbackStop:
        return _lib.paComplete
    except CallbackAbort:
        return _lib.paAbort
    return _lib.paContinue
```

只有**抛出**的控制异常会被翻译成 `paComplete` / `paAbort`；`return CallbackAbort` 等价于 `paContinue`，流根本不会停。因此一律写 `raise sd.CallbackStop` / `raise sd.CallbackAbort`。

**其二：不能在"最后一帧提交给 PortAudio"时判定播放结束。** 那一刻音频还在设备缓冲区里没出声。完成信号必须来自 `finished_callback`，文档对此有明确保证：

```text
For a stream providing audio output, if the stream callback raises
`CallbackStop`, or `stop()` is called, the stream finished callback
will not be called until all generated sample data has been played.
                            —— sounddevice.py:1820-1825
```

sounddevice 自己的 `sd.play()` / `sd.wait()` 就是这么实现的——`_CallbackContext.finished_callback`（`sounddevice.py:2650`）的主体就是 `self.event.set()`。

因此公共完成信号定为：

```text
inactive_event:  threading.Event   # 仅由 finished_callback 置位
abort_requested: bool              # 单向闩锁，只由主线程从 False 置 True，永不复位
termination_kind: Literal["running", "normal", "callback_error"]
                                   # 只由 callback 线程写，初始值为 running
```

- 送完最后一帧时 callback 先把 `termination_kind` 设为 `normal`，再 `raise CallbackStop`，**但不置位任何事件**。
- callback 错误设为 `callback_error` 并 `raise CallbackAbort`。这两个值都只在 callback 线程内写，彼此不竞争。
- 主线程要中止时（`BaseException` 或等待超时），先把 `abort_requested` 置 `True`，再调 `stream.abort()`。
- `finished_callback` 是 `inactive_event` 的唯一置位方：记录 `stream_inactive_offset_ms` 并 `inactive_event.set()`。它由 PortAudio 线程调用，只取一次单调时钟并置位，不做 I/O、不做合成。
- **完整播放的判据必须同时满足两条**：

  ```text
  completed_normally = (termination_kind == "normal") and (abort_requested is False)
  ```

  只有它为真时才写 `completed_normally=true` 并把 `finished_callback` 的时刻同时写入 `playback_drained_offset_ms`；否则 `completed_normally=false` 且 `playback_drained_offset_ms=None`，只保留 `stream_inactive_offset_ms`。
- 主线程等待 `inactive_event`，安全上限 = 音频时长 + 设备输出延迟（`stream.latency`）+ 余量。上限本身要覆盖输出延迟，否则会在正常播放尾段误判超时；等待超时后置 `abort_requested` 并 abort，返回 `playback_failed`。

**为什么要拆成两个变量而不是让主线程也写 `termination_kind`（rev 8 修正）**：§3.5.1 规定音频 callback 不取锁，所以任何两线程共写的字段都是"最后写入者胜"。考虑这个序列——callback 刚喂完最后一帧、置 `normal`、抛 `CallbackStop`，PortAudio 开始排空设备缓冲区（可达数十毫秒）；此时主线程超时，先置 `aborted` **再**调 `stream.abort()`，而在这两条语句之间 callback 又被调度了一次并把值改回 `normal`。结果是一次被截断的播放拿到 `completed_normally=true` 和非空的 `playback_drained_offset_ms`。

后果不只是元数据难看。§3.6.2 用 `full_playback_completed` 决定是否回退，这个假阳性会让**被截断的播放照常按 `audio_end` 计分**；而超时路径在 `--speech-on-error warn` 下对局还要继续，错误会留在数据里。用单向闩锁 + 两个条件相与就没有这个竞争：`abort_requested` 一旦为真永不复位，callback 无论何时写 `normal` 都无法推翻它。

这样既能用一个事件结束三条等待路径，又不会把“已经停止”误记成“完整播完”。如果在最后一帧提交时过早置位，§3.8 的 guard 会提前起算；如果把 abort 的 finished callback 误记为 drained，`audio_end` 又会错误地把残缺播放当成完整答案。

#### 3.5.3 中断要用 abort 而不是 stop（rev 3 修正）

rev 2 写"callback 用 `CallbackStop` 实现立刻停流"，两处都错了：

1. `CallbackStop` 的语义是 **播完已生成的缓冲区再停**（`paComplete`），不是立刻停。要立刻停必须 `raise CallbackAbort`（`paAbort`）或主线程调 `stream.abort()`——文档原文："If `CallbackAbort` is raised, the stream will finish as soon as possible. If `CallbackStop` is raised, the stream will continue until all buffers generated by the callback have been played."
2. `KeyboardInterrupt` 由 Python 投递给**主线程**，不投递给 PortAudio callback 线程。"callback 检查中断标志"这个描述本身就搞错了方向。

正确流程（主线程视角）：

```text
speak() 主线程在 inactive_event.wait() 上阻塞
  -> 捕获 KeyboardInterrupt / SystemExit / 其他 BaseException
  -> stream.abort()（丢弃未播缓冲区）
  -> best effort 等待 finished_callback 置位 inactive_event
  -> speak() 使用裸 raise 原样向上传播
  -> runtime 构造 interrupted/internal_error 元数据并调用 _finish_turn()
  -> runtime 再次裸 raise 原异常；KeyboardInterrupt 进入既有 user_interrupt 收尾路径
```

**`interrupted` / `internal_error` 不受 `--speech-on-error warn` 控制**：原异常永远重新抛出；`warn` 只对 `synthesis_failed` / `playback_failed` / `artifact_failed` 这三类预期运行时失败生效。

#### 3.5.4 失败策略

- **启动预检失败**（无设备、无法打开、后端不可用、模型缺失）= 启动错误，写 `status="startup_error"` manifest，不产生任何回合。
- **回合内合成、播放或可选音频持久化失败**：`speak()` 返回对应失败态。运行时按 `--speech-on-error {fail,warn}` 处理，两种情况下**回合记录都必须先落盘**（§3.11）。默认 `fail`。

### 3.6 截止时间语义：默认不变，但把口径显式化

现状：`_run_llm_turn` 用 `max(model_response.latency_ms, wall)`，即**文本就绪**时刻。人类侧的口径是"等待语音 + 说话时长 + 端点静音 + ASR"，即答案文本可判定时刻——它**包含了人类把答案说出口的时间**。严格对称的话机器侧也该包含播放时间，但改默认会让既有数据不可比。

方案：`--llm-deadline-basis {text,audio_end}`，**默认 `text`**。

- `text`：计分口径完全不变，播放时间只作为诊断字段。
- `audio_end`：见下面的 §3.6.1 公式与 §3.6.2 无音频回退。
- 无论选哪个，`text_ready_offset_ms` / `synthesis_ms` / `playback_start_offset_ms` / `first_dac_sample_offset_ms` / `playback_drained_offset_ms` / `stream_inactive_offset_ms` 都按实际可用性写进 trace，两种口径事后都能复原。
- 配置口径写进 manifest；**每回合实际生效的口径**写进该回合 trace 的 `effective_deadline_basis`（§3.6.2）。
- `--speech-output off` 配 `--llm-deadline-basis audio_end` **直接拒绝并退出 2**：整局都没有音频时该口径无定义，静默回退会让用户误以为自己在测对称口径。

#### 3.6.1 `audio_end` 的公式（rev 5 修正）

rev 3/rev 4 写的是 `playback_drained_s - llm_turn_start_s` 这一个原始墙钟差值。问题在于文本口径**不是**纯墙钟：

```python
# interactive_runtime.py:588
elapsed_ms = max(float(model_response.latency_ms), (self._now() - start_s) * 1000.0)
```

这个 `max()` 说明运行时把 `model_response.latency_ms` 当成**不可信的上报值**——`LocalChatModel` 是 Protocol（`domain/tasks/models.py:114`），任何实现都可以注入。当前具体实现 `LocalChatModelClient` 在 HTTP 调用外侧自测（`local_chat_client.py:70`），而 runtime 的 `start_s` 更早，所以这一个实现推不出 `reported > wall`；但契约不能建立在某个实现的巧合上。一旦某个实现上报了更大的值，纯墙钟的 `drained - start` 就会**小于** `text` 口径，得出"念完比没念还快"的荒谬结果。

因此固定为两段**未舍入**实测量相加：

```text
text_ready_ms  = max(reported_model_ms, measured_text_wall_ms)
audio_end_ms   = text_ready_ms + measured_text_to_drained_ms
                 # measured_text_to_drained_ms = playback_drained_s - text_ready_s
```

这保证 `audio_end_ms ≥ text_ready_ms` 恒成立。注意这与 rev 2 立下的"不要累加毫秒字段"并不冲突：**禁止的是累加 trace 里已经舍入过的展示字段**（`synthesis_ms + audio_duration_ms + …`），这里相加的是两个未经舍入的原始实测量。舍入只在写 trace 时发生一次。

#### 3.6.2 没有完整音频时的回退（rev 7 收紧）

回退条件不按状态名枚举，而按一个可验证的不变式判断：

```text
full_playback_completed =
    completed_normally is True
    and
    playback_drained_offset_ms is not None
```

只有 `full_playback_completed` 才表示人类完整听到了机器答案。`empty_text`、`cache_miss_skipped`、`synthesis_failed`、LLM 超时、出声前失败、播放一半后 abort、`interrupted`、`internal_error` 等路径都没有正常 drain，必须回退。相反，`artifact_failed` 发生在完整播放之后，仍然保留正常 drain，因此它的计分口径仍可使用 `audio_end`。

当请求口径是 `audio_end` 但 `full_playback_completed == false` 时：

- 该回合的延迟**回退到 `text` 口径**。
- 该回合 trace 写 `effective_deadline_basis: "text"`，同时 `deadline_basis: "audio_end"` 保留配置值。
- 另写 `deadline_basis_fallback_reason`，取稳定枚举值，例如 `empty_text`、`cache_miss_skipped`、`synthesis_failed`、`playback_incomplete`、`interrupted`、`internal_error`、`llm_timeout`。
- 运行摘要额外报告 `audio_end_fallback_turn_count`，并按 fallback reason 分组计数。

这样分析端可以显式区分"真的按 audio_end 计的回合"和"回退的回合"，而不是拿到一列口径混杂却无从分辨的延迟。落到实验解释上：如果回退回合占比很高，那这一局的 `audio_end` 结论本身就不成立，这个比例必须可见。

播放一旦开始不因截止时间中断（打断会让人类听到半截答案）：播放照常完成，事后判定超时。这与输入侧"超时是计分而非执行上界"的既有结论一致。合成音频的持久化耗时**不计入** `audio_end`（§5.3）。

### 3.7 人类回合只使用文字提示

不实现任何音频 cue，也不新增 cue 配置。人类回合继续使用现有 `build_human_prompt()` 和终端输出，例如 `ZipZapZopTask` 显示 `"17:"`。

- terminal 与 voice 输入都显示同一份文字 prompt。
- 文字 prompt 保持现有计时语义：打印耗时 `prompt_elapsed_ms` 继续计入人类回合延迟，不新增排除字段。
- TTS 不朗读当前数字、不播放提示音，也不为人类回合生成或缓存任何音频。

### 3.8 回声防护：一条统一的不变式

保护间隔必须位于**最后一段机器音频**之后，且**只在真的有麦克风要开**的时候才付。

```text
若下一个人类回合的 human_response_source.mode == "voice"
且此前实际播放过机器答案：
    机器答案 speak() 完整播放并排空
      -> sink.drain()          # 幂等兜底
      -> speech_guard_ms       # 一次
      -> 人类计时开始 + 打印文字 prompt + 开麦
若是人类先手，或此前没有实际播放机器音频：
    不付 guard，直接开始人类计时 + 打印文字 prompt + 开麦
否则（terminal 输入）：
    不 drain 等待、不付 guard，直接进入人类回合
```

要点：

- `speech_guard_ms` 由**运行时**在开麦边界执行一次，不由 `speak()` 自己 sleep。
- guard 只防护实际发生过的机器播放；运行时以该机器回合 `first_dac_sample_offset_ms is not None` 作为保守判据。人类先手、`empty_text`、`cache_miss_skipped`、出声前失败等没有扬声器音频的边界不等待；已经提交首个 DAC buffer 后才失败则仍支付 guard。
- **terminal 输入不付 guard**：没有麦克风就没有回声通路，每回合白等 200 ms 是纯损失。
- 由于 rev 3 取消了结果播报，一个机器回合内只有"答案"一段音频，不变式因此非常简单——不再存在"guard 之后又冒出一段音频"的风险。
- `drain()` 是协议方法，静默实现是 no-op，终端模式下这段编排零成本。
- 轮次门控已经存在：麦克风只在人类回合打开。本不变式在它之前再加一道。
- provenance 记录扬声器与麦克风设备，若指向同一物理设备（或均为内置）则打印一次外放风险提示，建议耳机。

### 3.9 保持运行时同步

不引入 asyncio、不引入通用线程池。合成留在当前回合内，PortAudio callback 线程仅用于填充缓冲区。若日后要与 `RealTimeMusicService` 同时跑，先基准 CPU 争用，再考虑进程边界。

### 3.10 时间偏移量的统一原点

- `SpeechPlayback` 内所有 `*_offset_ms` 的原点是 **`speak()` 入口**。
- 写入 trace 的所有 offset 原点是 **`_run_llm_turn` 的 `start_s`**（人类回合则是人类回合原点）。
- 运行时实现一个与 `_shift_voice_stage_offsets()`（`interactive_runtime.py:531`）对称的 `_shift_speech_offsets()`，同样跳过 `None` 与非有限值。
- `audio_end` 口径按 §3.6.1 的公式用未舍入的原始实测量计算，**不走这些平移并舍入后的 trace 字段**。平移只服务于 trace 的可读性与事后复原，不参与计分。

### 3.11 播放失败与中断都不得让回合消失

**问题**：LLM 已返回答案后，若朗读文本构造、合成、播放、音频持久化发生失败，或用户在 `_finish_turn()` 之前中断，这次真实发生过的模型调用就不会进入裁判、`response_trace.jsonl` 或 turn artifact——数据点凭空消失，而这恰是最需要被记录的异常样本。

**解法**（不重构 `_finish_turn` 的职责）：

```text
model_client.generate()
  -> 成功返回文本，记 text_ready_s / text_ready_ms
  -> 从这里开始进入 post-generation 保护区：
       try:
         terminal.write(LLM 原文)
         task.build_spoken_text()
         sink.speak()
       except BaseException as exc:
         构造 interrupted/internal_error 的 SpeechPlayback 占位记录
         计算 effective_deadline_basis
         _finish_turn(..., speech_output_metadata=...)
         裁判 + 状态推进 + trace + artifact 原子写入
         裸 raise 原异常
       else:
         计算 effective_deadline_basis
         _finish_turn(..., speech_output_metadata=...)
         status in {synthesis_failed, playback_failed, artifact_failed}
         and --speech-on-error == fail
           -> 抛 SpeechOutputError
         否则
           -> 继续下一回合
```

LLM timeout 分支没有成功返回模型文本，继续沿用现有 `_finish_turn(model_error=...)` 路径；启用语音输出时构造 `status="not_attempted"` 的 `speech_output_metadata`，启用 `audio_end` 时再写 `effective_deadline_basis="text"` 与 `deadline_basis_fallback_reason="llm_timeout"`，不调用 `speak()`。

这样：预期失败、中断，以及 `build_spoken_text()` / 播放层的程序异常都会留下含 LLM 原文、裁判结果和 `speech_output.status` / `.error` 的完整记录；`fail`、`user_interrupt`、`SystemExit` 与程序缺陷的异常身份都不变。唯一无法承诺“记录一定成功”的情况是 `_finish_turn()` 自身的持久化失败，此时沿用它已有的原子写入与回滚语义，不吞掉持久化异常。

### 3.12 启动失败 manifest 与错误文案接线

§2.4 列的三处现状都要改：

- `_best_effort_startup_manifest()` 增加 `speech_config: SpeechOutputConfig | None` 与可选的 `speech_output_sink` provenance 参数；语音输出工厂在 runtime 构造前失败时，manifest 里要能看到请求的 backend、device、model revision。
- CLI 不能再对所有基础设施错误统一打印 `voice input error`。按异常子类分派：输入/识别类打 `voice input error`，合成/播放类打 `speech output error`。
- `VoiceInfrastructureError` 的 docstring 从 "voice-input" 改为覆盖输入与输出。

关于基类命名：保留 `VoiceInfrastructureError`。它已经在 5 个文件、17 处被引用（含测试与 `scripts/qualify_voice_input.py`），而 §6 已经决定把 TTS 放进 `infrastructure/voice/` 包，"voice" 在本仓库本来就是覆盖双向的包名。放宽其 docstring，在其下新增 `SpeechOutputError` / `SpeechSynthesisError` / `SpeakerDeviceError` / `SpeakerPlaybackError` / `SpeechArtifactError` 子树，用户可见文案靠子类分派。该决定已记录在 §12。

## 4. CLI 与依赖接口

### 4.1 参数

```text
--speech-output {off,audio}            # 默认 off
--speech-backend {system,espeak_ng,kokoro,null}   # 默认 system
--speech-voice <name>
--speech-rate <multiplier>             # 相对后端默认语速的倍率，默认 1.0（§3.4.3）
--speaker-device <name-or-index>
--speech-model <repo-or-path>          # 仅 kokoro；阶段 0 前无隐式默认（§3.4.2）
--speech-model-cache <path>            # 仅 kokoro
--speech-model-revision <commit-or-tag># 仅 kokoro
--speech-local-files-only              # 仅 kokoro
--speech-synthesis-timeout-s 10.0
--speech-prewarm / --no-speech-prewarm # 默认 prewarm
--speech-cache-miss {synthesize,skip}  # 默认 synthesize
--speech-cache-max-entries 512
--speech-cache-max-bytes 67108864
--speech-guard-ms 200
--speech-on-error {fail,warn}          # 默认 fail
--speech-save-audio                    # 默认 false
--llm-deadline-basis {text,audio_end}  # 默认 text
```

（rev 3 删除 `--speech-announce`。）

**所有上述参数的 argparse 默认值必须是 `None`（或 `argparse.SUPPRESS`），真正的默认值由冻结的 `SpeechOutputConfig` 填充。** 否则无法区分"用户没传"和"用户显式传了一个恰好等于默认值的值"，下面的校验就做不了。这与 `cli.py:373` 处理 voice 参数的既有模式一致。

必须拒绝的组合（退出码 2，信息可操作）：

- `--speech-output off` + 任何显式传入的 `--speech-*` 参数；
- `--speech-output off` + `--llm-deadline-basis audio_end`；
- 非 `kokoro` 后端 + 任何 `--speech-model` / `--speech-model-*` / `--speech-local-files-only`；
- `--speech-backend kokoro` 但未同时显式提供 `--speech-model` 与 `--speech-model-revision`（§3.4.2：阶段 0 钉死默认快照之前，绝不允许隐式回退到上游 latest）。这条在阶段 0 选定默认值后即可放宽为"使用已钉死的默认值"。

新增 `streammuse-task speaker-devices` 子命令，与 `voice-devices` 对称：不要求 `--task`，不构造 LLM 客户端，不加载 TTS 模型。`main()` 在触碰 `args.task` 之前分派。

### 4.2 依赖分组

§1 声称输入输出正交，依赖分组必须兑现。播放需要的 `sounddevice` 目前只在 `voice` extra 里，会导致"人打字 + 机器说话"被迫安装 `faster-whisper` 和 `webrtcvad-wheels`。拆成：

```toml
[project.optional-dependencies]
speech = ["sounddevice>=0.5,<0.6"]                    # 播放，system/espeak_ng/null 后端只需这个
voice  = ["faster-whisper>=1.2,<1.3",
          "onnxruntime<1.24; python_version < '3.11'",
          "sounddevice>=0.5,<0.6",
          "webrtcvad-wheels>=2.0,<2.1"]               # STT，保持现状
tts-kokoro = ["kokoro>=0.9,<1.0",
              "soundfile>=0.12",
              "sounddevice>=0.5,<0.6"]
```

| 使用场景 | 需要的 extra |
|---|---|
| 人说话 + 机器打字（现状） | `voice` |
| 人打字 + 机器说话（`system`/`espeak_ng`） | `speech` |
| 完整双向语音（`system`/`espeak_ng`） | `voice,speech` |
| 完整双向语音（Kokoro 音质） | `voice,tts-kokoro` |

`speech` 里**不含 `soundfile`**：`system` 后端按 §2.5 实测直出 WAV 并用标准库 `wave` 读取。重新生成 `uv.lock`。

## 5. Trace 与产物变更

### 5.1 manifest 保持 schema_version 2

**不升 v3。** §1 要求"关闭时字段与今天完全一致"，而升版会让每一次默认运行的 manifest 都变化，两条要求直接冲突；同时 §2.3 那 4 处断言覆盖的是与语音输出无关的路径，改它们等于用测试改动迁就一个不必要的版本跃迁。

- `schema_version` 保持 `2`。
- `speech_output` 作为**可选新增对象**，**仅在 `--speech-output audio` 时出现**；关闭时 manifest 与今天逐字段相同。
- 内容：模式、后端、可执行文件路径与版本（或模型 ID/revision/快照/voice hash）、音色、`rate` 倍率与解析出的原生值及单位、扬声器设备、采样率、`prewarm_ms` / `prewarm_entry_count` / `prewarm_truncated`、`guard_ms`、cache 策略与上限、`llm_deadline_basis`、`on_error`、是否保存音频。
- 下游读取方按 key 存在与否处理，无需版本判断。

### 5.2 回合 trace

机器回合的 `metadata` 下新增唯一允许的键 `speech_output`，同样**仅在启用时出现**：

```json
{
  "metadata": {
    "speech_output": {
      "mode": "audio",
      "backend": "system",
      "status": "ok",
      "source_text": "ZipZap",
      "spoken_text": "Zip Zap",
      "cached": true,
      "synthesis_ms": 0.0,
      "audio_duration_ms": 853.0,
      "completed_normally": true,
      "text_ready_offset_ms": 412.0,
      "playback_start_offset_ms": 424.0,
      "first_dac_sample_offset_ms": 448.0,
      "playback_drained_offset_ms": 1301.0,
      "stream_inactive_offset_ms": 1301.0,
      "deadline_basis": "text",
      "effective_deadline_basis": "text",
      "deadline_basis_fallback_reason": null,
      "sample_rate_hz": 22050,
      "device": "MacBook Pro Speakers",
      "audio_artifact": null,
      "artifact_persistence_ms": 0.0,
      "error": null
    }
  }
}
```

- `_finish_turn()` 增加 `speech_output_metadata` 参数，运行时嵌入防御性副本。输出汇不得覆盖 `failure_reason`、token 字段、`deadline_mode`、`deadline_ms`、challenge 字段或 `model_error`。
- 所有 offset 原点是回合起点（§3.10）。`speech_guard_ms` 属于回合之间，记在下一个人类回合的 `human_input.guard_ms`。
- `InteractiveTurnRecord.latency_ms` 仍是计分的唯一权威值。`deadline_basis` 是配置值，`effective_deadline_basis` 是该回合实际生效的口径——无音频回退时两者不同（§3.6.2），运行摘要另报 `audio_end_fallback_turn_count`。
- 正常播放时 `completed_normally=true` 且 `stream_inactive_offset_ms == playback_drained_offset_ms`；abort、部分播放失败和中断时 `completed_normally=false`，只允许前者有值，后者必须为 `null`。
- 失败与中断回合同样有完整记录，`status` 为对应值且 `error` 非空（§3.11）。

### 5.3 保存合成音频的计时口径

`--speech-save-audio` 照搬输入侧 `artifact_persistence_ms` 的既有做法（`interactive_runtime.py:485-519`）：

- **播放完成之后**才写 WAV。播放前写会把磁盘 I/O 插进"文本就绪 → 首个可听样本"的关键路径上。
- 写入耗时记为 `artifact_persistence_ms`，**排除在 `audio_end` 口径之外**，与人类侧对称。该值要做有限性、非负、不超过实测墙钟时间的校验。
- 用临时文件 + 原子 `replace()`，与 `_write_turn_artifact` 的 `.tmp` 模式一致。
- 路径 `artifacts/turn/{n:04d}_turn_{id:04d}_llm.wav`，与人类侧 `_human.wav` 对称，存相对路径。
- 写入失败是预期基础设施失败：删除临时文件并返回 `status="artifact_failed"`、`completed_normally=true`、`audio_artifact=null` 和 JSON-safe `error`。因为失败发生在完整播放之后，正常的 `playback_drained_offset_ms` 与 `audio_end` 仍然保留；`--speech-on-error fail` 记录后退出，`warn` 记录后继续。

默认运行目录不含任何合成音频。

## 6. 计划中的代码改动

### 先做的重构（无行为变化）

- `src/streammuse/infrastructure/voice/audio.py`（新增）
  - 从 `microphone.py:697` 的 `MicrophoneCapture._pcm_to_float32` 抽出通用的
    `resample_float32(audio, source_rate_hz, target_rate_hz)`。现有实现是私有 staticmethod、目标 16 kHz 写死、且与 int16→float 转换耦合，输出侧无法直接复用。抽取后 `_pcm_to_float32` 只保留"字节 → float32"并调用新函数，麦克风侧行为与现有测试不变。
  - 把 `microphone.py:108` 的 `_PortAudioClockMapper` 泛化为 `PortAudioClockMapper`，时间戳字段名（`inputBufferAdcTime` / `outputBufferDacTime`）作为参数传入，两侧共用。

### 新增生产模块

- `application/tasks/speech_output.py` — 冻结校验的 `SpeechOutputConfig`；`SilentSpeechOutput`。
- `application/factories/speech_output_factory.py` — 延迟选择并组装。
- `infrastructure/voice/synthesizer.py` — 四个后端适配器，含 §3.4.1 加固与 §3.4.3 速率映射。
- `infrastructure/voice/speaker.py` — 输出设备枚举、预检、callback 播放、DAC 时间戳、§3.5.1 错误队列、§3.5.2 `finished_callback` 排空信号、§3.5.3 abort 中断、幂等关闭。
- `infrastructure/voice/speech_sink.py` — `AudioSpeechOutput`：预合成缓存 + 合成 + 播放 + 保存 + 结构化计时；实现 §3.1 的异常契约。

放在既有 `infrastructure/voice/` 而不是新建 `infrastructure/speech/`：可复用错误层级、`_json_safe()` 和上面抽出的音频工具。在 `voice/__init__.py` 增加 `SpeechOutputError` / `SpeechSynthesisError` / `SpeakerDeviceError` / `SpeakerPlaybackError` / `SpeechArtifactError`，并放宽 `VoiceInfrastructureError` 的 docstring（§3.12）。

### 需要修改的现有模块

- `domain/tasks/models.py` — 语音输出契约与 `SpeechRenderableTask`。
- `domain/tasks/zip_zap_zop.py` — `build_spoken_text()` / `speech_vocabulary()`。
- `domain/tasks/__init__.py` — 导出。
- `application/tasks/interactive_runtime.py` — 注入 sink；`_run_llm_turn` 播放与 §3.11 顺序；`_finish_turn` 元数据通道；`_shift_speech_offsets()`；deadline basis；§3.8 的条件 guard 编排；中断与生命周期。
- `application/tasks/__init__.py` — 导出。
- `presentation/task/cli.py` — 新参数（`None` 默认值）、四组互斥校验、工厂接线、`speaker-devices`、`_best_effort_startup_manifest` 扩参、错误文案按子类分派、`ExitStack` 关闭顺序（扬声器先于麦克风）。
- `pyproject.toml` / `uv.lock` — `speech` 与 `tts-kokoro` 组。
- 文档。

不改：`application/config/models.py`、`InputSourceFactory`、`OutputSinkFactory`、`RealTimeMusicService`、音乐 CLI、交互式 manifest 版本号、`_finish_turn` 的职责边界。

## 7. 实施阶段

### 阶段 0：TTS 后端资格验证

- [ ] 目标 Mac 上跑 `say` / `espeak-ng` / Kokoro：setup、合成、RTF、峰值 RSS。
- [ ] 用 §3.2 的实际朗读文本（`"Zip Zap"`、`"Zip Zap Zop"`、区间内数字）而非通用短语。
- [ ] 验证预合成路径：整表一次性合成的总耗时与内存，确认双重上限触发行为。
- [ ] 按 §7.1 盲测协议做可辨识度门槛测试。
- [x] 确认目标机器能以合成采样率打开输出设备，并能取到 `outputBufferDacTime`。
- [ ] 验证 `--speech-rate` 倍率到各后端原生单位的映射表。

#### 7.1 可辨识度盲测协议

老计划给 STT 定的语料协议很严（5 名说话人、每词 100 个留出样本、说话人隔离、置信区间），rev 1 给 TTS 只写"抽检 ≥99%"，同一份工作里两套标准说不过去。

但 TTS 与 STT 有本质差异：**合成是确定性的**，同一段文本每次产出同一波形。样本量不能靠重复渲染同一个词来堆——那只是把同一个观测复制 N 遍。可变维度只有听者、播放条件、上下文三个。

- 素材：每个候选后端对 7 个游戏词组合和一组具体数字各合成一份。数字集合 `D` 取下面两部分的去重并集：
  - 固定易混集合 `{13, 14, 15, 17, 18, 19, 30, 40, 50, 70, 80, 90}`；
  - 本局 `[start_number, start_number + max_turns)` 区间内额外选取的边界值与代表值。
- 记 `N = |D|`。素材总数是 `7 + N`，不得把数字数量写死成 10；实际 `D` 与 `N` 必须进入报告。
- 条件：≥3 种播放条件（耳机、笔记本外放近场、笔记本外放 1.5 m）。
- 听者：≥5 名，互相独立，事先不知道播放的是哪个词。
- 呈现：顺序随机化。**游戏词条目**用强制选择（7 个游戏词 + `无法辨识`）；**数字条目要求听者直接写下听到的具体数字**（或从数字列表中选具体值），不能只标一个笼统的"数字"选项。
- 规模：每后端至少 `(7 + N) × 3 种条件 × 5 名听者` 次判定。
- 报告：混淆矩阵 + Wilson 区间，游戏词与具体数字分开报；同时逐听者、逐播放条件列出错误数，避免只给一个聚合比例掩盖条件差异。

rev 4 把数字答案设计成单一的"数字"选项，只能验证听者知道这是个数字，验证不了把 `17` 听成 `70`。而在 Zip-Zap-Zop 里数字听错和游戏词听错**后果完全一样**：人类靠机器报出的数字推下一个数，听成 70 就会答 71，整局从此错位。所以数字必须按具体值判分。

门槛用**零误辨识**而不是 `≥99%`：本协议的实际样本量由 `N` 决定，不应用固定的 255 次判定宣称 99% 精度。而这是**选型门槛**不是产品指标——任何一次把 `Zip` 听成 `Zap`，或把 `17` 听成 `70`，都意味着人类玩家会据此答错，直接淘汰该后端/音色组合。

### 阶段 1：契约与静默默认

- [x] `audio.py` 重构（`resample_float32` + `PortAudioClockMapper`），麦克风侧行为与测试不变。
- [x] 领域契约、`SpeechOutputConfig`、`SilentSpeechOutput`、工厂骨架。
- [x] 注入 `InteractiveTaskRuntime`，默认静默；`drain()` 与条件 guard 在静默下为 no-op。
- [x] `_finish_turn` 的 `speech_output_metadata` 通道与保留字段保护。

退出标准：不装任何 TTS 依赖时全部现有测试通过；`--speech-output off` 下 trace 与 manifest **逐字段等价于改动前**，`schema_version` 仍为 2，4 处既有断言无需修改。

### 阶段 2：合成器与预合成缓存

- [x] 四个后端适配器、延迟导入、§3.4.1 加固、§3.4.2 Kokoro 开关、§3.4.3 速率映射、provenance、错误映射。
- [x] `prepare()` 预合成、双重上限、命中/未命中策略、耗时计量。
- [x] 任务侧 `build_spoken_text()` / `speech_vocabulary()`。

退出标准：假后端测试证明每局只加载一次、预合成在回合计时外、命中时 `synthesis_ms == 0`、未命中两种策略正确、超时/中断/异常路径下无临时文件与孤儿进程残留。

### 阶段 3：播放设备与时间戳

- [x] 输出设备枚举与预检、必要时重采样。
- [x] callback 播放 + `outputBufferDacTime` 映射 + §3.5.1 错误队列 + §3.5.2 `finished_callback` 排空信号 + §3.5.3 abort 中断 + 幂等关闭。
- [x] §3.1 异常契约的全分支覆盖。

退出标准：确定性 PCM 测试覆盖排空、abort 中断、callback 内异常、设备失败、重复关闭；DAC 偏移量在假时钟下可断言；目标 Mac 手动冒烟无流泄漏。

### 阶段 4：运行时集成

- [x] `_run_llm_turn` 按 §3.11 顺序接入，失败与中断回合先落盘后抛。
- [x] `_shift_speech_offsets()` 与统一原点。
- [x] `--llm-deadline-basis` 两种口径，`audio_end` 用 §3.6.1 的两段未舍入实测量相加，并排除持久化耗时。
- [x] §3.8 的条件 guard 由运行时在开麦边界统一执行。
- [x] trace / manifest 可选字段；`--speech-save-audio` 按 §5.3 口径；失败策略 `fail` / `warn`；`interrupted` 不受 `warn` 影响。

退出标准：假合成器 + 假扬声器跑通 `soft`/`hard`/`challenge`；两种口径的胜负与 trace 正确；播放失败与中断时回合记录完整存在。

### 阶段 5：CLI、文档与验证

- [x] CLI 参数（`None` 默认值）、四组互斥校验、`speaker-devices`、启动 manifest 扩参、错误文案分派。
- [x] 全量测试套件；分别记录本 feature 结果与既有失败。
- [x] 文档：安装（三个 extra 的场景对照表）、设备选择、耳机建议、计分口径、文字提示语义、隐私、故障诊断、明确说明不朗读结果。

### 阶段 6：真实对局与回声验收

- [ ] 目标 Mac 上耳机场景 20 回合 `soft` 双向对局。
- [ ] 外放场景回声误触发率测量（§10 门槛 4）。
- [ ] 发布分阶段 p50/p95：文本就绪 → 首个 DAC 样本 → 排空 → 开麦。
- [ ] 只有在上述数据支持后，才考虑 `hard`/`challenge` + `audio_end`。

## 8. 测试计划

### 领域测试

- `build_spoken_text`：单词、双词、三词、数字、负数、空、无法解析；
- 绝不改写错误答案（给定错误响应，朗读文本必须与之一致）；
- `speech_vocabulary` 的区间边界与去重；
- 不满足 `SpeechRenderableTask` 的任务被拒。

### 基础设施测试

- `resample_float32` 抽取后麦克风侧行为不变（现有测试全绿）+ 新的任意采样率对；
- `PortAudioClockMapper` 在输入/输出两种时间戳字段下的映射；
- 各后端：延迟导入、缺依赖/缺可执行文件/缺模型的可操作错误；Kokoro 在尚未钉死默认快照时缺 model 或 revision 必须拒绝，不能访问上游 latest；
- 子进程：`shell=False`、文本走 stdin、以 `-` 开头的文本不被当选项、超时、中断终止、临时文件在每条异常路径上都被删除；
- `--speech-rate` 倍率到各后端原生单位的映射与越界校验；
- 扬声器：设备枚举、不支持采样率、DAC 偏移量、重复关闭幂等；
- **控制流必须用 `raise` 而非 `return`**：假 sounddevice 断言 callback 抛出的是 `CallbackStop` / `CallbackAbort` 对象；返回值形式的实现必须让测试失败（否则真实环境里等价于 `paContinue`，流不会停）；
- **`inactive_event` 只由 `finished_callback` 置位**：送完最后一帧后事件必须仍未置位，直到假 `finished_callback` 被调用才置位；
- 正常 `CallbackStop` 下 `completed_normally is True` 且 `stream_inactive_offset_ms == playback_drained_offset_ms`；abort、callback 错误与中断下 `completed_normally is False`，只有 `stream_inactive_offset_ms` 有值，`playback_drained_offset_ms is None`；
- **abort 与最后一帧的竞争**：构造"主线程已置 `abort_requested`、随后 callback 才把 `termination_kind` 写成 `normal`"的交错序列，断言 `completed_normally is False`、`playback_drained_offset_ms is None`，且该回合在 `audio_end` 下回退到 `text`。这条测试必须能捕获"让主线程直接写 `termination_kind`"的实现（最后写入者胜会在此失败）；
- **abort 丢弃未播缓冲区**，且中断路径同样经由 `finished_callback` 置位 `inactive_event`；
- callback 内预期错误经队列回传并转换为 `playback_failed`；注入的 `TypeError` 保持原异常对象并向 runtime 传播；
- 等待上限覆盖 `stream.latency`：正常播放的尾段不得被误判为超时；
- 缓存：命中、未命中两种策略、条数上限、字节上限、截断标记；
- `speak()` 对合成失败、预期播放失败、音频持久化失败返回结构化状态且 `error` JSON 安全；`KeyboardInterrupt` / `SystemExit` / 未知程序异常在 abort 清理后原样抛出；
- 音频完整播放后写 WAV 失败返回 `artifact_failed` 且 `completed_normally is True`，保留正常 drain 时间、删除临时文件，且不把写盘耗时计入 `audio_end`。

### 应用层/运行时测试

- `--speech-output off` 下 trace 与 manifest 与基线逐字段相同，且不含 `speech_output` 键；`schema_version` 仍为 2；
- `text` 与 `audio_end` 两种口径下 `latency_ms` 与胜负判定；`audio_end` 用 §3.6.1 的两段未舍入实测量相加，且排除 `artifact_persistence_ms`；
- **`audio_end_ms ≥ text_ready_ms` 恒成立**，包括注入一个上报值大于墙钟的假 `LocalChatModel` 时；
- **没有正常 drain 的每一种路径都回退到 `text` 并写 `effective_deadline_basis` / `deadline_basis_fallback_reason`**：空文本、跳过、合成失败、出声前失败、部分播放后 abort、LLM 超时、中断与 `internal_error`；LLM 超时的 speech status 是 `not_attempted`，运行摘要按原因计数正确；
- `artifact_failed` 因为发生在正常 drain 之后，在请求 `audio_end` 时仍使用 `audio_end`，但按 `fail/warn` 执行失败策略；
- **异常身份保持**：`speak()` 抛 `SystemExit(2)` 时回合记录先落盘、随后裸 `raise` 的仍是原对象且退出码为 2；`KeyboardInterrupt` 与 `TypeError` 同样是原对象，且 `warn` 不能吞掉；
- `build_spoken_text()` 抛 `TypeError` 或此阶段发生 `KeyboardInterrupt` 时，已经返回的 LLM 回合仍先落盘；
- **播放失败（`fail`）时回合记录仍完整写入 `response_trace.jsonl` 与 turn artifact，`speech_output.error` 非空，之后才抛异常**；
- **`interrupted` 时回合记录完整写入，然后重新抛出原 `KeyboardInterrupt` 对象，且 `--speech-on-error warn` 不能吞掉它**；
- `warn` 下合成、播放或音频持久化失败继续对局；
- 预合成耗时不进入任何回合延迟；
- guard 只在下一个人类回合走语音且上一机器回合 `first_dac_sample_offset_ms is not None` 时执行一次；人类先手、无音频状态与 terminal 输入不付 guard；首个 DAC buffer 后失败仍支付 guard；
- voice 与 terminal 输入都只显示文字 prompt，不调用任何人类回合音频播放；
- `playback_*` 与 `stream_inactive_offset_ms` 的 offset 平移正确，`None` 与非有限值被跳过；
- `speech_output` 不能覆盖保留元数据键；
- 完成/退出/异常/`KeyboardInterrupt` 下扬声器与麦克风都恰好关闭一次，且扬声器先关。

### 表现层测试

- 默认静默；所有参数正确传到工厂；
- 四组拒绝组合各自报错并退出 2：`off` + 显式 `--speech-*`、`off` + `audio_end`、非 kokoro + `--speech-model` / `--speech-model-*`、`kokoro` 但缺 model 或 revision；
- `say` 后端的 provenance 用 `sw_vers` 而非 `--version`（本机 `say --version` 会报 unrecognized option）；`espeak-ng` 后端才用 `--version`；
- 显式传入等于默认值的参数也能被检出（验证 `None` 默认值模式）；
- 语音输出工厂启动失败时，manifest 含请求的 backend/device/revision；
- 输入类错误打 `voice input error`、输出类错误打 `speech output error`；
- `speaker-devices` 不要求任务、不加载模型；
- 缺 `speech` / `tts-kokoro` extra 时各只产生一条可操作信息。

### 集成/硬件测试

- 默认集成测试用假合成器 + 假扬声器，不碰真实设备、不下载模型；
- 显式启用的真实后端测试写 WAV 到临时目录、不播放（CI 无音频设备）；
- 手动硬件矩阵先覆盖目标 macOS。

## 9. 风险与缓解

| 优先级 | 风险 | 缓解 |
|---|---|---|
| P0 | 播放失败或中断导致已生成的机器回合从 trace 消失 | 运行时从 LLM 成功返回后用 `try/except BaseException` 保护整个后处理区，任何退出方式都先落盘再分派（§3.1 表 + §3.11）+ 专项测试 |
| P0 | 扬声器尾音被 VAD 当成人类语音起始 | §3.8 条件不变式 + 真实排空 + 运行时统一 guard + 耳机建议 + 外放误触发验收 |
| P0 | 在"最后一帧已提交"时判定播放结束，尾音仍在响就开麦，击穿回声防护 | §3.5.2 inactive 信号只来自 `finished_callback`；正常 `CallbackStop` 时才同时判定 drained + 专项测试 |
| P0 | 合成延迟污染机器回合计时 | 预合成缓存 + 默认 `text` 口径 + 分阶段时间点全量落盘 |
| P0 | Mac 上 Piper 不可用、Kokoro 加载 4.9 秒 | 默认 `system`（已实测）；预合成把加载成本移出对局 |
| P0 | `Zip`/`Zap`/`Zop` 合成后人耳分辨不清 | §7.1 盲测协议，零误辨识作为选型硬门槛 |
| P1 | `termination_kind` 跨线程共写，abort 与最后一帧竞争时截断播放被判为完整，错误按 `audio_end` 计分 | §3.5.2 拆出单向闩锁 `abort_requested`；`completed_normally` 需两条件相与 + 竞争序列专项测试 |
| P1 | 吞异常式的"永不抛出"契约把 `SystemExit(2)` 降级成中断、把程序缺陷伪装成播放失败 | §3.1 改为只转换预期内失败，其余原样重抛**原异常对象**；退出码与 traceback 保持 |
| P1 | `audio_end` 在无音频回合无定义，`warn` 下无法计算 latency | §3.6.2 回退到 `text` 并记 `effective_deadline_basis` + 摘要计回退回合数 |
| P1 | `audio_end` 用纯墙钟差值，可能小于 `text` 口径 | §3.6.1 两段未舍入实测量相加，保证 `audio_end ≥ text` |
| P1 | 把 abort 后的 inactive 误记成完整 drained，残缺播放错误参与 `audio_end` | §3.5.2 区分 `stream_inactive_offset_ms` 与 `playback_drained_offset_ms`；只有 normal termination 才设置后者 |
| P1 | callback 用 `return` 而非 `raise` 传控制信号，等价于 `paContinue`，流不会停 | §3.5.2 引 `_wrap_callback` 源码；测试断言抛出的是控制异常对象 |
| P1 | callback 内异常静默丢失，主线程永久等待 | §3.5.1 错误队列 + `inactive_event` + 覆盖 `stream.latency` 的等待上限；预期错误转状态，未知异常原样传到 runtime |
| P1 | 用 `CallbackStop` 当"立刻停止"，中断后仍播完缓冲区 | §3.5.3 明确用 `CallbackAbort`/`stream.abort()`，并有专项测试 |
| P1 | manifest 变更破坏默认运行与既有断言 | 保持 v2，`speech_output` 仅在启用时出现 |
| P1 | "首个可听样本"用提交时刻冒充 | callback + `outputBufferDacTime` + 字段名如实标注为估计值 |
| P1 | 计分口径变化导致与既有数据不可比 | 默认不变；`audio_end` 显式选择、写入 manifest、`off` 组合直接拒绝 |
| P1 | 依赖分组违背输入输出正交 | 拆出 `speech` extra，`system` 后端不需要 STT 依赖也不需要 `soundfile` |
| P1 | LLM 文本进入子进程命令行 | `shell=False` + stdin 传文本 + 绝对路径解析 |
| P1 | 输出侧失败打出 `voice input error`，误导排查 | §3.12 按异常子类分派文案 |
| P2 | 数字被听成易混值（17/70）却未被盲测捕获 | §7.1 数字条目按具体值判分，并覆盖已知易混对 |
| P2 | Kokoro 模型 ID 无人指定，"取上游最新"不可复现 | §3.4.2 补 `--speech-model`；钉死默认值之前，缺 model/revision 直接在启动校验拒绝（§4.1 第四组） |
| P2 | 保存 WAV 的 I/O 混进延迟 | §5.3 播放后保存 + `artifact_persistence_ms` 排除，照搬输入侧 |
| P2 | WAV 写入失败被误记成播放失败或丢失已完成的 `audio_end` | 独立 `artifact_failed` 状态；保留正常 drain，清理临时文件，再按 `fail/warn` 处理 |
| P2 | `--speech-rate` 单位跨后端不一致 | §3.4.3 统一为倍率，provenance 同时记原生值与单位 |
| P2 | Kokoro 模型资源不可复现 | model ID/revision/快照/voice hash 全进 provenance |
| P2 | STT + TTS + LLM 同机 CPU 争用 | 预合成把 TTS 移出回合；分阶段计时观测后再决定进程隔离 |

## 10. 验收门槛

1. `--speech-output off`（默认）下无任何行为、计时、trace 或 manifest 回归；`schema_version` 仍为 2；不需要任何 TTS 包。
2. 预合成在游戏计时器启动前完成；缓存命中回合 `synthesis_ms == 0.0`；双重上限触发时有明确标记。
3. §3.8 的条件不变式成立：下一回合为语音输入且上一机器回合已提交首个 DAC buffer 时执行"排空 → 一次 guard → 文字 prompt → 开麦"；人类先手、无音频状态与终端输入不付 guard。
4. **回声误触发**：外放场景 n ≥ 300 次"播放结束即采集"中 0 次误触发。0/300 的 **单侧** 95% Clopper-Pearson 上界为 0.994%，可支撑 ≤1% 的主张（0/200 的单侧上界是 1.487%，不足以支撑）。**报告必须写明是单侧**；若审稿要求双侧 95% 区间，同样的 ≤1% 门槛需要 n ≥ 368（0/300 的双侧上界是 1.222%）。耳机场景单独报告。
5. 合成、播放、音频持久化失败（`fail`）与用户中断（`interrupted`）时，该回合在 `response_trace.jsonl` 与 turn artifact 中均有完整记录，含 LLM 原文、裁判结果与 `speech_output.status`/`.error`；中断随后正常走 `user_interrupt` 收尾，程序缺陷保持原异常身份。
6. 任意退出路径不残留音频输出流、子进程、临时文件或模型资源。
7. `text` 与 `audio_end` 两种口径都能从同一份 trace 事后复原；请求口径写在 manifest，每回合实际口径与 fallback reason 写在 trace，回退计数写在运行摘要。
8. 默认运行目录不含合成音频；启用保存时其 I/O 耗时被排除在计分之外并单独记录。
9. 目标 Mac 上，缓存命中时"文本就绪 → `first_dac_sample_offset_ms`" p95 ≤ 150 ms。该指标基于 PortAudio DAC 时间戳，是**估计的**首个可听样本时刻；如需绝对验证，用回环录音单独校准一次并记录偏差。
10. §7.1 盲测协议下，选定后端在全部播放条件中对 `Zip`/`Zap`/`Zop` 及其组合、以及数字集合 `D` 的具体值均为**零误辨识**，并附实际 `N`、混淆矩阵、Wilson 区间和逐条件错误数。
11. 聚焦测试套件与仓库内更广泛的非硬件测试套件均通过。
12. 已记录安装（三个 extra 的场景对照）、设备选择、耳机建议、计分口径、文字提示语义、隐私、以及不朗读结果、不含音频 cue、AEC 与 barge-in。

## 11. 实施后建议运行的第一个命令

```bash
uv run --extra voice --extra speech streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --speech-output audio \
  --speech-backend system \
  --speech-guard-ms 200 \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

戴耳机，从 `soft` + `text` 口径开始。只有拿到目标设备的播放与回声数据后，才在真实对局中启用 `audio_end` 或 `hard`/`challenge`。

## 12. 决策记录

### 12.1 已定（2026-07-27 起，末次更新 2026-07-30）

| 决策 | 结论 | 说明 |
|---|---|---|
| TTS 朗读范围 | **只读 LLM 生成的答案** | 裁判结果、胜负只走终端打印与既有 log。同时消除了播报的事务性难题，见 §1 |
| 计分口径默认值 | **`text`** | 机器回合延迟仍以文本就绪计；本分支既有对局数据继续可比。`audio_end` 保留为显式可选的对称口径（§3.6） |
| 人类回合提示 | **只使用终端文字 prompt** | 不实现音频 cue；voice 与 terminal 输入保持同一份可见提示（§3.7） |
| 输出失败策略默认值 | **`fail`** | 合成、播放与音频持久化失败都不静默降级。有 §3.11 的保底记录，`fail` 无丢数据风险 |
| 错误基类命名 | **不改名** | 保留 `VoiceInfrastructureError`（5 文件 17 处引用），放宽 docstring，新增输出侧子类，用户可见文案靠子类分派（§3.12） |
| `speech` extra 命名 | **保持 `speech` / `voice` / `tts-kokoro`** | 不对既有 `voice` 做破坏性重命名。文档用 §4.2 的场景对照表消歧 |
| 实施节奏 | **阶段 1-5 已完成** | 软件实现、自动测试、CLI 与文档已落地；阶段 0 的真人盲测和阶段 6 的真实对局/回声验收仍需目标环境与参与者 |

### 12.2 待数据决定

- **默认后端**：当前定为 `system`（Mac `say`，§2.5 已实测可行）。这是**起始假设不是最终结论**——若 §7.1 盲测显示 `say` 对某个游戏词存在误辨识，按盲测结果换 Kokoro，默认 extra 组合相应改为 `tts-kokoro`。
- **`challenge` 阶段时间表是否需要为语音输出放宽**：取决于阶段 6 实测的"文本就绪 → 开麦"总间隔。

## 13. 外部实现参考

- python-sounddevice：`https://python-sounddevice.readthedocs.io/`。以下位置已在本机 0.5.5 源码核对——
  `sounddevice.py:1774-1784`（普通异常不传播到主线程）、
  `sounddevice.py:1820-1825`（正常 `CallbackStop` 时，`finished_callback` 在全部已生成音频播完后才触发）、
  `sounddevice.py:2650`（库自身 `sd.play()`/`sd.wait()` 就用 `finished_callback` 置位事件）、
  `sounddevice.py:2773-2782`（`_wrap_callback` 丢弃返回值，只识别抛出的控制异常）
- macOS `say(1)` 手册（`-f -` / `--data-format` / `-o` / `-v` / `-r`）
- espeak-ng 命令行参考：`https://github.com/espeak-ng/espeak-ng`
- Kokoro TTS（模型与 voice 资源选择）：`https://github.com/hexgrad/kokoro`
- Piper（Mac `espeak-ng-data` 路径问题跟踪）：`https://github.com/rhasspy/piper`
