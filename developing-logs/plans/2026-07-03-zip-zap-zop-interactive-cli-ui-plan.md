# ZipZapZop Interactive Terminal UI Plan

## 背景

当前 `streammuse-task` 已经支持两种非交互运行方式：

- `offline_benchmark`：模型尽快连续跑完若干 turn。
- `realtime_loop`：按固定 tick/deadline 连续跑若干 turn。

这两种模式都适合 benchmark 和 trace，但不适合真人在 terminal 里和 LLM 一人一轮玩游戏。现在用户只能在 run 结束后查看 `trace.jsonl`、`response_trace.jsonl` 或 `artifacts/task_turn/*.json`，交互感很弱。

目标是新增一个 terminal CLI UI，让 human 和 LLM 按回合交替回答 Zip-Zap-Zop，边玩边显示每轮结果，同时继续保存轻量 trace，方便之后复盘。

## 目标

新增一个交互式入口：

```bash
uv run streammuse-task play \
  --task zip_zap_zop \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-turns 30 \
  --deadline-ms 3000 \
  --output-dir task_runs
```

交互效果大致如下：

```text
StreamMUSE Zip-Zap-Zop
Commands: :help, :hint, :summary, :quit

[1] You > 1
    OK

[2] LLM thinking...
    LLM > 2
    OK  24.8 ms

[3] You > Zip
    OK

[4] LLM thinking...
    LLM > Zap
    OK  23.9 ms
```

## 非目标

这次不做完整 curses/TUI，不引入 `rich` / `textual` 等新依赖。先做一个稳定、可测试、无额外依赖的 readline-style terminal UI。

这次不改现有 `TaskRuntime.run()` 的 benchmark 语义；`run` 继续保持 batch/trace 模式，`play` 是新的交互模式。

这次不把 interactive UI 接到音乐 realtime loop，也不引入 prompt-continuation 相关逻辑。

## 用户体验设计

### 新增命令

在 `streammuse.presentation.task.cli` 里新增 `play` subcommand：

```bash
streammuse-task play --task zip_zap_zop [options]
```

核心参数：

- `--task zip_zap_zop`：第一版只支持这个任务。
- `--model-url`：OpenAI-compatible `/v1` base URL。
- `--model`：local model server 的模型名。
- `--max-turns`：总 turn 数，human 和 LLM turn 都计入。
- `--start-number`：从哪个数字开始。
- `--human-first` / `--llm-first`：谁先开始，默认 human first。
- `--deadline-ms`：每轮软 deadline。第一版不强制中断用户输入，只记录是否超时。
- `--max-tokens`：LLM 输出 token 上限，默认建议 `8`。
- `--temperature`：默认 `0.0`。
- `--output-dir`：保存 interactive trace 的目录。
- `--timeout-s`：传给 `LocalChatModelClientConfig.timeout_s`，控制 LLM 请求超时；它不属于 runtime config。
- `--history-limit`：传给具体 task/adapter，作为该 task 的上下文策略参数；runtime 不解释这个值。
- `--show-expected`：debug/practice 模式，显示 expected answer，默认关闭。

### Terminal 命令

交互中支持少量 colon commands：

- `:help`：显示规则和可用命令。
- `:hint`：调用 task 的 `build_hint(state, transcript)`；ZipZapZop 可显示当前数字是否是 3/4/5 的倍数，但 runtime 不知道具体规则。
- `:expected`：调用 task 的 `expected_for_state(state, transcript)`，仅 practice/debug 用。
- `:summary`：显示当前有效/无效次数、LLM latency、deadline miss。
- `:quit`：结束 session，写 summary。

### 回合规则

每个 turn 对应一个递增数字 `n`：

- human turn：runtime 负责渲染 terminal 框架，例如 `[n] You > `；task 的 `build_human_prompt()` 只返回用户需要回答的具体提示，例如 `1:`。
- LLM turn：runtime 调用 local chat client，让 LLM 回答当前数字。
- human 和 LLM 的答案都用同一套 referee 校验。
- 校验后无论对错，数字继续前进，避免 session 卡死。

默认不显示 expected answer，只显示：

```text
OK
```

或：

```text
MISS expected=Zip actual=3
```

是否显示 expected 可以通过 `--hide-expected-on-miss` 后续扩展；第一版建议 miss 时显示 expected，方便试玩和 debug。

## LLM Prompt 设计

当前 `ZipZapZopTask.build_turn()` 每轮只给：

```text
system: 规则
user: "n:"
```

交互模式建议保留这个简洁形式，但可以在 prompt 中加入最近几轮 transcript，让它更像在和人一起玩：

```text
system: You are playing Zip-Zap-Zop with a human. Respond with only your next value...
user: Recent turns:
      Human at 1: 1
      Assistant at 2: 2
      Human at 3: Zip
      Your turn. 4:
```

第一版 ZipZapZop task 可以默认使用 `history_limit=8`，避免 prompt 无限增长；但这个窗口是 ZipZapZop/task adapter 的上下文策略，不是 runtime 写死的策略。Runtime 只保存完整 transcript，并把 transcript 交给 task，由 task 决定给 LLM 看多少历史、是否压缩历史、是否隐藏某些 state。

## 代码结构计划

### 1. 新增 interactive runtime

建议新增：

```text
src/streammuse/application/tasks/interactive_runtime.py
```

核心类：

```python
@dataclass(frozen=True)
class InteractiveTaskRuntimeConfig:
    output_dir: str
    deadline_ms: float = 3000.0
    max_tokens: int = 8
    temperature: float = 0.0
    human_first: bool = True
    show_expected: bool = False

@dataclass(frozen=True)
class InteractiveTaskRunResult:
    output_dir: str
    task_name: str
    turn_count: int
    human_turn_count: int
    llm_turn_count: int
    valid_count: int
    invalid_count: int
    deadline_miss_count: int
```

核心 runtime：

```python
class InteractiveTaskRuntime:
    def play(self, task: InteractiveTask, *, max_turns: int) -> InteractiveTaskRunResult:
        ...
```

第一版 CLI registry 可以只暴露 `ZipZapZopTask`，但 runtime 的类型边界应该面向 `InteractiveTask` protocol。不是所有 `RealtimeTask` 都天然支持 human interactive，因此 interactive 能力应该是 task 显式实现的附加协议。`history_limit` 等上下文策略参数应传给具体 task/adapter，而不是写进 runtime 通用配置。

### 2. 抽象 terminal I/O，方便测试

不要直接在 runtime 里到处调用 `input()` 和 `print()`。建议定义很轻的 protocol：

```python
class TerminalIO(Protocol):
    def write(self, text: str) -> None: ...
    def prompt(self, text: str) -> str: ...
```

生产实现：

```python
class StdTerminalIO:
    def write(...): print(...)
    def prompt(...): return input(...)
```

测试里用 fake terminal，提前喂 human responses，检查输出内容。

### 3. ZipZapZop helper

当前 `ZipZapZopTask.expected_response(number)` 已经足够校验 human answer。

为了避免复制 prompt 规则，建议加一个 helper：

```python
ZipZapZopTask.rules_prompt() -> str
ZipZapZopTask.current_number(state: TaskState) -> int
ZipZapZopTask.build_human_prompt(state, transcript) -> str
ZipZapZopTask.build_llm_messages(state, transcript) -> list[ChatMessage]
ZipZapZopTask.build_hint(state, transcript) -> str | None
ZipZapZopTask.expected_for_state(state, transcript) -> str | None
```

这些 helper 应放在 task 类或 task-specific adapter 里，因为规则和上下文策略属于 task domain；不要把 ZipZapZop 的规则写进 `InteractiveTaskRuntime`。

### 4. CLI 接入

修改：

```text
src/streammuse/presentation/task/cli.py
```

保留现有：

```bash
streammuse-task run ...
```

新增：

```bash
streammuse-task play ...
```

`main()` 根据 `args.command` 分发到：

- `run_task(...)`
- `play_task(...)`

`play_task(...)` 创建：

- `ZipZapZopTask`
- `LocalChatModelClient`
- `InteractiveTaskRuntime`
- `StdTerminalIO`


为了避免 `run` 和 `play` 参数漂移，CLI parser 应提取公共参数 helper：

```python
def _add_common_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--deadline-ms", type=float, default=1000.0)
    parser.add_argument("--output-dir", default="task_runs")
    parser.add_argument("--start-number", type=int, default=1)
```

`run` 额外保留 `--runner`、`--tick-rate-hz`；`play` 额外加入 `--human-first` / `--llm-first`、`--show-expected`、`--history-limit`。`--timeout-s` 始终传给 `LocalChatModelClientConfig.timeout_s`。

## Trace / Output 设计

每次 interactive run 新建目录：

```text
task_runs/{task_name}_interactive_YYYYMMDD-HHMMSS_<id>/
  manifest.json
  response_trace.jsonl
  run_summary.json
  artifacts/turn/*.json
```

### response_trace.jsonl

交互模式的 `response_trace.jsonl` 建议比 batch 模式多一点字段，但仍保持轻量：

```json
{
  "turn_id": 0,
  "actor": "human",
  "number": 1,
  "prompt": "1:",
  "response": "1",
  "expected": "1",
  "is_valid": true,
  "latency_ms": 842.1,
  "deadline_missed": false
}
```

LLM turn：

```json
{
  "turn_id": 1,
  "actor": "llm",
  "number": 2,
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "response": "2",
  "expected": "2",
  "is_valid": true,
  "latency_ms": 24.8,
  "deadline_missed": false
}
```

### artifacts/turn

可选保存完整 turn payload，包括：

- state before/after
- prompt/messages
- response
- referee
- timing
- model token usage，仅 LLM turn 有

### manifest.json

记录：

- task name
- mode: `interactive`
- model URL/model name
- start number
- human first
- max turns
- deadline
- response trace path

`manifest.json` 必须在 `play()` 开始时先写出初始版本，结束时再 update summary 字段。不要依赖 `TaskRuntime.run()` 里的 `JsonlDebugTraceRecorder` 创建 manifest，因为 interactive runtime 第一版不一定复用该 recorder。

## Realtime / Deadline 策略

第一版采用软 deadline：

- human turn：记录用户从 prompt 出现到提交答案的耗时；超过 `deadline_ms` 则 `deadline_missed=true`，但不强制打断输入。
- LLM turn：记录模型请求耗时；超过 `deadline_ms` 则 `deadline_missed=true`。

原因：强制 timeout `input()` 在不同 terminal/OS 上比较麻烦，容易引入 fragile 行为。当前项目主要在 Linux/HPC 上跑，后续可以加 `--enforce-deadline`，用 `select.select()` 或独立 input thread 实现硬 deadline。

## 测试计划

### Unit tests

新增：

```text
tests/unit/application/tasks/test_interactive_runtime.py
```

覆盖：

- human 和 LLM 交替 turn。
- human-first / llm-first。
- valid / invalid 统计。
- deadline miss 统计。
- `:quit` 能提前结束。
- `:help` / `:hint` / `:expected` / `:summary` 不消耗 turn。
- `response_trace.jsonl` 字段正确。
- `manifest.json` 和 `run_summary.json` 正确写入。
- fake model client 收到正确 messages。
- transcript 被传给 task，并由 task 决定如何使用。

### CLI tests

扩展：

```text
tests/unit/presentation/task/test_task_cli.py
```

覆盖：

- `streammuse-task play --task zip_zap_zop ...` 能正确调用 `play_task()`。
- unsupported task 返回错误。
- `--llm-first` / `--human-first` 参数解析正确。
- common args helper 对 `run` 和 `play` 都生效，避免 parser 参数漂移。
- `--history-limit` 只传给 task/adapter，不进入 runtime 通用配置。

### Regression tests

继续跑：

```bash
uv run pytest tests/unit/application/tasks tests/unit/presentation/task -q
uv run pytest tests/unit tests/integration -q
```

## 手动验证命令

启动本地 OpenAI-compatible server 后：

```bash
uv run streammuse-task play \
  --task zip_zap_zop \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-turns 20 \
  --max-tokens 8 \
  --temperature 0 \
  --deadline-ms 3000 \
  --output-dir task_runs
```

检查输出：

```bash
LATEST=$(ls -td task_runs/zip_zap_zop_interactive_* | head -1)
cat "$LATEST/run_summary.json"
jq -r '.actor + " turn=" + (.turn_id|tostring) + " response=" + .response' "$LATEST/response_trace.jsonl"
```

## 实施顺序

1. 新增 `interactive_runtime.py` 和 terminal I/O abstraction。
2. 给 `ZipZapZopTask` 或 task-specific adapter 实现 `InteractiveTask` helpers，不把 ZipZapZop 规则写进 runtime。
3. 修改 `streammuse-task` CLI，新增 `play` subcommand。
4. 写 interactive response trace / summary / manifest。
5. 补 unit tests 和 CLI tests。
6. 跑 targeted tests。
7. 跑全量 unit/integration regression。
8. 用真实 local LLM server 手动玩一局，确认 terminal 体验和 trace 内容。

## 风险和注意点

- 不要让 interactive runtime 复用 `TaskRuntime.run()` 的 batch loop；两者交互模型不同，强行复用会让代码更绕。
- 不要在第一版引入 curses/rich/textual；先把交互语义跑稳。
- human input deadline 第一版只做软统计，不强制中断。
- LLM 输出可能带解释文本；需要继续建议 `temperature=0`、`max_tokens=8`，并在 prompt 里强调 only answer。
- 如果模型输出 `"Zip\n"`，当前 strip 后可接受；如果输出 `"Zip."`，默认不接受。是否做宽松 normalize 后续再决定。

## 多游戏扩展和上下文隔离补充

后续还会加入更多 interactive LLM game/task。不同游戏对上下文的需求可能完全不同：有的只需要当前数字，有的需要上一轮，有的需要完整对话历史，有的还有 hidden state 或只允许玩家看到部分信息。因此 interactive UI 不能把 ZipZapZop 的上下文策略写死在 runtime 里。

### 分层原则

交互系统需要分成三层：

1. Runtime 层只负责流程

`InteractiveTaskRuntime` 只负责：

- 决定当前 turn 是 human 还是 LLM。
- 显示 terminal prompt。
- 读取 human input。
- 调用 local chat model。
- 记录 latency / deadline miss。
- 写 `response_trace.jsonl`、artifact 和 summary。
- 调用 task 提供的方法做 prompt building、validation 和 state transition。

Runtime 不应该知道 ZipZapZop 的具体规则，也不应该决定某个游戏给 LLM 看多少历史。

2. Task 层决定游戏语义和上下文策略

每个 game/task 自己决定：

- human 看到什么 prompt。
- LLM 收到什么 messages。
- 是否需要完整 transcript。
- 是否只需要最近 N 轮。
- 是否需要把历史压缩成 summary。
- 是否存在 hidden state。
- human 和 LLM 的答案如何校验。
- 回合结束后如何更新 state。

例如：

- ZipZapZop：可以只用当前数字，或给最近几轮 transcript 让它更像在一起玩。
- Word chain：通常只需要上一个词和已用词集合。
- 20 Questions：需要完整问答历史，但不能把隐藏答案直接暴露给提问方。
- Memory game：可能要区分 public transcript 和 hidden referee state。
- Debate / negotiation：可能要完整上下文，或定期 summary 压缩。

3. Transcript 是 runtime 和 task 之间的通用数据结构

Runtime 维护完整 transcript，但不解释它。Task 可以基于 transcript 自己构造 prompt。

建议新增通用 turn record：

```python
@dataclass(frozen=True)
class InteractiveTurnRecord:
    turn_id: int
    actor: Literal["human", "llm"]
    number: int | None = None
    prompt: str | list[ChatMessage] | None = None
    response: str = ""
    expected: str | None = None
    is_valid: bool = False
    latency_ms: float | None = None
    deadline_missed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
```

`response_trace.jsonl` 可以由这个 record 的轻量投影写出。


### `state.history` 与 `transcript` 的关系

需要明确两份历史的边界，避免后续游戏维护两套互相打架的状态：

- `transcript` 是 interactive runtime 的完整审计日志，只存在于 interactive runtime，包含 actor、prompt、response、latency、valid、deadline 等复盘信息。
- `state.history` 是 task 需要推进游戏的最小游戏状态，batch 和 interactive 都可以使用。
- `ZipZapZopTask.advance_state()` 继续更新 `state.history`，但只存 response、valid、expected 等游戏需要的信息，不承担完整审计职责。
- `build_llm_messages(state, transcript)` 可以读取 `state.history` 或 `transcript`，由具体 task 自己决定。
- 不建议第一版把 `InteractiveTurnRecord` 直接塞进 `state.history`，因为这会改变 batch mode 的已有状态形状。

### 建议的 interactive task protocol

不要让 `InteractiveTaskRuntime` 直接依赖 `ZipZapZopTask` 的实现细节。建议定义一个 interactive-specific protocol，例如：

```python
class InteractiveTask(Protocol):
    name: str

    def initial_state(self) -> TaskState: ...

    def build_human_prompt(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str: ...

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]: ...

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: Literal["human", "llm"] | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult: ...

    def advance_state(
        self,
        state: TaskState,
        referee_result: TaskRefereeResult,
        response_text: str,
        *,
        actor: Literal["human", "llm"] | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskState: ...

    def expected_for_state(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None: ...

    def build_hint(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> str | None: ...
```

这样 runtime 的核心 loop 可以保持通用：

```python
state = task.initial_state()
transcript = []
for turn_id in range(max_turns):
    is_human_turn = (turn_id % 2 == 0) if human_first else (turn_id % 2 == 1)
    actor = "human" if is_human_turn else "llm"
    if actor == "human":
        prompt = task.build_human_prompt(state, transcript)
        response = terminal.prompt(prompt)
    else:
        messages = task.build_llm_messages(state, transcript)
        response = model_client.generate(messages, ...).text

    referee = task.validate_response(state, response, actor=actor, transcript=transcript)
    record = InteractiveTurnRecord(...)
    transcript.append(record)
    state = task.advance_state(state, referee, response, actor=actor, transcript=transcript)
```

### ZipZapZop 的第一版适配

第一版可以让 `ZipZapZopTask` 同时支持 batch `RealtimeTask` 和 interactive `InteractiveTask`。

建议增加 helper：

```python
def rules_prompt(self) -> str: ...
def current_number(self, state: TaskState) -> int: ...
def build_llm_messages(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> list[ChatMessage]: ...
def build_human_prompt(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str: ...
def build_hint(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None: ...
def expected_for_state(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None: ...
```

ZipZapZop 的上下文策略：

- 默认只需要当前数字。
- 可选 `history_limit` 用于把最近几轮加入 LLM prompt，让它更像在和 human 一起玩。
- 这个 `history_limit` 是 ZipZapZop task 的策略参数，不应该由 runtime 强制解释。

### 对原计划的调整

原计划中提到：

> 第一版实现可以先用固定窗口 `history_limit=8`，避免 prompt 无限增长。

需要改成：

> Runtime 可以提供默认 `history_limit` 配置，但是否使用、如何使用、给 LLM 看哪些历史，必须由具体 task/adapter 决定。Runtime 只保存完整 transcript，并把 transcript 传给 task。

这个调整可以防止后续新游戏被 ZipZapZop 的简单上下文模型限制住。

## Detailed Todo List

### Phase 0: 现状确认和边界保护

- [x] 确认当前 branch 的 `streammuse-task run` 仍然可用，作为 batch benchmark 路径。
- [x] 确认已有 `TaskRuntime.run()` 不被 interactive UI 改写。
- [x] 确认 interactive UI 不改 `RealTimeMusicService`、音乐 CLI、web server、prompt-continuation 相关代码。
- [x] 确认本地 OpenAI-compatible server 的基本连通性测试命令仍然记录在文档或最终说明里。

### Phase 1: 定义 interactive 通用数据结构

- [x] 新增 `InteractiveTurnRecord` dataclass。
- [x] 字段包含 `turn_id`、`actor`、`number`、`prompt`、`response`、`expected`、`is_valid`、`latency_ms`、`deadline_missed`、`metadata`。
- [x] 明确 `prompt` 可以是 human prompt string，也可以是 LLM messages list。
- [x] 决定 `InteractiveTurnRecord` 放置位置，建议放在 `src/streammuse/domain/tasks/models.py` 或新文件 `src/streammuse/domain/tasks/interactive.py`。
- [x] 更新 `src/streammuse/domain/tasks/__init__.py` export。
- [x] 写 unit test 覆盖 record 可序列化为 dict/json 的基本行为。

### Phase 2: 定义 interactive task protocol

- [x] 新增 `InteractiveTask` protocol。
- [x] protocol 包含 `initial_state()`。
- [x] protocol 包含 `build_human_prompt(state, transcript)`。
- [x] protocol 包含 `build_llm_messages(state, transcript)`。
- [x] protocol 包含兼容 batch 的 `validate_response(state, response_text, *, actor=None, transcript=None)`。
- [x] protocol 包含兼容 batch 的 `advance_state(state, referee_result, response_text, *, actor=None, transcript=None)`。
- [x] protocol 包含 `expected_for_state(state, transcript)`，供 `:expected` 使用。
- [x] protocol 包含 `build_hint(state, transcript)`，供 `:hint` 使用。
- [x] 确认 protocol 不强制具体 task 使用完整 transcript，只把 transcript 传进去。
- [x] 更新 type imports，避免和已有 `RealtimeTask` protocol 互相污染。

### Phase 3: 让 ZipZapZopTask 支持 interactive protocol

- [x] 给 `ZipZapZopTask` 添加 `rules_prompt()` helper。
- [x] 给 `ZipZapZopTask` 添加 `current_number(state)` helper。
- [x] 给 `ZipZapZopTask` 添加 `build_human_prompt(state, transcript)`。
- [x] 给 `ZipZapZopTask` 添加 `build_llm_messages(state, transcript)`。
- [x] 给 `ZipZapZopTask` 添加 `build_hint(state, transcript)`。
- [x] 给 `ZipZapZopTask` 添加 `expected_for_state(state, transcript)`。
- [x] 保留 `ZipZapZopTask.validate_response(state, response_text)` 的 batch 调用方式，并增加 keyword-only 可选参数 `actor=None, transcript=None`。
- [x] 保留 `ZipZapZopTask.advance_state(state, referee_result, response_text)` 的 batch 调用方式，并增加 keyword-only 可选参数 `actor=None, transcript=None`。
- [x] interactive runtime 调用时使用 keyword 参数，避免和 batch 位置参数混淆。
- [x] 把 human/llm response 都写入 `state.history` 的最小游戏状态。
- [x] 添加 `history_limit` 策略参数；默认值可以是 8，但只由 ZipZapZop 使用。
- [x] 确保 batch `build_turn()`、`validate_response()`、`advance_state()` 行为保持兼容，不破坏已有 `TaskRuntime.run()`。
- [x] 写 unit tests 覆盖 ZipZapZop 的 human prompt、LLM messages、history window、valid/invalid response。

### Phase 4: 设计 terminal I/O abstraction

- [x] 新增 `TerminalIO` protocol。
- [x] 新增 `StdTerminalIO` 实现，包装 `print()` 和 `input()`。
- [x] 新增 fake terminal I/O 用于测试，或在 test 内定义轻量 fake。
- [x] 确认 runtime 不直接调用裸 `input()`，方便测试和以后替换 UI。
- [x] 设计 terminal 输出格式：banner、turn prompt、OK/MISS、latency、summary。
- [x] 实现 colon commands 的解析函数，例如 `parse_terminal_command(text)`。
- [x] 支持 `:help`。
- [x] 支持 `:hint`，通过 task 的 `build_hint(...)` 获取内容。
- [x] 支持 `:expected`，通过 task 的 `expected_for_state(...)` 获取内容，只在 debug/practice 场景显示。
- [x] 支持 `:summary`，从 runtime 实时 counters 读取，不临时扫描 transcript。
- [x] 支持 `:quit`。

### Phase 5: 实现 InteractiveTaskRuntime

- [x] 新增 `src/streammuse/application/tasks/interactive_runtime.py`。
- [x] 新增 `InteractiveTaskRuntimeConfig`。
- [x] 新增 `InteractiveTaskRunResult`。
- [x] 新增 `InteractiveTaskRuntime.play(task, max_turns=...)`。
- [x] 用 inline 逻辑实现 human-first / llm-first：`is_human_turn = (turn_id % 2 == 0) if human_first else (turn_id % 2 == 1)`，不要为第一版凭空引入 actor_policy 类。
- [x] human turn: 调用 `task.build_human_prompt()`，读取 terminal input。
- [x] LLM turn: 调用 `task.build_llm_messages()`，调用 `LocalChatModelClient.generate()`。
- [x] 每 turn 记录 start/end time 和 latency。
- [x] 实现 soft deadline 统计；第一版不强制中断 human input。
- [x] 每 turn 调用 `task.validate_response()`。
- [x] 每 turn 调用 `task.advance_state()`。
- [x] 每 turn append `InteractiveTurnRecord` 到 transcript。
- [x] 支持 colon command 不消耗 turn。
- [x] `:quit` 能提前结束并写 summary。
- [x] runtime 出错时尽量写已有 trace/summary 或清晰报错。
- [x] runtime 内部维护实时 counters：`valid_count`、`invalid_count`、`deadline_miss_count`、`llm_latency_ms_sum`、`llm_turn_count`、`human_turn_count`。

### Phase 6: Interactive trace 和 artifacts

- [x] 每次 play 新建目录 `task_runs/{task_name}_interactive_<timestamp>_<id>/`。
- [x] 在 `play()` 开始时先写初始 `manifest.json`，结束时再 update summary 字段。
- [x] 写 `response_trace.jsonl`。
- [x] `response_trace.jsonl` 每行包含 actor、turn_id、number、prompt、response、expected、is_valid、latency_ms、deadline_missed。
- [x] 对 LLM turn 记录 token usage，如果 available。
- [x] 可选写 `artifacts/turn/*.json`，保存完整 turn payload。
- [x] 写 `run_summary.json`，包含 total/human/llm turns、valid/invalid、deadline miss、output_dir。
- [x] 确认 batch `TaskRuntime.run()` 的 `response_trace.jsonl` 不受 interactive schema 干扰，或者在文档里明确两者 schema 差异。

### Phase 7: CLI 接入

- [x] 新增 `_add_common_task_args(parser)`，让 `run` 和 `play` 共用 task/model/timeout/turn/output/start 参数。
- [x] 修改 `src/streammuse/presentation/task/cli.py`，新增 `play` subcommand。
- [x] 保留现有 `run` subcommand 参数和行为。
- [x] 给 `play` 加 `--task`。
- [x] 给 `play` 加 `--model-url`。
- [x] 给 `play` 加 `--model`。
- [x] 给 `play` 加 `--timeout-s`。
- [x] 给 `play` 加 `--max-turns`。
- [x] 给 `play` 加 `--max-tokens`，默认建议 8。
- [x] 给 `play` 加 `--temperature`，默认 0。
- [x] 给 `play` 加 `--deadline-ms`。
- [x] 给 `play` 加 `--output-dir`。
- [x] 给 `play` 加 `--start-number`。
- [x] 给 `play` 加 `--human-first` / `--llm-first`，默认 human first。
- [x] 给 `play` 加 `--history-limit`，传给具体 task/adapter，不放入 runtime 通用配置。
- [x] 给 `play` 加 `--show-expected`。
- [x] 实现 `play_task(...)` helper，方便测试 monkeypatch。
- [x] `main()` 根据 `args.command` 分发到 `run_task` 或 `play_task`。
- [x] unsupported task 输出清晰错误。

### Phase 8: Unit tests

- [x] 新增 `tests/unit/application/tasks/test_interactive_runtime.py`。
- [x] 测试 human-first 交替 turn。
- [x] 测试 llm-first 交替 turn。
- [x] 测试 human valid response。
- [x] 测试 human invalid response。
- [x] 测试 LLM valid response。
- [x] 测试 LLM invalid response。
- [x] 测试 soft deadline miss。
- [x] 测试 `:help` 不消耗 turn。
- [x] 测试 `:hint` 不消耗 turn。
- [x] 测试 `:expected` 不消耗 turn。
- [x] 测试 `:summary` 不消耗 turn。
- [x] 测试 `:quit` 提前结束。
- [x] 测试 response trace 每行字段完整。
- [x] 测试 manifest 和 summary 写入。
- [x] 测试 fake model client 收到 expected messages。
- [x] 测试 transcript 被传给 task，并由 task 决定如何使用。

### Phase 9: CLI tests

- [x] 扩展 `tests/unit/presentation/task/test_task_cli.py`。
- [x] 测试 common args helper 对 `run` 和 `play` 都添加基础参数。
- [x] 测试 `play` parser 能解析基础参数。
- [x] 测试 `--llm-first` / `--human-first` 互斥或优先级正确。
- [x] 测试 `main(["play", ...])` 调用 `play_task(...)`。
- [x] 测试 unsupported task 返回 non-zero。
- [x] 测试 `run` subcommand 既有测试仍然通过。

### Phase 10: Manual validation

Manual validation note: Phase 10 保留未勾选，因为它需要一个正在运行的真实 OpenAI-compatible LLM server 和真人试玩；本次已用 fake terminal/fake model 覆盖 runtime/CLI/trace 行为。


- [ ] 启动本地 OpenAI-compatible LLM server。
- [ ] 跑 `uv run streammuse-task play --task zip_zap_zop ...`。
- [ ] 人类先手玩至少 20 turn。
- [ ] LLM 先手玩至少 20 turn。
- [ ] 人为输入错误答案，确认 MISS 显示和 trace 记录正确。
- [ ] 输入 `:hint`，确认不消耗 turn。
- [ ] 输入 `:expected`，确认只在 practice/debug 场景显示 expected。
- [ ] 输入 `:summary`，确认统计正确。
- [ ] 输入 `:quit`，确认 summary 正常写出。
- [ ] 检查 `response_trace.jsonl` 可读。
- [ ] 检查 `run_summary.json` 可读。

### Phase 11: Regression tests

- [x] 跑 interactive targeted tests：

```bash
uv run pytest tests/unit/application/tasks tests/unit/presentation/task -q
```

- [x] 跑全量 unit/integration：

```bash
uv run pytest tests/unit tests/integration -q
```

- [x] 确认 `streammuse-task run` 的 offline benchmark 仍然可用。
- [x] 确认 `streammuse-task run` 的 realtime loop 仍然可用。
- [x] 确认音乐 realtime 相关 tests 没有被 interactive UI 影响。

### Phase 12: 文档和最终说明

- [x] 在最终实现报告或说明里写清楚 `run` 和 `play` 的区别。
- [x] 写清楚如何启动 local LLM server。
- [x] 写清楚 interactive trace 文件结构。
- [x] 写清楚 soft deadline 行为。
- [x] 写清楚每个 future game/task 应该自己控制上下文策略。
- [x] 给出一个完整可复制的 play 命令。

## Feedback Review Notes

针对外部 feedback 的处理结论如下：

1. `InteractiveTask` protocol 与现有 `RealtimeTask` 方法签名冲突：采纳。Plan 已改为保留 batch 兼容签名，并通过 keyword-only `actor=None, transcript=None` 扩展 `validate_response()` 和 `advance_state()`。

2. `state.history` 和 runtime `transcript` 关系不清：采纳。Plan 已新增分层说明：`transcript` 是 interactive runtime 的完整审计日志，`state.history` 是 task 的最小游戏状态，不建议第一版把 `InteractiveTurnRecord` 塞进 `state.history`。

3. `run` 和 `play` CLI 参数重复：采纳。Plan 已增加 `_add_common_task_args(parser)`，公共参数由 helper 添加，`run` 和 `play` 只维护各自特有参数。

4. `:hint` / `:expected` 需要 task protocol 支持：采纳方向，但调整命名。Feedback 建议 `expected_response(self, state)`，但 `ZipZapZopTask` 已有 `expected_response(number)` static method；为避免混淆，Plan 使用 `expected_for_state(state, transcript)`，并新增 `build_hint(state, transcript)`。

5. `actor_policy` 没有定义：采纳。Plan 已改为第一版直接 inline human-first / llm-first 逻辑，不额外引入 actor policy 类。

6. human prompt 渲染责任不清：采纳。Plan 已明确 runtime 负责 terminal 框架，例如 `[n] You > `；task 的 `build_human_prompt()` 只返回具体提示，例如 `1:`。

7. `manifest.json` 创建顺序：采纳。Plan 已明确 interactive `play()` 开始时先写初始 manifest，结束时再 update summary，不能依赖 batch recorder 的 manifest 创建行为。

8. run directory 命名 hardcode `zip_zap_zop`：采纳。Plan 已改为 `task_runs/{task_name}_interactive_<timestamp>_<id>/`。

9. `:summary` 的统计来源：采纳。Plan 已要求 runtime 维护实时 counters，`:summary` 直接读取 counters，不临时扫描 transcript。

10. `--timeout-s` 放置不清：采纳。Plan 已明确 `--timeout-s` 是 common CLI arg，并传给 `LocalChatModelClientConfig.timeout_s`，不属于 `InteractiveTaskRuntimeConfig`。
