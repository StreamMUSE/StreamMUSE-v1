# ZipZapZop Deadline Modes Plan

## 背景

当前 `streammuse-task play --task zip_zap_zop` 只有一种 soft deadline 行为：

- 每轮记录 latency。
- 如果超过 `--deadline-ms`，只标记 `deadline_missed=true`。
- 游戏继续进行。
- 结束后可以从 `response_trace.jsonl` / summary 里看到 deadline miss 数量。

现在希望游戏启动后先让用户选择三种模式：

1. Soft DDL：过时仍继续，结束后汇总超时。
2. Hard DDL：任何一方过时或答错，立刻结束，过时/答错者输。
3. Challenging Mode：从宽松 hard DDL 开始，例如 10s；如果连续 20 轮都按时且答对，就降低 deadline，例如降到 5s；继续递减，直到有人超时或答错输掉。

## 目标

新增 deadline mode 概念，让 `play zip_zap_zop` 支持三种玩法，同时保留 trace 可复盘能力。

目标用户体验：

```text
Select deadline mode:
1) Soft deadline
2) Hard deadline
3) Challenge mode
Choice [1-3]:
```

Soft mode 结束时：

```text
Summary: 30 turns, 28 valid, 2 invalid, 3 deadline misses
Deadline misses:
  turn=4 actor=llm number=4 latency=3250.4ms deadline=3000.0ms
  turn=11 actor=human number=11 latency=5221.7ms deadline=3000.0ms
```

Hard mode 超时时：

```text
Deadline missed: human lost at turn 11, number 11.
Winner: llm
```

答错时：

```text
Wrong answer: llm lost at turn 15, number 15. Expected ZipZop, got 15.
Winner: human
```

Challenge mode：

```text
Challenge stage 1: deadline 10000ms, target 20 turns
...
Stage passed. New deadline: 5000ms
...
Deadline missed: llm lost at turn 47, number 47.
Winner: human
```

## 非目标

这次不做 curses / textual UI。

这次不把 deadline mode 接入音乐 realtime 服务。

这次不改变 `streammuse-task run` 的 benchmark / realtime_loop 行为。

这次不让 runtime 理解 ZipZapZop 之外的规则；deadline mode 是 runtime 级别，游戏规则仍由 task 负责。

## 模式语义

### 1. Soft DDL

这是当前行为的增强版：

- 每一轮都有 deadline。
- 如果 human 或 LLM 超时，记录 miss，但游戏继续。
- `response_trace.jsonl` 每行继续记录 `deadline_missed`。
- `run_summary.json` 新增 deadline miss 明细列表。
- terminal 结束 summary 除了显示总数，还列出哪些 turn miss 了。

Soft DDL 不改变胜负规则；它只是记录 performance。

### 2. Hard DDL

Hard DDL 是真正的游戏失败条件：

- human 超过 deadline：human 输，游戏立即结束。
- LLM 超过 deadline：LLM 输，游戏立即结束。
- human 答错：human 输，游戏立即结束。
- LLM 答错：LLM 输，游戏立即结束。
- 超时 turn 仍写入 trace，`deadline_missed=true`。
- 答错 turn 仍写入 trace，`is_valid=false`、`failure_reason=EXPECTED_MISMATCH`。
- summary status 应写成类似 `deadline_loss` 或 `invalid_response_loss`，并记录 `loser` / `winner` / `loss_turn_id` / `loss_number`。

Hard DDL 对 human 和 LLM 都应该一致：谁过 deadline 或答错，谁输。如果同一 turn 同时超时且答错，第一版建议优先记录为 `deadline_loss`，同时在 turn trace 中保留 `is_valid=false`，因为 hard deadline 是更强的实时失败条件。

### 3. Challenge Mode

Challenge mode 是多 stage 的 hard DDL：

- 从一个比较宽松的 deadline 开始，例如 `10000ms`。
- 每个 stage 需要连续完成固定 turn 数，例如 `20`。
- 如果这 `20` 个 turn 内无人超时且无人答错，则进入下一 stage，deadline 变短。
- 如果任何一方超时，立即结束，超时者输。
- 如果任何一方答错，立即结束，答错者输。
- 游戏数字和 transcript 不重置，只是 deadline 变短。

建议第一版支持两种 deadline schedule：

1. 显式列表：

```bash
--challenge-deadline-ms-list 10000,5000,3000,2000,1000
```

2. 几何递减：

```bash
--challenge-start-deadline-ms 10000
--challenge-min-deadline-ms 1000
--challenge-factor 0.5
```

第一版优先实现显式列表；几何递减可以作为 fallback 或后续扩展。

默认建议：

- `challenge_deadline_ms_list = [10000, 5000, 3000, 2000, 1000]`
- `challenge_stage_turns = 20`

## CLI 设计

新增参数：

```bash
--deadline-mode {menu,soft,hard,challenge}
```

建议默认值需要谨慎：

- 如果严格满足“游戏一打开就先选模式”，默认应为 `menu`。
- 为了自动化测试和脚本稳定，也允许用户显式传 `--deadline-mode soft|hard|challenge` 跳过菜单。

新增 challenge 参数：

```bash
--challenge-stage-turns 20
--challenge-deadline-ms-list 10000,5000,3000,2000,1000
```

可选扩展参数：

```bash
--challenge-start-deadline-ms 10000
--challenge-min-deadline-ms 1000
--challenge-factor 0.5
```

第一版可以只实现 `--challenge-deadline-ms-list`，避免 schedule 推导逻辑过多。

示例：

```bash
uv run streammuse-task play \
  --task zip_zap_zop \
  --deadline-mode menu \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

直接 hard mode：

```bash
uv run streammuse-task play \
  --task zip_zap_zop \
  --deadline-mode hard \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

Challenge mode：

```bash
uv run streammuse-task play \
  --task zip_zap_zop \
  --deadline-mode challenge \
  --challenge-stage-turns 20 \
  --challenge-deadline-ms-list 10000,5000,3000,2000,1000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

## Runtime 设计

### DeadlineMode model

建议新增：

```python
DeadlineMode = Literal["menu", "soft", "hard", "challenge"]
```

并扩展 `InteractiveTaskRuntimeConfig`：

```python
@dataclass(frozen=True)
class InteractiveTaskRuntimeConfig:
    output_dir: str
    deadline_ms: float = 3000.0
    max_tokens: int = 8
    temperature: float = 0.0
    human_first: bool = True
    show_expected: bool = False
    deadline_mode: DeadlineMode = "menu"
    challenge_stage_turns: int = 20
    challenge_deadline_ms_list: tuple[float, ...] = (10000.0, 5000.0, 3000.0, 2000.0, 1000.0)
```

如果 mode 是 `menu`，`InteractiveTaskRuntime.play()` 开始后先调用 terminal 选择菜单，把结果解析成 `soft` / `hard` / `challenge`。

### Runtime state

新增内部 dataclass：

```python
@dataclass
class _DeadlineSessionState:
    mode: Literal["soft", "hard", "challenge"]
    current_deadline_ms: float
    stage_index: int = 0
    stage_turn_count: int = 0
    deadline_misses: list[dict[str, object]] = field(default_factory=list)
    winner: str | None = None
    loser: str | None = None
    stop_reason: str | None = None
```

每个 turn 完成后统一调用：

```python
_handle_turn_outcome(record, deadline_state, stats)
```

它负责：

- soft：append miss / invalid detail，继续。
- hard：如果 deadline miss 或 invalid，设置 loser/winner，停止；否则继续。
- challenge：如果 deadline miss 或 invalid，设置 loser/winner，停止；否则累计 stage turn，达到 stage target 后切换 deadline。

### Trace 扩展

`InteractiveTurnRecord.metadata` 里新增：

```json
{
  "deadline_mode": "challenge",
  "deadline_ms": 5000.0,
  "challenge_stage_index": 1,
  "challenge_stage_turn_count": 13
}
```

`run_summary.json` 新增：

```json
{
  "deadline_mode": "challenge",
  "final_deadline_ms": 5000.0,
  "deadline_misses": [...],
  "invalid_responses": [...],
  "winner": "human",
  "loser": "llm",
  "stop_reason": "deadline_loss"
}
```

`stop_reason` 第一版建议取值：

- `completed`：达到 `max_turns`。
- `user_quit`：用户输入 `:quit`。
- `deadline_loss`：hard/challenge 模式下某一方超时。
- `invalid_response_loss`：hard/challenge 模式下某一方答错。
- `error`：runtime 异常。

`manifest.json` 开始时写 mode / challenge config；结束时 update final status。

## Hard DDL 的技术问题

### Human input timeout

当前 `TerminalIO.prompt()` 基于 `input()`，不能自然 timeout。Hard DDL 需要真正限制 human 输入时间。

建议扩展 terminal protocol：

```python
class TerminalIO(Protocol):
    def write(self, text: str) -> None: ...
    def prompt(self, text: str) -> str: ...
    def prompt_with_timeout(self, text: str, timeout_s: float) -> TimedPromptResult: ...
```

结果类型：

```python
@dataclass(frozen=True)
class TimedPromptResult:
    text: str
    timed_out: bool = False
```

Linux terminal 第一版可以用 `select.select([sys.stdin], [], [], timeout_s)` 实现。测试里 fake terminal 可以直接模拟 timeout。

如果未来要兼容 Windows，再考虑 thread + queue fallback。

### LLM request timeout

当前 `LocalChatModel.generate()` 只用 client config 里的 `timeout_s`。Hard DDL 如果想“到点就输”，LLM 请求也应该能 per-turn timeout。

建议把接口扩展成兼容形式：

```python
def generate(
    self,
    messages: list[ChatMessage],
    *,
    max_tokens: int = 32,
    temperature: float = 0.0,
    timeout_s: float | None = None,
) -> ChatModelResponse:
    ...
```

如果 `timeout_s` 不传，保持原有行为。

Hard/challenge mode 下，runtime 调用 LLM 时传：

```python
timeout_s = current_deadline_ms / 1000.0
```

如果 requests timeout，则构造一个 deadline miss turn，LLM 输。

如果请求没有 timeout 但返回 latency 超过 deadline，也按 deadline miss 处理。

答错不需要额外 timeout 机制；runtime 在拿到 human/LLM response 后照常调用 `task.validate_response()`，并在 hard/challenge mode 下把 `is_valid=false` 当作立即失败条件。

## Terminal UI 行为

启动时如果 `deadline_mode == "menu"`：

```text
Select deadline mode:
1) Soft deadline - keep playing after misses; report them at the end.
2) Hard deadline - first timeout or wrong answer loses immediately.
3) Challenge mode - pass clean stages to reduce the deadline until someone times out or answers wrong.
Choice [1-3]:
```

每轮 prompt 显示当前 deadline：

```text
[11] You > 11:  (deadline 3000ms)
```

Challenge stage 切换时显示：

```text
Stage passed: 20 turns under 10000ms.
New deadline: 5000ms.
```

Hard/challenge 超时后显示：

```text
Deadline missed: llm used 3221.4ms > 3000.0ms.
Winner: human.
```

Hard/challenge 答错后显示：

```text
Wrong answer: human answered 15, expected ZipZop.
Winner: llm.
```

## 测试计划

### Unit tests: deadline state

- soft mode：miss 后继续。
- soft mode：summary 里列出 misses。
- soft mode：invalid response 后继续，并在 summary 里列出 invalid responses。
- hard mode：human timeout 后立即停止。
- hard mode：LLM timeout 后立即停止。
- hard mode：human invalid response 后立即停止。
- hard mode：LLM invalid response 后立即停止。
- hard mode：winner/loser 正确。
- challenge mode：stage 内无 miss，stage deadline 切换。
- challenge mode：切换 deadline 后 transcript/state 不重置。
- challenge mode：任一 turn miss 立即停止。
- challenge mode：任一 turn invalid response 立即停止。
- challenge mode：deadline schedule 用尽后的行为明确。

Schedule 用尽建议第一版策略：

- 如果已经到最后一个 deadline，继续使用最后一个 deadline，直到有人超时/答错输掉，或达到 `max_turns`。

### Unit tests: terminal

- menu 输入 `1` -> soft。
- menu 输入 `2` -> hard。
- menu 输入 `3` -> challenge。
- menu 输入非法值后重新询问。
- `prompt_with_timeout()` 正常输入。
- `prompt_with_timeout()` timeout。

### Unit tests: CLI

- `--deadline-mode soft` 解析正确。
- `--deadline-mode hard` 解析正确。
- `--deadline-mode challenge` 解析正确。
- `--deadline-mode menu` 解析正确。
- `--challenge-stage-turns` 解析正确。
- `--challenge-deadline-ms-list 10000,5000,3000` 解析成 tuple/list。
- 不传 mode 时是否默认 menu，按最终决策测试。

### Regression tests

- 原有 `streammuse-task run` 不受影响。
- 原有 soft-ish play 行为仍可通过 `--deadline-mode soft` 使用。
- `response_trace.jsonl` 仍可读。
- `run_summary.json` 仍可读。
- 音乐 realtime tests 不受影响。

## 实施 Todo

### Phase 0: 决策确认

- [ ] 确认 `play` 默认是否改为 `deadline_mode=menu`。
- [ ] 确认 challenge 默认 deadline list：建议 `10000,5000,3000,2000,1000`。
- [ ] 确认 challenge 每个 stage 默认 turn 数：建议 `20`。
- [ ] 确认 schedule 用尽后是否继续使用最后一个 deadline：建议继续使用。

### Phase 1: 数据结构

- [ ] 新增 `DeadlineMode` type。
- [ ] 新增 `TimedPromptResult`。
- [ ] 新增 `_DeadlineSessionState`。
- [ ] 扩展 `InteractiveTaskRuntimeConfig`。
- [ ] 扩展 `InteractiveTaskRunResult`，加入 mode、winner、loser、stop_reason、deadline_miss detail、invalid_response detail。

### Phase 2: Terminal timeout

- [ ] 扩展 `TerminalIO.prompt_with_timeout()`。
- [ ] 实现 `StdTerminalIO.prompt_with_timeout()`。
- [ ] 用 `select.select()` 实现 Linux timeout。
- [ ] fake terminal 支持 timeout 模拟。
- [ ] 保留 `prompt()` 兼容 soft mode 和普通 command 输入。

### Phase 3: LLM timeout

- [ ] 扩展 `LocalChatModel.generate(..., timeout_s=None)` protocol。
- [ ] 扩展 `LocalChatModelClient.generate()`，per-call timeout 覆盖 config timeout。
- [ ] runtime hard/challenge mode 下按 current deadline 传 timeout。
- [ ] 捕获 requests timeout，转成 deadline miss / loss。
- [ ] 保持 batch `TaskRuntime.run()` 不传 timeout，行为不变。

### Phase 4: Runtime deadline mode

- [ ] play 开始时解析 menu。
- [ ] soft mode 保持继续游戏，但记录 miss details 和 invalid response details。
- [ ] hard mode 任一 miss 立即停止。
- [ ] hard mode 任一 invalid response 立即停止。
- [ ] challenge mode 初始化 stage deadline。
- [ ] challenge mode 每 stage 成功后切换 deadline。
- [ ] challenge mode miss 后立即停止。
- [ ] challenge mode invalid response 后立即停止。
- [ ] 每 turn metadata 写 deadline mode / deadline / stage。
- [ ] summary 写 deadline_misses / invalid_responses / winner / loser / stop_reason。

### Phase 5: CLI

- [ ] 新增 `--deadline-mode`。
- [ ] 新增 `--challenge-stage-turns`。
- [ ] 新增 `--challenge-deadline-ms-list`。
- [ ] 解析 deadline list。
- [ ] 把 mode config 传给 `InteractiveTaskRuntimeConfig`。
- [ ] 更新 help 文案。

### Phase 6: Tests

- [ ] deadline state unit tests。
- [ ] terminal timeout unit tests。
- [ ] LLM timeout unit tests。
- [ ] CLI parser tests。
- [ ] interactive runtime soft/hard/challenge integration-style unit tests。
- [ ] 全量 unit/integration regression。

### Phase 7: 文档

- [ ] 更新 implementation report 或新增 deadline modes report。
- [ ] 写三种模式的运行命令。
- [ ] 写 trace 里如何查看 miss。
- [ ] 写 trace 里如何查看 invalid responses。
- [ ] 写 hard/challenge 的 winner/loser 规则，包括超时和答错两种失败条件。
