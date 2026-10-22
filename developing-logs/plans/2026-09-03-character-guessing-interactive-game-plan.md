# Character Guessing 按轮次双向语音游戏实现计划

> 日期：2026-09-03
>
> 状态：Implementation complete / 待实机 qualification
>
> 目标分支：`feature/game`
>
> 任务名：`character_guessing`

## 1. 目标

新增一个“猜人物”互动游戏：Human 先从一个明确的人物范围中私下选择一人，LLM 通过连续的
yes/no questions 缩小范围，也可以在有把握时提交具体人物猜测；Human 每轮只回答 yes 或 no。
当 Human 对一次正式猜测回答 yes 时，对局立即结束。

目标交互：

```text
Scope: World Historical Figures
Choose one person privately. Do not type the name.
Press Enter when ready.

[Round 1] LLM > Was this person born before 1900?
          You > Yes

[Round 2] LLM > Was this person primarily known as a scientist?
          You > Yes

[Round 3] LLM > Is your person Albert Einstein?
          You > Yes

LLM guessed the person in 3 rounds.
```

游戏必须同时支持：

- terminal text input；
- microphone + faster-whisper voice input；
- LLM answer TTS；
- 当前 task Web UI 和倒计时；
- `soft`、`hard`、`challenge` deadline modes；
- 完整 trace、turn artifact、latency breakdown 和 summary；
- 提前猜中后立即结束，而不是跑满固定 turn 数。

## 2. V1 明确规则

### 2.1 Target 只存在于 Human 心中

Human 从当前 scope 展示的候选名单中选择一人，但**不把人物名称输入程序**。

这样设计有三个原因：

1. target 不会意外进入 LLM prompt；
2. target 不会通过 `TaskState`、turn artifact、WebSocket snapshot 或 debug log 泄漏；
3. 规则与传统 twenty-questions game 一致，Human 是事实 oracle。

因此 V1 不会自动判断 Human 对事实问题的 yes/no 是否真实。最终猜测是否正确，以 Human 对
正式 `GUESS` 回答的 yes/no 为准。受控 benchmark 若需要 machine-verifiable ground truth，
应在后续增加独立 supervised mode，不能偷偷把 target 放进当前 `TaskState.data`。

### 2.2 Scope 是有限、显式、可复现的 catalog

V1 不允许任意自由文本 scope。仓库提供经过审核的英文人物 catalog，每个 scope 包含：

```text
scope_id
title
language
candidates[]:
  id
  display_name
  aliases[]
```

第一版至少提供一个可玩的 `world_historical_figures_en` scope，建议 40-60 人。名单应覆盖不同：

- 性别；
- 世纪；
- 地区；
- 职业/贡献；
- 政治、科学、艺术、文学等类别。

catalog 只定义“可以选和可以猜的人物”，不维护完整事实数据库。LLM 根据自身知识提问，Human
根据自己对所选人物的理解回答。人物事实存在争议时，以当前 Human 回答为本局事实。

V1 catalog 使用英文 display name 和 aliases，与当前 `tiny.en`、英文 prompt 和 TTS 路径一致。
中文人物、虚构人物、娱乐人物和自定义 catalog 留给后续 content pack，不在第一版混入。

### 2.3 一个 round 的精确定义

必须区分：

- **actor turn / attempt**：一次 LLM 输出，或一次 Human 输入；
- **logical round**：一个有效的 LLM `ASK`/`GUESS`，随后得到一个有效 Human yes/no answer。

一个 logical round 固定包含两个成功 actor turns：

```text
LLM action -> Human yes/no -> round completed
```

非法 LLM 输出或非法 Human 输入会记录为 attempt，但不推进 round，也不把状态切给对方。

### 2.4 对局终止

正常终止情况：

| 条件 | `stop_reason` | winner |
|---|---|---|
| Human 对 `GUESS` 回答 yes | `character_guessed` | `llm` |
| 完成 `max_rounds` 仍未猜中 | `round_limit_reached` | `human` |
| Human 输入 `:quit` | `user_quit` | 无 |
| hard/challenge 下某 actor 超时 | `deadline_loss` | 对方 |
| hard/challenge 下某 actor 输出非法 | `invalid_response_loss` | 对方 |
| soft 下同一 action 连续非法达到内部安全上限 | `retry_limit_reached` | 对方 |

默认 `max_rounds=20`。round limit 只能在 Human 完成一个 answer 后检查，不能停在 LLM 已经提问、
Human 尚未回答的半轮状态。

### 2.5 Deadline 语义

每个 actor turn 各自计时：

- LLM deadline：沿用现有 `text` 或 `audio_end` basis；
- Human deadline：在上一轮 TTS 完整 drain 和 speech guard 后开始；
- pre-game 选人和 ready confirmation 不计入 deadline；
- logical round latency 作为派生指标，等于该 round 的 LLM 和 Human 有效 turn latency 之和；
- challenge stage 按完成的 logical rounds 推进，不按 actor attempts 推进。

如果 hard/challenge 模式下 Human 在 deadline 之后才对 `GUESS` 回答 yes，deadline loss 优先，
不能再把该局标成正常猜中。soft 模式仍记录 deadline miss，但允许该 yes 完成本局。

## 3. 为什么不能直接复用当前严格交替循环

当前 `InteractiveTaskRuntime` 用 turn count 的奇偶决定 actor：

```text
human_first=True:  human -> llm -> human -> llm
human_first=False: llm -> human -> llm -> human
```

Character Guessing 需要额外能力：

1. 必须由 LLM 开始；
2. LLM 非法输出后应由 LLM 重试，不能让 Human 回答一个不存在的问题；
3. Human 非法 yes/no 后仍应由 Human 重试；
4. Human 对 guess 回答 yes 后必须立即结束；
5. round limit 必须在完整问答之后检查；
6. challenge progress 必须按 logical round 而不是每个 actor turn 计算。

这些是通用的 interactive flow 能力，不能用 `if task.name == ...` 散落在 runtime 中实现。

## 4. Domain 模型与协议扩展

目标文件：

- `src/streammuse/domain/tasks/models.py`
- `src/streammuse/domain/tasks/character_guessing.py`
- `src/streammuse/domain/tasks/character_scopes.py`
- `src/streammuse/domain/tasks/__init__.py`

### 4.1 Optional flow-control protocol

新增可选 protocol，不强迫 Zip-Zap-Zop 和 Animal Naming 改写：

```python
@dataclass(frozen=True)
class InteractiveFlowDecision:
    next_actor: InteractiveActor | None
    logical_round: int
    is_complete: bool = False
    stop_reason: str | None = None
    winner: InteractiveActor | None = None
    loser: InteractiveActor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FlowControlledInteractiveTask(Protocol):
    def flow_decision(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> InteractiveFlowDecision: ...
```

Runtime 行为：

- task 实现该 protocol 时，由 `flow_decision.next_actor` 决定 actor；
- task 未实现时，保持现有 `human_first + parity` 行为；
- 每次成功/失败 attempt 完成并 advance state 后重新读取 flow decision；
- `is_complete=True` 时应用 task stop reason/winner/loser 并停止；
- task completion 不得覆盖已经发生的 hard/challenge deadline/invalid loss。

### 4.2 Optional setup 和 public-view projection

为避免把人物选择伪装成第一个 Human turn，增加两个小型可选 protocol：

```python
@dataclass(frozen=True)
class InteractiveTaskSetup:
    title: str
    prompt: str
    choices: tuple[str, ...]

@runtime_checkable
class SetupAwareInteractiveTask(Protocol):
    def build_setup(self, state: TaskState) -> InteractiveTaskSetup: ...

@runtime_checkable
class PublicTaskStateProvider(Protocol):
    def public_state(self, state: TaskState) -> dict[str, Any]: ...
```

setup 只展示 scope 和候选人物，并等待 terminal Enter；用户不输入人物名。setup 不写成普通
response turn，也不进入 LLM history。Web UI 只接收 task 主动投影的 public state，禁止 runtime
直接把完整 `TaskState.data` 广播出去。

### 4.3 Character action

LLM 使用易解析且终端可读的两种 wire form：

```text
ASK: Was this person born before 1900?
GUESS: Albert Einstein
```

内部解析为：

```python
CharacterActionKind = Literal["ask", "guess"]

@dataclass(frozen=True)
class CharacterAction:
    kind: CharacterActionKind
    text: str
    canonical_character_id: str | None = None
```

不采用自由文本猜测，也不要求第一版使用 JSON mode。tagged format：

- 比 JSON 更短，适合当前较小的 `max_tokens`；
- 在 terminal/raw trace 中仍然容易阅读；
- 不依赖不同 OpenAI-compatible server 对 `response_format`/guided decoding 的支持；
- 可以通过纯 parser 严格拒绝 thinking、Markdown fence、多行 explanation 和多 action。

### 4.4 TaskState

`CharacterGuessingTask.initial_state()`：

```text
phase: awaiting_llm
scope_id
scope_title
candidate_ids
max_rounds
logical_round: 0
pending_action: null
rounds: []
asked_question_keys: []
guessed_character_ids: []
invalid_attempts_in_phase: 0
completed: false
stop_reason: null
winner: null
```

target 不得出现在 state 中。

`pending_action` 只保存最近一个有效 ASK/GUESS，供下一次 Human answer 使用。每次有效 Human
answer 后，将完整 pair 追加到 `rounds` 并清空 pending action。

### 4.5 状态转换

| 当前 phase | actor/input | valid | 下一个 phase |
|---|---|---:|---|
| `awaiting_llm` | LLM `ASK` | 是 | `awaiting_human` |
| `awaiting_llm` | LLM `GUESS` | 是 | `awaiting_human` |
| `awaiting_llm` | 非法 LLM output | 否 | `awaiting_llm` |
| `awaiting_human` | Human yes/no to ASK | 是 | `awaiting_llm`，round + 1 |
| `awaiting_human` | Human no to GUESS | 是 | `awaiting_llm`，round + 1 |
| `awaiting_human` | Human yes to GUESS | 是 | `finished`，round + 1 |
| `awaiting_human` | 非法 Human response | 否 | `awaiting_human` |

非法 attempt 不覆盖 `pending_action`，也不进入有效 Q/A history。

### 4.6 Referee 的 `is_valid` 含义

在该游戏中，`is_valid` 表示 protocol/规则合法，不表示机器已经猜对：

- Human 对普通问题回答 yes：valid；
- Human 对普通问题回答 no：valid；
- Human 对错误 guess 回答 no：valid；
- Human 对正确 guess 回答 yes：valid + task complete；
- LLM 发出合法但信息量很低的问题：V1 仍 structurally valid；
- LLM 输出 explanation、多 action 或 scope 外人物：invalid。

guess success、action kind 和 logical round 必须进入 referee metadata，不能从 `is_valid` 推断。

## 5. Character catalog

### 5.1 数据要求

每个人物：

```python
@dataclass(frozen=True)
class CharacterCandidate:
    character_id: str
    display_name: str
    aliases: tuple[str, ...] = ()

@dataclass(frozen=True)
class CharacterScope:
    scope_id: str
    title: str
    language: str
    candidates: tuple[CharacterCandidate, ...]
```

构造时 fail fast：

- scope 至少 2 人，第一版最多 100 人；
- id 唯一；
- canonicalized display name/alias 不能指向两个人；
- display name 非空且有明确长度上限；
- V1 名称只允许受控英文字符、空格、连字符、apostrophe 和 period；
- 顺序固定，不依赖 set/dict 的不稳定遍历；
- catalog 不包含生卒年、朝代、职业等 hidden ground truth。

### 5.2 选择体验

setup 同时显示：

- scope title；
- alphabetically sorted candidate names；
- “Choose one privately”；
- ready confirmation。

用户只按 Enter 确认。不得提供 `--target` CLI 参数，因为 shell history/process list 会暴露 target。

### 5.3 Prompt injection 边界

catalog 是 prompt data。V1 只加载仓库内置、代码审查过的 catalog，不接受任意外部 JSON。
以后增加 custom catalog 时，必须先设计 schema、字符/长度限制、provenance 和 prompt escaping，
不能直接把用户文件内容拼到 system prompt。

## 6. LLM 行为设计

### 6.1 System prompt

必须明确：

1. 目标人物一定来自给定 candidate list；
2. 只能输出一行 `ASK: ...` 或一行 `GUESS: ...`；
3. ASK 必须是 Human 可以回答 yes/no 的单一判断题；
4. 禁止 compound question、解释、thinking、Markdown 和编号；
5. 不重复已经问过的问题；
6. 不重复已经被回答 no 的 guess；
7. 尽量选择能够大致二分候选空间的问题；
8. 有足够把握时使用 GUESS；
9. GUESS 名称必须来自 candidate list；
10. Human 的历史回答是本局唯一事实来源，不与 Human 争辩。

### 6.2 User prompt/history

每次 LLM turn 发送：

```text
Scope: World Historical Figures
Candidates: ...
Completed rounds:
1. ASK "Was this person born before 1900?" -> YES
2. ASK "Was this person primarily known as a scientist?" -> YES
Rejected guesses: ...
Rounds remaining: 18
Next action:
```

使用全部有效 rounds，不使用当前 `history_limit=8` 截断关键事实。默认 `max_rounds=20` 下历史规模
有界；如果未来允许很大范围，必须引入显式 state summarization/evaluation，不能静默丢掉早期答案。

非法 attempts 可以作为 bounded protocol-repair message 提示最新错误，但不能混入事实 history。

### 6.3 LLM output validation

`ASK`：

- 单行；
- tag 大小写可规范化；
- question 非空、长度不超过 180 characters；
- 必须恰好以一个 `?` 结束；
- 以受控 yes/no auxiliary 开头，例如 `is/was/were/did/does/do/has/have/had/can/could/would`；
- normalized question 不能与本局已问问题完全重复；
- 不能同时包含 `ASK:`/`GUESS:` 第二个 action。

`GUESS`：

- 单行；
- 使用 display name 或唯一 alias；
- canonicalize 后必须存在于当前 scope；
- 不能重复已被回答 no 的人物；
- 由程序渲染成 spoken/display text：`Is your person <display_name>?`。

结构校验无法可靠判断一个问题是否真的高信息量、事实无争议或暗含两个子问题。V1 通过 prompt、
trace 和实机评估处理，不再调用第二个 LLM 充当 judge，避免增加延迟和不可重复判定。

### 6.4 Model parameters

Character Guessing 的输出比 Zip-Zap-Zop 长，不能沿用 `play` 的默认 `max_tokens=8`。

建议 task-specific defaults：

```text
max_tokens=64
temperature=0
```

保留 CLI override。64 tokens 足够一个短问题，但能限制 thinking runaway。必须用真实 Qwen server
验证 stop behavior；若模型持续输出 thinking，优先使用 server 支持的 no-thinking 参数或 guided
regex，而不是无限增加 `max_tokens`。

## 7. Human text/voice input

### 7.1 Canonical answer

domain 只接收 canonical `yes` 或 `no`。terminal 和 speech parser 共用一个纯规范化 helper。

V1 接受的短语建议：

| Canonical | 可接受输入 |
|---|---|
| `yes` | `yes`, `yeah`, `yep`, `correct`, `right`, `that's right` |
| `no` | `no`, `nope`, `incorrect`, `wrong`, `that's wrong` |

大小写、尾部标点和普通 whitespace 可忽略。不做 fuzzy matching，也不接受包含解释的长句。

`maybe`、`probably`、`unknown`、`sometimes` 在 V1 都是 invalid，因为它们会给 inference history
引入第三种语义。若实机显示 binary answer 经常不足，应单独扩展为 yes/no/unknown，而不是把它们
随意映射成 yes 或 no。

### 7.2 Speech context

```text
initial_prompt:
  "The speaker will answer a yes-or-no question using a short affirmative or negative phrase."

hotwords:
  yes, no, yeah, yep, nope, correct, incorrect, right, wrong
```

Human 不需要说人物名，因此 `faster-whisper tiny.en` 仍合适。建议 qualification 从
`--voice-max-utterance-ms 1000` 开始，VAD 继续使用当前 8-frame observation window。

### 7.3 Prompt 与麦克风时序

LLM TTS 完整 drain 后才开放 Human microphone，并应用当前 speech guard，防止扬声器问题被 STT
录回成为 answer。Human countdown 必须从实际可回答窗口开始，不能在 TTS 播放时提前倒计时。

## 8. TTS output

Character task 实现 `SpeechRenderableTask`：

- `ASK`：朗读 question body；
- `GUESS`：朗读程序生成的 `Is your person <name>?`；
- invalid LLM output：不朗读；
- Human yes/no：不走输出 TTS；
- `speech_vocabulary()` 返回空 tuple，因为问题是动态生成的，不能预热完整词表；
- TTS 文本来自严格 parser 输出，不把 raw model response 直接交给 shell/backend。

当前 `speech-cache-miss=synthesize` 才能支持动态问题。CLI 若选择 audio output 同时设置
`speech-cache-miss=skip`，应在启动时警告该游戏的动态问题将不会被朗读，或直接拒绝该组合。

## 9. Runtime 改造

目标文件：

- `src/streammuse/application/tasks/interactive_runtime.py`
- `src/streammuse/application/tasks/task_events.py`

### 9.1 Actor scheduling

主循环改为：

```text
flow-controlled task:
  decision = task.flow_decision(state, transcript)
  actor = decision.next_actor

legacy task:
  actor = existing human_first/parity rule
```

不能把 Character Guessing 写成 runtime 中的 task-name special case。

### 9.2 Completion application

每次 `_finish_turn()` 完成 state advance、artifact commit、trace append 和 stats update 后，runtime
重新读取 flow decision：

1. 先处理当前 hard/challenge deadline/invalid outcome；
2. 若没有 competitive loss，再应用 task completion；
3. 写入 `winner/loser/stop_reason`；
4. 发出 task progress 和最终 session event；
5. 停止下一 actor turn。

### 9.3 Invalid retry

Character state 在 invalid attempt 后保持当前 phase，所以 flow decision 返回同一个 actor。

- soft：允许同 actor 重试；连续 3 次 invalid 后 safety stop；
- hard/challenge：沿用当前 first invalid loses；
- timeout 在 soft 中记录后，如果没有 canonical response，仍按 invalid retry；
- retry attempt 有独立 deadline、turn artifact 和 trace row；
- logical round index 不增加。

### 9.4 Logical-round progress

扩展运行统计：

```text
attempt_turn_count
logical_round_count
question_count
guess_count
rejected_guess_count
successful_guess_round
```

为兼容旧消费者，现有 `turn_count` 继续表示 actor attempts。新增 `logical_round_count`，不能改变
旧字段含义。

challenge stage 对 flow-controlled task 使用 `logical_round_count` 增量；旧 task 继续使用当前 valid
actor-turn 增量。

### 9.5 Setup lifecycle

顺序：

```text
Web viewer ready
  -> initialize microphone/TTS/model clients
  -> emit session_setup with public scope
  -> terminal displays scope and waits for Enter
  -> emit session_config/running
  -> first LLM turn and countdown
```

setup wait 不打开麦克风、不调用 LLM、不计 turn latency。Ctrl-C 时仍写 startup/session summary，并
正常关闭 Web、input 和 speech resources。

## 10. CLI 设计

目标文件：

- `src/streammuse/presentation/task/cli.py`

新增：

```text
streammuse-task play \
  --task character_guessing \
  --character-scope world_historical_figures_en \
  --max-rounds 20 \
  ...
```

CLI 调整：

- 将 `character_guessing` 加入 interactive registry；
- 不加入普通 `run` benchmark registry，因为没有 Human oracle 就不能运行；
- 增加 `--character-scope`，仅对该 task 合法；
- 增加 `--max-rounds`，仅对该 task 合法，默认 20；
- `--max-turns` 对旧 task 保持原义；Character Guessing 使用内部 attempt safety budget，不让用户
  把 logical rounds 与 actor turns 混淆；
- parser 的 `human_first` 初始值改为未指定状态，再按 task 默认解析：旧游戏默认 Human first，
  Character Guessing 默认 LLM first；
- Character Guessing 若显式传 `--human-first`，启动前拒绝；
- task-specific `max_tokens` 默认 64，旧游戏仍为 8；
- `--start-number` 对 Character Guessing 无意义，显式提供时拒绝或在文档中不出现；
- manifest 保存 resolved scope、max rounds、model settings 和 first actor。

不要用一个不断增长的 `if task_name == ...` 集中解析所有规则。建立小型 task creation/config
mapping，使 task-specific CLI validation 可单测。

## 11. Terminal 和 Web UI

### 11.1 Terminal

终端按 logical round 显示：

```text
[Round 4] LLM thinking... (deadline 5000ms)
    LLM > Is your person a political leader?
    OK action=ASK  620.4 ms

[Round 4] You > Yes or no: (deadline 5000ms)
    ASR > raw="Yes." parsed="yes"
    OK answer=yes  1310.2 ms
```

raw tagged LLM output保留在 artifact，但面向玩家显示解析后的问题，不显示 `ASK:` protocol 噪声。
这需要 task 提供安全的 display projection；不能靠 terminal 对 task name 做字符串处理。

### 11.2 Task events

增加 task-neutral event：

```text
session_setup
task_progress
```

或等价的兼容扩展，但必须满足：

- setup 可以携带公开 scope/candidate list；
- progress 可以携带 phase、logical round、max rounds、action kind；
- `turn_finished` 继续保留 actor attempt 统计；
- 不直接广播完整 TaskState；
- event schema bump 后 snapshot reducer 能读取旧 event fixture；
- event queue drop/reconnect 后 snapshot 仍可恢复当前 pending question。

### 11.3 Web UI

在现有 task-neutral UI 上增加：

- scope title；
- setup candidate list；
- `Round N / 20`；
- 当前 stage：`LLM asking`、`Waiting for yes/no`、`LLM guessing`；
- 最新 question/guess；
- 最新 canonical Human answer；
- final result：猜中人物和 round count，或达到 round limit。

UI 仍然 read-only，不增加网页内 target 输入框。所有动态 question/name/ASR 内容继续使用
`textContent`，禁止 `innerHTML`。倒计时仍仅是 server event 的视觉投影，不参与计分。

## 12. Trace、artifact 和 summary

每个 actor attempt 保留当前 timing breakdown，并增加：

```text
logical_round
phase_before
phase_after
action_kind
display_text
canonical_character_id（仅 guess）
canonical_answer（仅 Human）
round_completed
task_completed
task_stop_reason
```

session summary 增加：

```text
scope_id
scope_size
logical_round_count
question_count
guess_count
rejected_guess_count
successful_guess
successful_character_id
successful_guess_round
```

target 未输入系统，因此 artifact 也不能声称记录了 objective ground truth。报告应把结果描述为
`human-confirmed guess`，不能写成 algorithmically verified correctness。

现有 breakdown analyzer 应继续把 Human turn 拆成 capture/VAD/ASR/parser/referee，把 LLM turn 拆成
HTTP inference/TTS/playback/referee；新增 round-level aggregation，但不破坏旧 trace schema。

## 13. 测试策略

### 13.1 Domain unit tests

新增 `tests/unit/domain/tasks/test_character_guessing_task.py`：

- catalog validation、stable ordering 和 alias collision；
- initial state 不含 target；
- setup public projection；
- LLM-first flow；
- ASK parse/validation；
- GUESS canonicalization 和 scope membership；
- duplicate question 和 repeated rejected guess；
- Human yes/no normalization；
- invalid LLM 保持 `awaiting_llm`；
- invalid Human 保持 `awaiting_human` 且 pending action 不丢失；
- ASK + yes/no 完成一轮；
- GUESS + no 返回下一轮；
- GUESS + yes 完成立即结束；
- max rounds 只在完整 Human answer 后结束；
- prompt 包含全部有效 history、剩余 round 和 rejected guesses；
- prompt、state、public view 和 trace projection 都没有 target field；
- spoken/display rendering 不使用 raw invalid model output。

### 13.2 Runtime tests

更新 `tests/unit/application/tasks/test_interactive_runtime.py`：

- legacy tasks 的 parity 行为完全不变；
- flow-controlled task 可以选择 next actor；
- invalid attempt 重试同一 actor；
- task completion 提前退出；
- deadline loss 优先于同 turn completion；
- round limit 不产生 half-round；
- challenge stage 按 logical round 推进；
- setup 不计 turn/deadline；
- Ctrl-C/setup error resource cleanup；
- summary 同时区分 turn count 和 round count；
- task event failure 仍不影响游戏。

### 13.3 Voice/TTS tests

- yes/no speech variants；
- explanation、repetition、empty/pathological transcript 拒绝；
- hotwords/prompt deterministic；
- ASK 和 GUESS spoken text；
- invalid model output 不朗读；
- dynamic speech cache miss 行为；
- TTS drain + guard 后才打开下一次 Human capture；
- Human deadline origin 不包含 LLM playback。

### 13.4 CLI tests

更新 `tests/unit/presentation/task/test_task_cli.py`：

- task registry 和 play-only restriction；
- default scope/max rounds/LLM first/max tokens；
- reject Human first；
- reject character-only flags on other tasks；
- reject invalid scope/round count；
- old Zip-Zap-Zop/Animal Naming defaults 不变；
- resolved manifest 包含正确 task config。

### 13.5 Web tests

- `session_setup`/`task_progress` reducer；
- reconnect snapshot 保留 pending action 和 round；
- final result projection；
- scope/name/question/ASR XSS payload 只作为 text；
- old event sequence/schema compatibility；
- mobile/desktop 页面无溢出和覆盖。

### 13.6 Hermetic integration

用 scripted fake model 运行完整序列：

```text
ASK male? -> yes
ASK scientist? -> yes
GUESS Newton -> no
GUESS Einstein -> yes
```

覆盖 terminal input、fake voice input、fake speech output、Web sink、trace、summary 和 early stop。
再加入一次 invalid LLM retry 和 invalid Human retry，确认 round count 不被 attempts 污染。

## 14. 实机 qualification

### 14.1 固定场景

- 使用真实 Qwen 27B endpoint；
- 使用 `faster-whisper tiny.en int8`；
- 启用当前 TTS 和 Web UI；
- 使用同一内置 scope；
- 至少选择 10 个不同人物，每个人物玩一局；
- 每局最多 20 rounds；
- 保存 voice audio 只在被试明确同意的 qualification run 中开启。

### 14.2 指标

- 猜中率；
- 猜中所需 rounds；
- question/guess 比例；
- duplicate/invalid LLM output；
- Human yes/no ASR accuracy；
- retry count；
- Human/LLM deadline miss；
- LLM text latency；
- Text -> first DAC；
- TTS playback -> microphone-open guard；
- full logical-round latency；
- ambiguous/compound/factually problematic question 数量。

### 14.3 Acceptance gate

- 完整 unit/integration suite 通过；
- Zip-Zap-Zop 和 Animal Naming 行为无 regression；
- 10 局均无 crash、hung microphone、Web disconnect 或 resource leak；
- final yes 后不再启动下一轮；
- round limit 后不遗留 unanswered LLM question；
- Human yes/no 的实机 parse accuracy 至少 95%；
- valid LLM protocol output 至少 95%，剩余可通过同-phase retry 恢复；
- trace 的 turn/round/action/completion 一致；
- target 没有出现在 prompt、trace、manifest、WebSocket 或 process arguments；
- TTS 回声不被下一 Human turn 误识别为 yes/no；
- Web UI 在 desktop/mobile 截图中无文本重叠。

猜中率不设为纯软件 correctness gate，因为它同时取决于模型知识、提问策略、人物难度和 Human
回答一致性；必须作为实验结果报告，而不是单元测试通过条件。

## 15. 分阶段 TODO List

### Phase 0：冻结当前任务框架

- [x] 保存 `feature/game` branch、commit、working tree 和当前 pytest baseline。
- [x] 用现有 regression tests 冻结 Zip-Zap-Zop、Animal Naming 的 interactive CLI defaults 和 trace contract。
- [ ] 确认现有 voice/TTS/Web UI 实机 smoke 仍能运行。
- [x] 冻结 flow-control 扩展前的 runtime behavior tests。

### Phase 1：Character scope 和 domain parser

- [x] 定义 `CharacterCandidate`、`CharacterScope` 和 validation。
- [x] 创建并审核 `world_historical_figures_en` catalog。
- [x] 实现 name/alias canonicalization。
- [x] 实现 `ASK:` / `GUESS:` parser。
- [x] 实现 yes/no parser。
- [x] 为 catalog 和 parser 添加完整 unit tests。

### Phase 2：通用 interactive flow contract

- [x] 新增 `InteractiveFlowDecision` 和 `FlowControlledInteractiveTask`。
- [x] 新增 setup 和 public-state optional protocols。
- [x] 在 runtime 中加入 task-owned actor scheduling，保留 legacy parity fallback。
- [x] 加入 task completion、round progress 和 precedence 规则。
- [x] 加入 same-actor retry 和 bounded retry safety。
- [x] 补 legacy/nonlegacy runtime regression tests。

### Phase 3：实现 CharacterGuessingTask

- [x] 实现 initial state、setup、public state 和 flow decision。
- [x] 实现 LLM prompt/history builder。
- [x] 实现 actor-aware response validation。
- [x] 实现 pending action 和完整 round state transition。
- [x] 实现 guess success、round limit 和 winner/stop reason。
- [x] 实现 hint/expected behavior。
- [x] 实现全部 domain state-machine tests。

### Phase 4：接入 Human voice 和 LLM TTS

- [x] 实现 SpeechContext 和 spoken yes/no parsing。
- [x] 实现 ASK/GUESS display 和 spoken rendering。
- [x] 保持 dynamic `speech_vocabulary=()`。
- [x] 校验 `speech-cache-miss` 配置。
- [x] 验证 TTS drain/guard/microphone/deadline 顺序。
- [x] 补 voice、speech 和 timing tests。

### Phase 5：CLI 和 task factory

- [x] 注册 `character_guessing` 为 play-only task。
- [x] 增加 `--character-scope` 和 `--max-rounds`。
- [x] 实现 task-specific first actor 和 max-token defaults。
- [x] 实现 incompatible option validation。
- [x] 更新 startup/manifest metadata。
- [x] 补 CLI 和 factory tests。

### Phase 6：Terminal、events 和 Web UI

- [x] 实现 setup/ready terminal flow。
- [x] 按 logical round 渲染 terminal 输出。
- [x] 增加 task-neutral setup/progress events。
- [x] 更新 reconnectable snapshot reducer。
- [x] 增加 scope、round、pending action 和 final result UI。
- [x] 完成安全、schema compatibility 和 responsive UI tests。

### Phase 7：Trace、summary 和 analysis

- [x] 扩展 turn artifact 的 phase/action/round metadata。
- [x] 扩展 session summary 的 character metrics。
- [x] 增加 logical-round latency aggregation。
- [x] 保持旧 trace reader 和 breakdown analyzer 兼容。
- [x] 增加 scripted full-session artifact assertions。

### Phase 8：完整自动化验证

- [x] 运行新增 domain/runtime/CLI/Web tests：99 passed。
- [x] 运行全部 task、voice、speech 和 task-view tests：422 passed。
- [x] 运行完整项目 `tests/` suite：772 passed、4 skipped、15 个既有 Lekai/robustness failures；裸 `pytest` 还会收集 vendored Transformers 并受本机 Keras 3 环境阻断。
- [x] 重复 retry/deadline/thread 相关测试，未发现本功能 flaky failure。
- [x] 检查 `git diff --check`，并构建 wheel 确认 Character modules 与 Web static package data。

### Phase 9：真实双向语音 qualification

- [ ] 启动真实 Qwen 27B endpoint 和 SSH forwarding。
- [ ] 完成至少 10 个人物、每局最多 20 rounds 的测试。
- [ ] 分析 guess rate、round count、LLM protocol、ASR 和 latency。
- [ ] 检查回声、VAD 截断、TTS pronunciation 和 Web countdown。
- [ ] 保存 qualification report 和代表性 trace。
- [ ] 根据数据决定是否增加 `unknown` answer、guided decoding 或更多 scopes。

## 16. 非目标

V1 不实现：

- 自动验证 Human 对历史事实的回答；
- 把 secret target 传给 LLM、Web UI 或 trace；
- 无范围的任意人物猜测；
- 互联网搜索或 knowledge retrieval；
- LLM-as-judge 判断 question 是否“好”；
- 自动 candidate elimination engine；
- yes/no 之外的 uncertain/probabilistic answers；
- 网页中输入 target；
- 多人房间或网络对战；
- 中文 STT/TTS 和中文人物 scope；
- 为新游戏单独复制一套 runtime、voice 或 Web server。

## 17. 主要风险

| 风险 | 后果 | 控制 |
|---|---|---|
| target 进入 TaskState | prompt/trace/UI 泄漏答案 | Human 心选；state/schema 测试禁止 target 字段 |
| 用 actor turn 当 round | 计数翻倍、半轮结束 | 显式 logical round，limit 只在 Human answer 后检查 |
| 继续用 parity scheduler | invalid 后轮到错误 actor | optional flow-control protocol |
| LLM 输出 thinking/explanation | 无法安全提问或 TTS | tagged protocol、短 max tokens、同-phase retry |
| LLM 重复或猜 scope 外人物 | 游戏停滞 | canonical catalog、duplicate validation、history |
| Human answer 模糊 | 推理 history 被污染 | strict yes/no whitelist，不做 fuzzy mapping |
| TTS 被麦克风录回 | 自动产生假 yes/no | drain + guard 后再打开 capture |
| dynamic TTS 未缓存 | 首轮播放延迟增加 | synthesize-on-miss，完整 breakdown measurement |
| hard deadline 与猜中同时发生 | winner 不一致 | deadline/invalid loss precedence 明确定义 |
| Web 直接广播 state | 未来 supervised target 泄漏 | task-owned public projection |
| scope 太大 | prompt/latency 增加 | V1 最多 100，默认 40-60，实测 token/latency |
| 问题含糊或事实有争议 | Human 难以回答 | prompt 约束、qualification 记录，未来再加 unknown |
