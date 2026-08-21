# StreamMUSE 交互式任务 Web 观察界面计划

日期：2026-08-13

目标分支：`feature/voice`（或另开 `feature/task-web`）

状态：已实现（rev 4）。

修订记录：

- rev 4（2026-08-13）：完成实现。服务端对每个 viewer 使用有界 snapshot subscription，连接建立与 snapshot 捕获在同一把锁内，避免 snapshot/live gap；溢出时直接以最新完整 snapshot 重同步。远程模式比 rev 3 更严格：所有 session 都使用随机 URL token，WebSocket 同时校验 token 与 Origin。

- rev 3（2026-08-13）：把 Web UI 改成启用后的强制开局门槛。新增 `server_ready` / `viewer_ready` 两阶段 handshake；只有 uvicorn 已接受连接、首个浏览器已收到并渲染 boot snapshot 后，才初始化 microphone / STT / TTS / LLM 并开始第一回合。相应取消“Web 启动失败后降级为无界面对局”的策略，并补充等待阶段的 Ctrl-C、startup manifest 和测试要求。
- rev 2（2026-08-13）：按 review 修正 8 处。其中 P0 一处：uvicorn 0.35.0 已无 `install_signal_handlers`，rev 1 的写法既错误又多余。另有三处结构性改动——事件发射点必须对齐真实计时起点（rev 1 会把 speech guard 的 200 ms 错算进时间轴）、新增服务端 snapshot（否则中途打开或重连的浏览器状态永久缺失）、启动就绪与关闭刷新协议（否则端口冲突静默、`session_finished` 丢失）。
- rev 1（2026-08-13）：初稿。

## 1. 目标与范围

给 `streammuse-task play` 加一个**可选启用的只读浏览器界面**，重点是**截止时间的时间轴条**。

范围约束：

- **默认关闭**。不加 `--web-ui` 时，行为、计时、trace 与今天一致。
- **只读游戏界面**。浏览器不接收作答、`:quit` 或其他游戏命令；只允许发送 `viewer_ready` 等传输层 ACK，人类作答仍走终端或麦克风。
- **零新依赖**。`fastapi` 与 `uvicorn` 已在基础依赖里，前端纯静态，无构建步骤、无 npm。
- **不引入前端框架**，vanilla HTML/CSS/JS。
- 倒计时**不显示具体数字**，用一条随时间收缩的时间轴表示。
- **前端保持"哑渲染器"**：所有状态归约放服务端（§3.4），JS 只做 DOM 赋值。这条同时解决了"没有 npm 就没法测前端逻辑"的矛盾。

明确**不在**本次范围内：浏览器输入、历史 session 回放、图表、多局看板。

## 2. 现状调研

### 2.1 运行时已经不是 rev 1 假设的样子

`interactive_runtime.py` 已从 1102 行增长到 **1854 行**，TTS 计划的产物已经落地。与本功能直接相关的是：

- `_apply_pending_speech_guard()`（`interactive_runtime.py:919`）—— 人类回合开头会阻塞最多 `speech_guard_ms`；
- `_new_timing_context(actor)`（`:661`）/ `_mark_timing()`（`:682`）—— 已经存在一套带命名锚点的分阶段计时设施。

**第二条是好消息**：事件发射不需要新造调用点，直接搭在已有的计时锚点上即可，两者天然同步。

### 2.2 音乐侧的 viewer 模式

`presentation/web/server.py`：FastAPI 单进程，`GET /` 发 `index.html`，`StaticFiles` 挂 `/css` `/js`，`WS /ws` 广播，read-only。envelope 是扁平 JSON 用 `type` 区分。前端 `main.js` 133 行，含指数退避重连。

**但它有一个不能照抄的缺陷**：`_broadcaster_loop`（`server.py:70`）**无论有没有浏览器连接都抽干队列**。音乐侧无所谓（钢琴卷帘本来就是流式的），任务侧不行，见 §3.4。

### 2.3 两个任务都可交互，UI 不能只为 Zip-Zap-Zop 设计

```python
# presentation/task/cli.py:34
INTERACTIVE_TASKS = {"animal_naming", "zip_zap_zop"}
```

`animal_naming` 现在**已经可以 play**。它和 ZZZ 的形状不同：

| | zip_zap_zop | animal_naming |
|---|---|---|
| `state.data` | `current_number` | `used_animals` / `attempted_animals` |
| `_number_from_state()` | 整数 | **`None`** |
| `build_human_prompt()` | `"17:"` | `"Name one unused animal:"` |
| `failure_reason` | `EXPECTED_MISMATCH` | `UNKNOWN_ANIMAL` / `REPEATED_ANIMAL` / `EMPTY_RESPONSE` |

rev 1 把"大字号显示 `number`"当成界面中心，对 `animal_naming` 直接是空的。

### 2.4 依赖现状

`fastapi>=0.116.1` 与 `uvicorn>=0.35.0` 已在基础依赖。`package.json` 只服务 VitePress。**本功能零新增依赖。**

## 3. 设计决策

### 3.1 线程方向：runtime 留主线程，uvicorn 进后台线程

`runtime.play()` 是阻塞到底的 while 循环，而 `KeyboardInterrupt` 只投递给主线程。如果把 runtime 放后台线程，现有的 `user_interrupt` → 写 manifest → 返回 130 这条路径会失效。所以 runtime 不动，uvicorn 让位。

#### 3.1.1 不要碰 `install_signal_handlers`（rev 2 修正）

rev 1 要求 `server.install_signal_handlers = False`。**这是错的**：本机 `uvicorn 0.35.0` 的 `Server` 根本没有这个属性，赋值只会产生一个无人读取的野生字段；而在更早的版本里它是**方法**，赋成 `False` 会导致调用时 `bool is not callable`。

实际上什么都不用做，因为 uvicorn 自己会判断：

```python
# uvicorn/server.py — capture_signals()
@contextlib.contextmanager
def capture_signals(self):
    # Signals can only be listened to from the main thread.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    ...
```

后台线程调用 `server.run()` 时会直接跳过信号注册。**这一行为必须由专项测试钉住**，否则将来升级 uvicorn 时会静默回归。

### 3.2 启动就绪：主线程预先 bind socket + 强制 Viewer Ready Gate

`server.run()` 在线程内部才绑定端口，所以端口占用发生在线程里，主线程无从得知——rev 1 说"启动失败降级为警告"，却没给出**知道它失败了**的机制。

`uvicorn.Server.run()` 支持传入已绑定的 socket：

```python
Server.run(self, sockets: list[socket.socket] | None = None) -> None
```

端口先由主线程绑定：

```text
主线程    socket.bind((host, port))      ← 端口冲突在这里同步暴露
          失败 → 写 startup_error manifest → 关闭已创建资源 → 不开始对局
          成功 → 起线程 server.run(sockets=[sock])
```

预 bind 解决的是端口冲突，但“socket 已绑定”不等于 uvicorn 已完成 lifespan，也不等于浏览器已经加载界面。因此启用 `--web-ui` 后必须经过两个 ready gate：

```text
Gate 1: server_ready
  uvicorn 已完成 startup，并且已开始接受 HTTP / WebSocket 连接

Gate 2: viewer_ready
  首个浏览器已完成：
    GET index/static assets
      -> 建立 WebSocket
      -> 收到 boot snapshot
      -> 使用 textContent 完成首次 DOM render
      -> 发回 {"type": "viewer_ready", "session_id": ...}
```

完整启动顺序固定为：

```text
解析并校验 CLI / 创建 run_dir
  -> 预 bind socket
  -> 启动 uvicorn 后台线程
  -> bounded 等待 server_ready；线程提前退出或超时都视为 startup_error
  -> 打印 Task Web UI URL 和 "Waiting for viewer..."
  -> 等待首个合法 WebSocket 的 viewer_ready ACK
  -> 初始化 HumanResponseSource / microphone / STT
  -> 初始化 SpeechOutputSink / TTS
  -> 构造 LLM client
  -> 进入 runtime.play()
  -> 发 session_config
  -> 开始第一回合
```

boot snapshot 的状态为 `waiting_for_game`，只包含 Web session identity、task name 和连接状态，不依赖尚未初始化的 STT / TTS / LLM。`viewer_ready` 是传输层 ACK，不进入 task state、计分、trace 或 Human input protocol，因此不改变“浏览器不提供游戏输入”的约束。

V1 等待首个 viewer 时不设置自动超时：显式启用 `--web-ui` 就表示本局需要这个显示界面。用户可以用 Ctrl-C 取消；该路径必须关闭 server/socket，写 `status="user_interrupt"` manifest，并由 CLI 返回 130。

首个合法 ACK 到达时必须确认该连接仍处于 active connection set。Gate 一旦释放，之后浏览器断线**不得暂停、重置或改变对局**；重连只依靠 §3.4 snapshot 恢复显示。

这比单纯的 `ready_event + startup_error` 更完整：预 bind 同步解决端口冲突，`server_ready` 解决 uvicorn 启动状态，`viewer_ready` 保证第一回合之前页面真的可以显示。

#### 3.2.1 启动前与开局后的故障策略不同

```text
未启用 --web-ui：
  Web 代码和资源完全不创建，现有行为不变。

已启用 --web-ui、第一回合开始前：
  Web server / static assets / WebSocket / viewer_ready 是 required dependency；
  初始化失败时不开始对局，并记录 startup_error 或 user_interrupt。

第一回合开始后：
  Web UI 退化为 best-effort observer；
  断线、重连、队列溢出或渲染失败不得暂停或改变对局。
```

### 3.3 事件发射点必须对齐真实计时起点（rev 2 修正）

rev 1 说"在 `_run_human_turn` / `_run_llm_turn` 开头发事件"。查了实际代码，计分起点根本不在函数开头：

```python
# _run_human_turn（:431 起）
timing = self._new_timing_context("human")
self._apply_pending_speech_guard()        # ← 最多阻塞 speech_guard_ms（默认 200）
prompt_text = task.build_human_prompt(...)
while True:
    ...
    turn_started_s = self._now()          # :459 ← 真正的计分起点
```

```python
# _run_llm_turn（:930 起）
self.terminal.write("[n] LLM thinking...")
messages = task.build_llm_messages(...)
start_s = self._now()                     # :951 ← 真正的计分起点
```

在函数开头启动时间轴，人类回合会把 **200 ms 的 speech guard** 算进 deadline，UI 比真实计分快一截。challenge 最后一挡是 1000 ms，这是 20% 的偏差。

还有第二个问题：`turn_started_s` 在 `while True:` **循环体内**。终端命令（`:hint` / `:expected`）走 `continue` 后会重新取时间——**同一个 turn 会有多个计时窗口**。

修正：

- 事件改名 `turn_attempt_started`，带 `attempt_index`（从 0 起，每次 `continue` 递增）。
- 紧邻 `turn_started_s` / `start_s` 发射，**搭在已有的 `_mark_timing` 锚点上**（§2.1），不新造调用点。
- 终端命令导致重试时重发事件，前端重置时间轴。
- 专项测试断言：UI 收到的 `deadline_ms` 与该回合 `InteractiveTurnRecord.metadata["deadline_ms"]` 相同，且事件时刻与 `turn_started` 锚点一致。

### 3.4 服务端 snapshot：中途打开和重连必须能恢复

音乐侧的广播循环无条件抽干队列。任务侧照抄会导致：

- 首个 viewer 已释放 gate 后，第二个浏览器才打开 → 收不到此前的 `session_config`；
- 断线重连 → 不知道当前第几回合、什么挡位、比分多少；
- 断线期间发生 `turn_finished` → 页面状态永久缺失；
- 队列溢出 → 无法恢复一致状态。

对局只有 20 回合、每回合几秒，第二个观察者中途打开或首个 viewer 断线重连都不是边缘情况。

因此在服务端加一个 latest-state reducer：

```text
event → reduce 进 TaskViewSnapshot（最新状态，覆盖式）
      → 向每个 viewer 的有界 subscription 追加最新完整 snapshot

WS 连接建立与 snapshot 捕获在同一把锁内
  → 先发完整 snapshot
  → 首次 DOM render
  → viewer_ready ACK
  → 后续仍发服务端已归约的完整 snapshot
```

`TaskViewSnapshot` 至少含：`session_id`、`event_seq`、`schema_version`、config、当前 turn（含 `attempt_index` 与剩余时间的服务端计算基准）、challenge stage、累计 stats、最后一次 human/llm 应答、session status。

首个浏览器收到的是 `status="waiting_for_game"` 的 boot snapshot。它的 `viewer_ready` ACK 只释放一次开局 gate；后续重连仍先收 snapshot，但不再影响 runtime 生命周期。

**顺带解决前端测试问题**：状态归约在服务端就意味着它是 **Python 代码**，可以用现有 pytest 覆盖。前端退化成"收到 snapshot/事件 → 往 DOM 赋值"的哑渲染器，剩下的 JS 逻辑少到手动验收足够。这样既满足了 review 第 8 条对前端逻辑的担忧，又不用为了测 JS 引入 npm。

### 3.5 类型化事件契约（rev 2 修正）

rev 1 的协议里没有 `asr()` 方法，事件清单里却要求发 `asr` —— 自相矛盾。而且六个 `dict[str, Any]` 参数与本仓库到处使用冻结 dataclass 的风格不符。

改为单一入口 + 类型化 envelope：

```python
# domain/tasks/models.py
TaskViewEventType = Literal[
    "session_config", "turn_attempt_started", "asr",
    "turn_finished", "stage_changed", "speech_output", "session_finished",
]

@dataclass(frozen=True)
class TaskViewEvent:
    type: TaskViewEventType
    session_id: str
    event_seq: int
    payload: dict[str, Any]        # 已经 JSON 安全，用 _json_safe() 处理过

class TaskEventSink(Protocol):
    def emit(self, event: TaskViewEvent) -> None: ...
    def close(self) -> None: ...
```

单一 `emit()` 让新增事件类型不必改协议，也让 §3.6 的异常包装只需要一处。

### 3.6 事件清单（task-neutral）

字段设计要同时容纳 `zip_zap_zop` 与 `animal_naming`：

| 事件 | payload 关键字段 |
|---|---|
| `session_config` | `task`、`deadline_mode`、`deadline_ms`、`challenge_deadline_ms_list`、`max_turns`、`human_first`、`human_input_mode`、`speech_output_mode`、`show_expected` |
| `turn_attempt_started` | `turn_id`、`attempt_index`、`actor`、`prompt`、`display_value`、`deadline_ms`、`stage_index` |
| `asr`（仅语音） | `turn_id`、`raw_transcript`、`canonical_response`、`parse_status` |
| `speech_output`（仅启用 TTS） | `turn_id`、`status`（合成中/播放中/完成/失败） |
| `turn_finished` | `turn_id`、`actor`、`response`、`is_valid`、`failure_reason`、`latency_ms`、`deadline_missed`、`expected`(条件) |
| `stage_changed` | `stage_index`、`old_deadline_ms`、`new_deadline_ms` |
| `session_finished` | `stop_reason`、`winner`、`loser`、各项计数 |

两个 task-neutral 的关键字段：

- **`prompt`** —— 直接取 `build_human_prompt()` 的返回值。ZZZ 是 `"17:"`，animal naming 是 `"Name one unused animal:"`。UI 不需要知道是哪个任务。
- **`display_value`** —— 需要突出显示的值，ZZZ 是数字，animal naming 为 `None`（UI 此时只显示 prompt）。

`failure_reason` 直接取自 `TaskRefereeResult`，让 UI 能区分 `UNKNOWN_ANIMAL` 和 `REPEATED_ANIMAL`。

`speech_output` 事件的作用：TTS 启用后，"文本就绪"到"播放结束"之间有几百毫秒，没有这个事件界面看起来像卡住了。

`expected` **仅当 `--show-expected` 开启时才下发**——不是前端隐藏，是**服务端根本不发**，否则从 DevTools 就能看到答案。

### 3.7 异常隔离放在运行时侧（rev 2 修正）

rev 1 说"sink 内部吞掉错误"，测试却要求"注入一个总是抛异常的假 sink，对局仍正常"——如果运行时直接调 `sink.emit(...)`，假 sink 的异常照样逃逸，两者矛盾。

包装放在**运行时**：

```python
def _emit_task_event(self, event: TaskViewEvent) -> None:
    try:
        self.task_event_sink.emit(event)
    except Exception:              # 只捕 Exception
        self._task_event_error_count += 1
```

- **只捕 `Exception`**，不捕 `BaseException`。`KeyboardInterrupt` / `SystemExit` 必须原样穿透——这与 TTS 计划 §3.1 的结论一致。
- `close()` 的异常同样不得覆盖对局本身已在飞行的 primary error，沿用 `_preserving_close` 的语义。
- 错误计数写进 run summary，便于事后发现"UI 一直在悄悄失败"。

### 3.8 关闭顺序：先刷新再退出

rev 1 没定义关闭顺序，存在竞态：`session_finished` 入队 → `ExitStack` 立刻关 uvicorn → broadcaster 还没来得及发。

固定为：

```text
runtime 发 session_finished
  -> flush：等待 live queue 抽干或超时（上限 ~500 ms）
  -> 关闭所有 WebSocket 连接
  -> server.should_exit = True
  -> thread.join(timeout=...)      ← 必须有上限
  -> 超时则放弃（线程是 daemon，不阻塞进程退出）
```

### 3.9 时间轴条

`turn_attempt_started` 带 `deadline_ms`，前端用 `requestAnimationFrame` 驱动一条从满到空的条。

- **不显示毫秒数字。** 这是你定的，而且顺带消解了一个真问题：浏览器时钟、WS 推送、渲染都有滞后，显示精确数字会让被试锚定到有偏的值上；challenge 最后一挡 1000 ms，50–150 ms 就是 5–15% 的预算。视觉条无法被精确锚定。
- **超时后进入 overrun 态**（反向增长 + 变色），不停在 0。`soft` 模式超时仍继续对局并记 deadline miss，停住会误导。
- **纯装饰，绝不回流计分。** 权威值永远是 `InteractiveTurnRecord.latency_ms`。前端不上报任何时间。
- 收到 `turn_finished` 或新的 `turn_attempt_started` 时，**必须取消旧的 rAF 句柄**，否则两个动画会同时改同一个 DOM 节点。
- 显示滞后不做每回合 RTT 测量（过度设计），改为在目标机器上一次性测量并记录。

### 3.10 界面

一屏、无滚动、无 canvas：

```text
┌──────────────────────────────────────────────────────┐
│ StreamMUSE · zip_zap_zop             ● Connected     │
├──────────────────────────────────────────────────────┤
│  Name one unused animal:        ← prompt（永远显示）  │
│                    17            ← display_value(可空)│
│   ████████████████████░░░░░░░░░░░░░░░░               │
│   YOU   ZipZap                              ✓        │
│   LLM   ─                          ♪ speaking        │
├──────────────────────────────────────────────────────┤
│ soft · stage 2/5 · 12 turns · 10 ok · 2 miss         │
└──────────────────────────────────────────────────────┘
```

`display_value` 为 `None` 时该行整体隐藏，界面退化成 prompt 主导——`animal_naming` 自然可用。

### 3.11 安全与隐私

浏览器页面里的动态文本**来自 LLM 输出和 ASR 转写，都是不可信输入**。这和 TTS 计划 §3.4.1 判定"朗读文本不可信、必须走 stdin"是同一条理由的另一面。

- **所有动态文本一律 `textContent`，全面禁用 `innerHTML`**。这是核心缓解。
- **WebSocket 校验 `Origin`**：WS 不受同源策略保护，任何网页都能连本机 WS。不匹配就拒绝握手。
- **CSP** 禁止外部 script 与 inline script。
- **`--web-host` 传非 loopback 地址时拒绝**，除非显式加 `--web-allow-remote`。
- 文档写明：启用 Web UI 时 **raw ASR 转写会显示在页面上**，可能含私人语音内容；与语音输入侧"默认不持久化音频"的隐私立场保持一致。

每个 session 生成随机 URL token，浏览器从 query string 带给 WebSocket；握手同时校验 token 与 Origin。这样 `--web-allow-remote` 不会只靠一个容易误配的 Origin 规则保护。默认仍只绑定 loopback，远程监听必须显式启用。

## 4. CLI

```text
--web-ui                 # 默认关闭
--web-host 127.0.0.1
--web-port 8002          # 避开音乐 viewer 8001 与推理服务 8000
--web-allow-remote       # 允许非 loopback 绑定，默认关闭
```

argparse 默认值为 `None`，真实默认由冻结配置填充，从而能拒绝"未开 `--web-ui` 却传了 `--web-*`"。不新增子命令。

启用后 CLI 必须在初始化游戏资源之前输出：

```text
Task Web UI: http://127.0.0.1:8002/?token=<session-token>
Waiting for viewer...
```

收到首个合法 `viewer_ready` 后再输出 `Viewer ready. Initializing game...`。这些是启动状态提示，不进入 trace。

## 5. 代码改动

### 新增

- `domain/tasks/models.py` — `TaskViewEvent`、`TaskViewEventType`、`TaskEventSink`（追加）。
- `application/tasks/task_events.py` — `NullTaskEventSink`、`TaskWebConfig`。
- `infrastructure/task_view/__init__.py`
- `infrastructure/task_view/snapshot.py` — `TaskViewSnapshot` + `reduce(snapshot, event)`，纯函数，**Python 可测**。
- `infrastructure/task_view/websocket.py` — `QueueTaskEventSink`：reduce + 有界 live queue + 溢出计数。
- `presentation/task_web/server.py` — 预 bind socket、FastAPI app、`server_ready` / `viewer_ready` coordination、连接时先发 snapshot、广播循环、关闭协议。
- `presentation/task_web/static/` — `index.html` / `css/style.css` / `js/main.js`

### 修改

- `application/tasks/interactive_runtime.py` — 注入 sink（默认 Null）；`_emit_task_event()` 包装；在 §3.3 对齐后的锚点上发事件。**现有 `terminal.write` 一行不动。**
- `presentation/task/cli.py` — 四个新参数、预 bind、Web ready gate、在 ready 后才构造游戏资源、`ExitStack` 生命周期、等待阶段的 startup manifest。
- 文档。

不改：`_finish_turn` 职责边界、manifest schema、trace 结构、音乐侧任何东西。

## 6. 实施阶段

**阶段 1** 契约与默认关闭：`TaskViewEvent` / `TaskEventSink` / `NullTaskEventSink` / `_emit_task_event` 包装 / 对齐锚点的发射点。
退出标准：不开 `--web-ui` 时现有测试全绿。

**阶段 2** snapshot 与 sink：`reduce()` 纯函数 + 有界队列 + 溢出计数。
退出标准：Python 测试覆盖全部事件类型的归约，含乱序与溢出后恢复。

**阶段 3** 服务器与开局门槛：预 bind、后台线程、`server_ready`、连接时发 boot snapshot、`viewer_ready` ACK、ready 后才初始化游戏资源、关闭协议。
退出标准：端口冲突同步暴露并在对局前失败；没有 viewer ACK 时不初始化 STT/TTS/LLM；Ctrl-C 仍走 `user_interrupt` 且返回 130；`session_finished` 不丢。

**阶段 4** 前端：哑渲染器 + 时间轴条 + rAF 取消。

**阶段 5** 验证与文档：`soft` 与 `challenge` 各一局；第二个 viewer 中途打开与首个 viewer 断线重连；一次性测量显示滞后。

## 7. 测试计划

### 应用层

- 默认 `NullTaskEventSink` 时行为无变化。**判据措辞修正**：不是"trace/manifest 逐字段相同"——`latency_ms`、`run_id`、`output_dir` 天然每次不同——而是"在固定时钟与 run_id 注入下，确定性字段相同"。
- `turn_attempt_started` 的 `deadline_ms` 与该回合 record 的 `metadata["deadline_ms"]` 相同；发射时刻与 `turn_started` 锚点一致，**不包含 speech guard 与 prompt build**；
- 终端命令 `:hint` 触发重试时重发事件且 `attempt_index` 递增；
- 假 sink 每次 `emit` 都抛 `Exception` → 对局正常跑完，错误计数正确；
- 假 sink 抛 `KeyboardInterrupt` → **原样穿透**，仍走 `user_interrupt`；
- `--show-expected` 关闭时 `turn_finished` 的 payload **不含** `expected` 键；
- `animal_naming` 下 `display_value is None` 且 `prompt`、`failure_reason` 正确。

### 基础设施

- `reduce()` 对每类事件的归约；乱序/重复 `event_seq` 的处理；溢出后 snapshot 仍一致；
- 队列有界、溢出计数；payload JSON 安全（`None`、非有限浮点）。

### 表现层

- **uvicorn 在后台线程不注册信号**（钉住 `capture_signals` 的行为，防升级回归）；
- 端口占用 → 同步暴露 → startup_error manifest → 不构造 Human/STT/TTS/LLM 资源 → CLI 非零退出；
- `server_ready` 之前不接受 viewer ready，`server_ready` 超时或线程提前退出时不开始对局；
- 合法浏览器连接后先收到 `waiting_for_game` boot snapshot；只有匹配 `session_id` 的 `viewer_ready` ACK 才释放 gate；
- malformed、错误 session 或非 `viewer_ready` 入站消息不会释放 gate，也不会进入游戏输入路径；
- `viewer_ready` 前 HumanInputFactory、SpeechOutputFactory 和 LLM client 均未构造或启动；
- 等待 viewer 时 Ctrl-C → server/socket 各关闭一次 → `user_interrupt` manifest → CLI 返回 130；
- gate 释放后浏览器断线不暂停对局；重连只恢复 snapshot；
- 第二个 viewer 中途连接先收到 snapshot；首个 viewer 重连后状态完整；
- `session_finished` 在关闭前被刷新出去；
- `join` 有上限，超时不阻塞进程退出；
- 三条退出路径（正常/异常/`KeyboardInterrupt`）都关闭服务器一次，且 `KeyboardInterrupt` 仍产生 `user_interrupt` manifest 并返回 130；
- 未开 `--web-ui` 却传 `--web-*` → 退出 2；非 loopback 且无 `--web-allow-remote` → 退出 2;
- WS `Origin` 不匹配被拒。

### 前端

状态归约已移到服务端（§3.4），前端只剩 DOM 赋值、首次 render 后发送 `viewer_ready` 和 rAF，不引入 npm 测试栈。手动验收清单：页面未完成首次 render 时游戏保持等待；ready 后才开始资源初始化和第一回合；中途断线不暂停；重连恢复；`turn_finished` 后旧 rAF 被取消；soft 超时进 overrun；session 结束后时间轴停止；长 ASR/LLM 文本不撑破布局；`--show-expected` 关闭时 DevTools 里搜不到答案。

## 8. 风险与缓解

| 优先级 | 风险 | 缓解 |
|---|---|---|
| P0 | 线程方向搞反，`KeyboardInterrupt` 收尾失效 | §3.1 runtime 留主线程 + 专项测试断言返回 130 |
| P0 | 时间轴起点早于真实计分起点，把 200 ms guard 算进 deadline | §3.3 对齐 `turn_started_s` / `start_s` 锚点 + 断言与 record 一致 |
| P0 | 被试页面尚未加载，第一回合已经开始 | §3.2 `server_ready` + `viewer_ready` 双 gate；ready 前不初始化游戏资源 |
| P1 | 中途打开/重连的浏览器状态永久缺失 | §3.4 服务端 snapshot，连接时先发全量 |
| P1 | 端口冲突或 uvicorn 启动失败后仍开始实验 | §3.2 主线程预 bind + bounded `server_ready`；失败写 startup_error 并终止本局 |
| P1 | `session_finished` 在关闭竞态中丢失 | §3.8 固定"发 → flush → 关连接 → should_exit → bounded join" |
| P1 | 假 sink 异常逃逸，与测试预期矛盾 | §3.7 运行时侧包装，只捕 `Exception` |
| P1 | LLM/ASR 文本进入 DOM 造成注入 | §3.11 全面 `textContent` + CSP + Origin 校验 |
| P1 | 前端解析终端字符串 | 结构化事件，不做 `TeeTerminalIO` |
| P2 | UI 泄题 | `expected` 服务端条件下发，不是前端隐藏 |
| P2 | 旧 rAF 未取消，两个动画抢同一节点 | §3.9 切换回合时显式取消句柄 |
| P2 | 升级 uvicorn 后信号行为回归 | 专项测试钉住后台线程不注册信号 |

## 9. 验收门槛

1. 不加 `--web-ui` 时，在固定时钟与 run_id 下确定性字段与基线相同，且不需要浏览器。
2. 零新增依赖，零构建步骤，`uv run` 仍是唯一启动命令。
3. 浏览器在 gameplay 层只读，仅允许 transport-level ACK；人类作答路径完全不变。
4. Ctrl-C 仍产生 `user_interrupt` manifest 并返回 130。
5. 启用 `--web-ui` 时，必须完成 `server_ready` 和首个 `viewer_ready` handshake 后才初始化游戏资源并开始第一回合。
6. Web 启动失败时本局在第一回合前明确失败并留下 startup manifest；开局后的 Web 故障不改变对局结果与 trace。
7. 时间轴起点与 `InteractiveTurnRecord` 的计时起点一致，误差在文档规定容差内。
8. 中途断线不暂停对局，重连后能得到完整当前状态。
9. `session_finished` 在进程退出前一定送达。
10. `animal_naming` 与 `zip_zap_zop` 用同一套界面均可正常显示。
11. 显示滞后量级已在目标机器上测量并写入文档。

## 10. 建议的第一个命令

```bash
uv run --extra voice streammuse-task play \
  --task zip_zap_zop \
  --human-input voice \
  --web-ui \
  --deadline-mode soft \
  --deadline-ms 3000 \
  --model-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-7B-Instruct
```

命令启动后先打开 terminal 打印的 URL。CLI 会停在 `Waiting for viewer...`；浏览器完成首次 snapshot render 并回传 `viewer_ready` 后，才初始化语音和模型资源并开始第一回合。之后麦克风不碰 stdin，终端完全让给日志，被试只看浏览器。

## 11. 实施 Todo（完成状态）

- [x] Phase 1：增加类型化 `TaskViewEvent` / `TaskEventSink`、默认 `NullTaskEventSink` 和运行时异常隔离。
- [x] Phase 1：将 human/LLM attempt 事件对齐真实 deadline 起点，增加独立 `deadline_window_started` timing anchor。
- [x] Phase 1：发送 session、ASR、TTS、turn、challenge stage 与 session finished 事件；`expected` 仅在 `--show-expected` 时下发。
- [x] Phase 2：实现纯 Python snapshot reducer、JSON 安全转换、乱序/重复防护和 task-neutral 状态。
- [x] Phase 2：实现每 viewer 有界 subscription；连接与初始 snapshot 原子化，溢出后以最新 snapshot 恢复。
- [x] Phase 3：实现主线程预 bind、后台 uvicorn、bounded `server_ready`、强制 `viewer_ready` gate 和 startup manifest。
- [x] Phase 3：确保 gate 前不构造 Human/STT、TTS 或 LLM 资源；开局后 viewer 故障不影响 runtime。
- [x] Phase 3：实现 flush、关闭 WebSocket、bounded thread join，以及正常断线的 async task 回收。
- [x] Phase 3：增加 CSP、`textContent`、Origin 校验与随机 session token；非 loopback 仍需 `--web-allow-remote`。
- [x] Phase 4：实现 vanilla HTML/CSS/JS 哑渲染器、responsive layout、deadline/overrun 时间轴、rAF 取消和重连。
- [x] Phase 5：增加 reducer、overflow、事件顺序、异常隔离、CLI gate、token/Origin、端口冲突和真实后台 uvicorn 测试。
- [x] Phase 5：补充 README、文档索引、CLI reference 与完整用户指南。
- [ ] Phase 5 手工实验：在目标机器上连接真实 microphone/STT、TTS 与远端 LLM，各完成一局 soft 和 challenge，并记录显示滞后。该项需要实验硬件和正在运行的 LLM 服务，不由自动化测试伪造通过。
