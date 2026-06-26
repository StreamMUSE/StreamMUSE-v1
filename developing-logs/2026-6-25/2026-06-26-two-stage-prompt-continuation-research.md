# Two-Stage (Prompt + Continuation) 架构研究（2026-06-26）

分支：`integrate-prompt-continuation-switch`。本文梳理 two-stage（prompt + continuation）模型在本仓库里 **offline** 和 **realtime** 两条路径分别是怎么做的，以及它解决了什么问题。

## 0. 为什么要 two-stage —— 解决冷启动空塌缩

单阶段 sliding-window（`lekai` / `LekaiHttpBackend`）在贪心（top_k=1, temp=0）下会**冷启动塌缩成空伴奏**：第 0 拍贪心选 empty marker(169)，空伴奏反馈进 context，之后每拍继续选空（自我强化空循环，见 `2026-06-25` replay 研究）。

two-stage 用一个**独立的 prompt 模型**先对开头的旋律窗口一次性生成一段"种子伴奏"，再把它作为 history 注入续写模型，从而绕过冷启动。结构上是：

```
melody[0 : prompt_length]  ──► [Stage 1: Prompt 模型]  ──► prompt 伴奏（一次性、全注意力）
                                          │ inject_history
                                          ▼
melody[prompt_length : ...] ──► [Stage 2: Continuation 模型 = 原 sliding-window backend] ──► 续写伴奏
```

关键点：**两个不同的模型 / checkpoint**。Stage 1 是 `lekai_prompt_continuation/prompt_model/`（自带 tokenizer `PianoMusicTokenizer`），Stage 2 复用现有 `lekai_model` 的 `LekaiHttpBackend`。

## 1. 组件地图

推理层 `src/streammuse/infrastructure/inference/lekai_prompt_continuation/`：

| 文件 | 职责 |
|---|---|
| `prompt_engine.py` (450) | **Stage 1**：加载并运行 prompt 模型，`generate_prompt_accompaniment(melody, 0, prompt_length_ticks)` 一次性生成种子伴奏 |
| `continuation_engine.py` (72) | **Stage 2**：薄封装，内部就是 `LekaiHttpBackend`（原单阶段滑窗） |
| `scheduler.py` (359) | **核心协调器**：单后台 worker 线程，串行跑 prompt → 续写 catch-up；offline/realtime 都用它 |
| `catchup_state.py` (81) | 纯状态机（无模型依赖）：以"拍"为单位记 melody/acc 进度，回答"还要生成几拍才能播" |
| `engine.py` (307) | `LekaiPromptContinuationEngine`：组装 prompt+continuation+scheduler，暴露同步 `generate()` 和异步 `start/append/playable` 两套接口 |
| `backend.py` (167) | `LekaiPromptContinuationBackend`：HTTP 请求边界，门面，转发给 engine |
| `prompt_extension_engine.py` / `prompt_extension_scheduler.py` | **变体**：让 prompt 输出延伸 `prompt_extension_ticks` 再交棒，平滑过渡（"bridge"） |
| `token_conversion.py` (95) | 事件拷贝 + `event_representation_summary`（sha256 digest，用于 client/server 表示一致性校验） |
| `prompt_model/` | Stage 1 的模型本体（model/inference/my_tokenizer/PianoDataset/config/Token2Midi…） |

应用 / 表现层：

| 文件 | 职责 |
|---|---|
| `application/services/prompt_continuation_realtime_service.py` (835) | **realtime 客户端编排**：3 线程驱动 start/append/poll/playable + wall-clock 播放 |
| `infrastructure/inference/prompt_continuation_http_client.py` (196) | realtime 控制流 HTTP client（`/prompt_continuation/*`） |
| `presentation/cli/cli.py` | `--continuation-mode prompt_continuation` 时切到上面这条路径 |
| `infrastructure/inference/server_lekai.py` | 同一个 server 同时挂 `lekai` 和 `lekai_prompt_continuation` 两个 backend，按 `model_name` 路由 |

## 2. Stage 1：Prompt 模型（`LekaiPromptEngine`）

`generate_prompt_accompaniment(melody_events, prompt_start_tick=0, prompt_length_ticks)`：

1. `_build_melody_prompt_tokens`：把前 `prompt_length_ticks` 的旋律编码成
   `[BOS, ts_token, bpm_token]` + 每个 bar/beat 的 `[bar_token, beat_marker, melody_tokens]` 交错序列（用 prompt_model 自己的 `PianoMusicTokenizer`，按小节切 pianoroll）。
2. `model.generate_music(...)`：**一次性自回归生成**整段（全注意力，非滑窗）。采样默认 `temperature=1.1, top_p=0.95, top_k=0`，可被 `LEKAI_PROMPT_*` env 覆盖；支持 `LEKAI_PROMPT_SEED` 固定随机种子。
3. `_decode_accompaniment_events`：解析生成 token → 取 acc beats → 还原成 pianoroll → 截到 `prompt_length_ticks` → 转 `note_on/note_off` 事件。
4. `last_generated_acc_beats()`：记录实际产出的伴奏拍数（可能少于请求，下游据此校正 `actual_prompt_length_ticks`）。

条件窗口长度可调：`LEKAI_PROMPT_CONDITION_BEATS` / `_CONDITION_BARS` / `_NUM_BARS`（默认就是 prompt_length 本身）。启动时 `warmup()` 跑一次 dummy 两小节，支付首调用开销。

## 3. Stage 2：Continuation（`LekaiContinuationEngine`）

就是原单阶段 `LekaiHttpBackend` 的薄封装（`generate / inject_history / clear_history` 直接转发）。不同点只是**它被 prompt 伴奏种过 history**，所以不再从空开始。续写本身仍是逐拍滑窗（`LEKAI_PROMPT_CONTEXT_BEATS` 窗口、每拍预测 acc[k]、自身输出反馈）。

## 4. 核心协调器：`LekaiPromptContinuationScheduler` + `CatchUpState`

这是 offline 和 realtime **共用**的引擎，理解它就理解了全部。

`CatchUpState`（以拍计）：
- `melody_history_beats`：已观测到的旋律拍数。
- `accompaniment_history_beats`：已生成的伴奏拍数。
- `beats_needed_for_playback() = max(0, melody_beats + lookahead(=1) − acc_beats)`：还差几拍。
- `is_playback_ready()`：伴奏领先观测旋律 ≥1 拍即可播。

`Scheduler`（单 worker 线程，串行化模型调用）：
- `start(melody, prompt_length_ticks, …, observed_until_tick)` → 后台跑 `_run_prompt_then_catchup`。
- `_run_prompt_then_catchup`：① 调 prompt_engine 生成种子伴奏；② `inject_history` 注入续写模型；③ 进入 `_run_catchup_loop`。
- `_run_catchup_loop`：只要 `beats_needed_for_playback() > 0`，就调 `continuation_engine.generate` 续写一个 chunk（默认 1 拍），把新 melody 增量喂进去，追到位就置 `phase="ready"`。
- `append_melody(events, observed_until_tick)`：随用户演奏不断追加旋律、抬高 `melody_history_beats`；若 prompt 已 ready 但又有新旋律，会**重启** catch-up loop。
- `observed_until_tick` 很关键：用户在边界附近**休止**时，靠它（而非最大事件 tick）告诉调度器"旋律已经走到这一拍了"，避免 catch-up 卡住。
- 查询接口：`playable_accompaniment()`（ready 时返回）/ `raw_accompaniment_history()`（全量）/ `prompt_accompaniment_history()`（只 Stage 1）。

> 偏移语义：续写的 `generation_start_tick = accompaniment_history_beats × 4`，所以续写总是从"已生成伴奏的末尾"接着写，melody 只传**自上次发送以来的增量**。

## 5. Offline 两阶段怎么做

**没有独立的 offline 算法**——offline = **同步驱动同一个 scheduler**，不带 wall-clock。参考 `scripts/benchmark_lekai_spark.py::benchmark_scheduler`：

```python
backend = LekaiPromptContinuationBackend(prompt_checkpoint_path=…, continuation_checkpoint_path=…)
backend.start_prompt_catchup(melody_events=prompt_events,         # 前 prompt_length 的旋律
                             prompt_length_ticks=…, observed_until_tick=prompt_length_ticks, …)
backend.append_melody_events(append_events, observed_until_tick=observed_until_tick)  # 其余旋律一次性追加
while not status["is_playback_ready"]:        # 同步轮询直到 catch-up 完成
    status = backend.scheduler_status(); time.sleep(0.05)
playable   = backend.playable_accompaniment()
raw_hist   = backend.raw_accompaniment_history()
```

也就是说：把整条旋律拆成"prompt 窗 + 其余"，`start` 后 `append` 全部，轮询到 ready，读 `raw_accompaniment_history` 当作离线结果。

另有更细粒度的 offline 入口（`benchmark_prompt` 只跑 Stage 1、`benchmark_continuation` 只跑 Stage 2），用于分别基准测试两个模型。`engine.generate(...)` 还提供一个**纯同步一次性**接口（prompt 生成 → inject → 续写一段），走 `/generate_accompaniment` 路由（server line 275），但 two-stage 的主用法是 scheduler 控制流。

> 历史参照：5 月的 `scripts/prepare_and_compare_lekai_prompt_alignment.py` 把 CLI 产出的 `prompt_continuation_prompt_history.json` 和**外部 RT codebase**（`/data/home/yuanxin/RT-accompanimentV2`）的原始 prompt 参考做对齐比对——这是把本仓库实现对标原始 RT 实现的验证脚本（外部路径本机可能不存在）。

## 6. Realtime 两阶段怎么做

`PromptContinuationRealtimeService`（client 侧 3 线程）+ server 的 `/prompt_continuation/*` 控制流 API。

**HTTP 控制流 API**（非"每窗一请求"，而是 start/append/poll/playable）：
- `POST /prompt_continuation/start`：初始旋律 + `prompt_length_ticks` + `observed_until_tick`，触发后台 prompt 生成。
- `POST /prompt_continuation/append_melody`：随演奏追加旋律 + 新的 `observed_until_tick`。
- `GET  /prompt_continuation/status`：轮询 catch-up 状态机快照。
- `GET  /prompt_continuation/playable`：取"可播"伴奏（领先 1 拍那部分）。
- `GET  /prompt_continuation/raw_history` / `/prompt_history`：全量 / 仅 Stage 1。

**client 3 线程**：
- `_input_worker`：从 InputSource 读旋律，按 wall-clock 戳 tick，进队列。
- `_protocol_worker`：① 旋律累到 prompt 窗后发 `start`；② 之后持续 `append_melody`；③ 轮询 `status`，发现 catch-up 推进（playable marker 变化）就拉 `playable()` 伴奏，丢给 tick 线程排播。
- `_tick_thread`：wall-clock 推进，播放旋律 + 已排好的伴奏。

**表示一致性校验（representation loop）**：client 每次拿到 playable 伴奏会算一个 `event_representation_summary` 的 sha256 digest，和 server 回传的 digest 比对（`playable_representation_match`）。`LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP=1` 时不一致直接抛错——保证 client/server 对"播了什么"达成一致（近期几个 commit 就是在做这个 trace/loop）。

CLI 接线（`cli.py:193`）：`--continuation-mode prompt_continuation` → 用 `PromptContinuationRealtimeService` + `PromptContinuationHttpClient`；否则走旧的 `RealTimeMusicService`。realtime（真人 midi_device）和"模拟"（midi_file input）走的是**同一个 service**，只是 InputSource 不同。

## 7. 接线与配置

- **选择**：CLI `--continuation-mode {standard|prompt_continuation}`（默认 standard）；`--prompt-length-ticks`（默认 32 = 8 拍）。`ContinuationMode` 在 `application/config/models.py`。
- **Server**：同一进程同时构造两个 backend（`backend` = 单阶段；`prompt_continuation_backend` = 两阶段），`_select_backend(model_name)` 按 `model_name` 路由（`lekai` vs `lekai_prompt_continuation`）。
- **Checkpoint env**（server 端）：
  - `LEKAI_PROMPT_CHECKPOINT_PATH` → Stage 1 prompt 模型；
  - `LEKAI_CONTINUATION_CHECKPOINT_PATH`（或 `LEKAI_PROMPT_CONTINUATION_CHECKPOINT_PATH` / 复用 `LEKAI_CHECKPOINT_PATH`）→ Stage 2 续写模型。
  - 两个模型独立加载、独立采样参数（Stage 1 用 `LEKAI_PROMPT_*`，Stage 2 用 `LEKAI_RT_*`）。
  - `LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1` 强制两个模型都必须真加载（否则 fallback 静默返回空）。
- **变体**：`LEKAI_PROMPT_CONTINUATION_ENGINE=prompt_extension`（别名 bridge/extension…）启用延伸变体，`LEKAI_PROMPT_CONTINUATION_EXTENSION_TICKS/BEATS` 控制延伸长度。

## 8. Offline vs Realtime 一句话对照

| 维度 | Offline（two-stage） | Realtime（two-stage） |
|---|---|---|
| 驱动 | 同步轮询 scheduler（无 wall-clock） | 3 线程 + wall-clock 播放 |
| 旋律喂入 | prompt 窗 + 其余一次性 `append` | 随演奏增量 `append` |
| 取结果 | 轮询到 ready，读 `raw_accompaniment_history` | 轮询 `status`，推进时拉 `playable` 排播 |
| 协调器 | **同一个** `LekaiPromptContinuationScheduler` | **同一个** | 
| 入口 | `benchmark_lekai_spark.py` / `engine.generate` | `PromptContinuationRealtimeService` + `/prompt_continuation/*` |

**本质**：two-stage 的"算法"只有一套（prompt 种子 + 续写 catch-up，由 scheduler 实现）。offline 和 realtime 只是**喂旋律的时机**和**取结果的方式**不同——一个同步灌满+等完，一个 wall-clock 增量+边追边播。

## 9. 待确认 / 后续可深挖

- prompt 模型与续写模型是否必须是**配套训练**的两个 checkpoint（prompt 输出的 token 表示要能被续写模型当 history 吃下）。`prompt_model/my_tokenizer.py`(696) 与 `lekai_model/my_tokenizer.py` 的 vocab/marker 对齐关系值得核对。
- `prompt_extension`（bridge）变体相对标准变体的实际收益（过渡处是否更平滑），看 `prompt_extension_scheduler.py`。
- representation digest loop 在 client/server 不一致时的真实触发场景。
- offline 的 `engine.generate`（`/generate_accompaniment` 同步一次性）与 scheduler 控制流两条 offline 入口产出的伴奏是否一致。
- 与之前 `2026-06-25` 的单阶段 replay 结论结合：two-stage 正是为了解决那次发现的"贪心冷启动塌缩"——可在此分支上重跑 replay 验证 two-stage 是否让真人 session 变得可（贪心）复现。
