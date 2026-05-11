# Lekai Prompt-Continuation 功能研究报告

分支：`merge-lekai-prompt-stanley-system`

---

## 概述

本分支新增了一个名为 `lekai_prompt_continuation` 的**两阶段推理模式**，核心思路如下：

1. **Prompt 阶段**：使用一个专门的离线风格 LLaMA 模型，对用户输入的前 N 拍旋律一次性生成对应的伴奏。
2. **Continuation 阶段**：Prompt 模型生成完毕后，已有的实时 Lekai 模型以 Prompt 伴奏为历史，继续逐拍生成后续伴奏。

**引入原因**：旧的实时推理路径从 tick 0 开始生成，几乎没有旋律上下文，导致开头几拍的伴奏质量偏低。Prompt 模型能看到完整的 N 拍旋律窗口，从而产生有音乐根基的开头；Continuation 模型则基于这段高质量历史继续推理，保持风格一致性。

---

## 系统架构

```
streammuse-cli
  → MidiFileInput / 用户实时输入
  → PromptContinuationRealtimeService（客户端，3 个线程）
  → PromptContinuationHttpClient
  → server_lekai.py（FastAPI 服务端）
  → LekaiPromptContinuationBackend
  → LekaiPromptContinuationEngine
  → LekaiPromptContinuationScheduler（单 worker 后台线程）
       → LekaiPromptEngine          （Prompt 模型）
       → LekaiContinuationEngine    （Continuation = 现有 LekaiHttpBackend）
```

---

## 关键组件

### 1. `LekaiPromptEngine` — `lekai_prompt_continuation/prompt_engine.py`

加载并运行 Prompt 模型（`prompt_model/` 下的 `PianoLLaMA` LLaMA 模型）。

**生成时的处理流程：**
- 接收前 N 拍旋律事件。
- 转换为 piano-roll，再用 **Prompt 模型专用 tokenizer**（`prompt_model/my_tokenizer.py`）编码为 token 序列。
- 序列格式：`[BOS][TS][BPM] [bar][beat][TRK_MEL][mel...][TRK_ACC][acc...]...`
- 进行一次完整的自回归生成，得到 Prompt 窗口内的伴奏 token。
- 将生成的 token 解码回 piano-roll，再转换为事件列表。

关键环境变量：
```
LEKAI_PROMPT_CHECKPOINT_PATH    # checkpoint .safetensors 文件路径
LEKAI_PROMPT_TEMPERATURE        # 默认 1.1
LEKAI_PROMPT_TOP_K              # 默认 0（不使用 top-k）
LEKAI_PROMPT_TOP_P              # 默认 0.95
LEKAI_PROMPT_TIME_SIGNATURE_INDEX  # 默认 4
LEKAI_PROMPT_BPM / LEKAI_DEFAULT_BPM  # 默认 120
LEKAI_PROMPT_WARMUP             # 默认 True（启动时预热）
```

### 2. `LekaiContinuationEngine` — `lekai_prompt_continuation/continuation_engine.py`

对现有 `LekaiHttpBackend` 的轻量封装。Prompt 阶段结束后，通过 `inject_history()` 将 Prompt 伴奏注入为历史，之后调用 backend 的常规 `generate()` 循环。

关键环境变量（与现有实时 Lekai 相同）：
```
LEKAI_CONTINUATION_CHECKPOINT_PATH
LEKAI_RT_TEMPERATURE       # 默认 0.8
LEKAI_RT_TOP_K             # 默认 50
LEKAI_RT_TOP_P             # 默认 0.98
LEKAI_RT_REPETITION_PENALTY  # 默认 1.2
```

### 3. `LekaiPromptContinuationScheduler` — `lekai_prompt_continuation/scheduler.py`

核心调度器。使用 `ThreadPoolExecutor(max_workers=1)` 序列化模型调用，同时允许 HTTP 层持续接收旋律追加请求。

**阶段状态机：**
```
idle
  → prompt_running    （调用 start()，Prompt 模型在后台运行）
  → catchup_running   （Prompt 完成，inject_history 完成，正在运行追赶循环）
  → ready             （追赶完成，可播放伴奏已就绪）
  → failed            （任何未处理异常）
```

**`start()` 流程：**
1. 保存 Prompt 旋律输入，重置所有状态。
2. 从 `observed_until_tick` 更新 `melody_history_beats`。
3. 将 `_run_prompt_then_catchup(run_id)` 提交到线程池。

**`_run_prompt_then_catchup()`（后台线程）：**
1. 调用 `prompt_engine.generate_prompt_accompaniment(...)`。
2. 根据 Prompt 输出长度更新 `accompaniment_history_beats`。
3. 调用 `continuation_engine.inject_history(melody, prompt_acc, prompt_length_ticks)`。
4. 将阶段设为 `catchup_running`。
5. 调用 `_run_catchup_loop(run_id)`。

**`_run_catchup_loop()`（后台线程，接续运行）：**
- 循环直到 `beats_needed_for_playback() == 0`。
- 每次迭代：调用 `continuation_engine.generate(chunk_beats * 4 帧)`。
- 将结果追加到 `_accompaniment_history`，递增 `accompaniment_history_beats`。

**`append_melody()`（HTTP 线程）：**
- 扩展 `_melody_history`，更新 `melody_history_beats`。
- 若当前阶段为 `ready` 但追加后又需要更多拍，则重新启动追赶循环。

### 4. `CatchUpState` — `lekai_prompt_continuation/catchup_state.py`

纯节拍计数逻辑。可播放条件为：
```
accompaniment_history_beats >= melody_history_beats + 1
```
`+1` 保证始终有一拍超前可播放（等于历史长度只代表追上了过去，没有下一拍可输出）。

### 5. `LekaiPromptContinuationBackend` — `lekai_prompt_continuation/backend.py`

面向 HTTP 请求的边界层，全部委托给 `LekaiPromptContinuationEngine`，暴露以下接口：
- `start_prompt_catchup()`、`append_melody_events()`、`scheduler_status()`
- `playable_accompaniment()`、`raw_accompaniment_history()`、`prompt_accompaniment_history()`
- `generate()`（旧 `/generate_accompaniment` 路由仍可用）

### 6. 服务端 HTTP 路由 — `infrastructure/inference/server_lekai.py`

在原有 `/generate_accompaniment` 基础上新增以下端点：

| 端点 | 方法 | 用途 |
|---|---|---|
| `/prompt_continuation/start` | POST | 以初始旋律窗口启动 Prompt 模型 |
| `/prompt_continuation/append_melody` | POST | 运行中追加更多旋律 |
| `/prompt_continuation/status` | GET | 轮询调度器阶段和追赶状态 |
| `/prompt_continuation/playable` | GET | 获取可播放伴奏（未就绪时返回空） |
| `/prompt_continuation/raw_history` | GET | Prompt+Continuation 完整伴奏历史 |
| `/prompt_continuation/prompt_history` | GET | 仅 Prompt 模型生成的伴奏 |
| `/prompt_continuation/runtime_info` | GET | 模型加载/预热状态 |

服务器在启动时加载两个模型：
```python
backend = LekaiHttpBackend(checkpoint_path=_ENV_CHECKPOINT_PATH)
prompt_continuation_backend = LekaiPromptContinuationBackend(
    checkpoint_path=_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH,
    prompt_checkpoint_path=_ENV_PROMPT_CHECKPOINT_PATH,
    continuation_checkpoint_path=_ENV_CONTINUATION_CHECKPOINT_PATH,
)
```

### 7. `PromptContinuationHttpClient` — `infrastructure/inference/prompt_continuation_http_client.py`

客户端 HTTP 客户端，对应上述端点。与 `HttpInferenceClient` 独立，因为控制流（start/append/poll/fetch）与普通单次请求-响应模式有本质区别。

### 8. `PromptContinuationRealtimeService` — `application/services/prompt_continuation_realtime_service.py`

客户端实时编排，运行 3 个线程：

**Input worker**：从 `InputSource` 读取事件，打上当前 tick 时间戳，放入 `_event_q`。

**Tick loop**：逐 tick 推进音乐时间。
- 排空 `_event_q`，输出用户事件，将事件按 tick 分类到 `_prompt_events`（tick < prompt_length_ticks）或 `_pending_append_events`。
- 每 tick 以 `observed_until_tick = tick + 1`：调用 `_maybe_enqueue_start()`（Prompt 窗口结束后触发一次）和 `_maybe_enqueue_append()`（之后每 `generation_interval_ticks` 触发一次）。
- 排空 `_playable_q`，调用 `_schedule_playable()`。
- 播放 `PlaybackScheduler` 中当前 tick 的已调度事件。

**Protocol worker**：处理 `_control_q` 中的 start/append 动作，轮询后端状态，获取可播放伴奏，将结果放入 `_playable_q`。

**播放调度策略**——两种模式：
- *默认（配对-仅未来）*：配对 note_on/note_off，丢弃已完全过去的事件，将延伸到现在的持续音截断到 current_tick。
- *Recover-late 模式*（`LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1`）：逐事件调度，若迟到则调度到 current_tick。可选上限 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` 控制过旧的 note_on 是否丢弃。

---

## Tokenizer 边界——重要

Prompt 模型和 Continuation 模型使用**不同的 token 序列格式**，不能直接共用 token。

**Prompt 模型 tokenizer**（`prompt_model/my_tokenizer.py`）：
```
[BOS][TS][BPM] [bar][beat][TRK_MEL][mel...][TRK_ACC][acc...] [beat]... [bar]... [EOS]
```
每拍内旋律在前，伴奏在后。这是"无条件联合生成"格式。

**Continuation 模型 tokenizer**（`lekai_continuation_model/`，RT offline 布局）：
```
[BOS][TS][BPM] [bar] [beat] acc...TRK_ACC mel...TRK_MEL ...
```
伴奏在前的排列顺序。Token ID 与 `lekai_model/` 相同，但拍级结构不同。

**转换路径**：Prompt 输出 token → `parse_generated_sequence()` → acc 拍 token 列表 → `decode_beats_to_pianoroll()` → `pianoroll_to_events()` → 事件 → `inject_history()` 注入 Continuation backend → 用 Continuation tokenizer 重新编码。

---

## Prompt 长度策略

```
4/4 曲目 → 8 拍 Prompt  （prompt_length_ticks = 32）
2/4 曲目 → 8 拍 Prompt  （8 能被 beats_per_bar=2 整除）
3/4 曲目 → 6 拍 Prompt  （必须能被 beats_per_bar=3 整除）
```

约束 `prompt_beats % beats_per_bar == 0` 来自 Prompt 模型的数据准备要求。

CLI 参数：`--prompt-length-ticks`（默认 32 = 8 拍）。Demo 脚本使用 `PROMPT_BEATS=auto`，按拍号自动选择。

---

## CLI 使用方法

用 `--model-name lekai_prompt_continuation` 激活：
```bash
uv run streammuse-cli \
  --model-name lekai_prompt_continuation \
  --server-url http://localhost:8001 \
  --input-mode midi_file --midi-file path/to/song.mid \
  --prompt-length-ticks 32 \
  --output-type session --log-dir realtime_runs/my_run
```

一键 Demo：
```bash
DEVICE=0 IDS="6217163" MAX_TICKS=96 \
  scripts/run_lekai_prompt_continuation_realtime_demo.sh
```

---

## 采样参数

与旧模式（只有一套参数）不同，这里 Prompt 和 Continuation 各有独立的一套参数：

| 阶段 | Temperature | top_k | top_p | rep_penalty |
|---|---|---|---|---|
| Prompt | 1.1 | 0 | 0.95 | 1.0 |
| Continuation（实时） | 0.8 | 50 | 0.98 | 1.2 |

对应环境变量：`LEKAI_PROMPT_TEMPERATURE`、`LEKAI_PROMPT_TOP_K`、`LEKAI_PROMPT_TOP_P`；`LEKAI_RT_TEMPERATURE`、`LEKAI_RT_TOP_K`、`LEKAI_RT_TOP_P`、`LEKAI_RT_REPETITION_PENALTY`。

---

## 会话输出文件

使用 `--output-type session` 时，除常规文件外还会额外写入：

| 文件 | 内容 |
|---|---|
| `combined.mid` | 经调度策略处理后的可听实时输出 |
| `prompt_continuation_prompt_history.mid/json` | 仅 Prompt 模型生成的伴奏 |
| `prompt_continuation_raw_history.mid/json` | Prompt+Continuation 完整历史 |
| `prompt_continuation_*_status.json` | 对应时刻的后端状态快照 |
| `prompt_continuation_client_trace.jsonl` | 客户端调度 trace（设置 `LEKAI_PROMPT_CONTINUATION_TRACE_PATH` 后生效） |

---

## 验证与测试

严格真实模型模式：
```bash
export LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1
export LEKAI_DISABLE_FALLBACK=1
```

批量验证脚本（`scripts/run_cli_prompt_alignment_batch.sh`）检查项：
- 服务器健康且真实模型已加载
- 无 fallback 激活
- Prompt 阶段 CLI 输出与 RT offline 参考输出的事件 SHA 完全一致

---

## 与旧实时路径的对比

| 对比维度 | 旧版（`lekai`） | 新版（`lekai_prompt_continuation`） |
|---|---|---|
| 从 tick 0 开始推理 | 立即开始，几乎无上下文 | 等待 Prompt 窗口结束后再开始 |
| 前 N 拍伴奏 | 逐拍生成，上下文很薄 | Prompt 模型一次性生成，拥有完整窗口上下文 |
| 客户端流程 | 每拍尾触发一次请求 | Start → Append 循环 → 轮询 → 获取可播放 |
| 服务类 | `RealTimeMusicService` | `PromptContinuationRealtimeService` |
| 模型数量 | 1 个 | 2 个（Prompt + Continuation） |
| Tokenizer | `lekai_model/my_tokenizer.py` | Prompt：`prompt_model/my_tokenizer.py`；Continuation：`lekai_continuation_model/` |
| 播放延迟 | 无（立即输出，可能为空） | 追赶完成前保持静默 |

---

## 待解决的开放问题（来自 DESIGN.md）

- 前端在追赶完成后，应只返回第一拍可播放伴奏，还是返回多拍的块？
- 前端应使用 HTTP 轮询还是切换到 WebSocket 来接收就绪通知？
- 追赶延迟最大允许多少？超出后是回退为静音还是保持上一拍？
- `RECOVER_LATE_MAX_TICKS` 是策略参数而非模型修复，需要通过听感测试调优。
