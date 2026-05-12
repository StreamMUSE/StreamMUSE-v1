# Lekai Prompt+Continuation Realtime Inference Workflow 研究报告

分支：`merge-lekai-prompt-stanley-system`  
主题：`lekai_prompt_continuation` 实时推理从输入到模型再到播放/日志的完整工作流  
日期：2026-05-12  

## 一句话概括

`lekai_prompt_continuation` 的实时推理不是普通的“一次请求生成一次伴奏”。它是一个 **客户端驱动的 HTTP polling 两阶段流程**：

1. 客户端先实时收集前 `prompt_length_ticks` 的用户旋律。
2. prompt window 到达后，客户端发送 `/prompt_continuation/start`。
3. 后端后台线程运行 Prompt 模型，生成 prompt window 内的开头伴奏。
4. Prompt 伴奏被注入 Continuation 模型的历史。
5. 用户继续演奏时，客户端周期性发送 `/prompt_continuation/append_melody`。
6. 后端 Continuation 模型逐拍生成伴奏，直到伴奏历史比旋律历史多 1 拍。
7. 客户端轮询 `/prompt_continuation/status`，ready 后拉取 `/prompt_continuation/playable`。
8. 客户端把返回的伴奏历史按本地播放策略排进 `PlaybackScheduler`，最后输出到 MIDI/audio/session log。

整体设计是：**Prompt 阶段解决开头上下文不足，Continuation 阶段追赶实时输入，前端负责拉取可播放历史并决定迟到事件怎么播放。**

## 关键文件

| 层级 | 文件 | 职责 |
|---|---|---|
| CLI 入口 | `src/streammuse/presentation/cli/cli.py` | 根据 `--model-name lekai_prompt_continuation` 创建特殊 realtime service |
| 客户端实时服务 | `src/streammuse/application/services/prompt_continuation_realtime_service.py` | 三线程实时编排：输入、tick loop、HTTP protocol |
| HTTP client | `src/streammuse/infrastructure/inference/prompt_continuation_http_client.py` | 封装 `/prompt_continuation/*` 请求 |
| HTTP server | `src/streammuse/infrastructure/inference/server_lekai.py` | FastAPI 路由和 request/response schema |
| Backend wrapper | `src/streammuse/infrastructure/inference/lekai_prompt_continuation/backend.py` | API 边界层，委托给 engine |
| Engine | `src/streammuse/infrastructure/inference/lekai_prompt_continuation/engine.py` | 拥有 prompt engine、continuation engine、scheduler |
| Scheduler | `src/streammuse/infrastructure/inference/lekai_prompt_continuation/scheduler.py` | 后端单 worker 状态机，负责 prompt + catch-up |
| Prompt engine | `src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py` | 构造 prompt tokens，调用 Prompt 模型，解码开头伴奏 |
| Continuation engine | `src/streammuse/infrastructure/inference/lekai_prompt_continuation/continuation_engine.py` | `LekaiHttpBackend` 的薄封装 |
| Continuation backend | `src/streammuse/infrastructure/inference/lekai_http_backend.py` | 加载 continuation checkpoint，逐拍构造 prompt 并生成 acc |

## 整体架构

```text
streammuse-cli
  -> InputSourceFactory
  -> PromptContinuationRealtimeService
     - input_thread
     - tick_thread
     - protocol_thread
  -> PromptContinuationHttpClient
  -> FastAPI server_lekai.py
  -> LekaiPromptContinuationBackend
  -> LekaiPromptContinuationEngine
  -> LekaiPromptContinuationScheduler
     -> LekaiPromptEngine
     -> LekaiContinuationEngine
        -> LekaiHttpBackend
           -> PianoContinuationAdapter / continuation checkpoint
```

## 数据结构

系统内部统一用 `MusicalEvent` 表示音符事件：

```python
@dataclass(frozen=True)
class MusicalEvent:
    tick: int
    pitch: int
    event_type: EventType
    velocity: int = 100
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False
    source: str = "unknown"
    backup_level: int = 0
```

HTTP payload 则是简化后的 JSON：

```json
{
  "type": "note_on",
  "pitch": 60,
  "tick": 0,
  "velocity": 100
}
```

`PromptContinuationHttpClient` 使用 `event_to_dict()` 把 `MusicalEvent` 转成 JSON，服务端再转为 `EventPayload`：

```python
def _melody_payload(notes: List[MelodyNoteEvent]) -> List[EventPayload]:
    return [
        {"type": note.type, "pitch": int(note.pitch), "tick": int(note.tick)}
        for note in notes
    ]
```

注意：服务端 schema 的 `MelodyNoteEvent` 只保留 `type/pitch/tick`，velocity/channel/program 在 melody request 里不会进入 `_melody_payload()`。

## 时间单位

这个功能假设 Lekai tokenization 固定为：

```python
TIMESTEPS_PER_BEAT = 4
```

所以：

- 1 beat = 4 ticks
- `prompt_length_ticks=32` = 8 beats
- `generation_interval_ticks=4` = 每 1 beat 发送一次 append
- continuation catch-up 默认每次生成 1 beat

客户端的真实时间和 tick 互转由 `Tempo` 负责：

```python
def seconds_to_tick(self, seconds: float) -> int:
    return int(seconds / self.seconds_per_tick)

def tick_to_seconds(self, tick: int) -> float:
    return tick * self.seconds_per_tick
```

如果输入是 MIDI 文件，`MidiFileInput` 会按 MIDI 文件时间睡眠并吐出 `tick=0` 的事件；真正的 tick 是 `PromptContinuationRealtimeService._input_worker()` 根据当前 wall-clock elapsed 重新打上的：

```python
elapsed = self._now() - start
tick = self._tempo.seconds_to_tick(elapsed)
stamped = MusicalEvent(
    tick=tick,
    pitch=ev.pitch,
    event_type=ev.event_type,
    velocity=ev.velocity,
    source="user",
)
```

这点很重要：**服务端看到的 tick 是客户端 realtime service 打出来的 tick，不一定是原 MIDI 文件中的原始 tick。**

## 第 0 步：服务器启动

`server_lekai.py` 在模块 import 时创建两个 backend：

```python
backend = LekaiHttpBackend(checkpoint_path=_ENV_CHECKPOINT_PATH)
prompt_continuation_backend = LekaiPromptContinuationBackend(
    checkpoint_path=_ENV_PROMPT_CONTINUATION_CHECKPOINT_PATH,
    prompt_checkpoint_path=_ENV_PROMPT_CHECKPOINT_PATH,
    continuation_checkpoint_path=_ENV_CONTINUATION_CHECKPOINT_PATH,
)
```

相关环境变量：

| 变量 | 用途 |
|---|---|
| `LEKAI_PROMPT_CHECKPOINT_PATH` | Prompt 模型 checkpoint |
| `LEKAI_CONTINUATION_CHECKPOINT_PATH` | Continuation 模型 checkpoint |
| `LEKAI_PROMPT_CONTINUATION_CHECKPOINT_PATH` | 可作为 continuation fallback checkpoint |
| `LEKAI_DEVICE` / `LEKAI_PROMPT_DEVICE` | continuation / prompt 的设备 |
| `LEKAI_DTYPE` / `LEKAI_PROMPT_DTYPE` | continuation / prompt 的 dtype |
| `LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS` | 要求两个真实模型都加载成功，否则启动失败 |
| `LEKAI_DISABLE_FALLBACK` | 禁止 continuation 走 rule stub |

`LekaiPromptContinuationBackend` 创建 `LekaiPromptContinuationEngine`：

```python
self._engine = engine or LekaiPromptContinuationEngine(
    checkpoint_path=checkpoint_path,
    prompt_checkpoint_path=prompt_checkpoint_path,
    continuation_checkpoint_path=continuation_checkpoint_path,
)
```

Engine 内部创建：

```python
self._prompt_engine = prompt_engine or LekaiPromptEngine(
    checkpoint_path=prompt_checkpoint_path
)
self._continuation_engine = continuation_engine or LekaiContinuationEngine(
    checkpoint_path=resolved_continuation_checkpoint
)
self._scheduler = LekaiPromptContinuationScheduler(
    prompt_engine=self._prompt_engine,
    continuation_engine=self._continuation_engine,
)
```

Prompt engine 如果拿到 checkpoint，会立即加载模型并默认 warmup：

```python
if checkpoint_path:
    self._load_model(checkpoint_path)
...
if self._env_bool("LEKAI_PROMPT_WARMUP", True):
    self.warmup()
```

Continuation engine 是薄封装，真正加载 continuation checkpoint 的是 `LekaiHttpBackend`：

```python
class LekaiContinuationEngine:
    def __init__(..., checkpoint_path=None, backend=None):
        self._backend = backend or LekaiHttpBackend(checkpoint_path=checkpoint_path)
```

## 第 1 步：CLI 创建实时服务

CLI 判断 `model_name`：

```python
if config.inference.model_name == "lekai_prompt_continuation":
    prompt_client = PromptContinuationHttpClient(...)
    service = PromptContinuationRealtimeService(
        input_source=input_source,
        prompt_client=prompt_client,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        prompt_length_ticks=int(config.inference.prompt_length_ticks),
        generation_interval_ticks=config.inference.generation_interval_ticks,
    )
else:
    service = RealTimeMusicService(...)
```

这个分支不会创建普通 `InferenceEngineFactory` 的 `HttpInferenceClient`，因为 prompt+continuation 不是普通 `/generate_accompaniment` 请求模式。

`PromptContinuationHttpClient` 会把用户传入的 server URL 归一化。即使 CLI 传的是：

```text
http://127.0.0.1:8001/generate_accompaniment
```

client 也会去掉 `/generate_accompaniment`，之后拼：

```text
/prompt_continuation/start
/prompt_continuation/append_melody
/prompt_continuation/status
/prompt_continuation/playable
```

## 第 2 步：客户端启动三条线程

调用 `service.start()` 后创建三条 daemon thread：

```python
self._input_thread = threading.Thread(target=self._input_worker, daemon=True)
self._tick_thread = threading.Thread(target=self._tick_loop, kwargs={"max_ticks": max_ticks}, daemon=True)
self._protocol_thread = threading.Thread(target=self._protocol_worker, daemon=True)
```

三条线程之间通过三个 queue 通信：

| Queue | 写入方 | 读取方 | 内容 |
|---|---|---|---|
| `_event_q` | input thread | tick thread | 用户输入事件 |
| `_control_q` | tick thread | protocol thread | `start` / `append` 控制动作 |
| `_playable_q` | protocol thread | tick thread | 后端返回的可播放伴奏 |

还有两份客户端 melody buffer：

| Buffer | 内容 |
|---|---|
| `_prompt_events` | `tick < prompt_length_ticks` 的用户事件 |
| `_pending_append_events` | prompt window 后、尚未发送给后端的用户事件 |

## 第 3 步：input thread 读取输入并打 tick

input thread 只做一件事：从 `InputSource.read_events()` 读取事件，用当前 elapsed time 打上 tick，然后放进 `_event_q`。

```python
for ev in self._input.read_events():
    elapsed = self._now() - start
    tick = self._tempo.seconds_to_tick(elapsed)
    stamped = MusicalEvent(tick=tick, ..., source="user")
    self._event_q.put(stamped)
```

这让 MIDI file、MIDI device、keyboard 等输入源都变成同一种 realtime event stream。

## 第 4 步：tick thread 推进音乐时间

tick thread 是客户端的本地时钟。每个 tick 做这些事：

1. 等到当前 tick 对应的 wall-clock 时间。
2. 输出 tick event 到 output sink。
3. 留一个小 input buffer 时间。
4. drain `_event_q`，把用户事件分到 prompt 或 append buffer。
5. 判断是否要 enqueue start。
6. 判断是否要 enqueue append。
7. drain `_playable_q`，把可播放伴奏排进 `PlaybackScheduler`。
8. 从 `PlaybackScheduler` 取当前 tick 的事件并输出。

核心代码：

```python
while self._running:
    target_time = start + self._tempo.tick_to_seconds(tick)
    delay = target_time - self._now()
    if delay > 0:
        self._sleep(delay)

    self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
    self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)

    self._drain_user_events()
    observed_until_tick = tick + 1
    self._maybe_enqueue_start(observed_until_tick)
    self._maybe_enqueue_append(observed_until_tick)

    while True:
        accompaniment, _status = self._playable_q.get_nowait()
        self._schedule_playable(accompaniment, current_tick=tick)

    for event in self._scheduler.get_events_at_tick(tick):
        self._output.output_event(event, source=event.source)
```

`_drain_user_events()` 会立即把用户事件输出到 sink，并按 tick 分桶：

```python
if int(event.tick) < self._prompt_length_ticks:
    self._prompt_events.append(event)
else:
    self._pending_append_events.append(event)
```

所以 user melody 在 `combined.mid` 中是实时写入的，model accompaniment 则要等 playable 被调度后才写入。

## 第 5 步：prompt window 结束后 enqueue start

当 `observed_until_tick >= prompt_length_ticks` 时，tick thread 放入一个 `start` 控制动作：

```python
_ControlAction(
    kind="start",
    melody_events=list(self._prompt_events),
    observed_until_tick=self._prompt_length_ticks,
)
```

注意这里传的是：

- `melody_events`: prompt window 内的真实 note_on/note_off 事件
- `observed_until_tick`: 固定为 `prompt_length_ticks`

即使 prompt window 后面几 tick 是 rest，没有事件，`observed_until_tick` 也告诉后端用户已经走到了这个 tick。后端用它更新 `melody_history_beats`。

## 第 6 步：prompt window 后周期性 enqueue append

start 之后，每当 `(observed_until_tick - prompt_length_ticks) % generation_interval_ticks == 0`，tick thread 发送一次 append：

```python
_ControlAction(
    kind="append",
    melody_events=list(self._pending_append_events),
    observed_until_tick=observed_until_tick,
)
```

`melody_events` 可以为空。空 append 很重要，因为它表示“这段时间用户没有弹音，但时间确实前进了”。后端靠 `observed_until_tick` 更新 melody history beat count。

## 第 7 步：protocol thread 发送 HTTP 请求

protocol thread 启动时先清后端历史：

```python
self._client.clear_history()
```

这实际调用的是普通 `/clear_history`，server 会同时清普通 `backend` 和 `prompt_continuation_backend`。

拿到 `start` action 后，发送：

```python
self._client.start(
    melody_events=action.melody_events,
    prompt_length_ticks=self._prompt_length_ticks,
    generation_interval_ticks=self._generation_interval_ticks,
    observed_until_tick=action.observed_until_tick,
)
```

HTTP body：

```json
{
  "melody_notes": [...],
  "prompt_length_ticks": 32,
  "generation_interval_ticks": 4,
  "observed_until_tick": 32,
  "inference_mode": "sliding_window",
  "model_name": "lekai_prompt_continuation"
}
```

拿到 `append` action 后，发送：

```python
self._client.append_melody(
    melody_events=action.melody_events,
    observed_until_tick=action.observed_until_tick,
)
```

HTTP body：

```json
{
  "melody_notes": [...],
  "observed_until_tick": 36
}
```

protocol thread 只有在 `start` 已发送且至少发送过一次 prompt 之后的 append 后，才开始轮询 status：

```python
if self._protocol_started and self._append_sent_after_prompt:
    status = self._client.status()
```

ready 后，它用 marker 去重：

```python
marker = (
    int(self._append_generation),
    int(status.get("accompaniment_event_count", 0) or 0),
    int(status.get("continuation_calls", 0) or 0),
)
if marker != self._last_playable_marker:
    accompaniment, playable_status = self._client.playable()
    self._playable_q.put((accompaniment, playable_status))
```

marker 的含义：

- append generation 是否增加；
- 后端 accompaniment event count 是否增加；
- continuation call count 是否增加。

只要这些变化，就重新 fetch playable。

## 第 8 步：服务端 `/prompt_continuation/start`

FastAPI schema：

```python
class PromptContinuationStartRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    prompt_length_ticks: int = Field(gt=0)
    generation_interval_ticks: int = Field(gt=0)
    observed_until_tick: Optional[int] = Field(default=None, ge=0)
    inference_mode: str = "sliding_window"
    model_name: str = "lekai_prompt_continuation"
    checkpoint_path: Optional[str] = None
```

路由做三件事：

1. 校验 `model_name` 必须是 `lekai_prompt_continuation`。
2. 把 `melody_notes` 转成 `EventPayload`。
3. 调用 backend `start_prompt_catchup()`。

```python
status = prompt_continuation_backend.start_prompt_catchup(
    melody_events=_melody_payload(request.melody_notes),
    prompt_length_ticks=int(request.prompt_length_ticks),
    generation_interval_ticks=int(request.generation_interval_ticks),
    inference_mode=request.inference_mode,
    model_name=request.model_name,
    checkpoint_path=request.checkpoint_path,
    observed_until_tick=...,
)
```

## 第 9 步：scheduler.start 初始化后台任务

`LekaiPromptContinuationScheduler.start()` 是后端真正的状态起点。

它在 lock 内：

1. 检查是否已有运行中的 future。
2. 设置 phase 为 `prompt_running`。
3. 清空旧历史。
4. 保存 prompt melody 输入。
5. 重置 catch-up state。
6. 保存 prompt/generation 参数。
7. 增加 `run_id`。
8. 根据 `observed_until_tick` 更新 `melody_history_beats`。
9. 提交 `_run_prompt_then_catchup(run_id)` 到单 worker executor。

核心代码：

```python
self._phase = "prompt_running"
self._melody_history = copy_events(melody_events)
self._prompt_melody_input = copy_events(melody_events)
self._prompt_accompaniment_history = []
self._accompaniment_history = []
self._catchup_state.reset()
self._prompt_length_ticks = int(prompt_length_ticks)
self._generation_interval_ticks = int(generation_interval_ticks)
self._run_id += 1

observed_tick = int(observed_until_tick) if observed_until_tick is not None else ...
self._set_melody_observed_until(observed_tick)
self._future = self._executor.submit(self._run_prompt_then_catchup, run_id)
```

`_ticks_to_beats()` 使用向上取整：

```python
return (tick_count + TIMESTEPS_PER_BEAT - 1) // TIMESTEPS_PER_BEAT
```

所以 `observed_until_tick=32` 会变成 8 beats。

## 第 10 步：Prompt 模型生成开头伴奏

后台 worker 执行 `_run_prompt_then_catchup()`。第一步是复制 prompt melody snapshot：

```python
prompt_melody_input = copy_events(self._prompt_melody_input)
prompt_length_ticks = int(self._prompt_length_ticks)
```

然后调用：

```python
prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
    melody_events=prompt_melody_input,
    prompt_start_tick=0,
    prompt_length_ticks=prompt_length_ticks,
)
```

Prompt engine 内部流程如下。

### 10.1 决定 prompt condition 长度

默认 condition length 等于 `prompt_length_ticks`，也可以用 env 覆盖：

```python
condition_beats = self._env_positive_int("LEKAI_PROMPT_CONDITION_BEATS")
if condition_beats is not None:
    condition_ticks = int(condition_beats) * TIMESTEPS_PER_BEAT
    return max(1, min(int(prompt_length_ticks), int(condition_ticks)))
```

### 10.2 构造 Prompt 模型输入 tokens

Prompt engine 从环境变量取 metadata：

```python
time_signature_idx = self._env_int("LEKAI_PROMPT_TIME_SIGNATURE_INDEX", 4)
metadata = {
    "time_signature_idx": time_signature_idx,
    "bpm": self._env_int("LEKAI_PROMPT_BPM", self._env_int("LEKAI_DEFAULT_BPM", 120)),
    "num_measures": num_bars,
}
```

然后把 melody events 转成 piano-roll：

```python
melody_pr = self._converter.events_to_pianoroll(
    events=melody_events,
    start_tick=int(prompt_start_tick),
    end_tick=prompt_end_tick,
    active_pitches=active,
)
```

Prompt 模型需要 4-channel measure，其中前 2 个 channel 是 melody，后 2 个 channel 是空 accompaniment：

```python
measure = np.zeros((4, 88, timesteps_per_bar), dtype=np.uint8)
measure[:2] = melody_pr[:, :, start:end]
```

最后编码成：

```text
[BOS, TS, BPM]
bar, beat, mel_tokens
     beat, mel_tokens
bar, beat, mel_tokens
...
```

代码片段：

```python
ts_token = self._tokenizer.encode_time_sig(metadata["time_signature_idx"])
bpm_token = self._tokenizer.encode_bpm(metadata["bpm"])
_, _, _, measure_beats = self._tokenizer._encode_measures(measures, metadata)

mel_parts = []
for beats in measure_beats:
    mel_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
    for mel, _acc in beats:
        mel_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
        mel_parts.append(mel)

prefix = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)
prompt_tokens = torch.cat([prefix, torch.cat(mel_parts)])
```

### 10.3 调用 Prompt 模型自回归生成

采样参数来自 env：

```python
self._model.generate_music(
    initial_tokens=prompt_tokens,
    device=self._resolved_device,
    max_length=...,
    temperature=self._env_float("LEKAI_PROMPT_TEMPERATURE", 1.1),
    top_k=top_k,
    top_p=self._env_float("LEKAI_PROMPT_TOP_P", 0.95),
    repetition_penalty=self._env_float("LEKAI_PROMPT_REPETITION_PENALTY", 1.0),
)
```

如果设置了 `LEKAI_PROMPT_SEED` 或 `LEKAI_SEED`，Prompt engine 会在生成前重置 torch seed：

```python
torch.manual_seed(int(seed))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))
```

### 10.4 解码 Prompt 输出为 accompaniment events

Prompt model 的输出先解析成 acc beats：

```python
_mel_beats, acc_beats = self._tokenizer.parse_generated_sequence(generated_tokens.squeeze(0))
```

再解码为 piano-roll：

```python
acc_pr = self._tokenizer.decode_beats_to_pianoroll(
    acc_beats,
    track_marker_id=self._tokenizer.vocab.track_marker_acc,
)
```

如果生成超过 prompt window，会裁掉：

```python
if acc_pr.shape[2] > prompt_length_ticks:
    acc_pr = acc_pr[:, :, :prompt_length_ticks]
```

最后转为 event list：

```python
events, _active = self._converter.pianoroll_to_events(
    pianoroll=acc_pr,
    start_tick=int(prompt_start_tick),
    close_at_end=True,
    active_pitches=None,
)
```

`close_at_end=True` 表示 prompt 阶段的持续音会在 prompt window 结束时闭合。

## 第 11 步：Prompt 结果写入 scheduler 历史

Prompt 返回后，scheduler 更新三份状态：

```python
self._prompt_accompaniment_history = copy_events(prompt_accompaniment)
self._accompaniment_history = copy_events(prompt_accompaniment)
self._catchup_state.accompaniment_history_beats = max(
    self._catchup_state.accompaniment_history_beats,
    self._ticks_to_beats(actual_prompt_length_ticks),
)
self._phase = "catchup_running"
melody_snapshot = copy_events(self._melody_history)
self._continuation_sent_melody_event_count = len(self._melody_history)
```

这里的 `melody_snapshot` 很关键：它不只包含最初 prompt window 的旋律，也包含 prompt 运行期间 HTTP thread 通过 append 收到的后续 melody。

## 第 12 步：把 Prompt 伴奏注入 Continuation 历史

Scheduler 调用：

```python
self._continuation_engine.inject_history(
    melody_events=melody_snapshot,
    accompaniment_events=prompt_accompaniment,
    injection_length_ticks=actual_prompt_length_ticks,
)
```

Continuation engine 直接委托给 `LekaiHttpBackend.inject_history()`：

```python
self._melody_history = list(melody_events)
self._accompaniment_history = list(accompaniment_events)
self._injection_length_ticks = int(injection_length_ticks)
self._active_pitches = set()
```

这一步的结果是：

- continuation backend 的 melody history = prompt window + prompt 等待期间收到的后续 melody；
- continuation backend 的 accompaniment history = Prompt 模型生成的开头伴奏；
- 后续 continuation generation 会以这些 histories 为上下文。

## 第 13 步：catch-up loop 逐拍追赶

注入后，scheduler 进入 `_run_catchup_loop()`。

每次循环先看还需要多少 beat：

```python
beats_needed = int(self._catchup_state.beats_needed_for_playback())
if beats_needed <= 0:
    self._phase = "ready"
    return
```

catch-up 规则在 `CatchUpState` 中：

```python
def target_playable_accompaniment_beats(self) -> int:
    return self.melody_history_beats + self.playable_lookahead_beats

def beats_needed_for_playback(self) -> int:
    return max(0, self.target_playable_accompaniment_beats() - self.accompaniment_history_beats)
```

默认 `playable_lookahead_beats=1`，也就是说：

```text
playback ready <=> accompaniment_history_beats >= melody_history_beats + 1
```

每次 catch-up 默认只生成 1 beat：

```python
chunk_beats = min(self._max_continuation_chunk_beats, beats_needed)
generation_start_tick = (
    int(self._catchup_state.accompaniment_history_beats) * TIMESTEPS_PER_BEAT
)
```

然后取还没发给 continuation backend 的 melody increments：

```python
sent_melody_event_count = int(self._continuation_sent_melody_event_count)
melody_increment = copy_events(self._melody_history[sent_melody_event_count:])
next_sent_melody_event_count = len(self._melody_history)
```

调用 continuation：

```python
accompaniment, _timings = self._continuation_engine.generate(
    melody_events=melody_increment,
    generation_start_tick=generation_start_tick,
    generation_length_frames=chunk_beats * TIMESTEPS_PER_BEAT,
    generation_interval_ticks=generation_interval_ticks,
    prompt_length_ticks=self._prompt_length_ticks,
    inference_mode=inference_mode,
    model_name=model_name,
    checkpoint_path=checkpoint_path,
)
```

返回后追加历史：

```python
self._accompaniment_history.extend(copy_events(accompaniment))
self._catchup_state.accept_continuation_beats(chunk_beats)
self._continuation_sent_melody_event_count = max(..., next_sent_melody_event_count)
self._continuation_calls += 1
```

## 第 14 步：append_melody 和 catch-up 并发

HTTP `/append_melody` 可以在 prompt_running 或 catchup_running 时进入。

Scheduler append 只拿 lock 做轻量更新：

```python
self._melody_history.extend(copy_events(melody_events))
observed_tick = int(observed_until_tick) if observed_until_tick is not None else ...
self._set_melody_observed_until(observed_tick)
```

如果当前 phase 已经 ready，但新的 append 让 `beats_needed_for_playback() > 0`，scheduler 会重新启动 catch-up loop：

```python
if self._phase == "ready" and ... and self._catchup_state.beats_needed_for_playback() > 0:
    self._phase = "catchup_running"
    self._future = self._executor.submit(self._run_catchup_loop, self._run_id)
```

也就是说，ready 不是终态。用户继续演奏后，状态会从：

```text
ready -> catchup_running -> ready
```

循环切换。

## 第 15 步：Continuation backend 如何生成一拍

Continuation engine 只是包装：

```python
return self._backend.generate(...)
```

真正逻辑在 `LekaiHttpBackend.generate()`。

### 15.1 更新 runtime config 和 melody history

每次 generate：

```python
self.configure(BackendRuntimeConfig(...))
if bpm is not None:
    self._request_bpm = int(bpm)
self._melody_history.extend(melody_events)
```

因为 prompt+continuation 专用 path 的 append 只传 increments，所以这里会把新的 melody increments 追加到 continuation backend 的完整 melody history。

### 15.2 选择真实模型或 fallback

```python
if self._has_real_model():
    accompaniment = self._generate_with_model(...)
else:
    accompaniment = self._fallback_or_raise(...)
```

真实 prompt+continuation continuation checkpoint 会走 `_generate_with_interleaved_prompt()`。

### 15.3 构造 continuation prompt tokens

当前要生成的 beat：

```python
current_beat = int(generation_start_tick) // TIMESTEPS_PER_BEAT
num_beats_to_generate = max(1, int(generation_length_frames) // TIMESTEPS_PER_BEAT)
```

metadata 来自 env / request-level BPM：

```python
effective_bpm = self._request_bpm if self._request_bpm is not None else int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
time_signature_idx = int(os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4"))
measure_beats = self._measure_beats_from_time_signature_idx(time_signature_idx)
```

prefix：

```python
prefix = [
    torch.tensor([LEKAI_BOS_TOKEN], dtype=torch.long),
    torch.tensor([time_sig_token], dtype=torch.long),
    torch.tensor([bpm_token], dtype=torch.long),
]
```

对每个 target beat，内部函数 `build_standard_offline_prompt(target_beat)` 从历史 events 重新编码上下文：

```python
for beat in range(start_beat, target_beat):
    if beat == start_beat or beat % measure_beats == 0:
        seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
    seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))

    acc_tokens, accompaniment_active = self._encode_beat_tokens(
        events=accompaniment_context_events,
        beat_start_tick=beat_start_tick,
        active_pitches=accompaniment_active,
        end_marker=LEKAI_ACC_END_TOKEN,
    )
    seq.append(acc_tokens)

    mel_tokens, melody_active = self._encode_beat_tokens(
        events=self._melody_history,
        beat_start_tick=beat_start_tick,
        active_pitches=melody_active,
        end_marker=LEKAI_MEL_END_TOKEN,
    )
    seq.append(mel_tokens)
```

结构等价于：

```text
[BOS, TS, BPM]
bar
beat acc_tokens mel_tokens
beat acc_tokens mel_tokens
...
bar?
target_beat
```

最后 prompt 以 target beat marker 结束，让模型接下来生成 target beat 的 acc tokens。

### 15.4 生成 target beat 的 acc tokens

```python
generated_beat_tokens = self._generate_part1_tokens_from_prompt(
    prompt_tokens,
    temperature=rt_temperature,
    top_k=rt_top_k,
    top_p=rt_top_p,
    repetition_penalty=rt_repetition_penalty,
)
```

采样参数来自 env：

| 变量 | 默认 |
|---|---|
| `LEKAI_RT_TEMPERATURE` | `1.1` |
| `LEKAI_RT_TOP_K` | `0` |
| `LEKAI_RT_TOP_P` | `0.95` |
| `LEKAI_RT_REPETITION_PENALTY` | `1.0` |

`_generate_part1_tokens_from_prompt()` 使用 KV cache，并一直采样到遇到 acc end marker、bar token 或 beat token：

```python
for _ in range(100):
    outputs = model(...)
    next_token = sample_token(...)
    generated = torch.cat([generated, next_token], dim=1)
    raw_tokens.append(token_val)
    if token_val in {part1_end_marker, bar_token, beat_token}:
        break
```

如果最后一个 token 是结构 token，会去掉：

```python
if valid_tokens and valid_tokens[-1] in {bar_token, beat_token}:
    valid_tokens.pop()
```

如果没有有效 token，则返回 empty token：

```python
if not valid_tokens:
    return [LEKAI_EMPTY_TOKEN]
```

### 15.5 acc tokens 解码成 events

一拍 acc tokens 先解码成 `(2, 88, 4)` piano-roll：

```python
beat_pianoroll = self._decode_acc_beat_tokens(generated_beat_tokens)
```

然后转成 events：

```python
active_snapshot = self._active_pitches_for_decode_boundary(
    accompaniment_context_events,
    beat_start_tick,
)
beat_events, next_active = self._converter.pianoroll_to_events(
    pianoroll=beat_pianoroll,
    start_tick=beat_start_tick,
    close_at_end=False,
    active_pitches=active_snapshot,
    emit_boundary_retrigger_off=False,
)
```

得到的 `normalized_beat_events` 会：

1. 加入本次 response；
2. 加入 `accompaniment_context_events`，供同一个 generate call 内后续 beat 继续作为上下文；
3. 函数返回后由 `generate()` 追加到 backend `_accompaniment_history`。

```python
generated_events.extend(normalized_beat_events)
accompaniment_context_events.extend(normalized_beat_events)
...
self._accompaniment_history.extend(accompaniment)
```

## 第 16 步：status API 返回什么

`/prompt_continuation/status` 返回 scheduler snapshot：

```python
{
    "phase": self._phase,
    "is_running": bool(future_running),
    "is_failed": self._phase == "failed",
    "error": self._error,
    "melody_event_count": len(self._melody_history),
    "accompaniment_event_count": len(self._accompaniment_history),
    "prompt_length_ticks": int(self._prompt_length_ticks),
    "generation_interval_ticks": int(self._generation_interval_ticks),
    "continuation_calls": int(self._continuation_calls),
    **snapshot,
}
```

Catch-up snapshot：

```python
{
    "melody_history_beats": ...,
    "accompaniment_history_beats": ...,
    "playable_lookahead_beats": 1,
    "target_playable_accompaniment_beats": melody + 1,
    "beats_needed_for_playback": ...,
    "is_history_aligned": ...,
    "is_playback_ready": ...,
}
```

Phase 状态：

```text
idle
prompt_running
catchup_running
ready
failed
```

## 第 17 步：playable API 返回什么

`/prompt_continuation/playable` 并不返回“增量伴奏”，而是在 ready 时返回 **完整 accompaniment history**：

```python
def playable_accompaniment(self) -> list[EventPayload]:
    with self._lock:
        if not self._catchup_state.is_playback_ready():
            return []
        return copy_events(self._accompaniment_history)
```

所以客户端每次 ready fetch 拿到的是从 tick 0 开始的 prompt + continuation 全历史。客户端本地用去重 set 避免重复调度。

相关 debug API：

| API | 返回 |
|---|---|
| `/prompt_continuation/playable` | ready 后的完整伴奏历史，否则空 |
| `/prompt_continuation/raw_history` | 无论 ready 与否，完整 prompt+continuation 历史 |
| `/prompt_continuation/prompt_history` | 仅 Prompt 模型生成的 prompt accompaniment |
| `/prompt_continuation/runtime_info` | prompt/continuation 模型加载状态和 scheduler 状态 |

## 第 18 步：客户端调度 playable events

protocol thread 把 playable 放进 `_playable_q` 后，tick thread 在当前 tick 取出并调度。

有两种策略。

### 18.1 默认策略：配对后只排未来/持续音

默认 `_recover_late_events=False`。

流程：

1. `_clip_playable_to_current_tick()` 删除已经完全过去的 note；
2. 如果 note_on 早于 current tick 但 note_off 还在未来，则把 note_on 克隆到 current tick；
3. `_pair_playable_events()` 配对 note_on/note_off；
4. 用 `_scheduled_model_note_keys` 避免重复调度同一 note；
5. 放入 `PlaybackScheduler`，schedule tick 使用原 event tick。

```python
events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(...)
note_pairs, skipped_unpaired = self._pair_playable_events(events_to_schedule)
for note_on, note_off in note_pairs:
    self._scheduler.schedule(model_event, int(event.tick))
```

### 18.2 Recover-late 策略：迟到事件调度到当前 tick

如果设置：

```bash
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
```

则逐 event 处理：

```python
schedule_tick = event_tick if event_tick >= current_tick else current_tick
```

如果设置：

```bash
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
```

过旧的 late `note_on` 会被丢弃：

```python
if current_tick - event_tick > self._recover_late_max_ticks \
   and event.event_type == EventType.NOTE_ON:
    dropped_too_late_note_on += 1
    continue
```

但 late `note_off` 不会因为这个上限被丢，目的是尽量关闭已发声的音。

## 第 19 步：PlaybackScheduler 输出事件

`PlaybackScheduler` 很简单，是一个 `tick -> list[MusicalEvent]` 的线程安全 dict：

```python
def schedule(self, event: MusicalEvent, tick: int) -> None:
    self._schedule.setdefault(tick, []).append(event)

def get_events_at_tick(self, tick: int) -> List[MusicalEvent]:
    return self._schedule.pop(tick, [])
```

tick thread 每 tick 执行：

```python
for event in self._scheduler.get_events_at_tick(tick):
    self._output.output_event(event, source=event.source)
```

这一步才是真正进入 output sink 的 model events。也就是说：

- backend raw history 里有的事件，不一定进入 `combined.mid`；
- 只有被本地调度策略保留下来、且 tick loop 走到对应 tick 的事件，才会被 output sink 记录为可听输出。

## 第 20 步：Session 输出和 raw history 保存

如果 `--output-type session`，`SessionLoggerOutputSink` 会写：

```text
combined.mid
events.jsonl
performance.json
...
```

`combined.mid` 只包含实时输出过的 user/model events。

CLI cleanup 时还会额外拉取后端 debug histories：

```python
raw_accompaniment, raw_status = prompt_client.raw_history()
save_accompaniment_snapshot(
    stem="prompt_continuation_raw_history",
    accompaniment=raw_accompaniment,
    status=raw_status,
)

prompt_accompaniment, prompt_status = prompt_client.prompt_history()
save_accompaniment_snapshot(
    stem="prompt_continuation_prompt_history",
    accompaniment=prompt_accompaniment,
    status=prompt_status,
)
```

每个 snapshot 保存：

```text
prompt_continuation_raw_history.json
prompt_continuation_raw_history_status.json
prompt_continuation_raw_history.mid

prompt_continuation_prompt_history.json
prompt_continuation_prompt_history_status.json
prompt_continuation_prompt_history.mid
```

这些 debug MIDI 会把完整 melody track 和完整 backend accompaniment history 一起写出，不受 live scheduling 丢弃策略影响。

保存完 raw/prompt history 后，CLI 才调用：

```python
prompt_client.clear_history()
```

清空服务端历史，并把清空前返回的 history 写到：

```text
melody_history.json
accompaniment_history.json
```

## 典型时间线示例

假设：

```text
ticks_per_beat = 4
prompt_length_ticks = 32
generation_interval_ticks = 4
max_continuation_chunk_beats = 1
```

### Tick 0-31

客户端只收集用户 melody：

```text
_prompt_events += user events whose tick < 32
```

后端没有开始模型推理，model output 静默。

### Tick 31 结束 / observed_until_tick=32

tick thread enqueue start：

```text
kind=start
melody_events=_prompt_events
observed_until_tick=32
```

protocol thread POST `/prompt_continuation/start`。

后端：

```text
phase = prompt_running
melody_history_beats = 8
submit _run_prompt_then_catchup
```

### Tick 36、40、44...

客户端每 4 ticks enqueue append：

```text
kind=append
melody_events=events since last append, possibly []
observed_until_tick=36/40/44...
```

后端 append 更新：

```text
melody_history += melody_events
melody_history_beats = ceil(observed_until_tick / 4)
```

### Prompt 完成

假设 prompt 生成 8 beats：

```text
prompt_accompaniment_history = prompt acc events
accompaniment_history = prompt acc events
accompaniment_history_beats = 8
phase = catchup_running
inject prompt acc into continuation backend
```

如果此时用户已经 observed 到 tick 44：

```text
melody_history_beats = ceil(44 / 4) = 11
target_playable_accompaniment_beats = 12
beats_needed = 12 - 8 = 4
```

### Catch-up continuation

Scheduler 会生成 4 次，每次 1 beat：

```text
generation_start_tick=32
generation_start_tick=36
generation_start_tick=40
generation_start_tick=44
```

完成后：

```text
accompaniment_history_beats = 12
phase = ready
is_playback_ready = true
```

### Client fetch playable

protocol thread 看到 ready：

```text
GET /prompt_continuation/playable
```

拿到完整 accompaniment history 后放到 `_playable_q`。

tick thread 在当前 tick 调度它。由于 prompt 部分 tick 0-31 已经过去，最终进入 `combined.mid` 的内容取决于默认策略或 recover-late 策略。

## 与普通 realtime service 的关键差异

普通 `RealTimeMusicService` 通常是：

```text
每隔 generation_interval_ticks:
  发一次 /generate_accompaniment
  直接把 response 中的未来事件排入 PlaybackScheduler
```

Prompt+Continuation 是：

```text
先等 prompt window 完成
start 后后端后台运行 prompt model
append 持续更新时间
后端 catch-up 直到伴奏多一拍
client polling ready
ready 后拉完整 playable history
本地再决定晚到事件如何处理
```

所以它本质上多了：

- prompt window 收集期；
- 后端 scheduler 状态机；
- prompt model 生成；
- prompt acc 注入 continuation；
- catch-up 规则；
- playable polling；
- live scheduling policy。

## 主要环境变量

### 模型加载

| 变量 | 说明 |
|---|---|
| `LEKAI_PROMPT_CHECKPOINT_PATH` | Prompt 模型 |
| `LEKAI_CONTINUATION_CHECKPOINT_PATH` | Continuation 模型 |
| `LEKAI_PROMPT_DEVICE` | Prompt 设备 |
| `LEKAI_DEVICE` | Continuation 设备 |
| `LEKAI_PROMPT_DTYPE` | Prompt dtype |
| `LEKAI_DTYPE` | Continuation dtype |
| `LEKAI_PROMPT_WARMUP` | Prompt 是否启动 warmup |
| `LEKAI_WARMUP_STEPS` | Continuation warmup steps |

### Prompt 阶段

| 变量 | 默认 | 说明 |
|---|---|---|
| `LEKAI_PROMPT_TIME_SIGNATURE_INDEX` | `4` | Prompt metadata TS token |
| `LEKAI_PROMPT_BPM` | `LEKAI_DEFAULT_BPM` 或 120 | Prompt metadata BPM token |
| `LEKAI_PROMPT_CONDITION_BEATS` | unset | 覆盖 prompt condition beats |
| `LEKAI_PROMPT_TEMPERATURE` | `1.1` | Prompt sampling |
| `LEKAI_PROMPT_TOP_K` | `0` | Prompt sampling |
| `LEKAI_PROMPT_TOP_P` | `0.95` | Prompt sampling |
| `LEKAI_PROMPT_REPETITION_PENALTY` | `1.0` | Prompt sampling |
| `LEKAI_PROMPT_MAX_NEW_TOKENS` | `2048` | Prompt 生成上限 |
| `LEKAI_PROMPT_SEED` / `LEKAI_SEED` | unset | Prompt 生成前重置全局 seed |

### Continuation 阶段

| 变量 | 默认 | 说明 |
|---|---|---|
| `LEKAI_TIME_SIGNATURE_INDEX` | `4` | Continuation prefix TS token |
| `LEKAI_DEFAULT_BPM` | `120` | Continuation prefix BPM token |
| `LEKAI_RT_TEMPERATURE` | `1.1` | Continuation sampling |
| `LEKAI_RT_TOP_K` | `0` | Continuation sampling |
| `LEKAI_RT_TOP_P` | `0.95` | Continuation sampling |
| `LEKAI_RT_REPETITION_PENALTY` | `1.0` | Continuation sampling |
| `LEKAI_PROMPT_CONTEXT_BEATS` | unset | 限制 continuation prompt 上下文长度 |
| `LEKAI_HISTORY_MAX_TICKS` | `max(512, gen_len*16)` | 后端历史裁剪窗口 |

### 客户端播放

| 变量 | 默认 | 说明 |
|---|---|---|
| `LEKAI_PROMPT_CONTINUATION_TRACE_PATH` | unset | 输出 client trace JSONL |
| `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` | off | late events 是否调度到当前 tick |
| `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` | unset | 太旧 note_on 的丢弃阈值 |

## 关键设计点

### 1. 后端模型调用是单 worker 串行的

Scheduler 使用：

```python
ThreadPoolExecutor(max_workers=1)
```

这避免 prompt model 和 continuation model 同时抢 GPU，也避免同一个 continuation backend history 被并发生成修改。

### 2. HTTP append 不阻塞模型调用

`append_melody()` 只拿 lock 追加 events 和更新 beat count，不直接跑模型。因此 prompt model 运行期间客户端还能持续 append。

### 3. `run_id` 用来废弃旧后台任务

`clear()` 或重新 start 会递增 `_run_id`。后台任务每个阶段检查：

```python
if not self._is_current_run(run_id):
    return
```

这可以避免旧 future 在 clear 后继续写状态。

### 4. playable 是“完整历史”，不是 delta

客户端每次拉取 playable 都拿到完整 accompaniment history，再靠 `_scheduled_model_event_keys` 和 `_scheduled_model_note_keys` 去重。

这让 API 简单，但导致客户端必须自己处理已经过去的事件、重复事件和 sustain clipping。

### 5. raw history 和 audible output 是两回事

后端 raw history：

```text
Prompt-generated acc + continuation-generated acc
```

audible `combined.mid`：

```text
经过客户端 late-event policy 后，实际 output_sink 收到的 model events
```

因此 workflow 中有两条输出线：

```text
后端 raw/prompt histories -> cleanup 时保存 debug MIDI/JSON
本地 PlaybackScheduler output -> combined.mid / audio / websocket
```

## 当前 workflow 的边界和注意事项

### Metadata 不是通过 start/append API 传的

Prompt 和 continuation 的 BPM/拍号主要来自环境变量。`PromptContinuationStartRequest` 目前没有 `bpm`、`time_signature_idx`、`beats_per_bar` 字段。

这意味着同一个 server 处理多首 metadata 不同的曲子时，除非外部重启 server 或改 env，否则模型 prefix metadata 可能不变。

### protocol worker 在第一次 append 后才 polling

代码条件是：

```python
if self._protocol_started and self._append_sent_after_prompt:
    status = self._client.status()
```

所以 start 刚发出、还没 append 时，即使后端已经 ready，客户端也不会 fetch playable。

### `/generate_accompaniment` 不是主路径

虽然 server 有 `_select_backend(model_name)` 支持 `lekai_prompt_continuation`，但真正 realtime prompt+continuation 走的是专用 `/prompt_continuation/*` endpoints。普通 `/generate_accompaniment` 是兼容/遗留单请求路径，不代表当前 realtime workflow。

### Continuation 上下文来自 events re-encode

Realtime continuation 每次生成 target beat 前，是把历史 melody/accompaniment events 重新编码成 tokens。它没有保存 continuation 模型原始生成 token history 作为下一拍 prompt。这对 workflow 很关键，也解释了为什么 raw history 是 event-level，不是 token-level。

## 推荐阅读顺序

如果之后要继续 debug 或改这个 workflow，建议按这个顺序读：

1. `prompt_continuation_realtime_service.py`
   - 先理解客户端三线程和 queue。
2. `scheduler.py`
   - 再理解后端 phase 和 catch-up。
3. `prompt_engine.py`
   - 看 prompt tokens 怎么构造、怎么解码。
4. `lekai_http_backend.py`
   - 看 continuation 每拍怎么从 event history 构造 prompt。
5. `cli.py`
   - 看 session/raw history 怎么保存。

## 最终总结

这个实时推理的核心不是模型单次调用，而是一个跨客户端和服务端的状态协议：

```text
client local time
  -> collect prompt window
  -> start
  -> append melody progress
  -> poll status
  -> fetch full playable history
  -> local scheduling policy
  -> audible/session output

server background scheduler
  -> prompt model generates initial acc
  -> inject prompt acc into continuation backend
  -> continuation model generates one beat at a time
  -> catch up until acc >= melody + 1 beat
  -> expose full playable/raw histories
```

它把“实时演奏”和“非实时模型生成”之间的差距放在 scheduler 和 playable policy 中处理。后端负责尽快补齐 raw accompaniment history，客户端负责决定这些可能迟到的历史事件最终如何进入实时播放。
