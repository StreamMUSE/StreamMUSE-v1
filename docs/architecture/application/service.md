---
title: application/service — RealTimeMusicService
description: RealTimeMusicService 三线程架构、count-in、beat-tail 推理触发与生命周期管理
---

# application/service — RealTimeMusicService

**源文件**：`src/streammuse/application/services/real_time_music_service.py`

`RealTimeMusicService` 是系统的核心编排服务，负责把输入事件、推理请求、模型输出、metronome 和日志输出对齐到同一条 tick-based 音乐时间线。

---

## `RealTimeServiceRuntime`

```python
@dataclass(frozen=True)
class RealTimeServiceRuntime:
    session_start_time: float
    timeline_start_time: float
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `session_start_time` | `float` | 服务启动的 Unix 时间戳，count-in 从这里开始 |
| `timeline_start_time` | `float` | 正式音乐时间线的 wall-clock 起点；若 `count_in_beats=0`，等于 `session_start_time` |

`timeline_start_time = session_start_time + tempo.tick_to_seconds(count_in_beats * ticks_per_beat)`。

---

## `RealTimeMusicService.__init__(...)`

```python
def __init__(
    self,
    *,
    input_source: InputSource,
    inference_engine: InferenceEngine,
    output_sink: OutputSink,
    tempo: Tempo,
    scheduler: PlaybackScheduler,
    generation_interval_ticks: int = 2,
    generation_length_frames: int = 20,
    count_in_beats: int = 0,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `input_source` | `InputSource` | 必填 | 音乐输入源 |
| `inference_engine` | `InferenceEngine` | 必填 | 推理引擎 |
| `output_sink` | `OutputSink` | 必填 | 输出 Sink，可是 composite |
| `tempo` | `Tempo` | 必填 | 速度配置，决定 tick 到秒的映射 |
| `scheduler` | `PlaybackScheduler` | 必填 | 模型事件调度器 |
| `generation_interval_ticks` | `int` | `2` | 透传到 HTTP 请求和日志的参数；当前 tick loop 不用它决定触发时刻 |
| `generation_length_frames` | `int` | `20` | 每次推理生成的帧数 |
| `count_in_beats` | `int` | `0` | 正式输入和推理前空转的拍数，必须 ≥ 0 |
| `now` | `Callable[[], float]` | `time.time` | 时间获取函数，测试中可替换 |
| `sleep` | `Callable[[float], None]` | `time.sleep` | 睡眠函数，测试中可替换 |

初始化时会计算：

```python
self._count_in_beats = int(count_in_beats)
self._count_in_ticks = self._count_in_beats * int(self._tempo.ticks_per_beat)
```

---

## 整体 Workflow

当前实时推理流程可以按以下顺序理解：

1. `start()` 设置 `_running=True`，记录 `session_start_time` 和 `timeline_start_time`。
2. 启动 `_input_worker`、`_tick_loop`、`_inference_worker` 三个 daemon 线程。
3. `_tick_loop` 先执行 `_run_count_in()`：只输出 metronome tick，不读取正式输入队列，不发送推理请求。
4. `_input_worker` 睡到 `timeline_start_time` 后才开始读取 input source；读到的事件按正式时间线打 `tick>=0`。
5. `_tick_loop` 从 `tick=0` 推进正式音乐时间线。
6. `tick=0` 时，如果 `_melody_history` 里已有注入或提前进入的旋律历史，则发送一次完整历史请求。
7. 每个 tick 会先输出 tick 状态，再等待 10% tick 时长作为输入缓冲窗口，然后 drain 用户输入。
8. 推理响应到达后，先清除 `generation_start_tick` 之后的旧 model 事件，再把新 model 事件 schedule 到 `PlaybackScheduler`。
9. 每个 tick 的播放阶段按顺序输出 metronome、user 事件、model 事件。
10. 在每拍最后一个 tick（默认 3、7、11...）发送下一拍请求：`generation_start_tick=tick+1`。即使这一拍没有新旋律事件，也会发送空增量，让 stateful server 能继续生成。

核心触发代码：

```python
# tick=0: fire once with full melody history (covers injected history).
if tick == 0:
    with self._melody_history_lock:
        notes_for_request = self._melody_history.copy()
    if notes_for_request:
        self._inference_request_queue.put((0, notes_for_request))

# Beat tail (tick 4n-1, n≥1): always send for next beat.
ticks_per_beat = self._tempo.ticks_per_beat
if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
    self._inference_request_queue.put((tick + 1, notes_for_next_request))
    notes_for_next_request = []
```

---

## `_run_count_in()` — count-in pre-roll

```python
def _run_count_in(self) -> None:
    if self._count_in_ticks <= 0:
        return
    self._output.output_status("count_in", f"{self._count_in_beats} beat(s)")
    for elapsed_tick in range(self._count_in_ticks):
        target_time = start + self._tempo.tick_to_seconds(elapsed_tick)
        self._sleep_until(target_time)
        count_tick = elapsed_tick - self._count_in_ticks
        self._output_metronome_tick(tick=count_tick, bar=bar, beat=beat)
```

关键点：

- count-in 使用负 tick：例如 4 拍 count-in、`ticks_per_beat=4` 时，会输出 `tick=-16..-1` 的 metronome ticks。
- 这个阶段不调用 `output_tick()`，不读取正式输入，不触发 `generate_accompaniment()`。
- 如果 MIDI 录制启用了 metronome，`MidiFileOutputSink` 会观察这些负 tick，并把正式 tick=0 的音乐整体向后平移，从而把 count-in 录在 MIDI 文件开头。

---

## `_input_worker()` — 输入线程

读取输入事件并打正式时间线 tick。

```python
start = self._timeline_start_time()
self._sleep_until(start)
for ev in self._input.read_events():
    elapsed = max(0.0, self._now() - start)
    tick = self._tempo.seconds_to_tick(elapsed)
    stamped = MusicalEvent(..., tick=tick, source="user")
    self._event_q.put(stamped)
    with self._melody_history_lock:
        self._melody_history.append(stamped)
```

行为：

1. 输入线程先等到 `timeline_start_time`，因此 count-in 期间不会把用户输入送入模型。
2. InputSource 自身通常发出 `tick=0` 的事件，真实 tick 由 service 根据 wall-clock 重新赋值。
3. 每个用户事件同时进入 `_event_q`（播放用）和 `_melody_history`（推理上下文用）。

---

## `_tick_loop(*, max_ticks)` — 时钟线程

正式 tick 循环的每 tick 顺序如下：

1. 睡到当前 tick 的绝对 wall-clock 边界。
2. 计算 `MusicalTime.from_tick(tick, tempo)` 并调用 `output_tick(tick, bar, beat)`。
3. `tick=0` 时用完整 `_melody_history` 触发一次请求。
4. 睡 `seconds_per_tick * 0.1`，给贴近 tick 边界到达的输入事件一个缓冲窗口。
5. drain `_event_q`，收集本 tick 要播放的用户事件，并追加到 `notes_for_next_request`。
6. drain `_inference_response_queue`，清理未来 model 事件并 schedule 新伴奏。
7. 输出 metronome tick，再输出 user 事件，再输出当前 tick 已调度的 model 事件。
8. 如果当前 tick 是每拍最后一个 tick，发送下一拍推理请求。

这意味着当前推理调度的音乐语义是“每拍末尾请求下一拍”，而不是旧文档中的“每 `generation_interval_ticks` ticks 请求一次”。

---

## `_inference_worker()` — 推理线程

推理线程采用 latest-only 队列策略，避免推理慢时堆积过期请求。

```python
generation_start_tick, melody_events = self._inference_request_queue.get(timeout=0.1)
merged_melody_events = list(melody_events)
while True:
    try:
        newer_tick, newer_events = self._inference_request_queue.get_nowait()
    except queue.Empty:
        break
    generation_start_tick = newer_tick
    if newer_events:
        merged_melody_events.extend(newer_events)
```

行为：

1. 至少取一个请求。
2. 继续 drain 队列，只保留最新的 `generation_start_tick`。
3. 被跳过请求里的 melody 增量会合并到 `merged_melody_events`，避免旋律事件丢失。
4. 调用 `inference_engine.generate_accompaniment(...)`。
5. 将 `(acc_events, generation_start_tick)` 放入 `_inference_response_queue`。
6. 调用 `output_stats(...)`，并在支持时调用 `log_inference(...)`。
7. 推理异常会转成 `output_status("error", ...)`，服务继续运行。

---

## `start(*, max_ticks=None) -> None`

```python
def start(self, *, max_ticks: Optional[int] = None) -> None:
```

启动服务：

1. 设置 `_running = True`。
2. 创建 `RealTimeServiceRuntime(session_start_time, timeline_start_time)`。
3. 调用 `output_sink.output_status("running", "")`。
4. 启动三个 daemon 线程。

`start()` 是非阻塞的；CLI 通过 `while service.running: time.sleep(0.1)` 保持主线程存活。

---

## `stop() -> None`

停止服务：

1. 设置 `_running = False`。
2. 向 `_inference_request_queue` 放入 dummy 条目唤醒 inference worker。
3. 调用 `input_source.close()`。
4. 调用 `output_sink.output_status("stopped", "")` 和 `output_sink.close()`。
5. Best-effort join 三个线程。
