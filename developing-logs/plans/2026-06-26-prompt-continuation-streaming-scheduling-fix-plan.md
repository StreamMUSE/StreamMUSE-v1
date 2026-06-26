# Prompt-Continuation Realtime Scheduling 修复计划

日期：2026-06-26

## 结论

当前 prompt+continuation realtime 的推理结果本身是可信的：之前 consistency test 已经证明 server 侧保存的 prompt history / raw continuation history 可以和 offline 逐事件对齐。真正导致 `combined.mid` 空、后半拍丢失、一个 beat 里前两个 tick 能播后两个 tick 播不出来的问题，主要在 client 侧 scheduling。

根因是 `PromptContinuationRealtimeService` 默认路径里使用了 `pair_playable_events`，它要求 `note_on` 和 `note_off` 在当前可见的 playable history 里组成完整 note pair 以后才 schedule。这对 streaming 场景是不成立的：realtime 每次拿到的是不断增长的 event stream，一个 `note_on` 可能先到，`note_off` 可能下一个请求才出现；同一 tick 的 `note_off -> note_on` 重触发也会被 pair 逻辑错误跳过。

修复方向：prompt-continuation 的 realtime playback 默认改成 `new_system_stanley` 那种 event-by-event streaming scheduler。也就是说，模型给出的每个 playable event 都独立 schedule；迟到事件按策略 recover 到当前 tick；不再要求先配成完整 note pair。

## new_system_stanley 的 scheduling 逻辑

在 `new_system_stanley:src/streammuse/application/services/real_time_music_service.py` 里，模型返回的伴奏事件是逐 event schedule 的：

```python
while True:
    try:
        acc_events, generation_start_tick = self._inference_response_queue.get_nowait()
    except queue.Empty:
        break

    self._scheduler.clear_future_events(from_tick=generation_start_tick, source="model")

    late_event_count = 0
    for ev in acc_events:
        backup_level = max(0, ev.tick - generation_start_tick)
        ev_model = MusicalEvent(
            tick=ev.tick,
            pitch=ev.pitch,
            event_type=ev.event_type,
            velocity=ev.velocity,
            channel=ev.channel,
            program=ev.program,
            is_placeholder=ev.is_placeholder,
            source="model",
            backup_level=backup_level,
        )
        schedule_tick = ev.tick if ev.tick >= tick else tick
        if ev.tick < tick:
            late_event_count += 1
        self._scheduler.schedule(ev_model, schedule_tick)
```

这里有几个关键点：

1. 不做 note pair 匹配。
2. 每个 `MusicalEvent` 独立 schedule。
3. 如果事件 tick 已经过了，就把它 schedule 到当前 tick，作为 late recovery。
4. 单阶段 inference 返回的是某个 generation window 的结果，所以它会先 `clear_future_events(from_tick=generation_start_tick, source="model")`，避免旧窗口的未来事件继续播放。

第 1-3 点应该直接迁移到 prompt-continuation。第 4 点不能直接照搬，因为 prompt-continuation 当前 `/playable` 返回的是 server 侧累计的 full playable history，不是“本次 generation window 的替换结果”。对 full history 来说，默认应该靠 event-level idempotency 去重，而不是每次清掉未来事件。

## 当前 prompt-continuation 的问题代码

当前 `src/streammuse/application/services/prompt_continuation_realtime_service.py` 的严格路径是：

```python
events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(
    accompaniment,
    current_tick=int(current_tick),
)

note_pairs, skipped_unpaired = self._pair_playable_events(events_to_schedule)
for note_on, note_off in note_pairs:
    ...
    for event in (note_on, note_off):
        self._scheduler.schedule(model_event, int(event.tick))
```

而 `_pair_playable_events()` 又有这两个限制：

```python
sorted_events = sorted(
    events,
    key=lambda event: (
        int(event.tick),
        0 if event.event_type == EventType.NOTE_OFF else 1,
    ),
)
...
if int(event.tick) > int(note_on.tick):
    pairs.append((note_on, event))
else:
    skipped_unpaired += 2
```

这会直接造成两个 bug：

1. 同一 tick 的 `note_off -> note_on` 重触发会被错误处理，因为排序后 `note_off` 先出现，此时没有 active note，先被记成 unmatched；后面的 `note_on` 如果本次 history 里还没有未来 `note_off`，也会被记成 unmatched。
2. 一个跨 chunk 的 note 会被丢掉：`note_on` 在本次 playable history 里出现，但 `note_off` 还没到，pair 失败；等 `note_off` 后面到了，`note_on` 可能已经变成过去事件，严格路径又不会正确恢复。

典型 trace 是 `current_tick=52` 时 server raw history 已经有：

```json
{"type": "note_off", "pitch": 60, "tick": 52}
{"type": "note_off", "pitch": 64, "tick": 52}
{"type": "note_off", "pitch": 67, "tick": 52}
{"type": "note_on", "pitch": 60, "tick": 52}
{"type": "note_on", "pitch": 64, "tick": 52}
{"type": "note_on", "pitch": 67, "tick": 52}
```

但是 pair 路径会得到 `future_note_on_count=3`、`scheduled_event_count=0`、`skipped_unpaired=3`。也就是说，模型已经生成了可以播放的事件，scheduler 自己把它们跳过了。

## 目标行为

prompt-continuation realtime playback 应满足：

1. 默认不再用 `pair_playable_events` 决定是否 schedule。
2. `note_on` 可以在没有对应 `note_off` 的情况下先被 schedule。
3. 后续收到对应 `note_off` 时，再独立 schedule `note_off`。
4. 同一 tick 的 `note_off -> note_on` 重触发必须能完整 schedule。
5. 重复拉取 full playable history 时不能重复 schedule 已经处理过的 event。
6. late event 的处理策略要显式：可以 recover 到当前 tick，也可以按窗口丢弃，但不能因为没有 pair 而丢弃未来可播放事件。
7. raw history / offline consistency 的逻辑不应该被改变；这次只修 client playback scheduling。

## 设计方案

### 1. 引入明确的 scheduling mode

现在 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=0` 会走 pair-based strict path，这个耦合是不合理的。建议新增：

```text
LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events
```

取值建议：

```text
streaming_events      默认值，逐 event 调度
paired_future_only    legacy/diagnostic only，不作为 realtime 默认路径
```

`LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` 以后只控制 late event 是否被移动到当前 tick，不再决定是否使用 pair scheduler。

### 2. 将现有 recover-late 路径重构为 streaming scheduler

当前代码里已经有 `_schedule_playable_recover_late()`，它本质上就是 event-by-event scheduler。不要再新增另一份平行实现；更好的做法是把它重构/重命名为：

```python
def _schedule_playable_streaming_events(
    self,
    accompaniment: list[MusicalEvent],
    *,
    current_tick: int,
) -> None:
    ...
```

然后让 `_schedule_playable()` 默认调用它。`LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` 只在这个方法内部决定“过去 tick 的 event 是否 recover 到当前 tick”，不再决定 pair scheduler 和 streaming scheduler 的分支。

核心流程：

1. 过滤 placeholder / pitch=-1，并按 `(tick, note_off_before_note_on, pitch, channel, program)` 排序。
2. 统计 input tick 分布；`future_event_count` 定义为 `event.tick >= current_tick` 的 usable event 数量，`future_note_on_count` 定义为同一条件下 `NOTE_ON and velocity > 0` 的数量。streaming 和 legacy pair 两种模式都应该输出这些字段，方便 trace 对比。
3. 先做 sustain rehydrate：对 `note_on.tick < current_tick < note_off.tick` 的 active span，在 `current_tick` clone 一个 note_on，并保留原 note_off。
4. 再对 rehydrated events 和原始 full history events 做 event-level scheduling。
5. 用 Counter 记录已经处理过的 event occurrence，保证 full history 重复拉取时不会重复 schedule。
6. sustain rehydrate 后必须把原始 note_on 标记为 handled，避免同一个长音同时被 rehydrated clone 和 late recovery 触发两次。

核心伪代码：

```python
usable_events = sorted(
    [e for e in accompaniment if not e.is_placeholder and e.pitch != -1],
    key=lambda ev: (
        int(ev.tick),
        0 if ev.event_type == EventType.NOTE_OFF else 1,
        int(ev.pitch),
        int(ev.channel),
        int(ev.program),
    ),
)

rehydrated_events, consumed_original_note_on_keys = self._rehydrate_sustaining_notes(
    usable_events,
    current_tick=current_tick,
)

# 原始 note_on 已经被 rehydrated clone 代表，后面不能再作为 late note_on recover 一次。
for key in consumed_original_note_on_keys:
    self._handled_model_event_counts[key] += 1

seen_in_payload = Counter()
for event in [*rehydrated_events, *usable_events]:
    key = self._event_key(event)
    seen_in_payload[key] += 1
    occurrence = seen_in_payload[key]

    if self._handled_model_event_counts[key] >= occurrence:
        skipped_duplicate += 1
        continue

    event_tick = int(event.tick)
    if event_tick < current_tick:
        late_event_count += 1
        if not self._recover_late_events:
            self._handled_model_event_counts[key] += 1
            dropped_past += 1
            continue
        if self._would_drop_late_note_on(event, current_tick=current_tick):
            self._handled_model_event_counts[key] += 1
            dropped_too_late_note_on += 1
            continue
        schedule_tick = current_tick
    else:
        schedule_tick = event_tick

    model_event = MusicalEvent(..., source="model", backup_level=max(0, event_tick - current_tick))
    self._scheduler.schedule(model_event, schedule_tick)
    self._handled_model_event_counts[key] += 1
    scheduled += 1
```

注意这里使用当前代码里真实存在的 `_would_drop_late_note_on()`，不是新造一个 `_should_drop_too_late_note_on()`。它只会 drop `NOTE_ON` 且 `velocity > 0` 的 too-late event；late `NOTE_OFF` 默认不会 drop，如果开启 recover，会被 schedule 到当前 tick。这可能会切断一个正在播放的 note，但通常比丢掉后续 note-on 更可接受。

这里建议把当前的 `self._scheduled_model_event_keys: set[...]` 改成类似：

```python
self._handled_model_event_counts: Counter[EventKey] = Counter()
```

原因是 `/playable` 返回 full history，可能包含完全相同 key 的重复 event。用 set 会把合法重复 event 合并掉；用 per-payload occurrence counter 可以做到：

1. 同一份 full history 重复拉取时不会重复 schedule。
2. full history 后面新增一个 key 完全相同的 event 时，也可以按 occurrence 处理。
3. too-late 被主动 drop 的 event 也可以标记为 handled，避免每个 tick 重复统计/重复尝试。

### 3. 扩展 event key、note identity 和 note span 状态

当前 `_event_key()` 是：

```python
return (
    int(event.tick),
    int(event.pitch),
    str(event.event_type.value),
    int(event.velocity),
)
```

建议改成：

```python
return (
    int(event.tick),
    int(event.pitch),
    str(event.event_type.value),
    int(event.velocity),
    int(event.channel),
    int(event.program),
)
```

这样不会把不同 channel/program 上同 tick 同 pitch 的事件误判成 duplicate。

除了 event-level Counter，还需要保留 note identity / note span 级别的状态：

```python
note_identity = (pitch, channel, program)
note_span_key = (note_on_tick, note_off_tick, pitch, channel, program)
```

原因是 sustain rehydrate 会 clone 一个新的 `note_on(current_tick)`。如果只看 event key，它和原始 `note_on(original_tick)` 不是同一个 key，后续 full history 再次拉取时可能重复 retrigger 同一个 sustain span。建议保留一个类似 `_rehydrated_model_note_span_keys` 的集合，记录已经 rehydrate 过的 note span；同时，如果原始 note_on 已经被 schedule/handled 过，也不要再 rehydrate 这个 span。

### 4. Sustain notes 在 streaming scheduler 中必须保留

默认 streaming scheduler 必须处理已经开始但尚未结束的 sustain note：

```text
note_on.tick < current_tick < note_off.tick
```

这类 note 的原始 `note_on` 已经过了，不能 schedule 回过去的 tick；但 `note_off` 还在未来。如果不补一个当前 tick 的 note_on，realtime catch-up 时会只播放未来的 note_off，听起来就是 sustain note 丢失。

建议实现一个不依赖 pair scheduler 的 helper：

```python
def _rehydrate_sustaining_notes(
    self,
    events: list[MusicalEvent],
    *,
    current_tick: int,
) -> tuple[list[MusicalEvent], list[EventKey]]:
    ...
```

语义：

1. 按 `(pitch, channel, program)` 维护 active note map。
2. 找到 `note_on.tick < current_tick < note_off.tick` 的 span。
3. 如果这个 span 没有 rehydrate 过，且原始 note_on 没有被处理/播放过，则 clone 一个 `note_on` 到 `current_tick`。
4. 把原始 note_on 的 event key 标记为 handled，避免主循环后面又把它当 late note_on recover 到当前 tick。
5. 保留原始 future `note_off`，让 event-level scheduler 正常 schedule 它。
6. 标记这个 `note_span_key` 已经 rehydrate，避免下一次 full history 拉取时在 `current_tick+1` 再次触发同一个 sustain。

这个逻辑默认开启，不受 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` 控制；它处理的是“已经开始的 sustain”，不是“迟到事件是否 recover”。

`_clip_playable_to_current_tick()` 的 sustain 语义可以复用，但它现在和 pair-based scheduler 绑得太紧。实现时建议二选一：

1. 把 sustain 部分抽成 `_rehydrate_sustaining_notes()`，streaming scheduler 复用它；`_clip_playable_to_current_tick()` 只留给 legacy `paired_future_only`。
2. 或者把 `_clip_playable_to_current_tick()` 拆成 `_drop_fully_past_events()` 和 `_rehydrate_sustaining_notes()`，streaming scheduler 只使用后者。

### 5. 暂时不要默认 clear_future_events

`new_system_stanley` 的这句在单阶段里是正确的：

```python
self._scheduler.clear_future_events(from_tick=generation_start_tick, source="model")
```

但是 prompt-continuation 现在没有明确的 `generation_start_tick` replacement window，client 拿到的是累计 history。如果每次从某个 tick 开始清 future events，很容易把之前已经正确 schedule 的 prompt extension 或 continuation future events 清掉。

所以第一版修复不要默认调用 `clear_future_events()`。如果未来 server 的 `/playable` 改成返回“本次新增 segment + segment_start_tick”，再加一个 explicit replacement mode：

```text
LEKAI_PROMPT_CONTINUATION_PLAYABLE_MODE=full_history | segment_delta
```

只有 `segment_delta` 才考虑按 `segment_start_tick` 清 future model events。

### 6. 保留 pair 方法但退出默认路径

短期可以保留 `_pair_playable_events()`，但只允许在 legacy/diagnostic mode 使用：

```python
if self._scheduling_mode == "paired_future_only":
    self._schedule_playable_paired_future_only(...)
else:
    self._schedule_playable_streaming_events(...)
```

这样便于对比旧行为，也能降低一次性删除代码的风险。默认值必须是 `streaming_events`。

## 测试计划

### Phase 0：加失败用例，先复现 bug

在 `tests/unit/application/test_prompt_continuation_realtime_service.py` 增加纯 synthetic unit tests，不依赖真实模型：

1. 同 tick 重触发：

```python
events = [
    note_off(tick=52, pitch=60),
    note_on(tick=52, pitch=60),
]
service._schedule_playable(events, current_tick=52)
assert scheduler.get_events_at_tick(52) == [note_off_60, note_on_60]
```

2. 跨 chunk note：

```python
service._schedule_playable([note_on(tick=52, pitch=60)], current_tick=52)
assert tick_52_has_note_on

service._schedule_playable(
    [note_on(tick=52, pitch=60), note_off(tick=56, pitch=60)],
    current_tick=53,
)
assert tick_56_has_note_off
assert no_duplicate_note_on
```

3. full history 重复拉取：

```python
same_history = [note_on(52, 60), note_off(56, 60)]
service._schedule_playable(same_history, current_tick=52)
service._schedule_playable(same_history, current_tick=53)
assert events_are_not_scheduled_twice
```

4. late recovery：

```python
service._schedule_playable([note_on(tick=50, pitch=60)], current_tick=52)
assert note_on_is_scheduled_at_52_when_recover_late_enabled
```

5. too-late bound：

```python
service._schedule_playable([note_on(tick=40, pitch=60)], current_tick=52)
assert too_late_note_on_is_dropped_when_bound_enabled
assert repeated_call_does_not_recount_or_reschedule_it
```

6. sustain note 在当前 tick 之前开始，但还没有结束：

```python
service._schedule_playable(
    [note_on(tick=48, pitch=60), note_off(tick=56, pitch=60)],
    current_tick=52,
)
assert tick_52_has_rehydrated_note_on_60
assert tick_56_has_note_off_60
assert tick_48_has_no_note_on
```

7. sustain rehydrate 不能因为 full history 重复拉取而重复触发：

```python
history = [note_on(48, 60), note_off(56, 60)]
service._schedule_playable(history, current_tick=52)
service._schedule_playable(history, current_tick=53)
assert only_one_rehydrated_note_on_for_this_span
assert note_off_at_56_is_scheduled_once
```

这些测试应该在当前 pair path 下至少有同 tick、跨 chunk、sustain 三类失败。

### Phase 1：实现 streaming scheduler

修改 `PromptContinuationRealtimeService`：

1. 新增 `_scheduling_mode`，默认 `streaming_events`。
2. 新增 `_handled_model_event_counts`，替代或补充 `_scheduled_model_event_keys`。
3. 新增 `_rehydrated_model_note_span_keys` 或等价状态，避免 sustain span 重复 retrigger。
4. 将 `_schedule_playable_recover_late()` 泛化为 `_schedule_playable_streaming_events()`，避免两份 event-by-event 代码。
5. `_schedule_playable()` 默认调用 streaming scheduler。
6. `recover_late_events` 只控制 late event 是否 recover，不再控制 pair/streaming 分支。
7. 明确 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` 的默认值：未设置时，如果 bounded recovery 开启，默认使用 `generation_interval_ticks`；如果 bounded recovery 关闭，则没有上限。
8. trace 里输出：

```text
mode=streaming_events
scheduled_event_count
late_event_count
dropped_past
dropped_too_late_note_on
skipped_duplicate
placeholder_count
input_min_tick/input_max_tick
future_event_count/future_note_on_count
```

默认路径不再输出 `skipped_unpaired`，因为默认路径不做 pair。

### Phase 2：更新 consistency runner 环境变量

当前 consistency runner 里有：

```text
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=0
```

修复后这不应该切回 pair path。需要确认并更新：

1. tests 里如果想关闭 late recovery，就仍然保留 `RECOVER_LATE_EVENTS=0`。
2. 必须显式 export `LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events`。
3. 在测试 setup、`runtime_info` 或 trace 里 assert 实际 mode 不是 `paired_future_only`，避免默认值写错时测试 silent 跑旧路径。
4. 不允许 consistency test 通过环境变量意外回到 `paired_future_only`。

### Phase 3：回归验证

先跑 unit：

```bash
uv run pytest tests/unit/application/test_prompt_continuation_realtime_service.py -q
```

再跑 two-stage consistency，重点确认 raw history 对齐不受影响：

```bash
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

最后重跑 prompt extension sweep 中之前最容易出问题的配置：

1. song 4
2. tempo 120
3. prompt extension 1/2/3/4 beats
4. `max_ticks=128`

验收时看三类 artifact：

1. realtime `combined.mid` 里 model accompaniment 不再全空。
2. trace 中 `mode=streaming_events`，不再出现默认路径的 `skipped_unpaired`。
3. 到了 catch-up 之后，不再系统性出现一个 beat 里前两个 tick schedule 上、后两个 tick schedule 不上的现象。

### Phase 4：更新文档/report

更新之前的 two-stage consistency implementation report，补一节：

1. raw history consistency 已经证明 inference/context 一致。
2. 旧 playback failure 是 scheduling 层问题，不是 continuation model context 问题。
3. 新 scheduler 与 `new_system_stanley` 的关系：
   - 相同点：逐 event 调度、late event recover 到当前 tick。
   - 不同点：prompt-continuation full history 模式下默认不清 future events。
4. sustain rehydrate 的语义：只补“已经开始且未结束、但原始 note_on 没有被实际处理过”的 note span，避免重复 retrigger。

## 验收标准

1. 默认 prompt-continuation realtime scheduling 不再调用 `_pair_playable_events()`。
2. 同 tick 的 `note_off -> note_on` 能正确播放。
3. 跨 chunk 的 `note_on` 不再因为暂时没有 `note_off` 被丢掉。
4. catch-up 时已经开始但未结束的 sustain note 会在当前 tick rehydrate，并保留未来 note_off。
5. 重复拉 full playable history 不会重复 schedule 同一个 event，也不会重复 retrigger 同一个 sustain span。
6. late event 的 recover/drop 行为由环境变量控制，并且 trace 可解释。
7. trace 在 streaming 和 legacy pair 两种模式下都输出 `future_event_count` / `future_note_on_count`。
8. two-stage raw history consistency test 保持 green。
9. prompt extension sweep 的 realtime `combined.mid` 不再空，且 catch-up 后 scheduling 缺口明显消失。

## 风险和注意点

1. 逐 event schedule 可能会播放没有对应 audible `note_on` 的 `note_off`。这通常是无害的，比丢掉后续 `note_on` 更可接受；MIDI sink 一般会忽略这种 redundant note_off。
2. 如果 late note_on recover 到当前 tick，节奏会被压缩。需要用 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` 控制最大 recover 窗口；未设置时，bounded recovery 默认使用 `generation_interval_ticks`，unbounded recovery 没有上限。
3. sustain rehydrate 必须用 note-span 状态去重，否则 full history 每次拉取都会把同一个长音重新触发一次。
4. 如果 full history 里确实有完全相同 key 的重复 event，set 去重会丢事件，所以建议用 `Counter` 而不是 set。
5. Counter 去重假设 `/playable` 返回的 full history 单调只增。若 server 未来支持 rollback 或修改历史事件，需要换成 sequence-id / version 机制。
6. 不要把 `clear_future_events()` 直接照搬过来。它适合单阶段 replacement window，不适合当前 prompt-continuation full history response。
7. 如果后续 server 改成 delta response，需要重新设计 replacement window 和 sequence id，而不是继续猜测 full history。

## 建议实施顺序

1. 先写 unit tests，把同 tick 重触发、跨 chunk note、sustain rehydrate 三类 bug 固化下来。
2. 重构现有 `_schedule_playable_recover_late()` 为 `streaming_events` scheduler，并让它成为默认。
3. 加入 event Counter 和 sustain note-span 去重状态。
4. 跑 unit tests，确认旧 bug 消失。
5. 跑 two-stage consistency，确认 inference/raw history 不受影响。
6. 跑 prompt extension sweep，听 realtime `combined.mid`，检查 trace。
7. 更新 report，记录这次修复把问题定位在 playback scheduler，而不是模型推理。


## TODO List

### Phase 0：先写失败用例，固定当前 bug

- [x] 在 `tests/unit/application/test_prompt_continuation_realtime_service.py` 增加 synthetic event helper，能方便构造 `note_on` / `note_off`。
- [x] 增加同 tick 重触发测试：`note_off(tick=52)` 后接 `note_on(tick=52)` 时，两个 event 都应该 schedule 到 tick 52。
- [x] 增加跨 chunk note 测试：先只收到 `note_on(tick=52)` 时也要 schedule；后续 full history 带上 `note_off(tick=56)` 时，只补 schedule `note_off`，不能重复 schedule 原 `note_on`。
- [x] 增加 full history 重复拉取测试：同一份 `[note_on(52), note_off(56)]` 连续传入两次，不应该重复 schedule。
- [x] 增加 late recovery 测试：`RECOVER_LATE_EVENTS=1` 时，迟到但仍在恢复窗口内的 `note_on` 应 schedule 到当前 tick。
- [x] 增加 too-late bound 测试：超过 `RECOVER_LATE_MAX_TICKS` 的 late `NOTE_ON` 应被 drop，并标记为 handled，重复调用不应重复统计或 schedule。
- [x] 增加 sustain rehydrate 测试：`note_on(48), note_off(56)` 在 `current_tick=52` 时，应 clone 一个 note_on 到 tick 52，并保留 tick 56 的 note_off。
- [x] 增加 sustain rehydrate 去重测试：同一个 sustain span 被 full history 重复返回时，只能 rehydrate 一次，不能在 tick 52、53、54 连续重复触发。
- [x] 确认上述测试在当前旧 pair path 下至少暴露同 tick、跨 chunk、sustain 三类失败。

### Phase 1：实现 streaming scheduler

- [x] 在 `PromptContinuationRealtimeService.__init__()` 中新增 `_scheduling_mode`，读取 `LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE`，默认 `streaming_events`。
- [x] 保留 `paired_future_only` 作为 legacy/diagnostic mode，但不作为默认路径。
- [x] 将 `_schedule_playable()` 改为按 `_scheduling_mode` dispatch：默认调用 streaming scheduler；只有显式 `paired_future_only` 才走旧 pair path。
- [x] 将现有 `_schedule_playable_recover_late()` 重构/重命名为 `_schedule_playable_streaming_events()`，避免两份 event-by-event 代码。
- [x] 在 streaming scheduler 内部使用 `RECOVER_LATE_EVENTS` 控制 late event 是否 recover 到当前 tick，而不是用它控制 scheduler mode。
- [x] 将 `_event_key()` 扩展为 `(tick, pitch, event_type, velocity, channel, program)`。
- [x] 用 `Counter[EventKey]` 或等价结构替代单纯 set 去重，支持 full history 中相同 event key 的多次 occurrence。
- [x] 把主动 drop 的 too-late event 也标记为 handled，避免每个 tick 重复处理。
- [x] 新增 `_rehydrated_model_note_span_keys` 或等价状态，记录已经 rehydrate 过的 sustain span。
- [x] 新增或抽取 `_rehydrate_sustaining_notes()`，按 `(pitch, channel, program)` 维护 active map，识别 `note_on.tick < current_tick < note_off.tick`。
- [x] rehydrate sustain 时 clone 一个 note_on 到 `current_tick`，保留原始 future note_off。
- [x] rehydrate sustain 后把原始 note_on 的 event key 标记为 handled，避免它又被 late recovery 触发一次。
- [x] 如果原始 note_on 已经被实际 schedule/handled，不能再 rehydrate 同一个 span。
- [x] 明确 `_clip_playable_to_current_tick()` 的归属：只保留给 legacy `paired_future_only`，或拆出 `_rehydrate_sustaining_notes()` 给 streaming scheduler 复用。
- [x] 保持默认不调用 `clear_future_events()`；只有未来 server 改成 segment/delta response 后再设计 replacement window。
- [x] 确认 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS` 默认语义：bounded recovery 开启且未设置时使用 `generation_interval_ticks`，unbounded recovery 没有上限。

### Phase 2：更新 trace、runtime 和 consistency runner

- [x] 在 streaming scheduler trace 中输出 `mode=streaming_events`。
- [x] 在 streaming 和 legacy pair 两种模式下都输出 `input_min_tick`、`input_max_tick`、`future_event_count`、`future_note_on_count`、`past_event_count`。
- [x] 在 trace 中输出 `scheduled_event_count`、`late_event_count`、`dropped_past`、`dropped_too_late_note_on`、`skipped_duplicate`、`placeholder_count`。
- [x] 默认 streaming path 不再输出或依赖 `skipped_unpaired`。
- [x] 更新 consistency runner 环境变量，显式设置 `LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events`。
- [x] 如果 consistency test 需要关闭 late recovery，继续设置 `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=0`，但确认它不会切回 pair path。
- [x] 在测试 setup、`runtime_info` 或 trace 里 assert 实际 scheduler mode 不是 `paired_future_only`。
- [x] 确认 `runtime_info` 若暴露 scheduler mode，返回的是 plain dict，测试里用 `.get()` 读取字段。

### Phase 3：运行测试和回归验证

- [x] 运行 prompt-continuation service unit tests：

```bash
uv run pytest tests/unit/application/test_prompt_continuation_realtime_service.py -q
```

- [x] 运行 two-stage consistency，确认 raw history / offline 对齐不受 scheduling 修复影响：

```bash
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

- [x] 重跑 prompt extension sweep 的重点配置：song 4、tempo 120、prompt extension 1/2/3/4 beats、`max_ticks=128`。
- [x] 检查 realtime `combined.mid`，确认 model accompaniment 不再全空。
- [x] 检查 trace，确认默认路径是 `mode=streaming_events`。
- [x] 检查 trace，确认 catch-up 后不再系统性出现一个 beat 里前两个 tick schedule 上、后两个 tick schedule 不上的情况。
- [x] 对比 raw history 和 scheduled output，确认剩余差异主要来自 late recovery/scheduling latency，而不是 pair/unpaired drop。
- [ ] 听至少一组修复后的 realtime `combined.mid`，确认 audible result 比修复前正常。（待人工听感确认；本轮已完成 Accompaniment 轨非空和 trace 事件级验证。）

### Phase 4：更新文档和报告

- [x] 更新 `developing-logs/reports/2026-06-26-two-stage-consistency-implementation-report.md`，说明这次定位为 playback scheduler 问题，不是模型 context/inference 问题。
- [x] 在 report 中记录旧 pair path 的失败模式：同 tick 重触发、跨 chunk note、sustain note。
- [x] 在 report 中记录新 streaming scheduler 与 `new_system_stanley` 的相同点：逐 event schedule、late event recover 到当前 tick。
- [x] 在 report 中记录新 streaming scheduler 与 `new_system_stanley` 的不同点：prompt-continuation full history 模式默认不 `clear_future_events()`。
- [x] 在 report 中记录 sustain rehydrate 的语义和去重策略。
- [x] 在 report 中记录最终跑过的测试命令、结果、主要 artifact 路径。

### Phase 5：最终验收

- [x] 默认 prompt-continuation realtime scheduling 不再调用 `_pair_playable_events()`。
- [x] 同 tick `note_off -> note_on` 可以正确播放。
- [x] 跨 chunk `note_on` 不再因为暂时没有 `note_off` 被丢掉。
- [x] catch-up 时 sustain note 会 rehydrate，并保留未来 note_off。
- [x] full history 重复拉取不会重复 schedule 同一个 event，也不会重复 retrigger 同一个 sustain span。
- [x] late event recover/drop 行为由环境变量控制，并且 trace 可解释。
- [x] two-stage raw history consistency test 保持 green。
- [x] prompt extension sweep 的 realtime `combined.mid` 不再空，catch-up 后 scheduling 缺口明显消失。
