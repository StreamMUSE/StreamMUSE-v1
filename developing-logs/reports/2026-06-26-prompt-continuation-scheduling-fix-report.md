# Prompt-Continuation Realtime Scheduling Fix 实现报告

日期：2026-06-26

## 1. 任务目标

本次任务是把 prompt+continuation realtime playback 的 scheduling 问题彻底修掉。

之前已经通过 two-stage consistency test 证明：server 侧保存的 prompt history / raw continuation history 可以和 offline driver 对齐，说明模型推理和 continuation 使用的 acc context 本身是对的。问题主要出在 client 侧 `PromptContinuationRealtimeService._schedule_playable()`：它把 streaming full history 强行配成 `note_on` / `note_off` pair，导致很多本来可以播放的事件被跳过，最终 `combined.mid` 里 Accompaniment 轨为空或严重缺拍。

这次修复的核心目标：

1. 默认 realtime playback 不再依赖 `pair_playable_events`。
2. 支持 streaming event-by-event scheduling。
3. 支持跨 chunk note：`note_on` 可以先 schedule，`note_off` 后到后再 schedule。
4. 支持同 tick `note_off -> note_on` 重触发。
5. 支持 sustain note rehydrate。
6. 支持 full history 重复拉取去重。
7. 保持 two-stage raw-history consistency green。
8. 让 prompt-extension realtime `combined.mid` 重新产生非空 Accompaniment 轨。

## 2. 根因回顾

旧逻辑在 `RECOVER_LATE_EVENTS=0` 时走 strict pair path：

```python
events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(
    accompaniment,
    current_tick=int(current_tick),
)

note_pairs, skipped_unpaired = self._pair_playable_events(events_to_schedule)
for note_on, note_off in note_pairs:
    ...
    self._scheduler.schedule(model_event, int(event.tick))
```

这个逻辑对 prompt-continuation realtime 不成立，因为 `/prompt_continuation/playable` 返回的是不断增长的 full event history，不是一个完整的、一次性可配对的 note list。

典型失败模式：

1. `note_on` 先到，matching `note_off` 下一次请求才到。旧 pair path 会把这个 `note_on` 记成 unpaired，然后跳过。
2. 同 tick 重触发：

```json
{"type": "note_off", "pitch": 60, "tick": 52}
{"type": "note_on", "pitch": 60, "tick": 52}
```

旧 path 会因为排序和 `note_off.tick > note_on.tick` 限制而跳过。

3. sustain note 在当前 tick 之前已经开始，但还没结束。原始 `note_on` 已经过了，不能 schedule 回过去；如果不在当前 tick rehydrate，一个仍在持续的 note 会完全丢失。

因此旧 trace 常见：

```text
scheduled_event_count = 0
skipped_unpaired > 0
combined.mid Accompaniment track empty
```

## 3. 实现概览

主要修改文件：

```text
src/streammuse/application/services/prompt_continuation_realtime_service.py
tests/unit/application/test_prompt_continuation_realtime_service.py
tests/consistency/two_stage_runners.py
tests/consistency/conftest.py
tests/consistency/test_two_stage_prompt_continuation_consistency.py
src/streammuse/infrastructure/inference/lekai_prompt_continuation/README.md
developing-logs/plans/2026-06-26-prompt-continuation-streaming-scheduling-fix-plan.md
developing-logs/reports/2026-06-26-two-stage-consistency-implementation-report.md
```

## 4. 新 scheduling mode

新增环境变量：

```text
LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE
```

取值：

```text
streaming_events      默认值，正常 realtime playback 使用
paired_future_only    legacy / diagnostic only
```

实现代码：

```python
raw_scheduling_mode = os.environ.get(
    "LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE",
    "streaming_events",
).strip().lower()
if raw_scheduling_mode not in {"streaming_events", "paired_future_only"}:
    raw_scheduling_mode = "streaming_events"
self._scheduling_mode = raw_scheduling_mode
```

`_schedule_playable()` 现在只根据 scheduling mode 分发：

```python
def _schedule_playable(self, accompaniment: list[MusicalEvent], *, current_tick: int) -> None:
    if self._scheduling_mode == "paired_future_only":
        self._schedule_playable_paired_future_only(accompaniment, current_tick=int(current_tick))
        return
    self._schedule_playable_streaming_events(accompaniment, current_tick=int(current_tick))
```

重要语义变化：

- `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` 不再决定是否走 pair path。
- `RECOVER_LATE_EVENTS=0` 只表示 late/past events 不 recover 到当前 tick。
- 当前 tick 和 future tick 的 event 仍然会按 streaming event 独立 schedule。

## 5. Streaming Event Scheduler

新增默认方法：

```python
def _schedule_playable_streaming_events(
    self,
    accompaniment: list[MusicalEvent],
    *,
    current_tick: int,
) -> None:
    """Schedule playable full history as a streaming event log."""
```

核心逻辑：

```python
usable_events = sorted(
    usable_events,
    key=lambda ev: (
        int(ev.tick),
        0 if ev.event_type == EventType.NOTE_OFF else 1,
        int(ev.pitch),
        int(ev.channel),
        int(ev.program),
    ),
)

seen_in_payload: Counter[EventKey] = Counter()
for event in [*rehydrated_events, *usable_events]:
    event_key = self._event_key(event)
    seen_in_payload[event_key] += 1
    occurrence = seen_in_payload[event_key]
    if self._handled_model_event_counts[event_key] >= occurrence:
        skipped_duplicate += 1
        continue

    event_tick = int(event.tick)
    schedule_tick = event_tick
    if event_tick < current_tick:
        late_event_count += 1
        if not self._recover_late_events:
            dropped_past += 1
            self._ensure_event_count(self._handled_model_event_counts, event_key, occurrence)
            continue
        if self._would_drop_late_note_on(event, current_tick=current_tick):
            dropped_too_late_note_on += 1
            self._ensure_event_count(self._handled_model_event_counts, event_key, occurrence)
            continue
        schedule_tick = current_tick

    model_event = self._to_model_event(event, current_tick=current_tick)
    self._scheduler.schedule(model_event, schedule_tick)
```

这个 scheduler 不做 note pair 匹配。它把 backend history 当成 event stream，因此：

- `note_on` 没有 matching `note_off` 也可以先 schedule。
- 后续 full history 里出现 `note_off` 时，会独立 schedule。
- 同 tick `note_off -> note_on` 不再因为零长度 pair 被跳过。
- old pair path 的 `skipped_unpaired` 不再参与默认播放逻辑。

## 6. Event Key 和 Counter 去重

旧去重 key 只有：

```python
(tick, pitch, event_type, velocity)
```

现在扩展为：

```python
EventKey = tuple[int, int, str, int, int, int]

def _event_key(event: MusicalEvent) -> EventKey:
    return (
        int(event.tick),
        int(event.pitch),
        str(event.event_type.value),
        int(event.velocity),
        int(event.channel),
        int(event.program),
    )
```

同时把 set 去重改为 Counter：

```python
self._handled_model_event_counts: Counter[EventKey] = Counter()
self._played_model_event_counts: Counter[EventKey] = Counter()
```

原因：

1. `/playable` 返回 full history，下一次会重复包含前一次的 events。
2. history 里可能存在完全相同 key 的重复 event。
3. set 会把合法重复 event 合并掉，Counter 可以按 occurrence 去重。

两个 Counter 的语义不同：

- `handled_model_event_counts`：这个 event occurrence 已经被处理过，后续 full history 不要重复处理。
- `played_model_event_counts`：这个 event occurrence 已经实际播放，或已经被 rehydrated clone 代表。

这个区分解决了一个关键边界：某个 late `note_on` 可能先被 drop/handled，但当时还没有 future `note_off`，后续 full history 补齐 span 后仍然需要允许 sustain rehydrate。

## 7. Sustain Note Rehydrate

新增状态：

```python
self._rehydrated_model_note_span_keys: set[NoteSpanKey] = set()
```

新增 helper：

```python
def _rehydrate_sustaining_notes(
    self,
    events: list[MusicalEvent],
    *,
    current_tick: int,
) -> tuple[list[MusicalEvent], Counter[EventKey]]:
```

核心逻辑：

```python
if not (note_on_tick < current_tick < note_off_tick):
    continue
span_key = self._note_span_key(note_on, event)
if span_key in self._rehydrated_model_note_span_keys:
    continue
note_on_key = self._event_key(note_on)
if self._played_model_event_counts[note_on_key] >= note_on_occurrence:
    continue
rehydrated.append(self._clone_event_at_tick(note_on, current_tick))
```

语义：

1. 如果一个 note 已经开始但尚未结束，即 `note_on.tick < current_tick < note_off.tick`，则在当前 tick clone 一个 replacement `note_on`。
2. 原始 future `note_off` 继续交给 streaming scheduler 正常 schedule。
3. rehydrate 后把原始 note_on 对应 occurrence 标记为 played/handled，避免它又被 late recovery 触发一次。
4. 用 `note_span_key` 防止 full history 每次 polling 都重复 retrigger 同一个 sustain note。

默认行为：

```python
self._rehydrate_active_notes = self._env_optional_bool(
    "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES"
)
if self._rehydrate_active_notes is None:
    self._rehydrate_active_notes = True
```

也就是说，新默认 streaming mode 会默认保留 sustain note。

## 8. Legacy Pair Path

旧 pair path 没有删除，而是改名为：

```python
def _schedule_playable_paired_future_only(...)
```

只有显式设置才会进入：

```bash
export LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=paired_future_only
```

保留它的目的只是诊断旧行为；正常 realtime playback 不应该使用。

## 9. Consistency Runner 更新

更新文件：

```text
tests/consistency/two_stage_runners.py
tests/consistency/conftest.py
tests/consistency/test_two_stage_prompt_continuation_consistency.py
```

runner 现在显式设置：

```python
env.update(
    {
        "LEKAI_PROMPT_CONTINUATION_TRACE_PATH": str(trace_path),
        "LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE": "streaming_events",
        "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS": "0",
        "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES": "1",
        "LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP": "1",
    }
)
```

trace summary 新增：

```python
"rehydrated_notes": rehydrated_notes,
"streaming_event_rows": streaming_event_rows,
"paired_future_only_rows": paired_future_only_rows,
```

test 里新增断言：

```python
assert schedule_counts["paired_future_only_rows"] == 0
if schedule_counts["schedule_rows"]:
    assert schedule_counts["streaming_event_rows"] == schedule_counts["schedule_rows"]
```

这样可以防止 consistency test 悄悄跑回旧 pair path。

## 10. Unit Tests

更新文件：

```text
tests/unit/application/test_prompt_continuation_realtime_service.py
```

新增/更新覆盖点：

1. 默认 scheduling mode 是 `streaming_events`。
2. past events 在 `RECOVER_LATE_EVENTS=0` 时会 drop/handled。
3. sustain note 会 rehydrate 到当前 tick。
4. 同 tick `note_off -> note_on` 会完整 schedule。
5. 跨 chunk note：先 schedule `note_on`，后续 full history 补 `note_off`。
6. full history 重复拉取不会重复 schedule。
7. sustain rehydrate 不会重复 retrigger。
8. bounded late recovery 会 drop too-late `NOTE_ON`。
9. unbounded recovery 保持可恢复旧行为。
10. bounded recovery 未显式设置 max 时默认用 `generation_interval_ticks`。

## 11. 文档更新

更新文件：

```text
src/streammuse/infrastructure/inference/lekai_prompt_continuation/README.md
```

文档现在明确：

- 默认 scheduler 是 `streaming_events`。
- `RECOVER_LATE_EVENTS` 只控制 late recovery，不控制 scheduler mode。
- sustain rehydrate 默认开启。
- `paired_future_only` 只作为 legacy/diagnostic mode。
- demo-style setting 推荐使用：

```bash
LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=1
LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY=1
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS=4
```

## 12. 验证结果

### 12.1 Unit tests

命令：

```bash
uv run pytest tests/unit/application/test_prompt_continuation_realtime_service.py -q
```

结果：

```text
16 passed in 0.31s
```

### 12.2 Two-stage consistency 真模型验证

命令：

```bash
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

结果：

```text
1 passed in 395.11s (0:06:35)
```

意义：

- scheduler fix 没有破坏 raw-history/offline consistency。
- realtime test trace 没有跑回 `paired_future_only`。
- server-side continuation context 仍和 offline 对齐。

### 12.3 Prompt-extension realtime sweep

输出目录：

```text
output/prompt_extension_sweep/20260626-105843/
```

汇总文件：

```text
output/prompt_extension_sweep/20260626-105843/summary.json
```

配置：

```text
song = 4
tempo = 120
prompt extension = 1/2/3/4 beats
max_ticks = 128
SCHEDULING_MODE = streaming_events
RECOVER_LATE_EVENTS = 0
REHYDRATE_ACTIVE_NOTES = 1
```

结果：

| extension | schedule_rows | streaming_rows | paired_rows | scheduled_events | rehydrated | skipped_unpaired | Accompaniment note_on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 beat | 23 | 23 | 0 | 158 | 0 | 0 | 79 |
| 2 beats | 23 | 23 | 0 | 96 | 7 | 0 | 49 |
| 3 beats | 23 | 23 | 0 | 140 | 15 | 0 | 76 |
| 4 beats | 23 | 23 | 0 | 147 | 24 | 0 | 81 |

结论：

- 四组都没有进入 `paired_future_only`。
- 四组 `skipped_unpaired = 0`。
- 四组 `scheduled_event_count > 0`。
- 四组 `combined.mid` 都有非空 `Accompaniment` track。

对应 MIDI：

```text
output/prompt_extension_sweep/20260626-105843/ext1_beats/cli/2026-06-26/session_105855/combined.mid
output/prompt_extension_sweep/20260626-105843/ext2_beats/cli/2026-06-26/session_105924/combined.mid
output/prompt_extension_sweep/20260626-105843/ext3_beats/cli/2026-06-26/session_105953/combined.mid
output/prompt_extension_sweep/20260626-105843/ext4_beats/cli/2026-06-26/session_110021/combined.mid
```

## 13. 当前限制和注意点

1. 我完成的是事件级 scheduling 修复和 MIDI artifact 非空验证，没有做主观听感判断。建议人工至少播放 ext3 或 ext4 的 `combined.mid`。
2. Counter 去重假设 `/playable` full history 是单调只增。如果未来 server 支持 rollback 或修改历史，需要引入 sequence-id / version 机制。
3. 默认不调用 `clear_future_events()` 是有意为之：prompt-continuation 当前拿到的是 full history，不是单阶段那种 replacement window。
4. late `NOTE_OFF` 默认不会被 too-late bound drop；它可以 recover 到当前 tick，用来关闭可能正在响的 note。
5. `paired_future_only` 保留只是为了诊断旧 bug，不建议作为 realtime demo 默认配置。

## 14. 最终结论

本次 scheduling fix 已经完成并通过验证。

最关键的行为变化是：prompt-continuation realtime playback 从 “pair-based note scheduling” 改成 “streaming event scheduling”。这和 `new_system_stanley` 的实时调度思路一致：模型输出的 event 应该逐个 schedule，late event 才由 recovery 策略处理，而不是要求每次 response 都包含完整 note pair。

修复后，之前导致 `combined.mid` 空/缺拍的 `skipped_unpaired` 问题在 prompt-extension sweep 中消失；四组 sweep 都产生了非空 Accompaniment 轨。
