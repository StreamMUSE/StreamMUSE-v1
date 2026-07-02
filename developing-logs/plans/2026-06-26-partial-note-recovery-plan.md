# 迟到伴奏音符的"部分补齐"（not-missing-all）机制 — 计划（2026-06-26）

## 背景与目标

实时链路里，一次推理返回的伴奏音符会被排进 `PlaybackScheduler` 等待播放（`real_time_music_service.py` `_tick_loop` step 6）。当推理慢、响应回来时某些音符的目标 tick 已经过去，需要一个策略。

**想要的行为（用户澄清）**：如果一个音**很长**（比如 4 个 tick），它的 note_on 所在的第一个 tick 没赶上，但这个音在当前 tick 仍在持续，就应该**从当前 tick 接着把它剩下的部分弹出来，弹到它本来该结束的地方**。即"只丢已经过去的头，保留剩余、结尾位置不变"。

**不想要的行为**：
- 把错过的 onset **整体挪到后面**补弹（late recovery 的做法，会"把前面没弹的放到后面弹"）；
- 一个音只要 onset 过点了就**整个丢掉**（长音被 missing all）。

## 当前状态（本 session 已改动，需再改）

`_tick_loop` step 6 的这段经历了两版：

1. **原实现（commit `cea763089`，2026-04-17，late recovery）**：
   ```python
   schedule_tick = ev.tick if ev.tick >= tick else tick   # 迟到事件挪到当前 tick
   ```
   问题见下。
2. **本 session 改成"直接丢"**：`ev.tick < tick` 的事件 `continue` 跳过。
   问题：长音的 note_on 被丢、note_off 留在未来 → 孤立 note_off，音等于没弹（正是要避免的 missing-all）。

→ **本计划要把这段替换成"配对 + 三分支"的部分补齐逻辑**，取代上面两版。

## 为什么两版都不对（问题分析）

调度器把音符表示为**两个独立事件**：`note_on@T` 与 `note_off@T+D`（已确认：HTTP 响应里 accompaniment 是 `{type, pitch, tick, velocity}` 列表，note_on / note_off 分开，`note_off.tick = note_on.tick + duration`）。设当前 tick = C。

对一个音 [on@T, off@T+D]：

| | 完全过去 (T+D ≤ C) | 进行中 (T < C < T+D) | 未来 (T ≥ C) |
|---|---|---|---|
| **late recovery（原）** | on、off 都挪到 C → C 处冒出一个早该结束的 blip；多个音的 onset 还会挤成和弦 ❌ | on→C, off 留 T+D → [C, T+D) ✅（这个 case 恰好对） | 原样 ✅ |
| **直接丢（本 session）** | 两个都丢 ✅ | on 丢、off 留 → 孤立 note_off，音没弹 ❌ | 原样 ✅ |
| **想要的** | 整音丢 | on 夹到 C、off 留 T+D → [C, T+D) | 原样 |

关键差异：想要的做法在**"完全过去"时丢整音、"进行中"时夹头留尾**，两版各只对了一半。

## 正确设计：按"音符"配对 + 三分支

不再逐事件判断，而是先把 note_on / note_off 配对成音符，再按区间处理。配对 key 不能只用 pitch，应该使用：

```python
EventKey = tuple[int, int, int]  # (pitch, channel, program)
```

当前 Lekai 多数情况下 channel/program 都是默认值，但 `RealTimeMusicService` 是通用实时调度层，后续如果多通道、多 program 或 MIDI device 输出保留通道信息，只按 pitch 会把不同声部错误配对。

```text
对每个音符 (on_tick, off_tick, key=(pitch, channel, program), ...):
    if off_tick <= C:                 # 完全过去
        丢弃（不排 on 也不排 off）
    elif on_tick < C:                 # 进行中（长音的头过了、尾还没到）
        note_on 排到 C；note_off 排到 off_tick（原位不动）
    else:                             # 未来
        note_on 排到 on_tick；note_off 排到 off_tick
```

要点：
- **note_off 永远排在它自然的 tick**（除了完全过去被整音丢），保证音在正确网格位置结束，不产生时间涂抹；
- **只有进行中音符的 onset 被夹到当前 tick**；
- 完全过去的音直接丢，不补播。

### 双时间语义：logical tick vs scheduled tick

进行中音符被部分补齐时，需要同时保留两个概念：

- `logical_tick` / `original_tick`：模型理论上生成的 tick，比如 note_on 本来在 `T`；
- `scheduled_tick` / `playback_tick`：client 实际播放/录制的 tick，比如被夹到当前 tick `C`。

实际播放路径必须使用 `scheduled_tick`。也就是说，夹到当前 tick 的 `MusicalEvent.tick` 应该改成 `C`，否则 Web UI、MIDI logger、audio output 会在"实际现在播放，但事件自称在过去"之间产生错乱。

为了保留理论输出，新增一个 sidecar scheduling trace（例如 `model_schedule_trace.jsonl`），每个事件记录：

```json
{
  "type": "note_on",
  "pitch": 60,
  "channel": 0,
  "program": 0,
  "logical_tick": 20,
  "scheduled_tick": 24,
  "policy": "clamped_partial_note"
}
```

如果需要听/比对两份 MIDI：

- `combined.mid` / session MIDI 继续表示**实际播放**，即使用 `scheduled_tick`；
- 额外导出 `theoretical_model.mid` 或从 trace 重建，表示**模型理论输出**，即使用 `logical_tick`。

这比把 live `MusicalEvent.tick` 保持为理论 tick 更安全，因为 live sink 当前只有一个 `tick` 字段，默认会把它理解为实际播放 tick。

### Sweep 输出档位：normal vs debug

后续 sweep 不再默认把所有 session/debug 文件都铺开，分成两档：

**normal sweep（默认）**：用于批量听结果和做快速质量对比，保留最少但完整的听感/调度信息。

```text
combined.mid
theoretical_model.mid
model_schedule_trace.jsonl
summary.json
aggregate_summary.json
run_config.json
```

含义：

- `combined.mid`：实际播放/录制结果，使用 `scheduled_tick`；
- `theoretical_model.mid`：模型理论输出，使用 `logical_tick`；
- `model_schedule_trace.jsonl`：logical tick、scheduled tick、policy 的逐事件 trace；
- `summary.json` / `aggregate_summary.json` / `run_config.json`：sweep 级别汇总和配置。

**debug sweep（显式开启）**：用于排查 inference、server context、session logger 或性能问题，在 normal 文件基础上额外保留完整诊断文件。

```text
events.jsonl
inferences.json
performance.json
statistics.csv
session_summary.txt
melody_history.json
accompaniment_history.json
prompt_continuation_history_status.json
prompt_continuation_prompt_history.json
prompt_continuation_raw_history.json
stdout.log
stderr.log
server.log
```

这样 normal sweep 目录会更清爽，但仍能直接听实际版本和理论版本；debug sweep 才保存完整 session/server 侧上下文。

## Phase 0：先把事件结构确认清楚（写代码前）

Lekai 后端用 `close_at_end=False` + server-side `self._active_pitches` 做跨 batch pianoroll 转事件，因此一批 `acc_events` 里出现"配不成对"的事件是正常结构，不是异常：

- open note：本批有 note_on，note_off 在后续 batch；
- isolated note_off：note_on 在更早 batch，本批只负责关掉；
- 同 tick retrigger：同 key 可能先 note_off 再 note_on。

**先跑一次 full-detail 推理看真实结构**，不是为了决定要不要支持这些情况，而是为了确认出现频率、排序和边界行为：

- [x] 0.1 起 server + 发一个多拍旋律请求（或用 `--inference-log-detail full` 的 realtime 短跑），dump 一批 acc_events，确认：
  - note_on / note_off 是否总是成对出现在**同一批**响应里；
  - 是否存在 **open note**（本批只有 note_on、note_off 在后续批）；
  - 是否存在 **孤立 note_off**（note_on 在更早的批、本批只有 off）；
  - 同一 `(pitch, channel, program)` 会不会在一批内**重叠**；
  - 同 tick 是否保持 note_off before note_on。
- [x] 0.2 记录结论到本文件，并把 open note / isolated note_off 当成正式测试场景。

## Phase 1：实现配对 + 三分支

- [x] 1.1 抽出一个纯 helper，不要把复杂逻辑全部塞进 `_tick_loop` 主循环。建议形态：

  ```python
  def _plan_model_events_for_playback(
      acc_events: list[MusicalEvent],
      *,
      current_tick: int,
      generation_start_tick: int,
      active_model_keys: set[EventKey],
  ) -> ModelSchedulePlan:
      ...
  ```

- [x] 1.2 在 helper 内按 `(pitch, channel, program)` 对 `acc_events` 做 note_on/note_off 配对；同 tick 排序必须保持 note_off before note_on。
- [x] 1.3 对配成对的音符套用三分支规则：
  - **完全过去**：`off_tick <= current_tick` → on/off 都不排，计入 `dropped_past_note_count`；
  - **进行中**：`on_tick < current_tick < off_tick` → note_on 的 `scheduled_tick=current_tick`，note_off 的 `scheduled_tick=off_tick`；
  - **未来**：`on_tick >= current_tick` → on/off 都按原 tick 排。
- [x] 1.4 夹到当前 tick 的 note_on 必须构造为 `MusicalEvent(tick=current_tick, ...)`。原始 tick 写入 trace 的 `logical_tick` 字段，不放在 live event 的 `tick` 字段里。
- [x] 1.5 open note / isolated note_off 作为正式逻辑处理：
  - **open note（只有 note_on）**：如果 `on_tick < current_tick`，note_on 夹到 `current_tick`；否则按原 tick 排。该 key 会在实际播放 note_on 后进入 client-side active set，等待后续 isolated note_off 关闭。
  - **isolated note_off（只有 note_off）**：如果 key 当前在 client-side active set 里，则关掉它；`off_tick <= current_tick` 时排到 `current_tick`，`off_tick > current_tick` 时按原 tick 排。若 key 不 active，则记录为 `orphan_note_off_count`，默认丢弃，避免凭空发 off 污染输出。
- [x] 1.6 在 `RealTimeMusicService` 维护 client-side `_active_model_note_keys: set[EventKey]`，表示**实际已经播放 note_on、但还没有实际播放 note_off** 的模型音。这个状态在 step 7 真正 output model event 时更新。
- [x] 1.7 保留观测计数，例如：
  - `clamped_onset_count`
  - `dropped_past_note_count`
  - `orphan_note_off_count`
  - `forced_note_off_count`
  - `scheduled_actual_event_count`
  - `theoretical_event_count`

  用 `output_status("debug", ...)` 发出摘要，文案如实描述（不要再叫 "Recovered"）。
- [x] 1.8 `backup_level` 建议继续按 logical tick 计算：`max(0, logical_tick - generation_start_tick)`。这样它仍表达模型输出相对 generation window 的位置，而不是被 client clamp 后的位置。
- [x] 1.9 新增 schedule trace：每个被 schedule/drop/force-off 的事件都记录 `logical_tick`、`scheduled_tick`、`policy`、`generation_start_tick`、`current_tick`、`key`。后续可从 trace 生成 `theoretical_model.mid`。

## Phase 2：边界与既有逻辑的交互

- [x] 2.1 **替换/扩展 `clear_future_events(from_tick=generation_start_tick, source="model")`**：当前 clear 会删除 `generation_start_tick` 之后的旧 model 事件。如果旧音的 note_on 已经播放、note_off 还在 future schedule 里，这一步会删掉 note_off，导致挂音。策略定为：
  1. 不直接盲 clear；先从 scheduler 中 pop/remove future model events，并拿到被删除事件列表；
  2. 对其中的 future note_off，如果它的 `(pitch, channel, program)` 在 `_active_model_note_keys` 中，说明这个音已经实际响过；
  3. 立刻合成一个 `note_off@current_tick`，在当前 tick 播放阶段输出，计入 `forced_note_off_count`；
  4. 再 schedule 新响应里的事件。

  需要在 `PlaybackScheduler` 增加一个非私有方法，例如：

  ```python
  pop_future_events(from_tick: int, source: str | None = None) -> list[MusicalEvent]
  ```

  避免在 service 里直接读 `_schedule` 私有字段。
- [x] 2.2 夹到当前 tick 的 note_on，和"当前 tick 已经在播放的事件"（step 7 播放 `get_events_at_tick(tick)`）的时序：确认夹进来的 note_on 会在**同一 tick 的播放阶段**被输出，而不是被跳过（step 6 在 step 7 之前，排进 `tick` 的事件应能被 step 7 取到）。
- [x] 2.3 同 key 若既有正在响的旧音、又夹进一个新 onset：必须先输出 forced note_off，再输出新 note_on，避免 double note_on 不 note_off。
- [x] 2.4 同 tick 事件顺序必须稳定：note_off 永远在 note_on 前；forced note_off 也要排在同 tick clamped/future note_on 前。
- [x] 2.5 如果新增 `theoretical_model.mid`，它只能用于诊断/听感对照，不参与 live playback；live playback 和 `combined.mid` 仍以 actual scheduled tick 为准。

## Phase 3：测试

- [x] 3.1 单元测试（纯 `_tick_loop` / 调度层，无需 GPU），构造响应事件直接喂进队列，锁住三分支：
  - 完全过去的音 → 不排期（scheduler 里没有）；
  - 进行中的长音 → note_on 的 `event.tick` 和 schedule tick 都是当前 tick，note_off 在原 off_tick；
  - 未来的音 → 原样；
  - open note 跨 batch：本批 note_on 被播放，后续 isolated note_off 能正确关掉；
  - isolated note_off 在 active set 中 → 关音；不在 active set 中 → 记录 orphan 并丢弃；
  - clear future 删除了已响音的 future note_off → 当前 tick 立刻 forced note_off；
  - 同 tick note_off before note_on；
  - same pitch different channel/program 不串配；
  - schedule trace 同时包含 logical_tick 和 scheduled_tick。
- [x] 3.2 输出测试：
  - `combined.mid` 记录 actual scheduled tick；
  - `theoretical_model.mid` 记录 logical tick，并且不影响实际 playback；
  - normal sweep 默认包含 `combined.mid`、`theoretical_model.mid`、`model_schedule_trace.jsonl` 和 sweep summary/config；
  - debug sweep 在 normal 文件基础上额外保留 session JSON、server histories、stdout/stderr/server log。
- [x] 3.3 回归：`uv run pytest tests/unit/application tests/integration -q` 全绿。
- [x] 3.4 必跑 end-to-end offline vs realtime consistency test，验证**赶得上的时候行为和以前一样**：
  - 跑现有 offline/realtime consistency 测试（慢 tempo / 足够长响应窗口，确保不会触发 partial recovery）；
  - 断言 offline 与 realtime 在公共窗口内仍一致；
  - 断言 `model_schedule_trace.jsonl` 中没有 `clamped_partial_note`、`dropped_past_note`、`forced_note_off` 等迟到策略；
  - 若跑 two-stage prompt+continuation consistency，也要确认 realtime raw/context history 与 offline 对齐逻辑不受影响。

## Phase 4：与一致性测试的关系（评估，不一定动）

- 本改动只在**推理跟不上调度**（快 tempo / GPU 忙）时才生效；慢 tempo（如一致性测试的 tempo 15）永不触发，所以 `tests/consistency/` 的 tempo-15 档必须继续全绿。
- 快 tempo 下，这套逻辑会改变迟到音符的落点（夹头留尾 vs 原来的整体挪动），可能让一致性测试的快 tempo 档表现变化。评估是否要：
  - [x] 4.1 在一致性测试的"零丢弃前置"里，同时检测 session 日志/`model_schedule_trace.jsonl` 的"部分补齐/丢弃/forced off"计数（目前只检测 `_inference_worker` 的 dropped stale request，见 [consistency-test.md](../../docs/developer-guide/consistency-test.md)）——快 tempo 失败能干净归类为"性能"而非"回归"。

## Phase 5：收尾

- [x] 5.1 更新 `docs/developer-guide/`（若有实时行为说明）描述新的迟到处理语义。
- [x] 5.2 更新 sweep 脚本/runner 的输出档位说明：normal 为默认，debug 需要显式开启。
- [x] 5.3 写执行 report 到 `developing-logs/reports/`。

## 已确认决策（开工前）

1. "完全过去"的音直接丢：已确认可以接受。结束时间已经过去的音不补播，避免迟到 blip 和错误和弦堆积。
2. `theoretical_model.mid` 第一版就放进 normal sweep：采用**每个 case 结束后从 `model_schedule_trace.jsonl` 重建**的方式生成，而不是 live playback 时额外挂一个理论 MIDI sink。这样 live 路径只负责 actual playback，理论 MIDI 与 trace 保持一一对应，也更容易回放/重建。
3. 夹到当前 tick 的 onset 不下调 velocity，也不在 live event 上额外做音色/力度标记；只在 trace 文件里用 `policy`、`logical_tick`、`scheduled_tick` 标清楚。


## 执行结果（2026-07-02）

- 已实现 partial-note recovery：完全过去的音整音丢弃，进行中的长音把 note_on 夹到当前 tick、note_off 保持 logical off tick，未来音保持原样。
- 已实现跨 batch open note / isolated note_off：按 `(pitch, channel, program)` 维护 `_active_model_note_keys`，并把 `off_tick == current_tick` 视为准点 `current_isolated_note_off`，不归类为 late。
- 已实现 `PlaybackScheduler.pop_future_events()`，在清理旧 future model events 时，对已实际响起但 future off 被删除的音立即输出 `forced_note_off`。
- 已实现 `model_schedule_trace.jsonl` 与 `theoretical_model.mid`：live/`combined.mid` 使用 actual scheduled tick，理论 MIDI 从 trace 的 logical tick 重建。
- 已实现 session artifact tier：CLI 默认 `debug`；batch sweep 默认 `normal`，保留核心 MIDI/trace，debug 档保留完整诊断文件。
- 已补充单元测试、输出测试、CLI config 测试和 consistency trace 断言。
- 已跑通过：`uv run pytest tests/unit tests/integration -q`（182 passed）和 tempo-15 end-to-end realtime/offline consistency（1 passed）。
