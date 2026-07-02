# Partial Note Recovery 实现报告（2026-07-02）

## 结论

本轮已按 `developing-logs/plans/2026-06-26-partial-note-recovery-plan.md` 完成迟到伴奏音符的 partial-note recovery：实时推理返回晚于目标 tick 时，不再把所有迟到事件整体挪到当前 tick，也不再简单丢掉 onset；而是按音符区间判断，保留仍在持续的长音尾部。

最终验证结果：

- `uv run pytest tests/unit tests/integration -q`：182 passed。
- tempo-15 end-to-end realtime/offline consistency：1 passed，耗时 345.84s。
- 通过的 consistency 产物在 `output/consistency/20260702-210355/`。

## 核心行为

模型事件现在先按 `(pitch, channel, program)` 组成 `EventKey`，再做 note_on/note_off 配对。对于一对音符 `[on_tick, off_tick)`，当前 tick 为 `C`：

```python
if off_tick <= C:
    # 整个音已经过去：丢掉 on/off，不补播
elif on_tick < C < off_tick:
    # 音还在持续：note_on clamp 到 C，note_off 保持 logical off_tick
else:
    # 未来音：原样调度
```

实现里保持了两个时间概念：

- `logical_tick`：模型理论输出 tick。
- `scheduled_tick`：实际播放/写入 `combined.mid` 的 tick。

因此 `combined.mid` 代表真实播放，`theoretical_model.mid` 从 trace 的 `logical_tick` 重建，用来听/查模型理论输出。

## 关键代码片段

### 1. 调度 helper

`RealTimeMusicService` 新增纯 helper `_plan_model_events_for_playback()`，把复杂逻辑从 `_tick_loop` 主循环拆出来：

```python
def _plan_model_events_for_playback(
    self,
    acc_events: List[MusicalEvent],
    *,
    current_tick: int,
    generation_start_tick: int,
    active_model_keys: set[EventKey],
) -> ModelSchedulePlan:
    ...
```

它负责：

- note_off before note_on 的稳定排序；
- 同 key note_on/note_off 配对；
- 完全过去 / 进行中 / 未来三分支；
- open note 与 isolated note_off；
- 输出 schedule trace row。

### 2. active model note 状态

service 维护实际已经播放、但还没有实际关闭的模型音：

```python
self._active_model_note_keys: set[EventKey] = set()
```

状态只在 step 7 真正 output model event 时更新，避免“计划排期了但还没播放”误算成 active。

### 3. clear future 的挂音修复

`PlaybackScheduler` 新增：

```python
def pop_future_events(self, from_tick: int, source: str | None = None) -> List[MusicalEvent]:
    ...
```

`_tick_loop` 清理旧 model future events 时会拿到被删除的事件。如果删除的是某个 active model note 的 future note_off，就在当前 tick 合成立刻关闭：

```python
if ev.event_type == EventType.NOTE_OFF and key in self._active_model_note_keys:
    forced = self._copy_model_event(ev, tick=tick, ...)
    forced_events.append(forced)
```

这样可以避免旧 note_on 已经响过、future note_off 被新响应覆盖后产生挂音。

### 4. schedule trace 与 theoretical MIDI

`SessionLoggerOutputSink` 新增 `log_model_schedule()`，每条模型事件写入：

```json
{
  "type": "note_on",
  "pitch": 60,
  "logical_tick": 20,
  "scheduled_tick": 24,
  "policy": "clamped_partial_note",
  "generation_start_tick": 16,
  "current_tick": 24
}
```

关闭 session 时，如果 trace 存在，会从 `logical_tick` 重建 `theoretical_model.mid`。`combined.mid` 不受影响，仍记录 actual scheduled playback。

## open note / isolated note_off

Lekai 的 server-side active pitch 机制会正常产生跨 batch 结构：本批可能只有 note_on，note_off 在后续 batch；也可能本批只有 note_off，用来关闭之前已经响起的音。因此这不是异常兜底，而是正式逻辑。

本轮实现的策略：

- open note：若 onset 已过去但音还应开始响，note_on clamp 到当前 tick；否则按原 tick。
- isolated note_off：只有 key 已在 `_active_model_note_keys` 中才输出。
- `off_tick < current_tick`：policy 为 `late_isolated_note_off`，实际排到当前 tick。
- `off_tick == current_tick`：policy 为 `current_isolated_note_off`，这是准点跨 batch off，不算 late。
- key 不 active：policy 为 `orphan_note_off`，默认丢弃。

这里首轮 consistency 曾经失败，是因为我最初把 `off_tick <= current_tick` 都标成 `late_isolated_note_off`。tempo 15 trace 显示大量 `logical_tick == scheduled_tick == current_tick` 的准点 off，被误判成 late。已修复为只有 `< current_tick` 才算 late。

## 输出档位

新增 `--session-artifact-tier {normal,debug}`：

- `streammuse-cli` 默认 `debug`，保持交互调试兼容性。
- `scripts/run_lekai_batch_client.py` 默认传 `normal`，让批量 sweep 目录更干净。

normal 档保留核心产物：

- `combined.mid`
- `theoretical_model.mid`
- `model_schedule_trace.jsonl`

debug 档额外保留完整 session/server 诊断文件，例如 `events.jsonl`、`inferences.json`、`performance.json`、`statistics.csv`、history、summary 等。

## 测试覆盖

新增/更新的测试覆盖了：

- `pop_future_events()` 返回并删除 future model events。
- 进行中长音 clamp onset、保留 note_off logical tick。
- 完全过去的音整音丢弃。
- 未来音原样调度。
- open note 与 active isolated note_off。
- same pitch 不同 channel/program 不串配。
- future note_off 被 clear 时强制输出 `forced_note_off`。
- SessionLogger 写 `model_schedule_trace.jsonl` 并重建 `theoretical_model.mid`。
- CompositeOutputSink fan-out `log_model_schedule()`。
- CLI config 解析 `--session-artifact-tier`。
- consistency 测试读取 trace，确认赶得上时没有 late/clamp/drop/force policy。

实际执行过的命令：

```bash
uv run pytest tests/unit/domain/timing/test_scheduler.py \
  tests/unit/application/test_real_time_music_service.py \
  tests/unit/infrastructure/output/test_output_sinks.py \
  tests/unit/presentation/test_cli_config_parser.py -q
# 50 passed

uv run pytest tests/unit/application tests/integration -q
# 38 passed

uv run pytest tests/unit tests/integration -q
# 182 passed

LEKAI_CHECKPOINT_PATH=/data/home/bowenzheng/mbzuai-projects/models/ModelLekai/epoch_4_1104_1204/model.safetensors \
STREAMMUSE_CONSISTENCY_TEMPOS=15 \
uv run pytest tests/consistency/test_realtime_offline_consistency.py -q -s
# 1 passed in 345.84s
```

## consistency 产物

通过的 end-to-end consistency run：

```text
output/consistency/20260702-210355/
├── offline_song4/004_4_generated.mid
├── server.log
└── song4_tempo15/2026-07-02/session_210417/
    ├── combined.mid
    ├── theoretical_model.mid
    ├── model_schedule_trace.jsonl
    ├── inferences.json
    ├── events.jsonl
    └── ...
```

trace 前几行只出现了 `future_open_note` 和 `current_isolated_note_off`，没有 `clamped_partial_note`、`dropped_past_note`、`forced_note_off` 等 late recovery policy。

## 文档更新

已更新：

- `docs/user-guide/session-logging.md`
- `docs/user-guide/output-types.md`
- `docs/reference/cli-reference.md`
- `docs/architecture/infrastructure/output/session_logger.md`
- `docs/developer-guide/consistency-test.md`

文档中明确了：

- `combined.mid` 是 actual scheduled playback；
- `theoretical_model.mid` 是 logical timeline；
- `model_schedule_trace.jsonl` 是诊断 late scheduling 的主依据；
- CLI 默认 debug，batch sweep 默认 normal；
- consistency 现在会检查 trace 中是否出现 late scheduling policy。

## 剩余注意点

- 本轮按用户要求重点验证了“赶得上”的 tempo 15 consistency；没有重跑默认 tempo 120，因为 120 档更容易受当前 GPU/系统负载影响，失败通常代表性能而非一致性回归。
- `theoretical_model.mid` 是从 trace 重建的诊断产物，不参与 live playback；真实听感仍以 `combined.mid` 为准。
- normal 档是为了批量 sweep 清爽；需要查 HTTP 请求、history、性能统计时要用 debug 档。
