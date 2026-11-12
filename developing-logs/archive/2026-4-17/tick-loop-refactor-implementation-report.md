# Tick Loop 重构实施报告

**日期**: 2026-04-17  
**基于计划**: `tick-loop-refactor-plan.md`  
**实施状态**: 完成  

---

## 1. 变更文件清单

| 文件 | 变更类型 |
|------|---------|
| `src/streammuse/domain/musical/events.py` | 新增字段 |
| `src/streammuse/application/services/real_time_music_service.py` | 重构逻辑 |
| `tests/unit/application/test_real_time_music_service_incremental.py` | 测试全量替换 |

---

## 2. 详细变更说明

### 2.1 `MusicalEvent` 新增 `backup_level` 字段

**文件**: `src/streammuse/domain/musical/events.py`

```python
# 新增字段（默认值 0，向后兼容）
backup_level: int = 0  # Ticks this note is offset from gen_start (0 = first frame)
```

**设计决策**：
- `MusicalEvent` 是 `frozen=True` 的不可变 dataclass，不能在创建后赋值。计划文档里 `ev.backup_level = ...` 的写法会在运行时抛 `FrozenInstanceError`。实际实现在构造 `MusicalEvent` 时直接传入 `backup_level`，而非事后赋值。
- 默认值 `0` 保证所有已有代码（`Note.to_events()`、`_input_worker` 等）无需改动。
- 计算公式与老系统一致：`backup_level = max(0, ev.tick - generation_start_tick)`，即伴奏音符在生成窗口内的 tick 偏移。

### 2.2 `RealTimeMusicService` — `_tick_loop` 重构

**文件**: `src/streammuse/application/services/real_time_music_service.py`

#### 2.2.1 移除 `_last_sent_index`

旧系统用 `_last_sent_index` 跟踪 `_melody_history` 的增量发送位置。新设计将增量跟踪职责移到 `_tick_loop` 内部的局部变量 `notes_for_next_request`，不再需要实例级别的游标。

**删除**：
```python
self._last_sent_index: int = 0
```

#### 2.2.2 新增类常量

```python
_INPUT_BUFFER_RATIO: float = 0.1
```

放在类级别而非方法内，便于子类覆盖或配置化。

#### 2.2.3 新的 `_tick_loop` 工作顺序

每个 tick 依次执行以下 8 步：

**Step 1 — 绝对时间同步**（保留原有逻辑）
```python
target_time = start + self._tempo.tick_to_seconds(tick)
delay = target_time - self._now()
if delay > 0:
    self._sleep(delay)
```

**Step 2 — 输出 tick 信息**（保留原有逻辑）

**Step 3 — tick=0 立即触发**（借鉴老系统，新增）
```python
if tick == 0:
    with self._melody_history_lock:
        notes_for_request = self._melody_history.copy()
    if notes_for_request:
        self._inference_request_queue.put((0, notes_for_request))
```
- 发送 `gen_start_tick=0`，覆盖 injection 预加载的历史数据。
- 仅在 `_melody_history` 非空时触发，避免无效请求。

**Step 4 — 输入缓冲窗口 sleep**（借鉴老系统，新增）
```python
self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)
```
- 在 120 BPM、4 ticks/beat 下，每 tick = 125ms，缓冲窗口约 12.5ms。
- 目的：tick 边界附近产生的用户输入有足够时间进入 `_event_q`，避免被归入下一个 tick。
- 绝对时间同步机制会在下一个 tick 补偿这 12.5ms，整体时序不漂移。

**Step 5 — 输入事件处理**（修改）
```python
ev = self._event_q.get_nowait()
self._output.output_event(ev, source="user")
notes_for_next_request.append(ev)   # 新增：累积到缓冲区
```
- 事件仍然立即 output（实时响应），同时累积到 `notes_for_next_request` 等待 4n-1 触发。

**Step 6 — 推理响应处理**（修改：加入 backup_level）
```python
backup_level = max(0, ev.tick - generation_start_tick)
ev_model = MusicalEvent(
    ...,
    source="model",
    backup_level=backup_level,
)
```
- `backup_level=0` 表示伴奏在生成窗口第一帧，即"准时"。
- `backup_level=k` 表示该音符在窗口内第 k 帧，值越大说明模型产生的前瞻越少。

**Step 7 — 播放 scheduled 事件**（保留原有逻辑）

**Step 8 — 4n-1 拍尾触发**（借鉴老系统，新增）
```python
ticks_per_beat = self._tempo.ticks_per_beat
if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
    if notes_for_next_request:
        self._inference_request_queue.put((tick + 1, notes_for_next_request))
        notes_for_next_request = []
```
- `tick > 0` 排除 tick=0（已由 Step 3 处理）。
- `gen_start_tick = tick + 1`，即下一拍的起始 tick，与老系统一致。
- 触发后清空 buffer，下一拍重新累积。
- 仅在有新事件时触发，避免无效请求。

#### 2.2.4 触发时序对比（ticks_per_beat=4，120 BPM）

| Tick | 本次实现 | 老系统 | 说明 |
|------|---------|--------|------|
| 0 | `put(0, history)` | `put(0, history)` | 立即生成，覆盖 injection |
| 3 | `put(4, buffer[0..3])` | `put(4, notes)` | 拍尾触发，gen_start=4 |
| 4 | — | — | 期望结果到达，开始播放 |
| 7 | `put(8, buffer[4..7])` | `put(8, notes)` | 下一拍尾触发 |

#### 2.2.5 `_event_to_log_dict` 新增 `backup_level`

```python
"backup_level": int(event.backup_level),
```

所有走 `log_inference` 路径的伴奏事件都会记录 `backup_level`，可在 `inferences.json` 中查看分布。

---

## 3. 计划中的问题与处理

### 3.1 frozen dataclass 赋值问题

**计划文档写法（错误）**：
```python
for ev in acc_events:
    ev.backup_level = max(0, ev.tick - gen_start_tick)  # FrozenInstanceError!
```

**实际实现**：在构造 `ev_model` 时直接传入 `backup_level`，无需事后赋值：
```python
ev_model = MusicalEvent(
    tick=ev.tick,
    ...
    backup_level=max(0, ev.tick - generation_start_tick),
)
```

### 3.2 `generation_interval_ticks` 参数

计划文档提到"配置参数 `generation_interval_ticks` 仍然有效，但优先级低于新的时序逻辑"。实际上新的 tick loop 不再使用该参数触发推理，触发逻辑完全由 `ticks_per_beat` 控制。参数仍保留在 `__init__` 中（用于 `_build_inference_log_payload` 的日志记录），不会影响功能。

---

## 4. 测试变更

**文件**: `tests/unit/application/test_real_time_music_service_incremental.py`

旧测试基于 `_last_sent_index` 的增量逻辑，全量替换为 6 个新测试：

| 测试名 | 验证内容 |
|--------|---------|
| `test_tick0_sends_full_melody_history` | tick=0 发送完整 melody_history，gen_start=0 |
| `test_tick0_skips_when_history_empty` | tick=0 history 为空时不入队 |
| `test_beat_tail_sends_buffered_events` | tick=3 发送 buffer 内事件，gen_start=4 |
| `test_beat_tail_skips_when_buffer_empty` | tick=3 buffer 为空时不入队 |
| `test_beat_tail_only_one_trigger_across_two_beats_with_no_new_events` | tick=3 触发后 buffer 清空，tick=7 无新事件不重复触发 |
| `test_inference_worker_latest_only_merges_events_and_keeps_latest_tick` | inference worker 的 latest-only 合并行为不变 |

**注**：`test_beat_tail_buffer_cleared_after_trigger`（验证"tick=3 后仅发送 tick=4 之后到达的事件"）无法用 `now=lambda: 0.0` 的假时间模拟，因为所有预置事件在 tick=0 即被 drain。改为用 `test_beat_tail_only_one_trigger_across_two_beats_with_no_new_events` 等价覆盖 buffer clear 语义。

---

## 5. 测试结果

```
tests/unit/application/test_real_time_music_service_incremental.py  6/6 passed
tests/  173/174 passed（唯一失败 test_load_model_mps_failure_falls_back_to_cpu 与本次改动无关）
```

---

## 6. 已知限制与后续工作

1. **backup_level 统计未暴露**：`output_stats()` 目前不输出 backup_level 的均值/分布，只记录在 `inferences.json` 中。后续可在 `MetricsCalculator` 中添加 average backup level 和 hit rate 计算。

2. **`generation_interval_ticks` 语义变化**：该参数不再控制触发频率，仅出现在日志中。如有用到 `generation_interval_ticks != ticks_per_beat` 的场景，行为会与之前不同，需注意。

3. **INPUT_BUFFER_RATIO 不可配置**：目前硬编码为 0.1，如需运行时调整需要手动改常量或扩展构造参数。
