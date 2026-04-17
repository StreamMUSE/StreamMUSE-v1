# Tick Loop 推理工作顺序重构计划

**日期**: 2026-04-17  
**目标**: 借鉴老系统 StreamMUSE 的推理触发时序设计，改进新系统 StreamMUSE-v1 的推理工作顺序，同时保留新系统的绝对时间计算机制。

---

## 1. 现状分析

### 1.1 老系统 (StreamMUSE client_lekai.py) 的推理时序

```python
# 核心设计：在拍尾触发，给服务器一整拍的处理时间

def tick_loop():
    tick_count = -1
    
    while True:
        tick_count += 1
        
        # === 位置 A：循环开头（仅 tick=0）===
        if tick_count == 0:
            # 立即触发第一次生成
            request_queue.put((gen_start_tick=0, events))
            is_trigger_tick = True
        
        time.sleep(seconds_per_tick * 0.1)  # 睡 10%
        
        # ... 处理输入、处理响应、播放事件 ...
        
        # === 位置 B：循环末尾（tick=3,7,11,...）===
        if not is_tick_zero and (tick_count % 4) == 3:  # 4n-1
            # 在拍尾触发，为下一拍生成
            gen_start_tick = tick_count + 1  # 下一拍开始
            request_queue.put((gen_start_tick, notes_for_next_request))
            notes_for_next_request = []  # 清空已发送的 notes
        
        time.sleep(seconds_per_tick * 0.9)  # 睡 90%
```

**设计精髓**：
- tick=0: 特殊处理，立即生成（gen_start=0）
- tick=3,7,11... (4n-1): 在拍尾触发，为下一拍生成（gen_start=4,8,12...）
- 给服务器 4 ticks（一拍）的处理时间
- 结果在下一拍开始时到达，实现无缝衔接

### 1.2 新系统 (StreamMUSE-v1) 的推理时序

```python
def _tick_loop(self, *, max_ticks):
    tick = 0
    last_generation_tick = -generation_interval_ticks
    
    while self._running:
        # 1. 绝对时间同步
        target_time = start + self._tempo.tick_to_seconds(tick)
        delay = target_time - self._now()
        if delay > 0:
            self._sleep(delay)
        
        # 2. 输出 tick 信息
        # 3. 处理用户输入
        
        # 4. 推理触发（在 tick 的"过程中"）
        if tick - last_generation_tick >= self._generation_interval_ticks:
            # 收集新增 melody 事件
            # 放入推理队列
            last_generation_tick = tick
        
        # 5. 处理推理响应
        # 6. 播放 scheduled 事件
        
        tick += 1
```

**当前问题**：
1. 推理触发发生在 tick 的"过程中"，没有区分"tick 开头"还是"tick 末尾"
2. 没有利用 backup_level 概念来量化和跟踪延迟
3. generation_start_tick 的计算逻辑简单（直接等于当前 tick）
4. 缺少 tick=0 的特殊处理逻辑

---

## 2. 问题识别

### 2.1 时序问题

**老系统优势**：
- tick=0 立即生成：用户开始弹奏的瞬间就有伴奏生成请求发出
- 4n-1 触发：在拍尾触发，给服务器完整一拍的处理时间
- 结果在 4n 到达：正好是新的一拍开始，无缝衔接

**新系统缺陷**：
- 如果 `generation_interval_ticks=4`，在 tick=0,4,8... 触发
- 这意味着在 tick=0 触发时，gen_start=0，但用户可能还没开始弹
- 伴奏生成和音乐播放的时序关系不够清晰

### 2.2 Backup Level 缺失

**老系统**：
```python
for n in newly_generated_notes:
    n["backup_level"] = max(0, int(n["tick"] - gen_start))
```
- backup_level=0: 伴奏准时到达（理想状态）
- backup_level=1: 伴奏延迟了 1 tick
- backup_level=2: 伴奏延迟了 2 ticks
- 用于量化系统的实时性能

**新系统**：
- 没有 backup_level 概念
- 迟到的事件只是被强制调度到当前 tick，没有记录延迟程度

### 2.3 事件调度粒度

**老系统**：
- 使用 `playback_schedule` 字典，key=tick, value=events
- 精确的 per-tick 调度

**新系统**：
- 使用 `PlaybackScheduler` 类
- 也是 per-tick 调度，但缺少 backup_level 记录

---

## 3. 改进方案设计

### 3.1 核心设计理念

**保留**：
- 新系统的绝对时间计算（target_time 同步）
- 新系统的分层架构（service/scheduler/output）
- 新系统的迟到恢复机制

**借鉴**：
- 老系统的推理触发位置（tick=0 开头，4n-1 末尾）
- 老系统的 backup_level 概念
- 老系统的 generation_start_tick 计算逻辑

### 3.2 新的 Tick 工作顺序

```python
def _tick_loop(self, *, max_ticks):
    tick = 0
    last_generation_tick = -self._generation_interval_ticks
    notes_for_next_request = []  # 新增：待发送的 melody buffer
    
    while self._running:
        # === 1. 绝对时间同步（保留）===
        target_time = start + self._tempo.tick_to_seconds(tick)
        delay = target_time - self._now()
        if delay > 0:
            self._sleep(delay)
        
        # === 2. 输出 tick 信息（保留）===
        mt = MusicalTime.from_tick(tick, self._tempo)
        self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
        
        # === 3. 在 tick=0 开头触发生成（借鉴老系统）===
        if tick == 0:
            # 收集所有当前 melody 历史
            with self._melody_history_lock:
                notes_for_request = self._melody_history.copy()
            if notes_for_request:
                self._inference_request_queue.put((0, notes_for_request))
                last_generation_tick = 0
        
        # === 4. 处理用户输入（保留，但修改 buffer 逻辑）===
        while True:
            try:
                ev = self._event_q.get_nowait()
            except queue.Empty:
                break
            self._output.output_event(ev, source="user")
            # 注意：不再立即发送到推理队列，而是累积到 buffer
            notes_for_next_request.append(ev)
        
        # === 5. 处理推理响应（保留，但增加 backup_level）===
        while True:
            try:
                acc_events, gen_start_tick = self._inference_response_queue.get_nowait()
            except queue.Empty:
                break
            
            # 计算 backup_level
            for ev in acc_events:
                ev.backup_level = max(0, ev.tick - gen_start_tick)
            
            # 清除旧伴奏（从 gen_start_tick 开始）
            self._scheduler.clear_future_events(from_tick=gen_start_tick, source="model")
            
            # 调度新伴奏
            for ev in acc_events:
                schedule_tick = ev.tick if ev.tick >= tick else tick
                self._scheduler.schedule(ev, schedule_tick)
        
        # === 6. 播放 scheduled 事件（保留）===
        for ev in self._scheduler.get_events_at_tick(tick):
            self._output.output_event(ev, source=ev.source)
        
        # === 7. 在 4n-1 末尾触发生成（借鉴老系统）===
        ticks_per_beat = self._tempo.ticks_per_beat
        if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
            # 在拍尾触发，为下一拍生成
            gen_start_tick = tick + 1  # 下一拍开始
            if notes_for_next_request:
                self._inference_request_queue.put((gen_start_tick, notes_for_next_request))
                notes_for_next_request = []  # 清空 buffer
                last_generation_tick = tick
        
        tick += 1
```

### 3.3 Backup Level 实现

```python
# 在 MusicalEvent 中添加 backup_level 字段
@dataclass
class MusicalEvent:
    tick: int
    pitch: int
    event_type: EventType
    velocity: int = 64
    channel: int = 0
    program: int = 0
    is_placeholder: bool = False
    source: str = "user"
    backup_level: int = 0  # 新增：延迟程度（0=准时，1=延迟1tick，etc）
```

### 3.4 推理触发时序对比

| Tick | 老系统 gen_start | 新系统 gen_start | 说明 |
|------|-----------------|-----------------|------|
| 0 | 0（开头触发） | 0（开头触发） | 特殊处理，立即生成 |
| 3 | - | 4（末尾触发） | 为第一拍结尾收集的 notes 生成 |
| 4 | - | - | 结果到达，开始播放 |
| 7 | 8（末尾触发） | 8（末尾触发） | 为第二拍生成 |
| 8 | - | - | 结果到达 |

---

## 4. 实施步骤

### Phase 1: 数据结构修改
1. 在 `MusicalEvent` 中添加 `backup_level` 字段
2. 修改 `PlaybackScheduler.schedule()` 支持 backup_level 存储
3. 更新 `_build_inference_log_payload` 记录 backup_level

### Phase 2: Tick Loop 重构
1. 添加 `notes_for_next_request` buffer
2. 实现 tick=0 开头的推理触发逻辑
3. 修改输入处理逻辑，不再立即触发推理，而是累积到 buffer
4. 实现 4n-1 末尾的推理触发逻辑
5. 在处理响应时计算并记录 backup_level

### Phase 3: 响应处理增强
1. 在 `_inference_worker` 中保留 gen_start_tick 信息
2. 在 tick loop 中计算 backup_level
3. 添加统计信息（average backup level, hit rate 等）

### Phase 4: 测试与验证
1. 单元测试：验证推理触发时序
2. 集成测试：验证 backup_level 计算正确性
3. 性能测试：对比老系统的延迟指标

---

## 5. 关键代码变更点

### 5.1 文件清单

| 文件 | 变更内容 |
|------|---------|
| `src/streammuse/domain/musical/events.py` | 添加 backup_level 字段 |
| `src/streammuse/application/services/real_time_music_service.py` | 重构 _tick_loop 的推理时序 |
| `src/streammuse/domain/timing/scheduler.py` | 支持 backup_level 存储 |

### 5.2 向后兼容性

- `backup_level` 默认为 0，不影响现有功能
- 推理触发逻辑变更只影响实时模式，不影响离线模式
- 配置参数 `generation_interval_ticks` 仍然有效，但优先级低于新的时序逻辑

---

## 6. 风险评估

### 6.1 风险点

1. **时序变更影响用户体验**
   - 推理触发从 tick"过程中"改为 tick 末尾
   - 可能改变用户感知的延迟

2. **Backup Level 计算准确性**
   - 如果网络延迟波动大，backup_level 可能不准确
   - 需要充分测试边界情况

3. **Buffer 溢出风险**
   - `notes_for_next_request` 如果在 4 ticks 内累积太多事件
   - 需要设置上限或定期清理

### 6.2 缓解措施

1. **A/B 测试**：对比新老系统的 latency 指标
2. **配置开关**：保留旧的推理触发逻辑作为 fallback
3. **监控告警**：当 backup_level > 2 时输出 warning

---

## 7. 成功标准

1. **功能正确性**
   - tick=0 在开头触发推理
   - tick=3,7,11... 在末尾触发推理
   - backup_level 计算正确

2. **性能指标**
   - 平均 backup_level <= 0.5
   - hit rate >= 95%
   - 不比老系统差

3. **代码质量**
   - 单元测试覆盖率 >= 80%
   - 通过所有集成测试

---

## 8. 附录：参考代码片段

### 8.1 老系统的推理触发（client_lekai.py:406-420, 673-687）

```python
# tick=0 特殊处理
is_tick_zero = tick_count == 0
if is_tick_zero:
    generation_start_tick = 0
    request_data = {...}
    inference_request_queue.put((request_data, request_data.copy()))
    notes_for_next_request = []
    is_trigger_tick = True

# ... 其他处理 ...

# 4n-1 末尾触发
if not is_tick_zero and (tick_count % ticks_per_beat) == (ticks_per_beat - 1):
    generation_start_tick = tick_count + 1
    request_data = {...}
    inference_request_queue.put((request_data, request_data.copy()))
    notes_for_next_request = []
    is_trigger_tick = True
```

### 8.2 老系统的 backup_level 计算（client_lekai.py:515-520）

```python
gen_start = request_data.get("generation_start_tick")
for n in newly_generated_notes:
    if gen_start is not None:
        n["backup_level"] = max(0, int(n["tick"] - gen_start))
    else:
        n["backup_level"] = 0
```

---

**计划制定者**: Kimi Code  
**审核状态**: 待审核  
**实施优先级**: 高
