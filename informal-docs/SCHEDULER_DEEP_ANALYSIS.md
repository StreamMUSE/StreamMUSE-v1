# PlaybackScheduler 深度分析报告

**文件**: `src/streammuse/domain/timing/scheduler.py`  
**行数**: 62行  
**复杂度**: 中等  
**关键性**: 高（影响整个实时播放系统）  
**分析日期**: 2026年3月11日

---

## 目录

1. [核心概念](#核心概念)
2. [设计目标](#设计目标)
3. [架构分析](#架构分析)
4. [方法深度解析](#方法深度解析)
5. [线程安全机制](#线程安全机制)
6. [实际使用场景](#实际使用场景)
7. [性能特征](#性能特征)
8. [潜在问题](#潜在问题)
9. [改进建议](#改进建议)
10. [总体评价](#总体评价)

---

## 核心概念

### PlaybackScheduler 是什么？

PlaybackScheduler 是一个**事件时序管理器**，负责：
- 📅 管理未来时刻（tick）的音乐事件播放计划
- 🎵 支持多源事件的独立取消（用户输入 vs AI生成）
- ⏱️ 保证线程安全的并发访问

### 类比理解

```
日历系统比喻：
┌─────────────────────────────────────────┐
│  PlaybackScheduler (日历)              │
├─────────────────────────────────────────┤
│  Tick 100: [Event1, Event2]             │
│  Tick 101: [Event3]                     │
│  Tick 105: [Event4, Event5]             │
│  Tick 110: [Event6]                     │
└─────────────────────────────────────────┘
       │
       ├─ schedule():     添加事件到某个日期
       ├─ get_events_at_tick(): 取出并删除某个日期的事件
       └─ clear_future_events(): 删除未来某类事件（选择性）
```

---

## 设计目标

### 1. 事件缓冲器

问题：音乐事件不是实时到达的，推理引擎可能一次生成多个未来事件。

```
无scheduler的情况:
Tick 100: [user_note]     ← 用户按下
Tick 101: (等待)           ← 用户没按
Tick 102: [model_acc]      ← AI生成的伴奏来了，需要回溯到Tick 102播放！❌
                             ↑ 时间已经过了，无法回溯
                             
有scheduler的情况:
Tick 100: [user_note] → schedule(user_note, 100)
Tick 101: (等待)
Tick 102: (等待) → 推理返回 [model_acc] → schedule(model_acc, 102)
          → get_events_at_tick(102) → 立即成功播放！✅
```

### 2. 事件替换机制

问题：当新推理结果比旧结果更准确时，需要**选择性地清除**旧的生成事件，但保留用户事件。

```
时间线:

初始推理 (generation_start_tick=100):
    Tick 100: [model_acc_v1]
    Tick 101: [model_acc_v1]
    Tick 102: [model_acc_v1]
    Tick 103: [model_acc_v1]

新推理 (generation_start_tick=102):
    Tick 102: [model_acc_v2]    ← 新版本更好
    Tick 103: [model_acc_v2]
    Tick 104: [model_acc_v2]

需要的操作:
    clear_future_events(from_tick=102, source="model")
    ↓
    删除: Tick 102, 103 的所有 source="model" 事件
    保留: Tick 102, 103 的 source != "model" 事件
    ↓
结果:
    Tick 102: [model_acc_v2]    ← 新版本
    Tick 103: [model_acc_v2]
    Tick 104: [model_acc_v2]
```

### 3. 线程安全回放

问题：和弦涉及多个线程：
- **Tick线程** (主): 读取和播放事件
- **推理线程**: 写入新事件，清除旧事件
- **输入线程**: 添加用户事件

---

## 架构分析

### 数据结构选择

```python
self._schedule: Dict[int, List[MusicalEvent]] = {}
```

#### 为什么选择 `Dict[int, List[MusicalEvent]]`？

| 选择 | 优点 | 缺点 |
|------|------|------|
| **Dict[int, List]** ✅ | O(1)查找/删除tick、支持稀疏时间、易于清除 | 无序遍历需要sorting |
| List[tuple] | 有序、内存连续 | O(n)查询、O(n)删除 |
| Heap | 优先级队列支持 | 复杂实现、难以选择性清除 |
| SortedDict | 有序查询 | 外部依赖、write O(log n) |
| Set | 去重 | 无序、不支持多个事件同一tick |

**结论**: Dict选择是正确的，trade-off很好。

### 线程安全策略

```python
self._lock = Lock()  # 互斥锁
```

#### 保护范围和级别

| 方法 | 临界区 | 并发冲突场景 |
|------|--------|------------|
| `schedule()` | 整个方法 | 推理线程 + Tick线程同时写 |
| `get_events_at_tick()` | 整个方法 | Tick线程读 + 推理线程写 |
| `clear_future_events()` | 整个方法 | 推理线程清除 + 另一推理线程新增 |

#### 锁的粒度分析

```
粒度太粗 (当前):
    with self._lock:
        ticks_to_consider = [t for t in self._schedule.keys() if t >= from_tick]
        ↓
        所有操作都在锁内执行
        缺点: 如果events很多，list comprehension 也被锁住
        
粒度太细（反面教材）:
    for tick in ticks_to_consider:
        with self._lock:
            ...
        ↓
        每个tick一个锁
        缺点: 死锁风险、竞态条件
        
当前是正确的 ✅
```

---

## 方法深度解析

### 方法1: `schedule(event, tick)`

```python
def schedule(self, event: MusicalEvent, tick: int) -> None:
    """Schedule event for playback at the given tick."""
    with self._lock:
        if tick not in self._schedule:
            self._schedule[tick] = []
        self._schedule[tick].append(event)
```

#### 执行流程

```
输入: MusicalEvent(pitch=60, tick=102, ...), tick=102

步骤1: 获取锁
       ↓
步骤2: 检查tick是否已在字典中
       - 若无: 创建空列表
       - 若有: 跳过
       ↓
步骤3: 将事件append到该tick的列表
       ↓
步骤4: 释放锁
       ↓
返回: None
```

#### 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| `in` 检查 | O(1) | Dict查找 |
| append | O(1) | 列表末尾插入 |
| 总体 | **O(1)** | 平均情况 |

#### 潜在问题

**问题1: Event被修改风险**

```python
# 用户可能在外部修改event对象
ev = MusicalEvent(pitch=60, ...)
scheduler.schedule(ev, tick=102)
setattr(ev, "source", "model")  # 修改后所有引用都变了！
```

**问题2: 保存的是引用，非副本**

```python
events_list = scheduler._schedule[102]
events_list[0].pitch = 100  # 直接修改了调度事件！
```

**解决方案**: 
```python
def schedule(self, event: MusicalEvent, tick: int) -> None:
    with self._lock:
        if tick not in self._schedule:
            self._schedule[tick] = []
        # 深拷贝以防外部修改
        self._schedule[tick].append(copy.deepcopy(event))
```

---

### 方法2: `get_events_at_tick(tick)`

```python
def get_events_at_tick(self, tick: int) -> List[MusicalEvent]:
    """Return all events scheduled for this tick and remove them from the schedule."""
    with self._lock:
        return self._schedule.pop(tick, [])
```

#### 执行流程

```
输入: tick=102

步骤1: 获取锁
       ↓
步骤2: pop(tick, [])
       - 若tick存在: 返回事件列表 + 删除该key
       - 若不存在: 返回空列表
       ↓
步骤3: 释放锁
       ↓
返回: List[MusicalEvent]
```

#### 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| pop | O(1) | Dict删除 |
| 总体 | **O(1)** | 独立于事件数量 |

#### 设计优雅点

✅ **原子性**: pop和删除是一个操作，不会出现"读出来又没删除"的竞态条件

✅ **简洁**: 一行代码完成"取出并清除"

✅ **容错**: `[])` 默认值处理tick不存在的情况

#### 使用场景

```python
# 在 _tick_loop 中被调用
for ev in self._scheduler.get_events_at_tick(tick):
    self._output.output_event(ev, source=source)
```

每一次循环迭代，只有这一个tick的事件被播放并移除。

---

### 方法3: `clear_future_events(from_tick, source=None)`

这是最复杂的方法。

```python
def clear_future_events(
    self,
    from_tick: int,
    source: str | None = None,
) -> None:
    """
    Remove events from from_tick onward.

    If source is provided, only remove events that have a matching
    'source' attribute (e.g. "model"); events without a source attribute
    are kept. If source is None, clear all events in the range.
    """
    with self._lock:
        ticks_to_consider = [t for t in self._schedule.keys() if t >= from_tick]
        for tick in ticks_to_consider:
            if source is None:
                del self._schedule[tick]
            else:
                self._schedule[tick] = [
                    e for e in self._schedule[tick]
                    if getattr(e, "source", None) != source
                ]
                if not self._schedule[tick]:
                    del self._schedule[tick]
```

#### 逻辑分支

```
输入: from_tick=102, source="model"

上游场景:
├─ 新推理 (generation_start_tick=102) 返回了伴奏事件。
├─ 需要清除从Tick 102开始的所有旧的model源事件。
└─ 但要保留用户输入的事件。

执行逻辑:

┌─────────────────────────────────────────┐
│  Step 1: 找出所有 >= from_tick 的tick   │
│  ticks_to_consider = [102, 103, 105]    │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  Step 2: 遍历每个tick                   │
└─────────────────────────────────────────┘
           ↓
     ┌─────────────────┐
     │ if source is None?
     └─────────────────┘
      ↙                 ↘
    YES (全删)         NO (选择性删)
     │                  │
   [删除整个tick]    [保留source!=<value>的事件]
                      │
                   过滤列表 = [
                       e for e in events
                       if getattr(e, "source", None) != source
                   ]
                      │
                   若过滤后为空→删除tick
```

#### 时间复杂度分析

```python
# 设:
# - K = 调度中的tick数
# - N = 某个tick的平均事件数
# - F = >= from_tick 的tick数（F <= K）

ticks_to_consider = [...]           # O(K) 遍历所有keys
for tick in ticks_to_consider:      # O(F) 遍历符合条件的ticks
    self._schedule[tick] = [        # O(N) 列表推导式
        e for e in self._schedule[tick]
        ...
    ]

总体时间复杂度: O(K + F*N)
```

**最坏情况**: 所有事件都在from_tick之后，且都需要过滤
```
复杂度 = O(K + K*N) = O(K*N)  # 其中K是tick数，N是每个tick的事件数
```

#### 设计问题1: getattr 的运行时成本

```python
if getattr(e, "source", None) != source
```

**分析**:
- `getattr()` 使用反射，每次都要查找属性
- 对于有N个事件的列表，被调用N次
- 相比直接访问 `e.source`，反射有约3-5倍的开销

**当前风险**: 如果事件列表很长（10000+），反复调用getattr会很慢

**改进方案**:
```python
# 方案1: 预先检查属性存在
for e in self._schedule[tick]:
    event_source = getattr(e, "source", None)
    if event_source != source:
        filtered.append(e)

# 方案2: 使用hasattr + 直接访问
for e in self._schedule[tick]:
    if not hasattr(e, "source") or e.source != source:
        filtered.append(e)

# 方案3: 要求所有事件都有source属性（架构级解决）
# MusicalEvent 中强制添加 source: str = "unknown"
```

#### 设计问题2: 原子性风险

```
时间线假设:

Tick线程 (T1):
    tick = 102
    events = scheduler.get_events_at_tick(102)     # 获取 = [E1, E2]
                                                     ↓ 时间流动 ↓
推理线程 (T2):
    clear_future_events(from_tick=102, source="model")  # 清除
    schedule(new_event, 102)                           # 新增
                                                     ↓ 时间流动 ↓
Tick线程 (T1):
    for ev in events:  # 播放 [E1, E2]
        output(ev)     # 问题: E1和E2可能已经被清除了！
```

**问题**: `get_events_at_tick()` 返回的列表脱离了scheduler的保护。

**但实际上OK吗?** 
✅ 既然已经pop出来了，就从scheduler中删除了。
❌ 但这意味着新的事件可能在这之间被清除，导致不一致。

---

## 线程安全机制

### 并发场景模拟

#### 场景1: 同时schedule和get_events

```
线程A (推理线程):
    1. clear_future_events(102, "model") - 在锁内执行 ⏱️ 100ms
    2. schedule(new_event, 102)          - 等待锁...

线程B (Tick线程):
    1. 等待锁...
    2. get_events_at_tick(102)           - 获得锁，pop出事件
    3. 播放事件

时间顺序:
100ms: A 开始clear (持锁)
150ms: B 试图get (等锁)
200ms: A clear完成，释放锁
201ms: B 获得锁，pop出102的事件
202ms: A 重新获得锁，schedule新事件到102
```

**安全性**: ✅ 1000% 安全，所有操作都互斥执行

#### 场景2: 竞态条件测试

```python
# 两个推理线程同时返回结果
推理线程1: clear(102, "model")  → 删除所有model事件
推理线程2: clear(102, "model")  → 删除... (已经被T1删了)

结果: 安全 ✅ (set.clear() 重复调用无害)
```

#### 场景3: 列表修改竞态

```python
# 推理线程返回事件，Tick线程同时遍历
推理线程:
    clear_future_events(102, "model")  
    schedule(E1, 102)  # _schedule[102] = [E1]

Tick线程:
    events = get_events_at_tick(102)   # pop: events = [E1]
    for ev in events:                  # 迭代 captured list
        output(ev)                     # 安全，事件已从dict中移除
```

**结论**: ✅ 锁的设计很好，没有发现竞态条件

---

## 实际使用场景

### 场景1: 用户输入 + AI生成的实时播放

```
时间轴:
┌─────────────────────────────────────────────────┐
│  Tick 0    Tick 100        Tick 200              │
│  |----------|----------|----------|              │
│             User presses key
│             输入线程: schedule(NOTE_ON, 100)
│                      schedule(NOTE_OFF, 150)
│
│             推理线程: generate() 获取 [100-149] 的旋律
│             返回伴奏: [105, 120, 135] 的低音和弦
│             推理线程: clear(100, "model")
│                      schedule(ACC1, 105)
│                      schedule(ACC1, 120)
│                      schedule(ACC1, 135)
│
│  Tick 100: get_events(100) → [NOTE_ON]   → output to speaker
│  Tick 105: get_events(105) → [ACC1]      → output to speaker
│  Tick 120: get_events(120) → [ACC1]      → output to speaker
│  Tick 135: get_events(135) → [ACC1]      → output to speaker
│  Tick 150: get_events(150) → [NOTE_OFF]  → output to speaker
└─────────────────────────────────────────────────┘
```

### 场景2: 推理结果更新时的事件替换

```
初始推理 (开始于 Tick 100):
    result_v1 = [
        Event(pitch=48, tick=100, duration=50),  # C  低音
        Event(pitch=55, tick=100, duration=50),  # G  中音
        Event(pitch=60, tick=125, duration=25),  # C  高音
    ]

时刻Tick150:
    推理引擎返回新结果 v2
    clear_future_events(100, source="model")  # 删除旧版本
    schedule 新版本
    
结果_v2 = [
    Event(pitch=48, tick=100, duration=50),  # C  (相同)
    Event(pitch=57, tick=100, duration=50),  # A  (改进！)
    Event(pitch=62, tick=125, duration=25),  # D  (改进！)
]

用户听到: v2版本（更好的和声）
```

---

## 性能特征

### 时间复杂度总结

| 操作 | 时间复杂度 | 空间复杂度 |
|------|----------|----------|
| `schedule()` | O(1) | O(1) |
| `get_events_at_tick()` | O(1) | O(n) |
| `clear_future_events()` | O(K + F×N) | O(1) |

其中:
- K = 调度中的总tick数
- F = >= from_tick的tick数
- N = 单个tick的平均事件数

### 在实时系统中的表现

```
假设条件:
- BPM = 120 (8.33ms per tick, 4 ticks per beat)
- 调度最多100个tick (约1.2秒的未来事件)
- 每个tick平均2个事件

最坏情况分析:
- schedule(): 0.001ms (Dict hash)
- get_events_at_tick(): 0.001ms (Dict pop)
- clear_future_events(50% ticks): 0.1ms (50 ticks × 2 events)

总体: < 0.2ms per tick ✅
Tick周期: 8.33ms
开销占比: 2.4% （完全可接受）
```

### 内存使用

```
假设:
- 100个future ticks
- 2个events per tick平均
- MusicalEvent大小 ≈ 200字节

最坏内存占用:
= 100 ticks × 2 events × 200 bytes + Dict overhead
≈ 40KB + 10KB overhead
≈ 50KB

实际可接受吗? ✅ 完全可以接受（现代系统的1毫秒）
```

---

## 潜在问题

### 问题1: 属性获取的脆弱性 ⚠️ 中风险

```python
# 当前实现使用 getattr 动态获取source属性
getattr(e, "source", None) != source

# 问题:
# 1. source不是MusicalEvent的正式属性
# 2. 通过setattr动态添加，容易出错
# 3. 如果忘记setattr，逻辑会失败

# 用户容易犯的错误:
ev = MusicalEvent(...)
scheduler.schedule(ev, 102)
# 忘记了: setattr(ev, "source", "model")
# 结果: clear(..., source="model") 无法清除这个事件！

# 实际代码:
ev_with_source = MusicalEvent(...)
setattr(ev_with_source, "source", "model")  # 硬转不可靠
scheduler.schedule(ev_with_source, ev.tick)
```

**改进方案**:
```python
# 选项1: 扩展MusicalEvent
@dataclass(frozen=True)
class MusicalEvent:
    ...
    source: str = "unknown"  # 新增字段，默认值

# 选项2: 创建ScheduledEvent包装
@dataclass(frozen=True)
class ScheduledEvent:
    event: MusicalEvent
    source: str  # 来源标记
    
# 选项3: 使用Protocol而非动态属性
from typing import Protocol
class HasSource(Protocol):
    source: str
```

---

### 问题2: Clear操作的语义歧义 ⚠️ 中风险

```python
clear_future_events(from_tick=102, source="model")

问题1: "未来"的定义
    >= from_tick 还是 > from_tick?
    当前: >= from_tick ✓ 清楚，但文档没说

问题2: source=None 的语义
    # 这两个调用的区别是什么？
    scheduler.clear_future_events(102, source=None)  # 清除所有
    scheduler.clear_future_events(102)               # 也是清除所有
    
    # 为什么要有两种写法？
    
问题3: "source" 参数注释不清楚
    文档说 "e.g. 'model'" 但没说标准值有哪些
    用户可能用 "user", "system", "ai" 等混乱命名
```

**改进方案**:
```python
from enum import Enum

class EventSource(Enum):
    """事件来源标准枚举"""
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"

def clear_future_events(
    self,
    from_tick: int,
    source: EventSource | None = None,
) -> None:
    """
    移除指定来源的未来事件。
    
    Args:
        from_tick: 起始刻度（包含），即 tick >= from_tick 的事件会被考虑
        source: 事件来源。如果为None，清除所有来源的事件。
    """
```

---

### 问题3: 没有事件优先级 ⚠️ 低风险

当前实现中，同一tick的事件播放顺序是不确定的（List顺序 = 添加顺序）。

```python
# 问题场景: 和弦的所有音符应该同时播放
scheduler.schedule(Note(pitch=48), 100)  # 第一个添加的 → 第一个播放
scheduler.schedule(Note(pitch=60), 100)  # 最后添加的 → 最后播放
# 结果: 可能不是完全同步 (FIFO)

# MIDI标准中，同一tick的所有音符应该有相同的演奏时间
```

**但实际影响**:
- ✅ 时间分辨率是tick级别，毫秒级足够了
- ✅ MIDI设备会立即全部处理
- ❌ 但逻辑上应该是完全并行的

**改进**:
```python
# 可以添加priority字段
@dataclass(frozen=True)
class ScheduledEvent:
    event: MusicalEvent
    priority: int = 0  # 优先级
    
# 播放时用 heapq 排序
```

---

### 问题4: 没有事件合并 ⚠️ 中风险

```python
# 如果同一个note被schedule两次会怎样？
scheduler.schedule(Note(pitch=60, tick=100), 100)
scheduler.schedule(Note(pitch=60, tick=100), 100)  # 重复

result = scheduler.get_events_at_tick(100)
# result = [Note(...), Note(...)]  # 两个相同的音符！

# 结果: 音频会播放两遍（错误）
```

**为什么会发生**: 
- 推理线程可能返回重复的伴奏
- 或者schedule前没有检查去重

**改进**:
```python
def schedule(self, event: MusicalEvent, tick: int) -> None:
    with self._lock:
        if tick not in self._schedule:
            self._schedule[tick] = set()  # 用Set而非List
        # 但MusicalEvent必须是hashable的
        self._schedule[tick].add(event)
```

---

### 问题5: 没有过期事件清理 ⚠️ 低风险

```python
# 如果一直不调用 get_events_at_tick(tick)会怎样？
scheduler.schedule(ev, 100)
scheduler.schedule(ev2, 101)
# tick 100, 101 的事件永远不会被取出和删除

# 结果: 内存泄漏（随着运行时间增长）
```

**为什么会发生**:
- 如果输出处理器失败，（可能不会继续播放
- 或者软件崩溃后重启，之前的schedule仍在dict中

**改进**:
```python
def cleanup_before_tick(self, tick: int) -> None:
    """清理所有过期的事件（已经过的tick）"""
    with self._lock:
        expired_ticks = [t for t in self._schedule.keys() if t < tick]
        for t in expired_ticks:
            del self._schedule[t]
```

---

## 改进建议

### 优先级高 🔴

#### 1. 使MusicalEvent包含source字段

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
    source: str = "unknown"  # ← 新增，避免动态属性
```

**影响**:
- 修复问题1（属性获取脆弱性）
- 使代码更type-safe
- 需要修改所有创建MusicalEvent的地方

---

#### 2. 标准化事件来源

```python
from enum import Enum

class EventSource(Enum):
    USER = "user"
    MODEL = "model"

# 更新MusicalEvent
source: EventSource = EventSource.USER  # 默认值

# 更新clear_future_events
def clear_future_events(
    self,
    from_tick: int,
    source: EventSource | None = None,
) -> None:
```

**影响**:
- 消除歧义，提高代码的can confidence
- 编译期检查，防止拼写错误

---

### 优先级中 🟡

#### 3. 优化clear_future_events的列表推导

```python
# 当前 O(F*N) 的操作
self._schedule[tick] = [
    e for e in self._schedule[tick]
    if getattr(e, "source", None) != source
]

# 改进: 使用filter而不是列表推导（微优化）
self._schedule[tick] = list(filter(
    lambda e: getattr(e, "source", None) != source,
    self._schedule[tick]
))

# 或者如果source是正式字段:
self._schedule[tick] = [
    e for e in self._schedule[tick]
    if e.source != source
]
```

---

#### 4. 添加过期事件清理

```python
def cleanup_expired_events(self, current_tick: int) -> None:
    """清理已过期（已播放）的事件"""
    with self._lock:
        expired = [t for t in self._schedule.keys() if t < current_tick]
        for t in expired:
            del self._schedule[t]
```

**在哪里调用**:
```python
# 在 _tick_loop 中
def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
    ...
    if tick % 100 == 0:  # 每100个tick清理一次
        self._scheduler.cleanup_expired_events(tick - 50)  # 清理50 tick之前的
```

---

### 优先级低 🟢

#### 5. 添加调试方法

```python
def get_schedule_info(self) -> Dict[str, int]:
    """返回调度统计信息"""
    with self._lock:
        return {
            "ticks_scheduled": len(self._schedule),
            "total_events": sum(len(evs) for evs in self._schedule.values()),
            "earliest_tick": min(self._schedule.keys()) if self._schedule else None,
            "latest_tick": max(self._schedule.keys()) if self._schedule else None,
        }

def dump_schedule(self, from_tick: int | None = None) -> str:
    """生成人类可读的调度状态，用于调试"""
    with self._lock:
        ticks = sorted(self._schedule.keys())
        if from_tick is not None:
            ticks = [t for t in ticks if t >= from_tick]
        
        lines = []
        for tick in ticks:
            events = self._schedule[tick]
            lines.append(f"Tick {tick}: {len(events)} events")
            for ev in events:
                lines.append(f"  - {ev.event_type.value} pitch={ev.pitch}")
        return "\n".join(lines)
```

**使用场景**: 调试实时播放问题

---

#### 6. 添加事件度量

```python
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class SchedulerMetrics:
    total_scheduled: int
    total_cleared: int
    max_queue_depth_ticks: int
    max_events_per_tick: int

class PlaybackScheduler:
    def __init__(self) -> None:
        self._schedule: Dict[int, List[MusicalEvent]] = {}
        self._lock = Lock()
        
        # 度量数据
        self._metrics_scheduled = 0
        self._metrics_cleared = 0
    
    def schedule(self, event: MusicalEvent, tick: int) -> None:
        with self._lock:
            if tick not in self._schedule:
                self._schedule[tick] = []
            self._schedule[tick].append(event)
            self._metrics_scheduled += 1
    
    def get_metrics(self) -> SchedulerMetrics:
        with self._lock:
            ticks = len(self._schedule)
            events_per_tick = [len(v) for v in self._schedule.values()]
            max_per_tick = max(events_per_tick) if events_per_tick else 0
            
            return SchedulerMetrics(
                total_scheduled=self._metrics_scheduled,
                total_cleared=self._metrics_cleared,
                max_queue_depth_ticks=ticks,
                max_events_per_tick=max_per_tick,
            )
```

---

## 总体评价

### 优点 ✅

| 优点 | 说明 | 评分 |
|------|------|------|
| **设计思路清晰** | 数据结构选择合理，符合use case | ⭐⭐⭐⭐⭐ |
| **线程安全** | 没有竞态条件，Lock使用恰当 | ⭐⭐⭐⭐⭐ |
| **代码简洁** | 62行代码实现核心功能 | ⭐⭐⭐⭐⭐ |
| **性能优秀** | O(1)操作居多，不是性能瓶颈 | ⭐⭐⭐⭐⭐ |
| **API直观** | 三个方法功能单一、易理解 | ⭐⭐⭐⭐☆ |

### 缺点 ❌

| 缺点 | 严重程度 | 改进难度 |
|------|---------|---------|
| source属性动态获取（反射） | 中 | 简单 |
| 事件去重策略缺失 | 中 | 中等 |
| 没有事件来源标准化 | 中 | 简单 |
| 无过期事件清理 | 低 | 简单 |
| 没有调试信息 | 低 | 简单 |

### 架构师总体评分

```
┌────────────────────────────────────┐
│ 整体评分: 8.5 / 10                 │
├────────────────────────────────────┤
│ 功能完整性:    ⭐⭐⭐⭐⭐  (5/5)   │
│ 代码质量:      ⭐⭐⭐⭐⭐  (5/5)   │
│ 线程安全:      ⭐⭐⭐⭐⭐  (5/5)   │
│ 可维护性:      ⭐⭐⭐⭐☆  (4/5)   │
│ 文档完整度:    ⭐⭐⭐☆☆  (3/5)   │
│ 监控可观测:    ⭐⭐☆☆☆  (2/5)   │
└────────────────────────────────────┘
```

---

## 深度问题思考

### 问: 为什么不用优先队列 (Heap) 而用Dict?

```python
# 堆方案:
heapq.heappush(self._heap, (tick, event))

好处:
- O(log n) 插入
- 自动排序

坏处:
- 选择性删除很难 (clear_future_events)
- 无法O(1)查询"tick 100有哪些事件"
- 需要额外的数据结构来追踪indices

Dict方案 (当前) ✅:
- O(1)插入、查询、删除
- 支持选择性清除
- 符合"按tick查询"的use case
```

**结论**: Dict是正确选择

---

### 问: 为什么MusicalEvent不直接包含source?

```python
# 当前方式: 动态属性
setattr(event, "source", "model")

# 问题: 
# 1. MusicalEvent在domain层，是纯数据对象
# 2. "source"是infrastructure层的概念
# 3. 混合会违反分层原则

# 但这造成了:
# - Type-unsafe
# - 反射开销
# - 容易出错
```

**改进哲学**:
```python
# 选项A: 升级MusicalEvent
@dataclass(frozen=True)
class MusicalEvent:
    ...
    source: str = "unknown"

# 选项B: 创建包装类型
@dataclass(frozen=True)
class ScheduledMusicalEvent:
    event: MusicalEvent
    source: str
    
    def __getattr__(self, name):
        return getattr(self.event, name)  # 代理

# 选项C: 保留current (trade-off接受的話)
# 权衡: 纯度 vs 实用性
```

---

### 问: clear_future_events的"未来"从何定义？

```python
# 当前: >= from_tick
ticks_to_consider = [t for t in self._schedule.keys() if t >= from_tick]

# 问题: 如果generation_start_tick=102
#      是否应该清除tick 102本身?

#      在 _tick_loop 中:
#      - 推理返回 generation_start_tick=102
#      - clear(102, "model") ← 是否包括102?
#      - schedule(102) ← 然后schedule新事件到102
#      
#      结果: tick 102 被清除又重新schedule
#           这是intentional吗?

# 答案: 是的！ (intentional)
# 原因: 推理返回时，可能102已经有旧的model事件
#      需要清除它们并用新的替换
```

**设计合理性**: ✅ OK

---

## 与系统的关键交互

### 1. RealTimeMusicService 中的使用

```python
def _tick_loop(self, *, max_ticks: Optional[int]) -> None:
    while self._running:
        # ... 时序逻辑 ...
        
        # 处理推理响应时
        if 有新推理结果:
            self._scheduler.clear_future_events(from_tick=..., source="model")
            for ev in acc_events:
                self._scheduler.schedule(ev_with_source, ev.tick)
        
        # 播放事件时
        for ev in self._scheduler.get_events_at_tick(tick):
            self._output.output_event(ev, source=source)
```

**设计意图**:
- scheduler 是推理响应和输出之间的**缓冲**
- 用于吸收推理的延迟
- 确保事件按tick精确播放

---

## 总结

PlaybackScheduler 是StreamMUSE系统中一个**设计良好但有小缺点**的组件：

### 核心强项
1. ✅ 简洁高效的事件管理
2. ✅ 完美的线程安全设计
3. ✅ 支持source-based selective clearing（独特优势）

### 需要改进的地方
1. ⚠️ source属性应该正式化（非动态）
2. ⚠️ 缺少事件来源的标准化枚举
3. ⚠️ 缺少过期事件清理机制

### 生产级建议
- 短期: 添加source作为MusicalEvent的正式字段
- 中期: 添加度量和调试方法
- 长期: 考虑事件优先级和去重

---

**报告完成**: 2026年3月11日  
**分析深度**: 架构师级别  
**建议实施**: 高优先级2项，中优先级3项，低优先级2项
