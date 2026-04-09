# StanleyInferenceEngine vs LegacyInferenceEngineStanley 关系详析

## 一句话总结

**StanleyInferenceEngine** 是一个**适配器（Adapter）**，它包装了**LegacyInferenceEngineStanley**（遗留推理引擎），使其符合项目的 `InferenceEngine` 领域接口。

```
┌─────────────────────────────────┐
│  Domain Layer                   │
│  (InferenceEngine Protocol)     │
└───────────────┬─────────────────┘
                │ implements
┌───────────────▼─────────────────┐
│  StanleyInferenceEngine         │  ← 新建的适配器
│  (Infrastructure Layer)         │
└───────────────┬─────────────────┘
                │ wraps & delegates
┌───────────────▼─────────────────┐
│  LegacyInferenceEngineStanley   │  ← 遗留的推理实现
│  (Legacy Code)                  │
└─────────────────────────────────┘
```

---

## 详细关系分析

### 1. 文件位置

| 类名 | 文件位置 | 用途 |
|------|---------|------|
| **LegacyInferenceEngineStanley** | `src/streammuse/infrastructure/inference/stanley_legacy.py` | 遗留的模型推理实现 |
| **StanleyInferenceEngine** | `src/streammuse/infrastructure/inference/stanley_engine.py` | 现代化的适配器包装 |

### 2. 职责分工

#### LegacyInferenceEngineStanley（遗留引擎）

**职责**: 直接执行模型推理

```python
class LegacyInferenceEngineStanley:
    """直接与RoFormer模型交互的遗留代码"""
    
    def __init__(self, *, checkpoint_path, model_size, ...):
        # 加载RoFormer Transformer模型
        self.model = RoFormerSymbolicTransformer.load_from_checkpoint(...)
        
        # 维护内部状态（duration-note格式）
        self.melody_history: list[dict] = []
        self.accompaniment_history: list[dict] = []
        self.injection_offset_ticks = 0
    
    def generate_accompaniment(
        self,
        melody_notes: list[dict],  # ← 输入：duration-note format
        generation_start_tick: int,
        acc_notes: Optional[list[dict]] = None,
        generation_length_frames: Optional[int] = None,
        prompt_length_ticks: Optional[int] = None,
    ) -> tuple[list[dict], float, float, float, float]:  # ← 输出：duration-note format
        # 1. Preprocess: notes → piano rolls → tensors
        # 2. Inference: RoFormer.global_sampling() 
        # 3. Postprocess: tensors → piano rolls → notes
        ...
```

**关键特性**:
- ✅ 直接管理模型状态（模型加载、GPU/CPU管理）
- ✅ 处理复杂的推理逻辑（piano roll编码、张量操作）
- ✅ 维护melody/accompaniment历史供连续生成用
- ✅ 支持注入（injection）机制（用于music injection特性）
- ❌ 但数据格式是 `dict[{"pitch", "tick", "duration", ...}]`（duration-note）
- ❌ 没有实现标准的 `InferenceEngine` 接口

---

#### StanleyInferenceEngine（适配器）

**职责**: 包装遗留引擎，提供清洁的领域接口

```python
class StanleyInferenceEngine(InferenceEngine):  # ← 实现domain接口
    """现代化适配器"""
    
    def __init__(self, *, config, legacy_engine=None):
        if legacy_engine is None:
            # 如果没提供，自动创建遗留引擎
            self._legacy = LegacyInferenceEngineStanley(...)
        else:
            # 测试时可以注入mock
            self._legacy = legacy_engine
    
    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],  # ← 输入：event stream
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:  # ← 输出：event stream
        # 1. Convert: MusicalEvent → Note dict → call legacy engine
        melody_notes = events_to_notes(melody_events, horizon_tick=generation_start_tick)
        melody_dicts = [...]
        
        # 2. Call legacy engine
        acc_notes, ... = self._legacy.generate_accompaniment(melody_dicts, ...)
        
        # 3. Convert: Note dict → MusicalEvent
        events = []
        for n in acc_notes:
            note = Note(...)
            events.extend(note.to_events())  # ← 转换为event stream
        
        return events, timing
```

**关键特性**:
- ✅ 实现 `InferenceEngine` 接口（符合规约）
- ✅ 数据格式转换：MusicalEvent ↔️ dict note
- ✅ 隐藏遗留引擎的复杂性
- ✅ 提供统一的API给 `RealTimeMusicService`
- ✅ 可以注入mock `legacy_engine`（便于测试）
- ❌ 依然依赖 `LegacyInferenceEngineStanley` 的实现

---

### 3. 数据流向图

```
StanleyInferenceEngine.generate_accompaniment():

输入:
    melody_events: List[MusicalEvent]
    ↓
┌──────────────────────────────────────┐
│ 转换步骤1: MusicalEvent → Note        │
│ events_to_notes(melody_events, ...)  │
└──────────────────────────────────────┘
    ↓
    melody_notes: List[Note]
    ↓
┌──────────────────────────────────────┐
│ 转换步骤2: Note → dict                │
│ [{"pitch": ..., "duration": ..., ...}]
└──────────────────────────────────────┘
    ↓
    melody_dicts: list[dict]
    ↓
┌──────────────────────────────────────┐
│ 委托给遗留引擎                         │
│ LegacyInferenceEngineStanley         │
│ .generate_accompaniment(melody_dicts)
└──────────────────────────────────────┘
    ↓
    acc_notes: list[dict]  (duration-note format)
    ↓
┌──────────────────────────────────────┐
│ 转换步骤3: dict → Note                │
│ Note(pitch=n["pitch"], ...)           │
└──────────────────────────────────────┘
    ↓
    Note对象列表
    ↓
┌──────────────────────────────────────┐
│ 转换步骤4: Note → MusicalEvent       │
│ note.to_events()  # 展开为NOTE_ON/OFF │
└──────────────────────────────────────┘
    ↓
输出:
    events: List[MusicalEvent]
    timing: TimingInfo
```

---

### 4. 关键交互点

#### 4.1 构造器注入模式

```python
class StanleyInferenceEngine(InferenceEngine):
    def __init__(
        self, 
        *, 
        config: StanleyInferenceConfig, 
        legacy_engine: _LegacyStanleyLike | None = None
    ) -> None:
        self._config = config
        
        if legacy_engine is None:
            # 生产环境：自动创建遗留引擎
            from streammuse.infrastructure.inference.stanley_legacy import LegacyInferenceEngineStanley
            self._legacy = LegacyInferenceEngineStanley(
                checkpoint_path=config.checkpoint_path,
                model_size=config.model_size,
                max_polyphony=config.max_polyphony,
                model_max_seq_len_frames=config.model_max_seq_len_frames,
                generation_length_frames=config.generation_length_frames,
            )
        else:
            # 测试环境：使用注入的mock
            self._legacy = legacy_engine
```

**优点**:
- ✅ 支持依赖注入（便于单元测试）
- ✅ 使用Protocol `_LegacyStanleyLike` 而非具体类型（鸭子类型）
- ✅ 延迟导入遗留模块（避免循环依赖）

---

#### 4.2 Protocol定义

```python
class _LegacyStanleyLike(Protocol):
    """定义遗留引擎必须实现的接口"""
    
    generation_length_frames: int
    
    def generate_accompaniment(
        self,
        melody_notes: list[dict],
        generation_start_tick: int,
        acc_notes: Optional[list[dict]] = None,
        generation_length_frames: Optional[int] = None,
        prompt_length_ticks: Optional[int] = None,
    ) -> tuple[list[dict], float, float, float, float]:
        ...
    
    def clear_history(self) -> None: ...
    
    def set_injection_offset(self, offset_ticks: int) -> None: ...
```

**作用**:
- ✅ 不依赖具体的 `LegacyInferenceEngineStanley` 类
- ✅ 允许用mock对象进行单元测试
- ✅ 清晰地定义了遗留引擎的"契约"

---

### 5. 历史信息与状态管理

#### LegacyInferenceEngineStanley（遗留引擎）维护历史

```python
class LegacyInferenceEngineStanley:
    def __init__(self, ...):
        self.melody_history: list[dict] = []              # ← 旋律历史
        self.accompaniment_history: list[dict] = []       # ← 伴奏历史
        self.injection_offset_ticks = 0                   # ← 注入偏移
    
    def set_injection_offset(self, offset_ticks: int) -> None:
        """设置注入偏移"""
        self.injection_offset_ticks = int(offset_ticks)
    
    def clear_history(self) -> None:
        """清除历史（用于新的曲目开始）"""
        self.melody_history = []
        self.accompaniment_history = []
    
    def generate_accompaniment(self, ...):
        # 逻辑中：
        # 1. 将新的旋律保存到 self.melody_history
        # 2. 从历史中提取"prompt"（上文）
        # 3. 使用prompt生成新的伴奏
        # 4. 更新 self.accompaniment_history
        self.melody_history.extend(absolute_melody)      # ← 追加新旋律
        # ... 推理 ...
        self.accompaniment_history = [...]  # ← 更新伴奏历史
```

**为什么需要历史?**

```
连续音乐生成的过程:

音乐事件时间线:
Tick:  0    10    20    30    40    50
       ├────┤────┤────┤────┤────┤
       用户输入     用户输入      用户输入

每次推理:
推理1 (Tick 10) : 上下文 [0-10] → 生成 [10-30] 的伴奏
推理2 (Tick 20) : 上下文 [0-20] → 生成 [20-40] 的伴奏
推理3 (Tick 30) : 上下文 [0-30] → 生成 [30-50] 的伴奏

每次推理都需要前面的历史作为"上下文"
这样才能生成连贯的、风格一致的伴奏。

如果不维护历史，每次推理的输入都是孤立的
→ 生成的伴奏可能风格不一致、不连贯
```

#### StanleyInferenceEngine（适配器）代理历史操作

```python
class StanleyInferenceEngine(InferenceEngine):
    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        """注入历史（用于music injection特性）"""
        self.clear_history()
        self.set_injection_offset(int(injection_length_ticks))
        
        # 转换格式后，直接修改遗留引擎的历史
        mel_notes = events_to_notes(melody_events, ...)
        acc_notes = events_to_notes(accompaniment_events, ...)
        
        if hasattr(self._legacy, "melody_history"):
            self._legacy.melody_history.extend([...])
        if hasattr(self._legacy, "accompaniment_history"):
            self._legacy.accompaniment_history.extend([...])
    
    def clear_history(self) -> None:
        """清除历史"""
        self._legacy.clear_history()  # ← 代理调用
    
    def set_injection_offset(self, offset_ticks: int) -> None:
        """设置注入偏移"""
        self._legacy.set_injection_offset(int(offset_ticks))  # ← 代理调用
```

**关键观察**: StanleyInferenceEngine 甚至直接修改遗留引擎的内部状态！

```python
# 在 inject_history 中
if hasattr(self._legacy, "melody_history"):
    self._legacy.melody_history.extend([...])  # ← 直接修改！
```

这表明：
- ✅ 适配器对遗留引擎的实现细节有很深的了解
- ❌ 但也意味着耦合度很高（依赖内部实现）

---

### 6. 数据格式对比

#### Duration-Note格式（遗留格式）

LegacyInferenceEngineStanley 使用的格式：

```python
melody_notes: list[dict] = [
    {
        "pitch": 60,
        "tick": 0,
        "duration": 10,      # ← 关键：显式的duration
        "velocity": 100,
        "program": 0,
    },
    {
        "pitch": 62,
        "tick": 10,
        "duration": 20,
        "velocity": 100,
        "program": 0,
    },
]
```

**高层表示**: 用"音符"的方式表示音乐
- 一个Note对象 = 一个声音事件（从开始到结束）

---

#### Event Stream格式（新格式）

StanleyInferenceEngine 使用的格式：

```python
melody_events: list[MusicalEvent] = [
    MusicalEvent(tick=0, event_type=EventType.NOTE_ON, pitch=60, ...),
    MusicalEvent(tick=10, event_type=EventType.NOTE_OFF, pitch=60, ...),
    MusicalEvent(tick=10, event_type=EventType.NOTE_ON, pitch=62, ...),
    MusicalEvent(tick=30, event_type=EventType.NOTE_OFF, pitch=62, ...),
]
```

**低层表示**: 用"事件"的方式表示音乐
- 每个Note = 两个Event（NOTE_ON 和 NOTE_OFF）

---

#### 格式转换

```python
# Duration-Note → Event Stream
def note_to_events(note: Note) -> List[MusicalEvent]:
    """一个Note展开为两个事件"""
    return [
        MusicalEvent(
            tick=note.tick, 
            event_type=EventType.NOTE_ON, 
            pitch=note.pitch, 
            ...
        ),
        MusicalEvent(
            tick=note.tick + note.duration,  # ← 计算NOTE_OFF的时刻
            event_type=EventType.NOTE_OFF, 
            pitch=note.pitch, 
            ...
        ),
    ]

# Event Stream → Duration-Note
def events_to_notes(events: List[MusicalEvent], ...) -> List[Note]:
    """配对NOTE_ON/OFF事件，重建Note对象"""
    for event in events:
        if event.event_type == EventType.NOTE_ON:
            # 找对应的NOTE_OFF
            matching_off = find_matching_note_off(event, events)
            duration = matching_off.tick - event.tick
            notes.append(Note(
                tick=event.tick,
                pitch=event.pitch,
                duration=duration,
                ...
            ))
```

---

### 7. 完整的调用链路

从RealTimeMusicService到模型推理：

```python
# 1. RealTimeMusicService（应用层）
class RealTimeMusicService:
    def __init__(self, inference_engine: InferenceEngine):
        self._inference_engine = inference_engine  # ← 期望InferenceEngine
    
    def _inference_worker(self):
        acc_events, timing = self._inference_engine.generate_accompaniment(
            melody_events=melody_excerpt,  # ← 是List[MusicalEvent]
            generation_start_tick=generation_start_tick,
            generation_length_frames=generation_length_frames,
        )
        # 继续处理acc_events...

# 2. StanleyInferenceEngine（适配器）
class StanleyInferenceEngine(InferenceEngine):
    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],  # ← 接收Event Stream
        ...
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        # 转换
        melody_notes = events_to_notes(melody_events, ...)  # → List[Note]
        melody_dicts = [...]  # → list[dict] (Duration-Note格式)
        
        # 委托
        acc_notes, ... = self._legacy.generate_accompaniment(
            melody_dicts, ...  # 传递dict格式
        )
        
        # 转换回来
        events = []
        for n in acc_notes:
            note = Note(...)
            events.extend(note.to_events())  # → List[MusicalEvent]
        
        return events, timing

# 3. LegacyInferenceEngineStanley（遗留引擎）
class LegacyInferenceEngineStanley:
    def generate_accompaniment(
        self,
        melody_notes: list[dict],  # ← 接收dict (Duration-Note)
        ...
    ) -> tuple[list[dict], ...]:
        # 实际的模型推理逻辑
        # 1. dict → piano roll tensor
        x_mel_raw = self._notes_to_rolls(...)
        # 2. tensor → 预处理
        x_mel, x_acc = self.model.preprocess(...)
        # 3. 模型推理
        output_tensors = self.model.global_sampling(...)
        # 4. 张量 → dict
        notes_output = self._tensors_to_notes(output_tensors, ...)
        
        return notes_output, ...
```

---

## 8. 设计模式识别

### MVC式分层

```
表现层 (Presentation)
    ↓ depends on
应用层 (Application)  - RealTimeMusicService
    ↓ depends on
领域层 (Domain)       - InferenceEngine Protocol
    ↓ implements & delegated
基础设施层 (Infrastructure)
    ├─ StanleyInferenceEngine    (适配器 - Adapter模式)
    │   └─ wraps
    └─ LegacyInferenceEngineStanley  (遗留实现)
```

### 适配器模式（Adapter Pattern）

**目的**: 使不兼容的接口能够协作

```
┌─────────────────────────────────────┐
│ 客户端 (RealTimeMusicService)      │
│ 期望: InferenceEngine interface     │
└─────────────────┬───────────────────┘
                  │
                  └────────────────────┐
                                       │ calls
                                       ↓
                          ┌────────────────────────────┐
                          │ StanleyInferenceEngine     │
                          │ (Adapter)                  │
                          │                            │
                          │ - 实现InferenceEngine     │
                          │ - 转换数据格式             │
                          │ - 调用遗留引擎             │
                          └────────────┬───────────────┘
                                       │ wraps
                                       ↓
                          ┌────────────────────────────┐
                          │ LegacyInferenceEngineStanley
                          │ (Incompatible Service)    │
                          │                            │
                          │ - dict[dict] API          │
                          │ - duration-note格式      │
                          │ - 复杂的内部状态           │
                          └────────────────────────────┘

问题被解决:
✅ 客户端使用统一的InferenceEngine接口
✅ 遗留引擎保持不变
✅ 格式转换由适配器处理
```

---

## 9. 为什么这样设计？

### 背景

项目的推理引擎可能最初是为了某个特定的AI/ML研究项目而开发的（命名为"Stanley"）。当StreamMUSE项目需要整合这个推理引擎时，面临以下问题：

1. **遗留代码的特性**
   - ❌ 使用duration-note格式（不是event stream）
   - ❌ 没有实现StreamMUSE的`InferenceEngine`接口
   - ❌ 包含复杂的张量操作和模型加载逻辑
   - ❌ 保留了大量研究代码

2. **架构目标**
   - ✅ StreamMUSE使用event stream格式
   - ✅ 需要遵循Clean Architecture分层原则
   - ✅ 应用层应该抽象化基础设施细节
   - ✅ 应该支持多种推理引擎的切换

### 解决方案：适配器模式

```
不改动遗留代码 +  新增适配器层  =  逻辑清晰、架构优雅
(保险)             (灵活)          (可维护)
```

**优点**:
- ✅ 保持遗留代码不动（降低风险）
- ✅ StanleyInferenceEngine可以单独测试、修改
- ✅ 其他推理引擎可以很容易地添加（只需新的Adapter）
- ✅ 应用层代码对推理引擎的具体实现一无所知

**缺点**:
- ❌ 多了一层抽象（性能开销微小）
- ❌ 格式转换有额外的计算（events↔notes转换）
- ❌ 代码有一定的冗余

---

## 10. 实际使用示例

### 生产环境（自动创建遗留引擎）

```python
# CLI 中的使用
config = StanleyInferenceConfig(
    checkpoint_path="/path/to/model.ckpt",
    model_size="0.12B",
    generation_length_frames=20,
)

# 自动创建遗留引擎
stanley_engine = StanleyInferenceEngine(config=config)

# 传递给应用服务
service = RealTimeMusicService(
    inference_engine=stanley_engine,
    ...
)
```

### 测试环境（注入mock）

```python
# 单元测试
class MockLegacyEngine:
    generation_length_frames = 20
    
    def generate_accompaniment(self, ...):
        return (mock_notes, 0.1, 0.2, 0.3, 0.4)
    
    def clear_history(self): pass
    def set_injection_offset(self, offset): pass

# 创建适配器，注入mock
mock_engine = MockLegacyEngine()
stanley_engine = StanleyInferenceEngine(
    config=config,
    legacy_engine=mock_engine  # ← 注入
)

# 现在可以测试StanleyInferenceEngine而不需要加载真实模型
```

---

## 总结表

| 维度 | LegacyInferenceEngineStanley | StanleyInferenceEngine |
|------|--------------------------------|------------------------|
| **职责** | 执行模型推理 | 适配和转换 |
| **输入格式** | dict (duration-note) | MusicalEvent (event stream) |
| **输出格式** | dict (duration-note) | MusicalEvent (event stream) |
| **实现接口** | _LegacyStanleyLike Protocol | InferenceEngine Protocol |
| **依赖关系** | 依赖 PyTorch + RoFormer | 依赖 LegacyInferenceEngineStanley |
| **历史管理** | 直接维护melody/acc_history | 代理调用遗留引擎 |
| **可测试性** | 难（需要真实模型和checkpoint） | 易（可注入mock） |
| **可替换性** | 低（深度集成） | 高（标准接口） |
| **代码行数** | 225行（复杂） | 140行（简洁） |

---

## 关键代码片段

### StanleyInferenceEngine 中的适配器模式

```python
def __init__(self, *, config: StanleyInferenceConfig, legacy_engine: _LegacyStanleyLike | None = None) -> None:
    """构造器注入模式 - 允许测试时注入mock"""
    if legacy_engine is None:
        # 生产环境：自动创建真实的遗留引擎
        self._legacy = LegacyInferenceEngineStanley(...)
    else:
        # 测试环境：使用注入的遗留引擎（可能是mock）
        self._legacy = legacy_engine

def generate_accompaniment(self, melody_events, ...) -> tuple[List[MusicalEvent], TimingInfo]:
    """适配过程：三步转换"""
    # 步骤1: MusicalEvent → Note
    melody_notes: list[Note] = events_to_notes(melody_events, ...)
    
    # 步骤2: Note → dict
    melody_dicts = [{...} for n in melody_notes]
    
    # 步骤3: 委托给遗留引擎
    acc_notes, preprocess_start, inf_start, inf_end, post_start = self._legacy.generate_accompaniment(
        melody_dicts, ...
    )
    
    # 步骤4: dict → Note → MusicalEvent
    events = []
    for n in acc_notes:
        note = Note(...)
        events.extend(note.to_events())  # ← 展开为NOTE_ON/OFF事件
    
    # 步骤5: 返回event stream格式
    return events, timing
```

---

**结语**: 这是一个很好的适配器模式教案例。通过这个设计，StreamMUSE成功地：
1. 整合了现有的推理引擎
2. 保持了架构的清晰性
3. 没有改动遗留代码
4. 允许未来轻松替换或添加新的推理引擎
