# Injection Mode Plan (Revised)

## 需求概述

实现音乐注入（Music Injection）功能：
1. CLI 支持 `--injection-file` 和 `--injection-length` 参数
2. Client 解析 MIDI，提取前 N ticks 作为 prompt
3. 调用已有的 `inject_history()` 注入到 Server
4. MIDI file input 从注入点之后开始播放
5. 明确 tick 坐标系：使用**绝对 tick**（方案 A）

---

## 关键设计决策

### Tick 坐标系（方案 A - 绝对坐标）

| 组件 | Tick 定义 | 示例（注入 16 ticks） |
|------|----------|---------------------|
| 注入的 events | 绝对 tick（从歌曲开头）| 0, 4, 8, 12... |
| Client 播放 | 绝对 tick（从歌曲开头）| 从 tick=16 开始 |
| `generation_start_tick` | 绝对 tick | 16, 20, 24... |
| Server 内部 | 直接使用绝对 tick | 无需 offset 转换 |

**优势**：
- Server 无需知道 injection offset，逻辑最简单
- Client 的 tick 和 Server 的 tick 完全一致
- 不依赖 `_injection_offset_ticks`（当前未使用）

---

## Feature 1: CLI 参数与配置

### 1.1 添加 CLI 参数

**文件**: `src/streammuse/presentation/cli/config_parser.py`

```python
parser.add_argument(
    "--injection-file",
    type=str,
    default=None,
    help="Path to melody MIDI file to inject as prompt",
)
parser.add_argument(
    "--injection-length",
    type=int,
    default=0,
    help="Number of ticks to inject (e.g., 16 for 4 beats)",
)
parser.add_argument(
    "--inject-acc-file",
    type=str,
    default=None,
    help="Optional: Path to accompaniment MIDI for injection (default: replace '/mel/' with '/acc/')",
)
```

### 1.2 添加 InputConfig 字段

**文件**: `src/streammuse/application/config/models.py`（注意：不是 `config.py`，是 `config/models.py`）

```python
@dataclass(frozen=True)
class InputConfig:
    type: Literal["midi_device", "keyboard", "midi_file", "list"]
    midi_device_name: Optional[str] = None
    midi_file_path: Optional[str] = None
    midi_file_delay_ticks: int = 0
    # NEW: Injection configuration
    injection_file: Optional[str] = None
    injection_length_ticks: int = 0
    injection_acc_file: Optional[str] = None
```

同步检查 `src/streammuse/application/config/__init__.py`，确认 `InputConfig` 从 `models.py` 正确导出。

### 1.3 修改 args_to_config() 填充配置

**文件**: `src/streammuse/presentation/cli/config_parser.py`

在 `args_to_config()` 函数中，构造 `InputConfig` 时添加：

```python
input_config = InputConfig(
    type=args.input_mode,
    midi_device_name=args.midi_device_name,
    midi_file_path=args.midi_file_path,
    midi_file_delay_ticks=int(args.midi_file_delay_ticks),
    # NEW: Injection configuration（用 getattr 兜底，保持测试兼容）
    injection_file=getattr(args, "injection_file", None),
    injection_length_ticks=int(getattr(args, "injection_length", 0) or 0),
    injection_acc_file=getattr(args, "inject_acc_file", None),
)
```

### 1.4 参数验证与 main() 调用时序

**文件**: `src/streammuse/presentation/cli/cli.py`

**重要**：injection 验证放在 `args_to_config()` 之后，从 `config.input` 读取（不直接用 `args`），避免 mock 测试中 MagicMock 属性意外为 truthy。`inference_engine` 必须在注入之前创建，`input_source` 在注入之后创建。

```python
def main() -> int:
    args = parse_args()
    config = args_to_config(args)

    # ... session_manager, output_sink 创建 ...

    # 创建 inference_engine（注入依赖它，必须在注入前）
    inference_engine = InferenceEngineFactory.create(config)

    # 执行注入（用 config.input 读取，不用 args）
    if config.input.injection_file:
        if config.input.type != "midi_file":
            print("Error: --injection-file is only supported with --input-mode midi_file")
            return 1
        if config.input.injection_length_ticks <= 0:
            print("Error: --injection-length must be positive")
            return 1
        if not os.path.exists(config.input.injection_file):
            print(f"Error: Injection file not found: {config.input.injection_file}")
            return 1
        injection_length = _perform_injection(inference_engine, config)
        if injection_length == 0:
            return 1

    # 创建 input_source（factory 会读 injection_length_ticks 设置 start_tick，必须在注入后）
    input_source = InputSourceFactory.create(config)

    # ... 后续 service 启动逻辑 ...
```

---

## Feature 2: MIDI 解析与注入

### 2.1 复用现有 _midi_to_notes

**不新建 midi_parser.py**，直接复用 `MidiFileInput._midi_to_notes()`：

```python
# In cli.py _perform_injection()
from streammuse.infrastructure.input.midi_file import MidiFileInput

notes, _resolution, _max_tick = MidiFileInput._midi_to_notes(
    midi_path=config.input.injection_file,
    beat_div=config.tempo.ticks_per_beat,
    min_pitch=0,
    max_pitch=127,
    program=None,
    max_tick=config.input.injection_length_ticks,  # 只读到 injection length
)
```

### 2.2 Notes 转 MusicalEvent

```python
from streammuse.domain.musical import EventType, MusicalEvent, Note

def _notes_to_musical_events(notes: List[Dict[str, int]], velocity: int = 80) -> List[MusicalEvent]:
    """Convert duration notes to event stream via domain Note.to_events()."""
    events: List[MusicalEvent] = []
    for raw in notes:
        note = Note(
            pitch=int(raw["pitch"]),
            tick=int(raw["tick"]),
            duration=max(1, int(raw.get("duration", 1))),
            velocity=int(velocity),
        )
        events.extend(note.to_events())

    events.sort(key=lambda e: (e.tick, 0 if e.event_type == EventType.NOTE_OFF else 1))
    return events
```

### 2.3 执行注入

```python
def _perform_injection(
    inference_engine: InferenceEngine,
    config: ApplicationConfig,
) -> int:
    """Inject music to server. Returns injection_length_ticks on success, 0 on failure."""
    
    injection_file = config.input.injection_file
    injection_length = config.input.injection_length_ticks
    
    # Derive acc file path
    acc_file = config.input.injection_acc_file
    if acc_file is None:
        # Convention: replace /mel/ with /acc/ in path
        acc_file = injection_file.replace("/mel/", "/acc/")
        if acc_file == injection_file:
            print(f"Warning: Cannot derive acc path from {injection_file}")
            acc_file = None
    
    print(f"Injecting from: {injection_file} (first {injection_length} ticks)")
    
    try:
        # Parse melody
        mel_notes, _, _ = MidiFileInput._midi_to_notes(
            injection_file,
            beat_div=config.tempo.ticks_per_beat,
            min_pitch=0,
            max_pitch=127,
            program=None,
            max_tick=injection_length,
        )
        mel_events = _notes_to_musical_events(mel_notes)
        
        # Parse accompaniment if available
        acc_events = []
        if acc_file and os.path.exists(acc_file):
            acc_notes, _, _ = MidiFileInput._midi_to_notes(
                acc_file,
                beat_div=config.tempo.ticks_per_beat,
                min_pitch=0,
                max_pitch=127,
                program=None,
                max_tick=injection_length,
            )
            acc_events = _notes_to_musical_events(acc_notes)
            print(f"  Loaded accompaniment: {len(acc_notes)} notes")
        
        # Clear server history and inject
        inference_engine.clear_history()
        inference_engine.inject_history(
            melody_events=mel_events,
            accompaniment_events=acc_events,
            injection_length_ticks=injection_length,
        )
        
        print(f"✓ Injected: {len(mel_notes)} melody notes, {len(acc_events)//2} acc notes")
        return injection_length
        
    except Exception as e:
        print(f"✗ Injection failed: {e}")
        return 0
```

---

## Feature 3: MIDI File Input 的 Offset 处理

### 3.1 修改 MidiFileInputConfig

**文件**: `src/streammuse/infrastructure/input/midi_file.py`

```python
@dataclass(frozen=True)
class MidiFileInputConfig:
    bpm: float
    ticks_per_beat: int
    delay_ticks: int = 0
    min_pitch: int = 0
    max_pitch: int = 127
    program: Optional[int] = None
    max_tick: Optional[int] = None
    # NEW: Skip events before this tick (injection offset)
    start_tick: int = 0  # Events before this tick are skipped
```

### 3.2 修改 read_events() 过滤逻辑

```python
def read_events(self) -> Iterator[MusicalEvent]:
    seconds_per_tick = self._config.seconds_per_tick()
    
    # Parse MIDI
    notes, _resolution, _max_tick = self._midi_to_notes(
        self._path,
        beat_div=self._config.ticks_per_beat,
        min_pitch=self._config.min_pitch,
        max_pitch=self._config.max_pitch,
        program=self._config.program,
        max_tick=self._config.max_tick,
    )
    
    # Filter: skip notes before start_tick (injection offset)
    # BUT: Keep absolute tick values for correct timing
    start_offset = self._config.start_tick
    effective_notes = [n for n in notes if n["tick"] >= start_offset]
    
    # Build schedule with ABSOLUTE ticks (do NOT subtract start_offset)
    # This ensures tick=16 note plays at absolute tick=16
    schedule: Dict[int, List[MusicalEvent]] = {}
    delay_offset = int(self._config.delay_ticks)
    
    for n in effective_notes:
        # Use absolute tick directly
        onset = int(n["tick"]) + delay_offset  # e.g., 16 + 0 = 16
        offset = onset + int(n["duration"])    # e.g., 16 + 4 = 20
        
        schedule.setdefault(onset, []).append(
            MusicalEvent(
                tick=0,  # Application layer assigns actual tick from timing
                pitch=int(n["pitch"]),
                event_type=EventType.NOTE_ON,
                velocity=self._velocity_default,
            )
        )
        schedule.setdefault(offset, []).append(
            MusicalEvent(
                tick=0,
                pitch=int(n["pitch"]),
                event_type=EventType.NOTE_OFF,
                velocity=0,
            )
        )
    
    # ... rest of timing logic unchanged ...
```

### 3.3 Factory 传递配置

**文件**: `src/streammuse/application/factories/input_factory.py`

```python
if cfg.type == "midi_file":
    midi_config = MidiFileInputConfig(
        bpm=float(tempo.bpm),
        ticks_per_beat=int(tempo.ticks_per_beat),
        delay_ticks=int(cfg.midi_file_delay_ticks),
        # NEW: Pass injection offset as start_tick
        start_tick=cfg.injection_length_ticks if cfg.injection_file else 0,
    )
    
    return MidiFileInput(
        midi_file_path=cfg.midi_file_path,
        config=midi_config,
    )
```

---

## Feature 4: 关键约束与注意事项（已存在但需验证）

### 4.1 适用范围：仅支持 midi_file 输入模式

绝对 tick 方案能在 `midi_file` 输入下正确闭环，是因为 `MidiFileInput.read_events()` 按绝对时间等待每个 note（`target_time = start + t * seconds_per_tick`），note at tick=16 会在正好 16 ticks 之后播放，与 service 的 `current_tick` 自然对齐。

`keyboard` 和 `midi_device` 输入没有这种时间对齐，service 的 tick 从 0 开始，`generation_start_tick` 在注入后第一轮仍是 0 或 2，会触发 zero-prompt fallback，无法利用注入上下文。

**第一版范围限制**：injection 功能仅与 `--input-mode midi_file` 组合使用。若用户指定了 `--injection-file` 但 `--input-mode` 不是 `midi_file`，CLI 应打印 error 并退出。

### 4.2 HTTP Client 与 Backend 已有能力，无需修改

`_injection_offset_ticks` 在 `http_client.py` 里只是状态字段，方案 A 不需要它。不修改。

`_trim_histories()` 的窗口下限是 `max(512, generation_length_frames * 16)`，默认配置下最小 512 ticks。injection 通常 ≤32 ticks，不会被裁掉。但如果用户将 `LEKAI_HISTORY_MAX_TICKS` 设置为小值，注入上下文可能丢失。正常使用下无问题，不修改 backend，但在文档中说明这一约束。

**验证**：当 client 发送 `generation_start_tick=16`，server 的 `prompt_start = max(0, 16 - 8) = 8`，正确包含注入的 0-15 ticks 作为上下文。

---

## 涉及的文件总结

| 文件 | 修改内容 |
|------|---------|
| `presentation/cli/config_parser.py` | 添加 `--injection-file`, `--injection-length`, `--inject-acc-file` |
| `application/config/models.py` | `InputConfig` 添加 `injection_file`, `injection_length_ticks`, `injection_acc_file` |
| `presentation/cli/cli.py` | `_perform_injection()`, `_notes_to_musical_events()`, 调用逻辑 |
| `infrastructure/input/midi_file.py` | `MidiFileInputConfig` 添加 `start_tick`, `read_events()` 过滤逻辑 |
| `application/factories/input_factory.py` | 根据 `injection_file` 设置 `start_tick` |

**注意**：不修改 `http_client.py` 和 `lekai_http_backend.py`，因为相关功能已实现。

---

## 使用示例

```bash
# 注入前 4 beats，从第 5 beat 开始实时演奏
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --injection-file prompts/inputs_lekai/mel/1.mid \
  --injection-length 16 \
  --server-url http://localhost:8000/generate_accompaniment \
  --output-type session
# 注意：--midi-file-path（不是 --midi-file-input）
# acc 文件自动推导为 prompts/inputs_lekai/acc/1.mid（replace /mel/ -> /acc/）
```

执行流程：
1. CLI 解析 `--injection-file`，提取前 16 ticks 的 notes
2. 调用 `inject_history()` 注入到 server（events 的 tick 为 0-15）
3. 创建 `MidiFileInput`，`start_tick=16`，跳过前 16 ticks
4. MIDI 播放从 tick=16 开始（绝对 tick）
5. `generation_start_tick` 从 16 开始发送（绝对 tick）
6. Server 使用 ticks 8-15 作为 prompt，生成 tick 16+ 的伴奏

---

## 测试计划

### 新增单元测试

**1. 参数解析测试** — `tests/unit/presentation/test_config_parser.py`
- `--injection-file` / `--injection-length` / `--inject-acc-file` 正确映射到 `InputConfig` 字段
- `injection_length_ticks=0` 时不赋值 `injection_file`

**2. MidiFileInput start_tick 跳过行为** — `tests/unit/infrastructure/input/test_midi_file_input.py`
- `start_tick=16`：tick < 16 的 note 不出现在 schedule 中
- tick=16 的 note 在绝对时间 `16 * seconds_per_tick` 播放（不是 `0 * seconds_per_tick`）
- `start_tick=0`（默认）：行为与原来完全一致

**3. generation_start_tick 对齐测试** — `tests/integration/`
- 注入 16 ticks 后，service 第一个 inference 请求的 `generation_start_tick >= 16`
- 使用 `ListInput` 或 mock MidiFileInput 构造可控时序

**4. clear_history 落盘测试** — 已有 debug1-plan 中规划，确认 `melody_history.json` 内容包含注入的 events

### 回归测试

- `test_cli_entry_point.py`：mock 中 `mock_args` 不含 `injection_file` 属性时，`config.input.injection_file` 为 `None`，`main()` 正常退出不报错

---

## 验收标准

1. ✅ CLI 支持 `--injection-file`, `--injection-length`, `--inject-acc-file`
2. ✅ Client 使用 `MidiFileInput._midi_to_notes()` 解析 MIDI（不引入 pretty_midi）
3. ✅ 调用 `inference_engine.inject_history()`（方法名正确，类型为 `List[MusicalEvent]`）
4. ✅ `MidiFileInput` 通过 `start_tick` 配置跳过注入部分（保持绝对 tick，不减 offset）
5. ✅ 使用绝对 tick 坐标系，不依赖 `_injection_offset_ticks`
6. ✅ 仅支持 `--input-mode midi_file`，其他模式显式报错
7. ✅ `InputConfig` 修改在 `config/models.py`，`args_to_config()` 用 `getattr` 兜底不破坏现有测试
8. ✅ 生成伴奏与注入音乐连贯

---

## Detailed Todo List（Phases + Tasks）

### Phase 0 — 环境确认与改动边界冻结

- [ ] P0-1 确认 `config/models.py::InputConfig` 当前字段列表，确认不存在 `injection_file` 字段（无重复添加风险）。
- [ ] P0-2 确认 `InputSourceFactory.create()` 的 `midi_file` 分支构造方式（`midi_file_path` 参数名、顶部 import 已有 `MidiFileInput`/`MidiFileInputConfig`）。
- [ ] P0-3 运行 `uv run pytest tests/ -q --tb=no` 确认 baseline 全绿，保留输出作为回归基线。
- [ ] P0-4 确认 `tests/unit/infrastructure/input/test_midi_file_input.py` 存在并覆盖了 `read_events()` 基本行为（后续回归依赖）。

**Exit Criteria**
- [ ] P0-E1 baseline 测试结果已记录，无预存失败。
- [ ] P0-E2 所有待修改文件路径已对照实际文件确认正确。

---

### Phase 1 — Feature 1：CLI 参数与配置层

- [ ] P1-1 在 `config_parser.py::parse_args()` 添加三个参数：`--injection-file`、`--injection-length`、`--inject-acc-file`。
- [ ] P1-2 在 `config/models.py::InputConfig` 添加三个字段：`injection_file: Optional[str] = None`、`injection_length_ticks: int = 0`、`injection_acc_file: Optional[str] = None`。
- [ ] P1-3 在 `config_parser.py::args_to_config()` 的 `InputConfig` 构造里填充三个新字段（用 `getattr` 兜底）。
- [ ] P1-4 确认 `config/__init__.py` 的 `__all__` 无需修改（`InputConfig` 已导出）。

**Exit Criteria**
- [ ] P1-E1 `uv run streammuse-cli --help` 显示三个新参数。
- [ ] P1-E2 现有集成测试（`test_cli_entry_point.py`）仍通过（`getattr` 兜底不破坏 mock）。

---

### Phase 2 — Feature 2：MIDI 解析与注入逻辑（cli.py）

- [ ] P2-1 在 `cli.py` 添加辅助函数 `_notes_to_musical_events(notes, velocity=80) -> List[MusicalEvent]`，内部使用 `Note.to_events()` 复用 domain 对象（不手写 MusicalEvent 构造）。
- [ ] P2-2 在 `cli.py` 添加 `_perform_injection(inference_engine, config) -> int`，实现：
  - 推导 acc 路径（replace `/mel/` → `/acc/`；若路径未变则跳过 acc，打印 warning）
  - 调用 `MidiFileInput._midi_to_notes()` 解析 melody（`max_tick=injection_length_ticks`）
  - 若 acc 文件存在，调用 `MidiFileInput._midi_to_notes()` 解析 acc
  - 调用 `inference_engine.clear_history()` 清空 server 状态
  - 调用 `inference_engine.inject_history(mel_events, acc_events, injection_length_ticks)`
  - 返回 `injection_length_ticks` 成功，`0` 失败（异常捕获后打印并返回 0）
- [ ] P2-3 在 `cli.py::main()` 中按正确顺序插入注入逻辑：
  - `inference_engine = InferenceEngineFactory.create(config)` 先创建
  - 验证 `injection_file` 非空时 `input_mode == "midi_file"`，否则打印错误 `return 1`
  - 验证 `injection_length_ticks > 0` 且 `os.path.exists(injection_file)`
  - 调用 `_perform_injection()`，返回 0 时 `return 1`
  - `input_source = InputSourceFactory.create(config)` 最后创建

**Exit Criteria**
- [ ] P2-E1 手动运行，console 打印 `✓ Injected: N melody notes, M acc notes`。
- [ ] P2-E2 指定不存在的 `--injection-file` 时，CLI 报错退出，不崩溃。
- [ ] P2-E3 指定 `--input-mode keyboard` + `--injection-file` 时，CLI 打印错误并退出。

---

### Phase 3 — Feature 3：MidiFileInput 的 start_tick 支持

- [ ] P3-1 在 `infrastructure/input/midi_file.py::MidiFileInputConfig` 添加 `start_tick: int = 0` 字段。
- [ ] P3-2 修改 `MidiFileInput.read_events()`：
  - 解析完 `notes` 后过滤掉 `note["tick"] < self._config.start_tick` 的 note
  - schedule key 使用绝对 tick（`int(n["tick"]) + delay_offset`），**不减 start_tick**
  - `ticks = sorted(schedule.keys())` 及后续 timing 循环逻辑不变
- [ ] P3-3 在 `application/factories/input_factory.py` 的 `midi_file` 分支里传入 `start_tick=cfg.injection_length_ticks if cfg.injection_file else 0`（去掉 inline import，顶部已有导入）。

**Exit Criteria**
- [ ] P3-E1 单测：`start_tick=16` 时 tick < 16 的 note 不出现在输出中。
- [ ] P3-E2 单测：tick=16 的 note 在绝对时间 `16 * seconds_per_tick` 播放（不是 0 时刻）。
- [ ] P3-E3 单测：`start_tick=0`（默认）时行为与原来完全一致（现有测试通过）。

---

### Phase 4 — 联调：绝对 tick 端到端验证

- [ ] P4-1 启动 fake server，运行 `--injection-file 1.mid --injection-length 16 --input-mode midi_file --midi-file-path 1.mid`。
- [ ] P4-2 检查 server 日志：`inject_history()` 收到的 events tick 范围为 0-15；第一次 `generate()` 的 `generation_start_tick >= 16`。
- [ ] P4-3 检查生成输出：前 16 ticks 无 model events，tick=16 起开始有伴奏生成。
- [ ] P4-4 验证历史窗口：`generation_start_tick=16`、`generation_length_frames=8` 时 `prompt_start=8`，注入的 8-15 ticks 在 prompt 内。

**Exit Criteria**
- [ ] P4-E1 联调 session 正常结束，无异常。
- [ ] P4-E2 伴奏从 tick=16 开始，不在注入段内生成。

---

### Phase 5 — 测试补齐与回归

- [ ] P5-1 **参数解析测试**（`tests/unit/presentation/test_config_parser.py` 或已有同类文件）：
  - `--injection-file foo.mid --injection-length 16` → `config.input.injection_file == "foo.mid"`, `injection_length_ticks == 16`
  - 不传 `--injection-file` → `config.input.injection_file is None`, `injection_length_ticks == 0`
- [ ] P5-2 **MidiFileInput start_tick 测试**（`tests/unit/infrastructure/input/test_midi_file_input.py`）：
  - `start_tick=16` 过滤行为（3 个子用例见 P3-E1/E2/E3）
- [ ] P5-3 **injection 流程 mock 测试**（`tests/integration/test_cli_entry_point.py` 或新增文件）：
  - mock `InferenceEngine.inject_history()` 被调用，入参类型为 `List[MusicalEvent]`
  - mock `InferenceEngine.clear_history()` 在 `inject_history()` 之前被调用
  - `--input-mode keyboard` + `--injection-file` 组合触发错误退出
- [ ] P5-4 运行全量测试 `uv run pytest tests/ -q --tb=short`，无新增失败。

**Exit Criteria**
- [ ] P5-E1 P5-1、P5-2、P5-3 三类测试全部通过。
- [ ] P5-E2 全量测试结果与 Phase 0 baseline 对比无回归。
