# StreamMUSE Formal Documentation Plan

> **目标**：在 `docs/` 目录下建立一套 formal documentation，结构参照成熟开源项目（如 Pydantic、FastAPI、Typer）的标准，内容完全对齐当前代码库实现。

---

## 文档结构总览

```
docs/
├── index.md
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── configuration.md
├── architecture/
│   ├── overview.md                        # 四层总览 + 数据流图
│   ├── domain/
│   │   ├── overview.md
│   │   ├── musical.md                     # MusicalEvent, EventType, converters, MusicalSequence
│   │   ├── timing.md                      # Tempo, PlaybackScheduler
│   │   ├── logging.md                     # SessionManager, MetricsCalculator, LogEvent, InferenceEvent
│   │   └── interfaces.md                  # InputSource, OutputSink, InferenceEngine, TimingInfo
│   ├── application/
│   │   ├── overview.md
│   │   ├── config.md                      # TempoConfig, InputConfig, OutputConfig, InferenceConfig, ApplicationConfig
│   │   ├── factories.md                   # InputSourceFactory, OutputSinkFactory, InferenceEngineFactory
│   │   └── service.md                     # RealTimeServiceRuntime, RealTimeMusicService
│   ├── infrastructure/
│   │   ├── overview.md
│   │   ├── input/
│   │   │   ├── overview.md
│   │   │   ├── keyboard.md                # KeyboardInputConfig, KeyboardInput
│   │   │   ├── midi-device.md             # MidiDeviceInput
│   │   │   ├── midi-file.md               # MidiFileInputConfig, MidiFileInput
│   │   │   └── list-input.md              # ListInput
│   │   ├── output/
│   │   │   ├── overview.md
│   │   │   ├── console.md                 # ConsoleOutputConfig, ConsoleOutputSink
│   │   │   ├── audio.md                   # AudioOutputConfig, AudioOutputSink
│   │   │   ├── midi-file.md               # MidiFileOutputConfig, MidiFileOutputSink
│   │   │   ├── websocket.md               # WebSocketOutputConfig, WebSocketOutputSink
│   │   │   ├── json-logger.md             # JsonLoggerOutputSink
│   │   │   ├── session-logger.md          # SessionLoggerOutputSink
│   │   │   └── composite.md               # CompositeOutputSink
│   │   └── inference/
│   │       ├── overview.md
│   │       ├── http-client.md             # HttpInferenceClientConfig, HttpInferenceClient
│   │       ├── stanley-engine.md          # StanleyInferenceConfig, StanleyInferenceEngine
│   │       ├── stanley-legacy.md          # LegacyInferenceEngineStanley
│   │       └── serialization.md           # event_to_dict, event_from_dict, timing_info_from_dict
│   └── presentation/
│       ├── overview.md
│       ├── cli.md                         # main(), cleanup(), signal_handler()
│       └── config-parser.md               # parse_args(), args_to_config(), env_to_config()
├── user-guide/
│   ├── input-modes.md
│   ├── output-types.md
│   ├── session-logging.md
│   └── music-injection.md
├── developer-guide/
│   ├── extending-inputs.md
│   ├── extending-outputs.md
│   ├── extending-engines.md
│   ├── inference-server.md
│   └── testing.md
└── reference/
    ├── cli.md
    └── protocols.md
```

---

## 各文件内容规划

### `docs/index.md` — 项目首页

**参照**：FastAPI 首页、Pydantic 首页  
**内容**：
- 一句话介绍（StreamMUSE = 实时 AI 伴奏生成）
- 核心特性列表（实时、Clean Architecture、多输入/输出、Session logging）
- 系统示意图（ASCII 或 Mermaid：User → CLI → Service → Inference Server → Output）
- 快速导航（Getting Started / Architecture / API Reference）
- 系统要求（Python ≥ 3.10、CUDA 可选、支持 macOS / Linux）

---

### `docs/getting-started/installation.md` — 安装

**内容**：
1. 克隆仓库
2. `uv sync`（安装主包依赖）
3. `cd transformers && pip install -e . && cd ..`（安装改版 transformers）
4. 验证安装：`uv run streammuse-cli --help`
5. 可选依赖说明（audio 输出需要 rtmidi / FluidSynth；GPU 推理需要 CUDA）

---

### `docs/getting-started/quickstart.md` — 5 分钟上手

**内容**：
1. 启动 fake 推理服务器（无需真实模型）：
   ```bash
   uv run python scripts/fake_inference_server.py
   ```
2. 启动 CLI（另一个终端），键盘模式 + 控制台输出：
   ```bash
   uv run streammuse-cli --input-mode keyboard
   ```
3. 键位说明（`a/s/d/f/g/h/j` = 音符，`space` = 持续音，`q` = 退出）
4. 观察终端输出，解释各字段含义
5. 下一步链接（→ Configuration, → Output Types）

---

### `docs/getting-started/configuration.md` — 配置项

**内容**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--input-mode` | enum | `keyboard` | 输入源类型 |
| `--output-type` | enum | `console` | 输出类型 |
| `--server-url` | str | `http://localhost:8000` | 推理服务器地址 |
| `--bpm` | int | `120` | 速度（每分钟节拍数）|
| `--injection-file` | path | None | 预热注入 MIDI 文件 |
| `--injection-length` | int | None | 注入帧数 |
| `--log-dir` | path | None | Session 日志目录（composite 模式用）|
| `--midi-file` | path | None | midi_file 输入模式的文件路径 |

所有配置最终汇聚为 `ApplicationConfig`（`application/config/models.py`）。

---

### `docs/architecture/overview.md` — 架构总览

**参照**：Django 架构页、FastAPI 内部设计文档  
**内容**：
1. **Clean Architecture 四层图**（Mermaid flowchart）：
   ```
   Presentation (CLI)
       ↓ creates ApplicationConfig
   Application (RealTimeMusicService + Factories)
       ↓ depends on interfaces only
   Domain (Protocols, MusicalEvent, Timing, Logging)
       ↑ implemented by
   Infrastructure (Inputs, Outputs, Inference engines)
   ```
2. **依赖规则**：内层不依赖外层，所有跨层通信通过 Protocol（duck typing）
3. **数据流**：从 MIDI/键盘输入 → MusicalEvent → 推理请求 → 伴奏输出，配 Mermaid sequence diagram
4. **三线程模型**（`_input_worker` / `_tick_loop` / `_inference_worker`）简要说明
5. 关键设计决策：为何用 Protocol 而非 ABC；为何不用 asyncio
6. 各子目录快速导航链接

---

### `docs/architecture/domain/` — Domain 层（子目录）

```
docs/architecture/domain/
├── overview.md
├── musical.md
├── timing.md
├── logging.md
└── interfaces.md
```

#### `domain/overview.md`
- Domain 层职责：定义核心业务对象与协议，不依赖任何外部框架
- 子模块一览（musical / timing / logging / interfaces）及其依赖关系图
- 不变性约定：`MusicalEvent` 是 frozen dataclass，tick 为全局单调递增

#### `domain/musical.md` — 源文件：`musical/events.py`, `converters.py`, `sequence.py`

**`EventType`（enum）**
- `NOTE_ON`：音符按下
- `NOTE_OFF`：音符抬起

**`MusicalEvent`（frozen dataclass）**
- 字段说明表：`tick` (int, 绝对音乐时间), `pitch` (int, MIDI 0–127，-1=非音符事件), `event_type` (EventType), `velocity` (int, 0–127), `channel` (int), `program` (int, MIDI 乐器编号), `source` (str, "user"/"model"), `is_placeholder` (bool)
- 特殊值约定（`pitch=-1` 用于节拍/状态事件）
- 不变性：`frozen=True`，创建后不可修改

**`converters.py`（模块级函数）**
- `midi_message_to_event(msg, tick, source, program) -> MusicalEvent`：将 mido `Message` 对象转换为 `MusicalEvent`，参数说明，返回值，不支持的消息类型处理
- `event_to_midi_message(event) -> mido.Message`：反向转换，pitch=-1 时的行为

**`MusicalSequence`**
- `__init__(events: List[MusicalEvent])`：构造，events 按 tick 排序
- `get_events_in_range(start_tick, end_tick) -> MusicalSequence`：返回 tick 在区间内的子序列（左闭右开）
- `quantize(quantization_ticks) -> MusicalSequence`：将所有事件量化到 `quantization_ticks` 的整数倍

#### `domain/timing.md` — 源文件：`timing/tempo.py`, `timing/scheduler.py`

**`Tempo`（dataclass）**
- `bpm` 字段
- `tick_duration_seconds(ticks_per_beat) -> float`：将单个 tick 转换为秒
- `ticks_to_seconds(ticks, ticks_per_beat) -> float`；`seconds_to_ticks(seconds, ticks_per_beat) -> int`

**`PlaybackScheduler`**
- `__init__(tempo, ticks_per_beat)`：初始状态
- `get_next_event(events) -> Optional[MusicalEvent]`：根据当前 tick 取出下一个应播放的事件
- `advance_tick() -> int`：递增 tick，返回新 tick 值
- `reset()`：重置调度器状态
- 调度语义：事件按 tick 顺序出队，与实时时钟对齐

#### `domain/logging.md` — 源文件：`logging/event_types.py`, `logging/session_manager.py`, `logging/metrics_calculator.py`

**`LogEvent`（dataclass）**
- 字段：`tick`, `timestamp`, `event_type`, `pitch`, `velocity`, `source`, `channel`
- 用途：记录单个音乐事件到日志

**`InferenceEvent`（dataclass）**
- 字段：`timestamp`, `request`, `response`, `latency_ms`, `server_process_ms`
- 用途：记录单次推理的完整 request/response 及延迟

**`SessionManager`**
- `__init__(base_log_dir: str = "logs")`：指定日志根目录
- `_generate_session_id() -> str`：生成 `YYYYMMDD-HHMMSS` 格式 ID（私有）
- `create_session_directory() -> Path`：在 `base_log_dir/session_<id>/` 创建目录，返回路径
- `save_config(config: Dict[str, Any]) -> None`：将配置写入 `session_config.json`
- `save_summary(summary: Dict[str, Any]) -> None`：将汇总数据写入 `statistics.csv`

**`MetricsCalculator`**
- `__init__()`：初始化内部累积器
- `add_event(event: LogEvent) -> None`：累积单个音乐事件
- `add_inference(inf: InferenceEvent) -> None`：累积单次推理记录
- `calculate_latency_stats() -> Dict[str, float]`：计算延迟 p50/p95/p99/mean/min/max
- `calculate_event_stats() -> Dict[str, Any]`：计算事件数量、user/model 比例等
- `calculate_music_stats() -> Dict[str, Any]`：计算音高分布、音域、音符密度等音乐统计
- `generate_performance_json(session_config) -> Dict[str, Any]`：聚合所有统计，生成完整 `performance.json` 结构
- `generate_statistics_csv() -> str`：生成 CSV 格式的 summary 字符串

#### `domain/interfaces.md` — 源文件：`interfaces/input.py`, `interfaces/output.py`, `interfaces/inference.py`, `interfaces/timing_info.py`

**`InputSource`（Protocol）**
- `read_events() -> Iterator[MusicalEvent]`：阻塞式事件流，直到 `close()` 调用或输入结束
- `close() -> None`：释放底层资源，使 `read_events()` 退出

**`OutputSink`（Protocol）**
- `output_event(event, source) -> None`：处理单个音乐事件（播放/录制/打印）
- `output_tick(tick, bar, beat) -> None`：每 tick 回调，用于显示时间进度
- `output_stats(round_trip_ms, server_process_ms, ...) -> None`：推理延迟统计回调
- `output_status(state, message="") -> None`：服务状态变更通知（如 "running"/"stopping"）
- `output_config(config: dict) -> None`：会话开始时传入配置摘要
- `close() -> None`：刷新缓冲区，关闭文件/端口

**`InferenceEngine`（Protocol）**
- `generate_accompaniment(melody_events, generation_start_tick, generation_length_frames) -> tuple[List[MusicalEvent], TimingInfo]`：核心推理接口；melody_events 为当前上下文，返回伴奏事件列表及时序信息
- `inject_history(melody_events, acc_events) -> None`：预填充模型历史（用于注入 MIDI prompts）
- `clear_history() -> None`：清空模型内部状态

**`TimingInfo`（dataclass/Protocol）**
- 字段说明：`server_process_ms`（服务端推理耗时）、`round_trip_ms`（完整往返时间）

---

### `docs/architecture/application/` — Application 层（子目录）

```
docs/architecture/application/
├── overview.md
├── config.md
├── factories.md
└── service.md
```

#### `application/overview.md`
- Application 层职责：编排 Domain 接口，不依赖具体 Infrastructure 实现
- 三组件关系：Config → Factories → Service
- 数据流：CLI args → `ApplicationConfig` → Factories 创建组件 → `RealTimeMusicService` 运行

#### `application/config.md` — 源文件：`config/models.py`

每个 dataclass 的字段表（名称、类型、默认值、描述）：

**`TempoConfig`**：`bpm` (float, 120.0), `ticks_per_beat` (int, 4)

**`InputConfig`**：`mode` (str, "keyboard"), `midi_file_path` (Optional[str]), `midi_device_name` (Optional[str])

**`OutputConfig`**：`output_type` (str, "console"), `midi_output_path` (Optional[str]), `websocket_host/port`, `log_dir` (Optional[str])

**`InferenceConfig`**：`server_url` (str), `generation_length_frames` (int, 20), `generation_interval_ticks` (int, 2), `injection_file` (Optional[str]), `injection_length` (Optional[int])

**`ApplicationConfig`**：组合上述四个 Config 的顶层配置对象

#### `application/factories.md` — 源文件：`factories/input_factory.py`, `factories/output_factory.py`, `factories/inference_factory.py`

**`InputSourceFactory`**
- `create(app_config: ApplicationConfig, *, list_events=None) -> InputSource`（静态方法）
- 支持类型：`keyboard` → `KeyboardInput`；`midi` → `MidiDeviceInput`；`midi_file` → `MidiFileInput`；`list` → `ListInput`（测试用）
- 分支逻辑与异常（未知 mode 时 raise `ValueError`）

**`OutputSinkFactory`**
- `create(config: OutputConfig, session_manager: Optional[SessionManager] = None) -> OutputSink`（静态方法）
- 支持类型：`console`/`audio`/`midi_file`/`websocket`/`json_log`/`session`/`composite`
- `composite` 模式：同时创建 `ConsoleOutputSink` + `SessionLoggerOutputSink`，包装为 `CompositeOutputSink`
- `session_manager` 参数：仅在 `json_log`/`session`/`composite` 时使用

**`InferenceEngineFactory`**
- `create(app_config: ApplicationConfig) -> InferenceEngine`（静态方法）
- 当前实现：始终返回 `HttpInferenceClient`（将来可扩展为 `StanleyInferenceEngine` 本地模式）

#### `application/service.md` — 源文件：`services/real_time_music_service.py`

**`RealTimeServiceRuntime`（dataclass）**
- 内部运行时状态容器：持有三个 `threading.Thread` 引用及停止标志

**`RealTimeMusicService`**
- `__init__(input_source, output_sink, inference_engine, config)`：依赖注入，接收 Domain 接口实例
- `running -> bool`（property）：服务是否运行中
- `_input_worker() -> None`（私有线程函数）：循环调用 `input_source.read_events()`，将事件放入 melody queue 和 playback queue
- `_tick_loop(*, max_ticks) -> None`（私有线程函数）：按 `Tempo` 精确睡眠，每 tick 从 playback queue 取事件调用 `output_sink.output_event()`；每 `generation_interval_ticks` ticks 向 inference queue 投入推理请求
- `_inference_worker() -> None`（私有线程函数）：从 inference queue 取推理请求，调用 `inference_engine.generate_accompaniment()`，将结果放入 playback queue；调用 `output_sink.output_stats()` 及 `output_sink.log_inference()`（hasattr 保护）
- `start(*, max_ticks=None) -> None`：启动三个线程，注入 MIDI history（若配置），阻塞直到 `stop()` 或 max_ticks 到达
- `stop() -> None`：设置停止标志，join 三个线程，调用 `output_sink.close()`
- 线程交互图（Mermaid sequence diagram）：展示三线程与三个队列之间的消息流

---

### `docs/architecture/infrastructure/` — Infrastructure 层（子目录）

```
docs/architecture/infrastructure/
├── overview.md
├── input/
│   ├── overview.md
│   ├── keyboard.md
│   ├── midi-device.md
│   ├── midi-file.md
│   └── list-input.md
├── output/
│   ├── overview.md
│   ├── console.md
│   ├── audio.md
│   ├── midi-file.md
│   ├── websocket.md
│   ├── json-logger.md
│   ├── session-logger.md
│   └── composite.md
└── inference/
    ├── overview.md
    ├── http-client.md
    ├── stanley-engine.md
    ├── stanley-legacy.md
    └── serialization.md
```

#### `infrastructure/overview.md`
- Infrastructure 层职责：实现 Domain 接口，对接外部 I/O（硬件、网络、文件系统）
- 三子包（input / output / inference）各自对应的 Protocol
- 选型说明：python-rtmidi / pynput / mido / requests / FluidSynth

#### `infrastructure/input/overview.md`
- 四个 `InputSource` 实现汇总表
- 共同契约：`read_events()` 是阻塞生成器；`close()` 后 `read_events()` 必须退出

#### `infrastructure/input/keyboard.md` — 源文件：`input/keyboard.py`

**`KeyboardInputConfig`（dataclass）**
- `key_note_map`：键盘按键 → MIDI pitch 映射表（默认：a=60, s=62, d=64, f=65, g=67, h=69, j=71）
- `velocity` (int, 80)

**`KeyboardInput`**
- `__init__(config, tempo_config, start_tick=0)`：启动 pynput listener 线程
- `_handle_key_down(char) -> None`：产生 NOTE_ON 事件，以 tick 为时间戳放入内部队列
- `_handle_key_up(char) -> None`：产生 NOTE_OFF 事件
- `_run_listener() -> None`：pynput 监听线程主函数，捕获 key press/release
- `read_events() -> Iterator[MusicalEvent]`：从内部队列取事件并 yield；遇到 `q` 键时 StopIteration
- `close() -> None`：停止 pynput listener，唤醒阻塞的 `read_events()`

#### `infrastructure/input/midi-device.md` — 源文件：`input/midi_device.py`

**`MidiDeviceInput`**
- `__init__(device_name=None)`：枚举 MIDI 端口；`device_name=None` 时自动选第一个可用端口
- `read_events() -> Iterator[MusicalEvent]`：轮询 rtmidi 消息，转换为 `MusicalEvent`（调用 `converters.midi_message_to_event()`）
- `close() -> None`：关闭 rtmidi 端口

#### `infrastructure/input/midi-file.md` — 源文件：`input/midi_file.py`

**`MidiFileInputConfig`（dataclass）**
- `file_path` (str)；`bpm` (float)；`ticks_per_beat` (int)
- `seconds_per_tick` (property)：根据 bpm 和 ticks_per_beat 计算

**`MidiFileInput`**
- `__init__(config)`：解析 MIDI 文件（mido），提取 track 0 的音符事件
- `_midi_to_notes(mid, ticks_per_beat) -> List[MusicalEvent]`：MIDI tick → 系统 tick 换算，生成有序 event 列表
- `read_events() -> Iterator[MusicalEvent]`：按 `seconds_per_tick` 精确睡眠，模拟实时演奏速度 yield 事件
- `close() -> None`：设置停止标志，中断 `read_events()` 的睡眠

#### `infrastructure/input/list-input.md` — 源文件：`input/list_input.py`

**`ListInput`**
- `__init__(events: List[MusicalEvent])`：持有预置事件列表，用于测试/回放
- `read_events() -> Iterator[MusicalEvent]`：顺序 yield 所有事件后结束（不阻塞）
- `close() -> None`：无操作

---

#### `infrastructure/output/overview.md`
- 七个 `OutputSink` 实现汇总表（类名、输出目标、是否有 `log_inference()`、配套 Config 类）
- `log_inference()` 设计说明：不在 `OutputSink` Protocol 中；`RealTimeMusicService` 通过 `hasattr` 调用；`CompositeOutputSink` fan-out 传播

#### `infrastructure/output/console.md` — 源文件：`output/console.py`

**`ConsoleOutputConfig`（dataclass）**：`show_events` (bool, True), `show_stats` (bool, True)

**`ConsoleOutputSink`**
- `__init__(config=None)`
- `output_event(event, source)`：格式化打印 tick/pitch/velocity/source
- `output_tick(tick, bar, beat)`：打印进度（可配置是否显示）
- `output_stats(round_trip_ms, server_process_ms, ...)`：打印延迟统计
- `output_status(state, message="")`：打印状态变更
- `output_config(config)`：打印初始配置摘要
- `close()`：无操作

#### `infrastructure/output/audio.md` — 源文件：`output/audio.py`

**`AudioOutputConfig`（dataclass）**：`soundfont_path` (Optional[str]), `port_name` (str, "StreamMUSE")

**`AudioOutputSink`**
- `__init__(config=None)`：延迟初始化 rtmidi 输出端口
- `_ensure_port() -> None`：懒加载 rtmidi 虚拟端口（首次 `output_event` 调用时）
- `output_event(event, source)`：将 `MusicalEvent` 转为 MIDI 消息，发送到 rtmidi 端口
- `output_tick / output_stats / output_status / output_config`：均为 no-op
- `close()`：关闭 rtmidi 端口

#### `infrastructure/output/midi-file.md` — 源文件：`output/midi_file.py`

**`MidiFileOutputConfig`（dataclass）**：`output_path` (str), `bpm` (float, 120.0), `ticks_per_beat` (int, 4)

**`MidiFileOutputSink`**
- `__init__(config)`：创建 mido `MidiFile`，初始化 track
- `output_event(event, source)`：将事件追加到 mido track（时间转换为 MIDI delta tick）
- `output_tick / output_stats / output_status / output_config`：均为 no-op
- `close()`：将 mido `MidiFile` 写入 `output_path`

#### `infrastructure/output/websocket.md` — 源文件：`output/websocket.py`

**`WebSocketOutputConfig`（dataclass）**：`host` (str), `port` (int, 8765)

**`WebSocketOutputSink`**
- `__init__(config=None)`：初始化 pending message 队列
- `_enqueue(message: Dict) -> None`：序列化为 JSON 字符串，加入队列
- `output_event(event, source)`：构造事件 JSON 并 enqueue
- `output_tick(tick, bar, beat)`：构造 tick JSON 并 enqueue
- `output_stats(...)`：构造 stats JSON 并 enqueue
- `output_status / output_config`：枚举并 enqueue
- `get_pending_messages() -> List[str]`：取出并清空待发消息（供外部 WebSocket server 调用）
- `close()`：清空队列

#### `infrastructure/output/json-logger.md` — 源文件：`output/json_logger.py`

**`JsonLoggerOutputSink`**
- `__init__(session_dir: Path)`：初始化 `MetricsCalculator`；创建/打开 `events.jsonl`
- `output_event(event, source)`：构造 `LogEvent`，追加到 `events.jsonl`（调用 `_append_jsonl`）
- `_get_event_type(event) -> EventType`：从 `MusicalEvent` 提取 `EventType` 枚举（私有）
- `log_inference(request, response, latency_ms, server_process_ms)`：构造 `InferenceEvent`，累积到 `self.inferences` 列表，调用 `MetricsCalculator.add_inference()`
- `_append_jsonl(obj: Dict) -> None`：JSON 序列化并写入 `events.jsonl`（私有）
- `save_metrics(session_config: Dict) -> None`：调用 `MetricsCalculator.generate_performance_json()`，写入 `performance.json`
- `save_inferences() -> None`：将 `self.inferences` 列表写入 `inferences.json`
- `output_tick / output_stats / output_status / output_config`：均为 no-op
- `close()`：调用 `save_metrics()` + `save_inferences()`，关闭文件

#### `infrastructure/output/session-logger.md` — 源文件：`output/session_logger.py`

**`SessionLoggerOutputSink`**
- `__init__(session_manager, tempo_config, app_config)`：调用 `session_manager.create_session_directory()`，创建 `MidiFileOutputSink` (→ `combined.mid`) 和 `JsonLoggerOutputSink`；保存初始配置
- `output_event(event, source)`：同时委托给 `MidiFileOutputSink` + `JsonLoggerOutputSink`
- `output_tick(tick, bar, beat)`：委托给 `JsonLoggerOutputSink`
- `output_stats(...)`：委托给 `JsonLoggerOutputSink`
- `output_status(state, message="")`：委托给 `JsonLoggerOutputSink`；若 state=="running" 则记录会话开始时间
- `output_config(config)`：委托给两个 sink；调用 `session_manager.save_config()`
- `log_inference(request, response, latency_ms, server_process_ms)`：委托给 `JsonLoggerOutputSink.log_inference()`
- `save_metrics(session_config) -> None`：委托给 `JsonLoggerOutputSink.save_metrics()`
- `close()`：调用 `MidiFileOutputSink.close()` + `JsonLoggerOutputSink.close()`；调用 `session_manager.save_summary()`

#### `infrastructure/output/composite.md` — 源文件：`output/composite.py`

**`CompositeOutputSink`**
- `__init__(sinks: List[OutputSink])`：持有多个 sink 列表
- `output_event / output_tick / output_stats / output_status / output_config`：完全 fan-out——对每个 sink 调用相同方法
- `log_inference(request, response, latency_ms, server_process_ms)`：对每个实现了 `log_inference()` 的 sink（通过 `hasattr` 检测）进行 fan-out
- `close()`：按顺序调用每个 sink 的 `close()`

---

#### `infrastructure/inference/overview.md`
- 两种推理模式：HTTP 模式（`HttpInferenceClient`）vs 本地模式（`StanleyInferenceEngine`）
- Stanley 引擎两层 adapter 说明
- 关键参数：context window 96 frames，ticks_per_beat 4，max polyphony 4

#### `infrastructure/inference/http-client.md` — 源文件：`inference/http_client.py`

**`HttpInferenceClientConfig`（dataclass）**：`server_url` (str), `timeout_seconds` (float, 10.0)

**`HttpInferenceClient`（implements `InferenceEngine`）**
- `__init__(config: HttpInferenceClientConfig)`
- `_endpoint(replacement_path: str) -> str`：拼接 `server_url + path`（私有）
- `generate_accompaniment(melody_events, generation_start_tick, generation_length_frames) -> tuple[List[MusicalEvent], TimingInfo]`：POST `/generate`，调用 `serialization.event_to_dict()` 序列化请求，调用 `serialization.event_from_dict()` + `timing_info_from_dict()` 反序列化响应
- `inject_history(melody_events, acc_events) -> None`：POST `/inject`，预填充服务端模型历史
- `set_injection_offset(offset_ticks: int) -> None`：POST `/set_injection_offset`
- `clear_history() -> None`：POST `/clear_history`
- `get_injection_status() -> Dict`：GET `/injection_status`，返回注入状态

#### `infrastructure/inference/stanley-engine.md` — 源文件：`inference/stanley_engine.py`

**`_LegacyStanleyLike`（local Protocol）**：定义旧引擎接口（`generate_accompaniment`, `clear_history`, `set_injection_offset`），用于类型约束和测试替换

**`StanleyInferenceConfig`（dataclass）**：`ticks_per_beat` (int, 4), `max_polyphony` (int, 4), 等

**`StanleyInferenceEngine`（implements `InferenceEngine`）**
- `__init__(*, config, legacy_engine=None)`：若 `legacy_engine=None`，自动创建 `LegacyInferenceEngineStanley`
- `generate_accompaniment(melody_events, generation_start_tick, generation_length_frames)`：将 `List[MusicalEvent]` 转为 duration-note dict list → 调用 legacy engine → 将返回的 dict list 转回 `List[MusicalEvent]`
- `inject_history(melody_events, acc_events) -> None`：转换后调用 legacy engine 注入
- `set_injection_offset(offset_ticks) -> None`：委托给 legacy engine
- `clear_history() -> None`：委托给 legacy engine

#### `infrastructure/inference/stanley-legacy.md` — 源文件：`inference/stanley_legacy.py`

**`LegacyInferenceEngineStanley`**（RoFormer 模型直接包装）
- `__init__(checkpoint_path, model_config, ...)`：加载模型权重，初始化 tokenizer，设置设备
- `set_injection_offset(offset_ticks) -> None`：记录注入偏移量，用于 tick 对齐
- `clear_history() -> None`：清空内部 piano-roll 历史张量
- `_notes_to_rolls(notes, max_tick, max_polyphony, program) -> torch.Tensor`：将 note dict list 转为钢琴卷帘 tensor（私有）
- `_tensors_to_notes(output_tensors, generation_start_tick) -> list[list[dict]]`：将模型输出 tensor 转回 note dict list（私有）
- `generate_accompaniment(melody_notes, generation_start_tick, generation_length_frames) -> list[list[dict]]`：端到端推理；组合 melody + history context，调用 RoFormer forward，返回伴奏 note dicts

#### `infrastructure/inference/serialization.md` — 源文件：`inference/serialization.py`

（模块级函数，无类）
- `event_to_dict(event: MusicalEvent) -> Dict[str, Any]`：将 `MusicalEvent` 序列化为 JSON-safe dict（EventType → 字符串，所有字段平铺）
- `event_from_dict(d: Dict[str, Any]) -> MusicalEvent`：从 dict 反序列化，字符串 → `EventType` 枚举
- `timing_info_from_dict(d: Dict[str, Any]) -> TimingInfo`：从 HTTP 响应 dict 构造 `TimingInfo`

---

### `docs/architecture/presentation/` — Presentation 层（子目录）

```
docs/architecture/presentation/
├── overview.md
├── cli.md
└── config-parser.md
```

#### `presentation/overview.md`
- Presentation 层职责：唯一的 Infrastructure 感知层入口，从外部参数（CLI args / 环境变量）构造完整的 `ApplicationConfig`
- 程序生命周期图（Mermaid）：解析 args → 创建组件 → 启动 Service → 捕获信号 → 优雅关闭

#### `presentation/cli.md` — 源文件：`presentation/cli/cli.py`

**模块级函数**
- `main() -> int`：入口点（`pyproject.toml` 中注册为 `streammuse-cli`）；调用 `parse_args()`，创建 `SessionManager`（若需要），调用三个 Factory 创建组件，初始化 `RealTimeMusicService`，注册信号处理，调用 `service.start()`，返回退出码
- `cleanup() -> None`（内部闭包）：调用 `service.stop()`，刷新输出 sink
- `signal_handler(sig, frame)`（内部闭包）：捕获 SIGINT/SIGTERM，调用 `cleanup()`

#### `presentation/config-parser.md` — 源文件：`presentation/cli/config_parser.py`

**模块级函数**
- `parse_args() -> argparse.Namespace`：定义所有 `--` 参数（参数表见 `reference/cli.md`），返回 parsed namespace
- `args_to_config(args: argparse.Namespace) -> ApplicationConfig`：将 argparse namespace 映射为嵌套的 `ApplicationConfig`；验证参数合法性（如 `midi_file` 模式必须提供 `--midi-file`）
- `env_to_config() -> Optional[ApplicationConfig]`：从环境变量读取配置，优先级低于 CLI args；若无相关 env var 则返回 `None`

---

### `docs/user-guide/input-modes.md` — 输入模式

**内容**：每种输入模式的详细说明、示例命令、使用场景、已知限制  
**覆盖**：`keyboard` / `midi` / `midi_file`

---

### `docs/user-guide/output-types.md` — 输出类型

**内容**：7 种输出类型的说明、示例命令、输出样例、依赖要求  
**覆盖**：`console` / `audio` / `midi_file` / `websocket` / `json_log` / `session` / `composite`

---

### `docs/user-guide/session-logging.md` — Session 日志

**内容**：
- 启用方式（`--output-type composite --log-dir logs`）
- Session 目录结构：
  ```
  logs/session_YYYYMMDD-HHMMSS/
  ├── events.jsonl
  ├── inferences.json
  ├── performance.json
  ├── statistics.csv
  ├── session_config.json
  └── combined.mid
  ```
- 每个文件的 schema / 字段说明（含示例 JSON）
- `performance.json` 中延迟百分位数的含义（p50/p95/p99）

---

### `docs/user-guide/music-injection.md` — 音乐注入

**内容**：
- 什么是注入（预填充模型历史，减少冷启动时间）
- 使用方式：`--injection-file <path> --injection-length <frames>`
- `prompts/` 目录中可用的文件列表（按调性分组）
- 建议：选择与演奏调性匹配的注入文件

---

### `docs/developer-guide/extending-inputs.md` — 扩展输入

**内容**：
1. 实现 `InputSource` Protocol（两个方法：`read_events`, `close`）
2. 在 `InputSourceFactory` 中注册新类型（添加 enum 值 + `elif` 分支）
3. 在 `InputConfig` 中添加配置字段（如需要）
4. 在 CLI 中暴露新的 `--input-mode` 选项
5. 编写单元测试（参照 `tests/unit/infrastructure/` 中已有测试）
6. 完整示例：实现一个 `OscInput`（从 OSC 协议读取）

---

### `docs/developer-guide/extending-outputs.md` — 扩展输出

**内容**：
1. 实现 `OutputSink` Protocol（7 个方法）
2. 可选：实现 `log_inference()` 方法（用于推理日志记录）
3. 在 `OutputSinkFactory` 中注册
4. 完整示例：实现一个 `HttpOutputSink`（POST 事件到外部服务）

---

### `docs/developer-guide/extending-engines.md` — 扩展推理引擎

**内容**：
1. 实现 `InferenceEngine` Protocol（三个方法）
2. 在 `InferenceEngineFactory` 中注册
3. `generate_accompaniment()` 的契约：输入 `List[MusicalEvent]`，返回 `(List[MusicalEvent], TimingInfo)`
4. 完整示例：实现一个 stub engine（始终返回固定音符序列）

---

### `docs/developer-guide/inference-server.md` — 推理服务器

**内容**：
- **Fake server**（开发用）：启动命令，行为（返回随机音符），用途
- **Real server**：环境变量（`CHECKPOINT_PATH`, `MODEL_MAX_SEQ_LEN_FRAMES`, `GENERATION_LENGTH_FRAMES`, `MODEL_SIZE`），启动命令
- REST API：`POST /generate` 请求/响应 JSON schema
- Stanley 引擎参数：context window (96 frames)、ticks/beat (4)、max polyphony (4)

---

### `docs/developer-guide/testing.md` — 测试

**内容**：
- 运行全部测试：`uv run pytest tests/`
- 测试目录结构（`tests/unit/` 按层分、`tests/integration/` 测 CLI 入口）
- 各层测试策略（domain：纯函数，无 mock；infrastructure：mock 外部依赖；integration：端到端 CLI）
- 添加新测试的惯例（命名、fixture 位置）
- CI 注意事项（音频相关测试在无声卡环境的跳过条件）

---

### `docs/reference/cli.md` — CLI 完整参考

**内容**：完整的参数表，包含：
- 参数名、短旗
- 类型 / 允许值
- 默认值
- 描述
- 用法示例

格式参照 Click / Typer 的 CLI reference 文档标准。

---

### `docs/reference/protocols.md` — Protocol / API 参考

**内容**：Domain 层四个 Protocol 的完整 API 文档（类似 Python API 文档格式）：

```python
class InputSource(Protocol):
    def read_events(self) -> Iterator[MusicalEvent]: ...
    def close(self) -> None: ...

class OutputSink(Protocol):
    def output_event(self, event: MusicalEvent, source: str) -> None: ...
    def output_tick(self, tick: int, bar: int, beat: int) -> None: ...
    def output_stats(self, round_trip_ms, server_process_ms, ...) -> None: ...
    def output_status(self, state: str, message: str = "") -> None: ...
    def output_config(self, config: dict) -> None: ...
    def close(self) -> None: ...

class InferenceEngine(Protocol):
    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
    ) -> tuple[List[MusicalEvent], TimingInfo]: ...
    def inject_history(self, ...) -> None: ...
    def clear_history(self) -> None: ...
```

每个方法包含：参数说明、返回值、副作用、已知实现类列表。

---

## 文件数量汇总

| 目录 | 文件数 |
|---|---|
| `docs/` (root) | 1 (`index.md`) |
| `docs/getting-started/` | 3 |
| `docs/architecture/` (root) | 1 (`overview.md`) |
| `docs/architecture/domain/` | 5 |
| `docs/architecture/application/` | 4 |
| `docs/architecture/infrastructure/` | 1 + 5 + 8 + 5 = 19 |
| `docs/architecture/presentation/` | 3 |
| `docs/user-guide/` | 4 |
| `docs/developer-guide/` | 5 |
| `docs/reference/` | 2 |
| **合计** | **47** |

---

## 写作优先级

优先写入**最常被访问**的文档，按以下顺序：

1. `docs/index.md` — 入口，最重要
2. `docs/getting-started/quickstart.md` — 新用户第一步
3. `docs/architecture/overview.md` — 全局视角，对开发者最有价值
4. `docs/architecture/domain/interfaces.md` — 所有扩展的起点，贡献者必读
5. `docs/architecture/application/service.md` — 三线程核心，调试时高频参考
6. `docs/reference/cli.md` — 使用频率最高的参考文档
7. `docs/user-guide/output-types.md` + `session-logging.md` — 功能最复杂
8. `docs/architecture/domain/` 其余文件（musical → timing → logging）
9. `docs/architecture/infrastructure/` 各具体实现
10. `docs/developer-guide/` 扩展指南

---

## 格式规范

- 每个 `.md` 文件顶部有 YAML frontmatter（`title`, `description`）——便于未来接入 MkDocs / Docusaurus
- 代码块指定语言（` ```python `, ` ```bash `）
- 表格用于参数/配置对比
- Mermaid 图用于架构图、数据流图（`architecture/overview.md` 必须包含）
- 内部链接用相对路径（`../architecture/overview.md`）
- 不使用第一人称叙述；风格简洁、技术性

---

## Todo List

### Phase 0 — Setup

- [x] Create `docs/` directory structure (all subdirectories: `getting-started/`, `architecture/domain/`, `architecture/application/`, `architecture/infrastructure/input/`, `architecture/infrastructure/output/`, `architecture/infrastructure/inference/`, `architecture/presentation/`, `user-guide/`, `developer-guide/`, `reference/`)

---

### Phase 1 — Entry Point & Getting Started (4 files)

- [x] **`docs/index.md`** — project intro paragraph; feature list; ASCII/Mermaid system diagram (User → CLI → Service → Inference → Output); system requirements table; quick-nav links to all sections
- [x] **`docs/getting-started/installation.md`** — clone repo; `uv sync`; `cd transformers && pip install -e .`; verify with `--help`; note optional deps (audio: rtmidi/FluidSynth; GPU: CUDA)
- [x] **`docs/getting-started/quickstart.md`** — start fake server; start CLI keyboard mode; keyboard note mapping table; explain terminal output fields; links to next steps
- [x] **`docs/getting-started/configuration.md`** — full parameter table (name, type, default, description) for all CLI flags; note that all flags map to `ApplicationConfig`

---

### Phase 2 — Architecture Overview (1 file)

- [x] **`docs/architecture/overview.md`** — Mermaid flowchart of 4 layers; dependency rule explanation; Mermaid sequence diagram of full data flow (keyboard → MusicalEvent → inference → output); 3-thread model summary; design decision rationale (Protocol over ABC; no asyncio); nav links to each sub-section

---

### Phase 3 — Architecture: Domain Layer (5 files)

- [x] **`docs/architecture/domain/overview.md`** — domain layer responsibilities; sub-module list with one-line descriptions; Mermaid dependency graph among musical/timing/logging/interfaces; immutability contract for `MusicalEvent`
- [x] **`docs/architecture/domain/musical.md`** — `EventType` enum values; `MusicalEvent` all 8 fields with types/constraints/special values (`pitch=-1`); `midi_message_to_event()` signature + params + return + edge cases; `event_to_midi_message()` signature + behavior on `pitch=-1`; `MusicalSequence.__init__`; `get_events_in_range()` semantics (half-open interval); `quantize()` behavior
- [x] **`docs/architecture/domain/timing.md`** — `Tempo` fields + `tick_duration_seconds()` + `ticks_to_seconds()` + `seconds_to_ticks()` with formulas; `PlaybackScheduler.__init__`; `get_next_event()` semantics; `advance_tick()` return value; `reset()` effect; scheduling alignment explanation
- [x] **`docs/architecture/domain/logging.md`** — `LogEvent` all fields; `InferenceEvent` all fields; `SessionManager.__init__`; `_generate_session_id()` format; `create_session_directory()` return; `save_config()` target file; `save_summary()` target file; `MetricsCalculator` all 7 methods (params, return types, what each calculates); output JSON structure for `generate_performance_json()`
- [x] **`docs/architecture/domain/interfaces.md`** — `InputSource` protocol: both methods with blocking/generator semantics, close contract; `OutputSink` protocol: all 6 methods with purpose; `InferenceEngine` protocol: `generate_accompaniment()` full signature + contract, `inject_history()`, `clear_history()`; `TimingInfo` fields; note on `log_inference()` (not in protocol, detected via `hasattr`)

---

### Phase 4 — Architecture: Application Layer (4 files)

- [x] **`docs/architecture/application/overview.md`** — layer responsibilities; how Config → Factories → Service flows; note that application layer never imports from infrastructure directly
- [x] **`docs/architecture/application/config.md`** — field tables for all 5 config dataclasses (`TempoConfig`, `InputConfig`, `OutputConfig`, `InferenceConfig`, `ApplicationConfig`); field name + type + default + description for each
- [x] **`docs/architecture/application/factories.md`** — `InputSourceFactory.create()` signature + all 4 supported modes with their concrete classes; `OutputSinkFactory.create()` signature + all 7 modes + `session_manager` param note + composite construction logic; `InferenceEngineFactory.create()` + current behavior (always `HttpInferenceClient`) + extensibility note; `ValueError` on unknown mode for each
- [x] **`docs/architecture/application/service.md`** — `RealTimeServiceRuntime` dataclass fields; `RealTimeMusicService.__init__()` params; `running` property; `_input_worker()` loop logic and queues it writes to; `_tick_loop()` sleep mechanism, playback scheduling, inference trigger condition; `_inference_worker()` inference call, result routing, `output_stats()` + `log_inference()` calls; `start()` injection logic + thread launch; `stop()` thread join + `close()`; Mermaid sequence diagram of 3-thread interaction with 3 queues

---

### Phase 5 — Architecture: Infrastructure — Input (5 files)

- [x] **`docs/architecture/infrastructure/input/overview.md`** — summary table of all 4 `InputSource` implementations; shared contract: `read_events()` is blocking generator, `close()` must unblock it
- [x] **`docs/architecture/infrastructure/input/keyboard.md`** — `KeyboardInputConfig` fields + default key→pitch map table; `KeyboardInput.__init__()` (starts listener thread); `_handle_key_down()` event creation; `_handle_key_up()` event creation; `_run_listener()` pynput mechanics; `read_events()` queue drain loop + `q` key exit; `close()` stopping mechanism
- [x] **`docs/architecture/infrastructure/input/midi-device.md`** — `MidiDeviceInput.__init__()` port auto-selection logic; `read_events()` rtmidi polling + `converters` call; `close()` port release; note on device_name=None behavior
- [x] **`docs/architecture/infrastructure/input/midi-file.md`** — `MidiFileInputConfig` fields + `seconds_per_tick` property formula; `MidiFileInput.__init__()` file parsing; `_midi_to_notes()` tick conversion logic; `read_events()` real-time sleep simulation; `close()` stop flag mechanism
- [x] **`docs/architecture/infrastructure/input/list-input.md`** — `ListInput.__init__()`; `read_events()` sequential yield semantics (non-blocking, terminates); `close()` no-op; use cases (testing, replay)

---

### Phase 6 — Architecture: Infrastructure — Output (8 files)

- [x] **`docs/architecture/infrastructure/output/overview.md`** — summary table of all 7 `OutputSink` implementations (class, output target, has `log_inference()`, config class); `log_inference()` design explanation (not in Protocol, `hasattr`-based, `CompositeOutputSink` fan-out)
- [x] **`docs/architecture/infrastructure/output/console.md`** — `ConsoleOutputConfig` fields; all 6 `OutputSink` methods: format strings used, which are no-ops
- [x] **`docs/architecture/infrastructure/output/audio.md`** — `AudioOutputConfig` fields; `_ensure_port()` lazy-init; `output_event()` MIDI message send; which methods are no-ops; `close()` port teardown; dependency note (python-rtmidi virtual port)
- [x] **`docs/architecture/infrastructure/output/midi-file.md`** — `MidiFileOutputConfig` fields; `__init__()` mido MidiFile creation; `output_event()` delta-tick calculation; `close()` file write; which methods are no-ops
- [x] **`docs/architecture/infrastructure/output/websocket.md`** — `WebSocketOutputConfig` fields; `_enqueue()` JSON serialization; all `output_*` methods and their JSON message shapes; `get_pending_messages()` usage pattern; `close()` queue clear
- [x] **`docs/architecture/infrastructure/output/json-logger.md`** — `__init__()` creates `MetricsCalculator` + opens `events.jsonl`; `output_event()` → `LogEvent` → `_append_jsonl()`; `_get_event_type()` note; `log_inference()` → `InferenceEvent` + `MetricsCalculator.add_inference()`; `_append_jsonl()` write mechanics; `save_metrics()` → `performance.json`; `save_inferences()` → `inferences.json`; `close()` sequence; which methods are no-ops
- [x] **`docs/architecture/infrastructure/output/session-logger.md`** — `__init__()` session dir creation + inner sink construction; delegation table (which method calls which inner sink(s)); `log_inference()` delegation chain; `save_metrics()` delegation; `close()` sequence including `session_manager.save_summary()`
- [x] **`docs/architecture/infrastructure/output/composite.md`** — `__init__(sinks)` list; complete fan-out behavior for all 6 protocol methods; `log_inference()` fan-out with `hasattr` guard; `close()` ordered teardown

---

### Phase 7 — Architecture: Infrastructure — Inference (5 files)

- [x] **`docs/architecture/infrastructure/inference/overview.md`** — two-mode diagram (HTTP vs local); two-layer adapter pattern (StanleyInferenceEngine wraps LegacyInferenceEngineStanley); Stanley model parameters table (context 96 frames, ticks_per_beat 4, max_polyphony 4)
- [x] **`docs/architecture/infrastructure/inference/http-client.md`** — `HttpInferenceClientConfig` fields; all 6 methods: endpoint paths, request/response JSON schema for each, serialization functions called; error handling (timeout, connection refused)
- [x] **`docs/architecture/infrastructure/inference/stanley-engine.md`** — `_LegacyStanleyLike` protocol definition and purpose; `StanleyInferenceConfig` fields; `StanleyInferenceEngine.__init__()` lazy legacy engine creation; `generate_accompaniment()` full conversion pipeline (`MusicalEvent` → dict → model → dict → `MusicalEvent`); `inject_history()`; `set_injection_offset()` + `clear_history()` delegation
- [x] **`docs/architecture/infrastructure/inference/stanley-legacy.md`** — `__init__()` checkpoint loading + device setup; `set_injection_offset()` tick alignment purpose; `clear_history()` tensor reset; `_notes_to_rolls()` piano-roll tensor shape and encoding; `_tensors_to_notes()` decoding logic; `generate_accompaniment()` context window assembly + RoFormer forward call
- [x] **`docs/architecture/infrastructure/inference/serialization.md`** — `event_to_dict()` field mapping table; `event_from_dict()` inverse mapping + `EventType` string→enum; `timing_info_from_dict()` fields extracted from response dict; note on JSON-safety (all values primitive types)

---

### Phase 8 — Architecture: Presentation Layer (3 files)

- [x] **`docs/architecture/presentation/overview.md`** — layer responsibilities; Mermaid lifecycle diagram (parse → create components → inject history → start service → signal → cleanup); entry point registration in `pyproject.toml`
- [x] **`docs/architecture/presentation/cli.md`** — `main()` step-by-step control flow (parse → session manager → factories → service init → signal register → start → return code); `cleanup()` closure: `service.stop()` call; `signal_handler()` closure: SIGINT/SIGTERM handling
- [x] **`docs/architecture/presentation/config-parser.md`** — `parse_args()` full argument list with types and metavars; `args_to_config()` mapping logic + validation rules (e.g. `midi_file` mode requires `--midi-file`); `env_to_config()` supported env vars + priority vs CLI args

---

### Phase 9 — User Guide (4 files)

- [x] **`docs/user-guide/input-modes.md`** — for each mode (`keyboard`, `midi`, `midi_file`): description, example command, key mapping / device selection / file path details, known limitations
- [x] **`docs/user-guide/output-types.md`** — for each of 7 types: description, example command, sample output/file, required dependencies
- [x] **`docs/user-guide/session-logging.md`** — enable command (`--output-type composite --log-dir logs`); full session directory structure; schema for each output file (`events.jsonl`, `inferences.json`, `performance.json`, `statistics.csv`, `session_config.json`, `combined.mid`); latency percentile interpretation
- [x] **`docs/user-guide/music-injection.md`** — what injection does (pre-warm model history); command syntax; table of available prompt files in `prompts/` by key; recommendation to match key

---

### Phase 10 — Developer Guide (5 files)

- [x] **`docs/developer-guide/extending-inputs.md`** — implement `InputSource` (2 methods); register in `InputSourceFactory` (add enum + elif); add `InputConfig` field if needed; expose CLI flag; write unit tests; full worked example: `OscInput`
- [x] **`docs/developer-guide/extending-outputs.md`** — implement `OutputSink` (6 methods); optionally implement `log_inference()`; register in `OutputSinkFactory`; full worked example: `HttpOutputSink`
- [x] **`docs/developer-guide/extending-engines.md`** — implement `InferenceEngine` (3 methods); register in `InferenceEngineFactory`; `generate_accompaniment()` contract details; full worked example: stub engine returning fixed notes
- [x] **`docs/developer-guide/inference-server.md`** — fake server: startup, behavior (random notes), use case; real server: all env vars, startup command; `POST /generate` request/response JSON schema; `POST /inject` schema; Stanley model parameters
- [x] **`docs/developer-guide/testing.md`** — run all tests; run quiet; test directory structure by layer; per-layer testing strategy (domain: pure functions; infrastructure: mock externals; integration: CLI end-to-end); naming conventions + fixture locations; audio test skip conditions

---

### Phase 11 — Reference (2 files)

- [x] **`docs/reference/cli.md`** — complete parameter table: short flag, long flag, type, choices, default, description, example; grouped by category (input / output / inference / session)
- [x] **`docs/reference/protocols.md`** — full API doc for all 4 Domain protocols + `TimingInfo`; for each method: signature, parameters, return type, contract/invariants, known implementing classes; `log_inference()` non-protocol extension pattern

---

### Phase 12 — Review & Polish

- [x] Verify all internal links resolve (relative paths between files)
- [x] Ensure every source file mentioned in architecture docs maps to an actual file in `src/streammuse/`
- [x] Validate all code signatures in docs match current code (grep against actual class/method definitions)
- [x] Add YAML frontmatter (`title`, `description`) to every `.md` file
- [x] Check all Mermaid diagrams render correctly (at minimum: `overview.md`, `service.md`, `presentation/overview.md`)
- [x] Confirm `docs/` `README` or `index.md` links cover all 47 files (no orphan pages)

---

*Plan version: 2026-03-11 | 对应代码版本: branch `new_system_stanley`, 98 tests passing*
