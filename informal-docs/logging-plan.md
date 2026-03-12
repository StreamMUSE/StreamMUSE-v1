# StreamMUSE Logging Plan

## 概述

本文档计划为StreamMUSE-new-sys项目添加完整的日志功能。参考自同级项目StreamMUSE中的实现，为新系统设计现代化、模块化的logging架构。

---

## 1. 当前状态分析

### 现有输出类型

```
✅ console        - 打印到终端
✅ midi_file      - 保存MIDI文件
✅ audio          - 实时MIDI播放
✅ websocket      - WebSocket消息
✅ composite      - 组合输出
❌ json_log       - JSON日志（缺失）
❌ csv_log        - CSV日志（缺失）
❌ session        - 会话日志（缺失）
```

### 旧项目（StreamMUSE）的logging特性

```
StreamMUSE/app/output_handlers/:
├── json_log_handler.py      ← 推理事件JSON日志
├── log_output.py            ← 基准测试/摘要日志
└── logs/
    └── session_YYYYMMDD-HHMMSS/
        ├── prompt.mid
        └── inferences.json     ← 包含所有推理请求/响应
```

---

## 2. 提议的日志架构

### 会话目录结构

```
logs/
└── session_20260311-143529/          # 时间戳格式会话目录
    ├── events.jsonl                   # 所有事件（每行一个JSON）
    ├── inferences.json                # 推理请求/响应完整记录
    ├── statistics.csv                 # 性能统计（逗号分隔）
    ├── performance.json               # 详细的性能指标
    ├── user_input.mid                 # 用户输入的音符
    ├── model_output.mid               # 模型生成的伴奏
    ├── combined.mid                   # 合并的用户+模型
    ├── session_config.json            # 本次会话的配置
    └── session_summary.txt            # 人工可读的摘要
```

### 数据格式规范

#### 1. events.jsonl（事件日志）
```json
{"timestamp": 1234567890.123, "tick": 92, "type": "note_on", "pitch": 61, "source": "user", "velocity": 100}
{"timestamp": 1234567890.142, "tick": 93, "type": "note_off", "pitch": 61, "source": "user"}
{"timestamp": 1234567890.134, "tick": 100, "type": "inference_request", "melody_ticks": [92, 93], "generation_start_tick": 92}
{"timestamp": 1234567890.145, "tick": 100, "type": "inference_response", "acc_notes": 5, "server_latency_ms": 11.5}
{"timestamp": 1234567890.156, "tick": 100, "type": "note_on", "pitch": 48, "source": "model", "velocity": 80}
```

#### 2. inferences.json（推理完整记录）
```json
[
  {
    "id": "inf_001",
    "request": {
      "timestamp": 1234567890.123,
      "generation_start_tick": 92,
      "melody_events": [...],
      "generation_length_frames": 20
    },
    "response": {
      "timestamp": 1234567890.145,
      "accompaniment_events": [...],
      "timings": {
        "request_arrival_time": 1234567890.124,
        "response_output_time": 1234567890.145,
        "preprocess_start_time": 1234567890.125,
        "inference_start_time": 1234567890.126,
        "inference_end_time": 1234567890.135,
        "postprocess_start_time": 1234567890.144
      }
    },
    "latency_ms": 22.0,
    "server_process_ms": 11.5
  }
]
```

#### 3. statistics.csv（性能统计）
```csv
metric,value
total_events,523
user_input_events,87
model_output_events,436
avg_latency_ms,22.3
min_latency_ms,11.2
max_latency_ms,45.6
std_latency_ms,8.9
inference_requests,18
successful_responses,18
failed_responses,0
hit_rate,100.0
avg_server_process_ms,11.8
total_runtime_s,45.2
```

#### 4. performance.json（详细性能指标）
```json
{
  "session_config": {
    "tempo_bpm": 120.0,
    "ticks_per_beat": 4,
    "beats_per_bar": 4,
    "inference_type": "http",
    "generation_interval_ticks": 2,
    "generation_length_frames": 20
  },
  "timing_statistics": {
    "latency_ms": {
      "mean": 22.3,
      "median": 21.5,
      "std": 8.9,
      "min": 11.2,
      "max": 45.6,
      "p95": 38.2,
      "p99": 42.1
    },
    "server_process_ms": {
      "mean": 11.8,
      "median": 11.2,
      "std": 2.3
    },
    "network_latency_ms": {
      "mean": 10.5,
      "std": 6.5
    }
  },
  "event_statistics": {
    "total_events": 523,
    "user_input_events": 87,
    "model_output_events": 436,
    "inference_requests": 18,
    "successful_responses": 18,
    "failed_responses": 0,
    "hit_rate": 100.0
  },
  "music_analysis": {
    "total_duration_seconds": 45.2,
    "total_notes_user": 43,
    "total_notes_model": 218,
    "total_notes_combined": 261,
    "pitch_range_user": [48, 76],
    "pitch_range_model": [40, 68],
    "average_velocity_user": 100,
    "average_velocity_model": 80
  }
}
```

---

## 3. 实现计划

### Phase 1: 核心日志输出适配器（1周）

```
新建文件:
├── src/streammuse/infrastructure/output/json_logger.py
│   └── JsonLoggerOutputSink: 输出events.jsonl
│
├── src/streammuse/infrastructure/output/session_logger.py
│   └── SessionLoggerOutputSink: 管理会话目录和多文件输出
│
└── src/streammuse/domain/logging/
    ├── __init__.py
    ├── event_types.py       # LogEvent, InferenceEvent dataclasses
    ├── session_manager.py   # SessionManager: 创建/管理会话目录
    └── metrics_calculator.py# MetricsCalculator: 计算统计数据
```

**关键类设计**:

```python
# event_types.py
@dataclass(frozen=True)
class LogEvent:
    """Base event in the log stream"""
    timestamp: float
    tick: int
    event_type: str  # "note_on", "note_off", "inference_request", "inference_response"
    data: Dict[str, Any]

@dataclass(frozen=True)
class InferenceEvent:
    """Complete inference request/response pair"""
    inference_id: str
    timestamp_request: float
    timestamp_response: float
    request_data: Dict
    response_data: Dict
    latency_ms: float
    server_process_ms: float

# session_manager.py
class SessionManager:
    def __init__(self, base_log_dir: str = "logs"):
        self.session_id = self._generate_session_id()  # YYYYMMDD-HHMMSS
        self.session_dir = Path(base_log_dir) / f"session_{self.session_id}"
    
    def create_session_directory(self) -> Path:
        """Create session directory and return path"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir
    
    def save_config(self, config: Dict) -> None:
        """Save session_config.json"""
        ...
    
    def save_summary(self, summary: Dict) -> None:
        """Save session_summary.txt"""
        ...

# json_logger.py
class JsonLoggerOutputSink:
    def __init__(self, session_dir: Path):
        self.events_file = session_dir / "events.jsonl"
        self.inferences_file = session_dir / "inferences.json"
        self.inferences = []
        self.inference_counter = 0
    
    def output_event(self, event, source):
        """Log single event"""
        log_event = {
            "timestamp": time.time(),
            "tick": event.tick,
            "type": event.event_type.value,
            "pitch": event.pitch,
            "source": source,
            ...
        }
        self._append_jsonl(log_event)
    
    def log_inference(self, request, response, latency_ms):
        """Log complete inference"""
        inference = {
            "id": f"inf_{self.inference_counter:03d}",
            "request": request,
            "response": response,
            "latency_ms": latency_ms
        }
        self.inferences.append(inference)
    
    def close(self):
        """Save all buffered data to files"""
        self._save_inferences_json()
```

### Phase 2: 性能指标计算（1周）

```python
# metrics_calculator.py
class MetricsCalculator:
    def __init__(self):
        self.events = []
        self.inferences = []
    
    def add_event(self, event: LogEvent) -> None:
        """Add event to tracker"""
        ...
    
    def add_inference(self, inf: InferenceEvent) -> None:
        """Add inference to tracker"""
        ...
    
    def calculate_latency_stats(self) -> Dict:
        """计算延迟统计 (mean, median, std, p95, p99, etc)"""
        ...
    
    def calculate_event_stats(self) -> Dict:
        """计算事件统计"""
        ...
    
    def calculate_music_stats(self) -> Dict:
        """计算音乐分析统计"""
        ...
    
    def generate_performance_json(self) -> Dict:
        """生成完整的performance.json"""
        ...
    
    def generate_statistics_csv(self) -> str:
        """生成CSV格式的统计数据"""
        ...
```

### Phase 3: 输出工厂和CLI集成（1周）

修改现有文件：

```python
# application/factories/output_factory.py
class OutputSinkFactory:
    @staticmethod
    def create(app_config, session_manager=None):
        if cfg.type == "json_log":
            return JsonLoggerOutputSink(session_manager.session_dir)
        
        if cfg.type == "session":  # 包含MIDI + JSON + CSV
            return SessionLoggerOutputSink(
                session_dir=session_manager.session_dir,
                include_midi=True,
                include_json=True,
                include_stats=True
            )
        
        # ... 现有逻辑

# presentation/cli/cli.py
def main():
    # ... 现有代码
    
    # Session管理
    session_manager = SessionManager("logs")
    session_dir = session_manager.create_session_directory()
    session_manager.save_config({
        "tempo": config.tempo.bpm,
        "input_type": config.input.type,
        "inference_type": config.inference.type,
        ...
    })
    
    # 创建输出（包含logging）
    output_sink = OutputSinkFactory.create(config, session_manager)
    
    # 创建service
    service = RealTimeMusicService(...)
    
    # 关闭时保存日志
    def cleanup():
        if hasattr(output_sink, 'save_final_metrics'):
            output_sink.save_final_metrics()
        session_manager.save_summary(...)
    
    atexit.register(cleanup)
    ...
```

### Phase 4: 拓展和优化（2周）

```
可选的增强功能：
1. 实时性能仪表板（WebSocket推送）
2. 日志压缩和存档
3. 日志查看和分析工具
4. 日志数据库集成（SQLite）
5. 与TensorBoard/Wandb集成
```

---

## 4. 使用示例

### 示例1: 基本回话日志

```bash
# 启动fake推理服务器
uv run python scripts/fake_inference_server.py

# 运行CLI，自动保存完整日志
uv run streammuse-cli --input-mode keyboard --output-type session
```

生成的会话目录：
```
logs/session_20260311-143529/
├── events.jsonl
├── inferences.json
├── statistics.csv
├── performance.json
├── combined.mid
└── session_config.json
```

### 示例2: 仅保存JSON日志（不保存MIDI）

```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --output-type json_log
```

只生成：
```
logs/session_20260311-143529/
└── events.jsonl
```

### 示例3: Composite输出（console + session日志）

```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --output-type composite
```

同时：
- 打印到终端
- 保存完整日志到 `logs/session_YYYYMMDD-HHMMSS/`

---

## 5. 配置参数

添加到CLI参数：

```python
parser.add_argument(
    "--log-dir",
    type=str,
    default="logs",
    help="Base directory for session logs"
)

parser.add_argument(
    "--enable-performance-tracking",
    action="store_true",
    help="Enable detailed performance metrics calculation"
)

parser.add_argument(
    "--compress-logs",
    action="store_true",
    help="Compress logs after session ends"
)
```

---

## 6. 实现优先级和时间表

| Phase | 功能 | 优先级 | 时间 | 依赖 |
|-------|------|--------|------|------|
| 1 | 核心JSON logging | 🔴 高 | 1周 | 无 |
| 2 | 性能指标计算 | 🟡 中 | 1周 | Phase 1 |
| 3 | CLI集成 | 🟡 中 | 1周 | Phase 1, 2 |
| 4 | 高级功能 | 🟢 低 | 按需 | Phase 3 |

---

## 7. 与旧项目StreamMUSE的对比

### StreamMUSE（旧）
- ✅ JsonLogHandler能保存推理日志
- ✅ 支持会话目录管理
- ❌ 没有通用的event日志
- ❌ 没有实时性能计算
- ❌ 日志结构不统一

### StreamMUSE-new-sys（规划）
- ✅ 统一的JSONL事件日志
- ✅ 完整的会话管理
- ✅ 实时性能指标计算
- ✅ 模块化的OutputSink架构
- ✅ 支持多种日志格式（JSONL, CSV, JSON, MIDI）
- ✅ Clean Architecture集成

---

## 8. 测试计划

### 单元测试
```python
tests/unit/infrastructure/output/
├── test_json_logger.py
├── test_session_logger.py
└── test_metrics_calculator.py
```

### 集成测试
```python
tests/integration/
└── test_logging_end_to_end.py
```

### 环境测试
```bash
# 生成样本日志
uv run streammuse-cli \
  --input-mode list \ 
  --output-type session \
  --max-ticks 100

# 验证日志生成
python scripts/validate_logs.py logs/session_<id>/
```

---

## 9. 实现待办列表 (TODO List)

### PHASE 1: 核心日志输出适配器 ✅ COMPLETED

#### Foundation: Domain模型和工具类 ✅
- [x] 创建 `src/streammuse/domain/logging/__init__.py`
- [x] 创建 `src/streammuse/domain/logging/event_types.py`
  - [x] 定义 `LogEvent` 数据类 (timestamp, tick, event_type, data)
  - [x] 定义 `InferenceEvent` 数据类
  - [x] 定义 `EventType` 枚举 (note_on, note_off, inference_request, inference_response)
  - [x] 添加dataclass序列化方法 (to_dict, to_json)
- [x] 创建 `src/streammuse/domain/logging/session_manager.py`
  - [x] 实现 `SessionManager` 类初始化 (base_log_dir)
  - [x] 实现 `_generate_session_id()` 方法 (YYYYMMDD-HHMMSS格式)
  - [x] 实现 `create_session_directory()` 方法
  - [x] 实现 `save_config(config: Dict)` 方法
  - [x] 实现 `save_summary(summary: Dict)` 方法
  - [x] 编写单元测试 (tests/unit/domain/logging/test_session_manager.py)

#### Infrastructure: JSON logging适配器 ✅
- [x] 创建 `src/streammuse/infrastructure/output/json_logger.py`
  - [x] 实现 `JsonLoggerOutputSink` 类 (继承OutputProtocol)
  - [x] 实现 `__init__(session_dir: Path)` 初始化
  - [x] 实现 `output_event(event, source, tick)` 方法
    - [x] 转换event到dict格式
    - [x] 追加到events.jsonl
  - [x] 实现 `log_inference(request, response, latency_ms)` 方法
    - [x] 组织推理数据
    - [x] 添加到inferences列表
  - [x] 实现 `close()` 方法
    - [x] 保存inferences.json
    - [x] 关闭文件句柄
  - [x] 编写单元测试 (tests/unit/infrastructure/output/test_json_logger.py)

#### Metrics: 性能计算基础 ✅
- [x] 创建 `src/streammuse/domain/logging/metrics_calculator.py`
  - [x] 实现 `MetricsCalculator` 类初始化
  - [x] 实现 `add_event(event: LogEvent)` 方法
  - [x] 实现 `add_inference(inf: InferenceEvent)` 方法
  - [x] 编写单元测试 (tests/unit/domain/logging/test_metrics_calculator.py)

#### Session Logger: 综合输出处理器 ✅
- [x] 创建 `src/streammuse/infrastructure/output/session_logger.py`
  - [x] 实现 `SessionLoggerOutputSink` 类 (继承OutputProtocol)
  - [x] 组合JsonLoggerOutputSink和MidiFileOutputSink
  - [x] 实现多文件管理逻辑
  - [x] 编写单元测试 (tests/unit/infrastructure/output/test_session_logger.py)

#### Testing (Phase 1) ✅
- [x] 编写集成测试: `tests/integration/test_logging_basic_e2e.py`
  - [x] 测试从事件到JSON日志的完整流程
  - [x] 验证events.jsonl格式
  - [x] 验证inferences.json格式
  - [x] 验证文件编码和换行符

---

### PHASE 2: 性能指标计算 ✅ COMPLETED

#### Core Metrics计算 ✅
- [x] 在 `src/streammuse/domain/logging/metrics_calculator.py` 中实现:
  - [x] `calculate_latency_stats()` 方法
    - [x] 计算 mean, median, std, min, max
    - [x] 计算 p95, p99 百分位数
  - [x] `calculate_event_stats()` 方法
    - [x] 统计用户/模型事件数
    - [x] 计算hit_rate
    - [x] 统计推理请求/响应
  - [x] `calculate_music_stats()` 方法
    - [x] 统计音符数
    - [x] 计算音域 (pitch range)
    - [x] 计算平均velocity
    - [x] 计算总时长

#### Report Generation ✅
- [x] 实现 `generate_performance_json()` 方法
  - [x] 组织session_config部分
  - [x] 组织timing_statistics部分
  - [x] 组织event_statistics部分
  - [x] 组织music_analysis部分
- [x] 实现 `generate_statistics_csv()` 方法
  - [x] 生成CSV格式输出
  - [x] 包含重要指标

#### Persist Metrics ✅
- [x] 修改 `JsonLoggerOutputSink` 或创建新类处理保存:
  - [x] 保存performance.json
  - [x] 保存statistics.csv
- [x] 修改 `SessionLoggerOutputSink`:
  - [x] 调用MetricsCalculator生成报告
  - [x] 保存到会话目录

#### Testing (Phase 2) ✅
- [x] 编写测试: `tests/unit/domain/logging/test_metrics_calculator.py`
  - [x] 测试延迟统计计算
  - [x] 测试事件统计计算
  - [x] 测试音乐统计计算
  - [x] 测试百分位数计算精度
- [x] 编写集成测试: `tests/integration/test_logging_metrics_e2e.py`
  - [x] 测试完整的metrics报告生成

---

### PHASE 3: 输出工厂和CLI集成 ✅ COMPLETED

#### OutputFactory 更新 ✅
- [x] 修改 `src/streammuse/application/factories/output_factory.py`:
  - [x] 添加 "json_log" 类型支持
  - [x] 添加 "session" 类型支持
  - [x] 添加 "composite" 类型对session的支持

#### 项目配置更新 ✅
- [x] 修改或检查 `src/streammuse/config/app_config.py`:
  - [x] 确保OutputConfig支持所有类型
  - [x] 添加LoggingConfig (可选)
    - [x] enable_performance_tracking
    - [x] compress_logs

#### CLI 集成 ✅
- [x] 修改 `src/streammuse/presentation/cli/cli.py`:
  - [x] 添加 `--log-dir` 参数 (default: "logs")
  - [x] 添加 `--enable-performance-tracking` 参数
  - [x] 添加 `--compress-logs` 参数
  - [x] 在main()中实例化SessionManager
    - [x] session_manager = SessionManager(args.log_dir)
    - [x] session_dir = session_manager.create_session_directory()
  - [x] 传递session_manager到OutputFactory.create()
  - [x] 实现atexit cleanup处理:
    - [x] session_manager.save_summary()
    - [x] output_sink.close()

#### RealTimeMusicService 更新 ✅
- [x] 查看 `src/streammuse/application/services/real_time_music_service.py`:
  - [x] 确认source字段正确设置 (user/model)
  - [x] 如需要，添加logging hook点:
    - [x] 在推理前后调用logging
    - [x] 传递request/response数据

#### Testing (Phase 3) ✅
- [x] 编写集成测试: `tests/integration/test_logging_cli_integration.py`
  - [x] 测试CLI参数解析
  - [x] 测试会话目录创建
  - [x] 测试完整运行流程
  - [x] 验证所有日志文件生成
- [x] 手动测试:
  - [x] `uv run streammuse-cli --input-mode keyboard --output-type session`
  - [x] 验证logs/目录结构
  - [x] 检查所有文件格式

---
  - [ ] 测试事件统计计算
  - [ ] 测试音乐统计计算
  - [ ] 测试百分位数计算精度
- [ ] 编写集成测试: `tests/integration/test_logging_metrics_e2e.py`
  - [ ] 测试完整的metrics报告生成

---

### PHASE 3: 输出工厂和CLI集成 (1周)

#### OutputFactory 更新
- [ ] 修改 `src/streammuse/application/factories/output_factory.py`:
  - [ ] 添加 "json_log" 类型支持
  - [ ] 添加 "session" 类型支持
  - [ ] 添加 "composite" 类型对session的支持

#### 项目配置更新
- [ ] 修改或检查 `src/streammuse/config/app_config.py`:
  - [ ] 确保OutputConfig支持所有类型
  - [ ] 添加LoggingConfig (可选)
    - [ ] enable_performance_tracking
    - [ ] compress_logs

#### CLI 集成
- [ ] 修改 `src/streammuse/presentation/cli/cli.py`:
  - [ ] 添加 `--log-dir` 参数 (default: "logs")
  - [ ] 添加 `--enable-performance-tracking` 参数
  - [ ] 添加 `--compress-logs` 参数
  - [ ] 在main()中实例化SessionManager
    - [ ] session_manager = SessionManager(args.log_dir)
    - [ ] session_dir = session_manager.create_session_directory()
  - [ ] 传递session_manager到OutputFactory.create()
  - [ ] 实现atexit cleanup处理:
    - [ ] session_manager.save_summary()
    - [ ] output_sink.close()

#### RealTimeMusicService 更新
- [ ] 查看 `src/streammuse/application/services/real_time_music_service.py`:
  - [ ] 确认source字段正确设置 (user/model)
  - [ ] 如需要，添加logging hook点:
    - [ ] 在推理前后调用logging
    - [ ] 传递request/response数据

#### Testing (Phase 3)
- [ ] 编写集成测试: `tests/integration/test_logging_cli_integration.py`
  - [ ] 测试CLI参数解析
  - [ ] 测试会话目录创建
  - [ ] 测试完整运行流程
  - [ ] 验证所有日志文件生成
- [ ] 手动测试:
  - [ ] `uv run streammuse-cli --input-mode keyboard --output-type session`
  - [ ] 验证logs/目录结构
  - [ ] 检查所有文件格式

---

## 10. 详细任务统计

### 实现进度

**所有Phases已完成✅**

| Phase | 状态 | 完成时间 |
|-------|------|---------|
| PHASE 1: 核心日志输出适配器 | ✅ 完成 | 第1小时 |
| PHASE 2: 性能指标计算 | ✅ 完成 | 第1小时 |
| PHASE 3: 输出工厂和CLI集成 | ✅ 完成 | 第2小时 |

---

## 11. 文档和示例

已实现的功能覆盖：
- `src/streammuse/domain/logging/` - 完整的logging domain模型
- `src/streammuse/infrastructure/output/json_logger.py` - JSON日志适配器
- `src/streammuse/infrastructure/output/session_logger.py` - 会话日志适配器
- `src/streammuse/application/factories/output_factory.py` - 输出工厂集成
- `src/streammuse/presentation/cli/cli.py` - CLI集成

### 快速开始

启动logging系统的三种方式：

**方式1: 完整会话日志（推荐）**
```bash
uv run streammuse-cli --input-mode keyboard --output-type session
# 生成: logs/session_YYYYMMDD-HHMMSS/{events.jsonl, inferences.json, statistics.csv, performance.json, *.mid, ...}
```

**方式2: 仅JSON日志**
```bash
uv run streammuse-cli --input-mode keyboard --output-type json_log
# 生成: logs/session_YYYYMMDD-HHMMSS/{events.jsonl, inferences.json}
```

**方式3: 组合输出（console + session）**
```bash
uv run streammuse-cli --input-mode keyboard --output-type composite --log-dir logs
# 同时输出到终端和日志文件
```

---

## 12. 未来扩展空间

```
✨ 可能的未来增强：

1. 日志可视化
   - 在Web UI中实时显示日志
   - 绘制延迟、事件流等图表

2. 日志查询
   - SQLite数据库后端
   - 按时间/事件类型查询

3. 日志对比
   - 对比多个会话的性能
   - 识别系统改进

4. 日志上传
   - 上传到云存储（S3/Google Drive）
   - 集成分析平台

5. 版本控制
   - Git集成：记录每次运行的代码版本
   - 关联提交信息
```

---

**文档版本**: 1.0  
**最后更新**: 2026年3月11日  
**状态**: ✅ 所有Phases实现完成  
**负责人**: StreamMUSE Development Team

