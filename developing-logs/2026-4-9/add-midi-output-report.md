# Simulator 直接产出双轨 MIDI 实施报告

更新时间：2026-04-09  
对应计划：`developing-logs/2026-4-9/add-midi-output-plan.md`  
执行范围：Phase 1.0 / 1 / 2 / 3 / 4

---

## 1. 执行摘要

本次实现已完成计划中的核心目标：

1. 默认输出增强：`console` / `audio` / `websocket` 现会自动附加 `MidiFileOutputSink`，输出 `combined.mid`。
2. 会话目录重构：`SessionManager` 目录结构改为 `logs/YYYY-MM-DD/session_HHMMSS/`。
3. BPM 透传修复：`SessionLoggerOutputSink` 不再硬编码 `120/4`，改为使用 `TempoConfig`。
4. MIDI 轨道语义化：默认轨道名改为 `Melody` / `Accompaniment`。
5. 回归测试与文档同步：新增单元/集成测试，更新 CLI help 与用户文档。

最终测试状态：

- `uv run pytest tests/ -q`：`160 passed, 1 warning`

---

## 2. 代码变更总览

### 2.1 核心实现文件

- `src/streammuse/domain/logging/session_manager.py`
- `src/streammuse/infrastructure/output/session_logger.py`
- `src/streammuse/application/factories/output_factory.py`
- `src/streammuse/presentation/cli/cli.py`
- `src/streammuse/infrastructure/output/midi_file.py`
- `src/streammuse/presentation/cli/config_parser.py`

### 2.2 新增测试文件

- `tests/unit/application/factories/test_output_factory.py`
- `tests/unit/domain/logging/test_session_manager.py`
- `tests/integration/test_simulator_midi_output.py`

### 2.3 调整测试文件

- `tests/integration/test_cli_entry_point.py`（补充 `log_dir` mock 参数以适配新 CLI 行为）

### 2.4 文档更新文件

- `docs/user-guide/running-realtime.md`
- `docs/reference/cli-reference.md`
- `docs/user-guide/output-types.md`（额外补齐一致性，避免与新行为冲突）

---

## 3. 按计划逐项完成情况

## Phase 1.0：Session 目录按日期分组

### Task 1.0.1 / 1.0.2 / 1.0.3 / 1.0.4

已完成，`SessionManager` 行为如下：

1. `session_id` 改为 `%H%M%S`（如 `143052`）。
2. 新目录结构为 `base_log_dir / YYYY-MM-DD / session_<session_id>`。
3. `create_session_directory()` 使用 `mkdir(parents=True, exist_ok=True)`。
4. `get_session_dir()` 返回新的日期分层目录。

实现位置：

- `session_timestamp/date_str/session_id` 初始化逻辑
- `create_session_directory()`
- `get_session_dir()`

### Task 1.0.5（向后兼容 legacy 路径）

已完成。

策略：

1. 在 `SessionManager` 内维护 `_legacy_session_dir = base/session_<session_id>`。
2. 若 legacy 目录存在且新目录不存在，则优先复用 legacy 目录。
3. `get_session_dir()` 也包含 fallback 判断。

---

## Phase 1：BPM 修复 + 自动附加 MIDI

### Task 1.1.1 / 1.1.2

已完成。

`SessionLoggerOutputSink.__init__` 新增参数：

- `bpm: float = 120.0`
- `ticks_per_beat: int = 4`

并将原先硬编码：

- `MidiFileOutputConfig(bpm=120.0, ticks_per_beat=4, ...)`

改为：

- `MidiFileOutputConfig(bpm=float(bpm), ticks_per_beat=int(ticks_per_beat), ...)`

### Task 1.1.3

已完成。

`OutputSinkFactory` 的 `session` 和 `composite` 分支均已传入：

- `bpm=float(tempo.bpm)`
- `ticks_per_beat=int(tempo.ticks_per_beat)`

### Task 1.2.1 / 1.2.2 / 1.2.3

已完成。

在 `OutputSinkFactory` 新增 `_attach_auto_midi_if_needed(...)`：

1. 当 `session_manager is None` 时，不附加 MIDI。
2. 当 `session_manager` 存在时，创建 `MidiFileOutputSink(MidiFileOutputConfig(...combined.mid))`。
3. 返回 `CompositeOutputSink([base_sink, auto_midi_sink])`。

并应用到：

- `console`
- `audio`
- `websocket`

### Task 1.2.4（CLI 创建 session_manager 条件）

已完成。

`cli.py` 已从：

- `if output.type in ["json_log", "session", "composite"]`

调整为：

- `if output.type != "midi_file"`

即：除 `midi_file` 外都会创建 session 目录并写 `session_config.json`。

### Task 1.2.5（json_log 行为不变）

已完成。

`OutputSinkFactory` 的 `json_log` 分支仍直接返回 `JsonLoggerOutputSink`，不会附加 MIDI sink。

---

## Phase 2：轨道名语义化

### Task 2.1 / 2.2

已完成。

`MidiFileOutputConfig` 默认值修改：

- `user_track_name: "User" -> "Melody"`
- `model_track_name: "Model" -> "Accompaniment"`

该修改是全局默认值，覆盖 `midi_file/session/composite/自动附加` 所有路径。

### Task 2.3

已通过自动化集成测试验证（见 Phase 3 测试结果）：

1. 生成的 MIDI 包含两轨。
2. 轨道名为 `Melody` 和 `Accompaniment`。

---

## Phase 3：测试实现

### Task 3.1（工厂测试）

已完成，新增 `tests/unit/application/factories/test_output_factory.py`：

1. `console + session_manager` -> `CompositeOutputSink(Console + MidiFile)`。
2. `audio + session_manager` -> `CompositeOutputSink(Audio + MidiFile)`。
3. `json_log` -> `JsonLoggerOutputSink`（不附加 MIDI）。
4. `session` -> `SessionLoggerOutputSink`（不重复附加 MIDI）。
5. `console + session_manager=None` -> 仅 `ConsoleOutputSink`。

### Task 3.2（BPM 透传测试）

已完成：

1. 单测验证 `SessionLoggerOutputSink` 使用传入的 `bpm/ticks_per_beat`。
2. 单测验证自动附加 `MidiFileOutputSink` 的配置来自 `TempoConfig`。

### Task 3.3（集成测试）

已完成，新增 `tests/integration/test_simulator_midi_output.py`：

1. 短 `max_ticks` 运行后，session 目录存在 `combined.mid`。
2. `combined.mid` 包含两轨且两轨均有事件。
3. 轨道名为 `Melody` 与 `Accompaniment`。
4. 额外校验 tempo header 约等于 110 BPM（容忍 `pretty_midi` 序列化误差，阈值 `< 0.01`）。

---

## Phase 4：文档与 help

### Task 4.1

已完成，`running-realtime.md` 新增“默认 MIDI 产物与日志目录结构”说明，明确：

1. 新目录结构 `logs/YYYY-MM-DD/session_HHMMSS/`。
2. 各 `output_type` 的 MIDI 产物行为。

### Task 4.2

已完成，`cli-reference.md`：

1. 更新 `--output-type` 描述，标注自动 MIDI。
2. 更新 `--log-dir` 描述，标注日期分层目录。
3. 新增输出类型与 `combined.mid` 对照表。

### Task 4.3

已完成，`config_parser.py` 的 `--output-type` help 文案已更新，提示自动 MIDI 行为与 `json_log` 例外。

---

## 4. 验证记录

执行过的关键验证：

1. 针对新增/受影响测试：
   - `uv run pytest tests/unit/application/factories/test_output_factory.py tests/unit/domain/logging/test_session_manager.py tests/integration/test_simulator_midi_output.py tests/integration/test_cli_entry_point.py -q`
   - 结果：`10 passed`
2. 全量回归：
   - `uv run pytest tests/ -q`
   - 结果：`160 passed, 1 warning`

说明：

- `warning` 来自 `pretty_midi` 依赖中 `pkg_resources` 弃用提示，不是本次功能回归。

---

## 5. 与原计划的对照结论

按计划定义的核心 DoD 判定：

1. `console/audio/websocket` 自动产出 `combined.mid`：✅ 已实现 + 单测覆盖
2. `json_log` 行为不变（无 MIDI）：✅ 已实现 + 单测覆盖
3. `session/composite` 行为不变且 BPM 透传：✅ 已实现 + 单测覆盖
4. 轨道名为 `Melody/Accompaniment`：✅ 已实现 + 集成测试覆盖
5. 测试通过、文档同步：✅ 已完成

---

## 6. 额外说明

1. 本次额外同步了 `docs/user-guide/output-types.md`，用于消除旧文档中的 `User/Model` 与“console 不产出 MIDI”等过时描述。
2. 计划中建议的手工命令（4.1/4.2/4.3）对应行为已通过自动化测试验证；若需要，可再执行一次真实 CLI 流程做人工验收截图。
