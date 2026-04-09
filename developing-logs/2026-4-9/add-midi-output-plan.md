# Simulator 直接产出双轨 MIDI 计划

更新时间：2026-04-09
目标：运行 real-time service 后，自动得到一个可播放 MIDI 文件，包含两条轨道：
1. Melody（用户输入）
2. Accompaniment（模型生成）

## 0. 背景与问题定义

当前现象：
1. 运行 simulator 后，session 目录里主要有 `inferences.json` 与 `session_config.json`。
2. 缺少可直接听的 MIDI 产物，不利于快速回放和结果比对。

现有代码事实（As-Is）：
1. 项目已存在 `MidiFileOutputSink`（`infrastructure/output/midi_file.py`），支持 user/model 两轨写入。
2. `SessionLoggerOutputSink`（`infrastructure/output/session_logger.py`）组合了 MIDI + JSON 双写能力，默认写 `combined.mid`。
3. `OutputSinkFactory`（`application/factories/output_factory.py`）根据 `output_type` 分发不同 sink。
4. `CompositeOutputSink`（`infrastructure/output/composite.py`）可将多个 sink 组合，事件广播给所有子 sink。

结论：
核心缺口不是"不会写双轨 MIDI"，而是"默认行为没有自动附加 MIDI 输出"。

## 1. 目标行为（To-Be）

**日志目录结构调整**：
当前结构：`logs/session_YYYYMMDD-HHMMSS/...`
目标结构：`logs/YYYY-MM-DD/session_HHMMSS/...`（外层日期 + 内层纯时间）

动机：
1. 长期运行后，`logs/` 目录下可能有数千个 session 目录，按日期分组便于浏览和归档。
2. 方便按日期批量清理旧日志（直接删除整个日期目录）。
3. 与常见的日志系统（如 Linux logrotate、应用日志）结构保持一致。

**默认行为**：除 `output_type=json_log` 外，系统自动附加一个 `MidiFileOutputSink`，将事件同步写入 MIDI 文件。

具体规则：

| `output_type` | 当前行为 | 修改后行为 |
|---|---|---|
| `console` | 仅终端输出 | 终端输出 + 自动附加 MIDI |
| `audio` | 仅实时播放 | 实时播放 + 自动附加 MIDI |
| `midi_file` | 已有 MIDI 输出 | 不变（用户显式指定了 MIDI 路径） |
| `websocket` | 仅 WebSocket 推送 | WebSocket + 自动附加 MIDI |
| `json_log` | 仅 JSON 日志 | **不变**（纯日志场景，不附加 MIDI） |
| `session` | 已有 JSON + MIDI | 不变（SessionLogger 已包含 MIDI） |
| `composite` | Console + Session | 不变（Session 已包含 MIDI） |

MIDI 输出规格：
1. 两条轨道：`Melody`（source=user）、`Accompaniment`（source=model）
2. BPM 和 ticks_per_beat 从 `TempoConfig` 透传（不硬编码）
3. 默认输出路径：session 目录下 `combined.mid`（需要 `SessionManager`）
4. Session 目录新位置：`logs/<date>/session_xxx/` 而非 `logs/session_xxx/`

## 2. 设计原则

1. 默认体验优先：无需额外参数即可拿到 MIDI。
2. 不破坏现有行为：`json_log`、`session`、`composite`、`midi_file` 保持原逻辑。
3. 轨道语义清晰：使用 `Melody/Accompaniment` 而非 `User/Model`。

## 3. 详细实施计划

### Phase 1：修复 BPM 硬编码 + 工厂自动附加 MIDI

**目标**：让 MIDI 输出使用正确的 BPM，并让工厂自动为需要的 output_type 附加 MIDI sink。

#### Task 1.1：修复 SessionLoggerOutputSink 的 BPM 硬编码

**问题**：`session_logger.py:30-33` 硬编码 `bpm=120.0, ticks_per_beat=4`。如果用户 MIDI 是 110 BPM，输出时值会偏快 ~9%。

- [ ] **1.1.1** 修改 `SessionLoggerOutputSink.__init__` 签名，新增 `bpm: float = 120.0` 和 `ticks_per_beat: int = 4` 参数
- [ ] **1.1.2** 将硬编码的 `MidiFileOutputConfig(bpm=120.0, ticks_per_beat=4, ...)` 改为使用传入参数
- [ ] **1.1.3** 更新 `OutputSinkFactory` 中所有创建 `SessionLoggerOutputSink` 的地方（`session` 和 `composite` 分支），传入 `tempo.bpm` 和 `tempo.ticks_per_beat`
- [ ] **1.1.4** 验证：用 110 BPM 的 MIDI 文件跑 simulator，`combined.mid` 的 tempo header 应为 110

涉及文件：
- `src/streammuse/infrastructure/output/session_logger.py`
- `src/streammuse/application/factories/output_factory.py`

#### Task 1.0：Session 目录按日期分组（前置修改）

**目标**：修改 `SessionManager`，使其创建的 session 目录包含日期层级，并简化 session 命名。

**命名变更**：
- 旧：`session_20260409-143052`（含完整日期时间）
- 新：`session_143052`（仅时间，外层目录已有日期）

**实施细节**：
- [ ] **1.0.1** 修改 `SessionManager` 的 `__init__` 和 `_generate_session_id()`：
  - `session_id` 改为仅时间格式：`"%H%M%S"`（如 `143052`）
  - 保留完整时间戳用于其他用途（如 `session_timestamp`）
- [ ] **1.0.2** 修改 `create_session_directory()` 方法：
  - 获取当前日期：`date_str = datetime.now().strftime("%Y-%m-%d")`（如 `2026-04-09`）
  - 在 `log_dir` 下创建日期目录：`date_dir = log_dir / date_str`
  - 在日期目录下创建 session 目录：`session_dir = date_dir / f"session_{self.session_id}"`
- [ ] **1.0.3** 更新 `SessionManager` 的 `get_session_dir()` 返回新的路径
- [ ] **1.0.4** 确保日期目录不存在时自动创建（`mkdir(parents=True, exist_ok=True)`）
- [ ] **1.0.5** 向后兼容：如果旧路径的 session 存在，仍然可以读取（resume 场景）

**涉及文件**：
- `src/streammuse/domain/logging/session_manager.py`

**验证**：
- [ ] P0-E1 运行后 session 目录为 `logs/2026-04-09/session_143052/`
- [ ] P0-E2 `session_manager.get_session_dir()` 返回的路径包含日期层级
- [ ] P0-E3 session 目录名仅包含时间（如 `session_143052`，不含日期前缀）

---

#### Task 1.2：工厂自动附加 MIDI sink

**核心逻辑**：在 `OutputSinkFactory.create()` 中，对于 `console`、`audio`、`websocket` 三种 output_type，用 `CompositeOutputSink` 将原 sink 与一个 `MidiFileOutputSink` 组合。

**关键**：工厂需要读 `app_config.input.type` 来判断 input 模式——这不需要改签名，因为 `app_config` 已包含 `input` 配置。

- [ ] **1.2.1** 在 `OutputSinkFactory.create()` 中，对 `console`/`audio`/`websocket` 分支，在返回前检查是否需要附加 MIDI：
  ```python
  # 需要自动 MIDI 的条件：
  # 1. output_type 不是 json_log / session / composite / midi_file（这些要么已有 MIDI，要么是纯日志）
  # 2. session_manager 存在（需要有目录可写）
  ```
- [ ] **1.2.2** 如果满足条件，创建 `MidiFileOutputSink`（使用 `TempoConfig` 的 bpm/ticks_per_beat），用 `CompositeOutputSink` 包装
- [ ] **1.2.3** MIDI 输出路径：`session_manager.get_session_dir() / "combined.mid"`（注意：现在路径已经是 `logs/<date>/session_xxx/`）

**注意**：这要求使用 `console`/`audio`/`websocket` 时也创建 `SessionManager`。当前 `cli.py:34` 只在 `output_type in ["json_log", "session", "composite"]` 时创建 session_manager。需要扩展这个条件。

- [ ] **1.2.4** 修改 `cli.py:34`：将 session_manager 创建条件从特定 output_type 改为**始终创建**（或排除 `midi_file` 和 `json_log`）
  ```python
  # 改为：除了 json_log 和 midi_file，都创建 session_manager
  if config.output.type not in ["json_log", "midi_file"]:
      session_manager = SessionManager(args.log_dir)
      session_manager.create_session_directory()
      ...
  ```
  **等等——`json_log` 本身也需要 session_manager**（当前逻辑就是这样）。所以更准确的改法是：
  ```python
  # 原来：只在 json_log/session/composite 时创建
  # 改为：除了 midi_file 以外都创建（因为其他 type 要么需要 JSON 日志，要么需要自动 MIDI）
  if config.output.type != "midi_file":
      session_manager = SessionManager(args.log_dir)
      ...
  ```
- [ ] **1.2.5** 验证 `json_log` 路径不受影响：`output_type=json_log` 时工厂仍返回 `JsonLoggerOutputSink`（不附加 MIDI）

**cli.py cleanup 兼容性注意**：`cli.py:72-73` 的 `isinstance(output_sink, SessionLoggerOutputSink)` 检查在新逻辑下仍然正确——自动附加 MIDI 用的是 `CompositeOutputSink` 包 `MidiFileOutputSink`，不是 `SessionLoggerOutputSink`，所以不会触发 `save_metrics`。只有 `session` 和 `composite` 分支才会触发。这是期望行为，无需修改。

涉及文件：
- `src/streammuse/application/factories/output_factory.py`
- `src/streammuse/presentation/cli/cli.py`

#### Phase 1 验收

- [ ] P1-E1 `output_type=console` + `input_mode=midi_file` 运行后，session 目录出现 `combined.mid`
- [ ] P1-E2 `output_type=json_log` 运行后，行为不变（无 `combined.mid`）
- [ ] P1-E3 `output_type=session` 运行后，`combined.mid` 的 BPM 与输入文件一致（不再硬编码 120）
- [ ] P1-E4 `output_type=composite` 运行后，行为不变

---

### Phase 2：轨道名语义化

**目标**：MIDI 轨道名从 `User/Model` 改为 `Melody/Accompaniment`。

- [ ] **2.1** 修改 `MidiFileOutputConfig` 的默认值：
  - `user_track_name: str = "User"` → `"Melody"`
  - `model_track_name: str = "Model"` → `"Accompaniment"`
- [ ] **2.2** 这是**全局默认值修改**，会影响所有使用 `MidiFileOutputConfig` 的场景（`midi_file`、`session`、`composite`、自动附加）。这是期望行为，因为 `Melody/Accompaniment` 比 `User/Model` 对所有场景都更直观。
- [ ] **2.3** 验证：导出的 MIDI 用 DAW 或 `pretty_midi` 打开，确认两条轨道分别命名为 `Melody` 和 `Accompaniment`

涉及文件：
- `src/streammuse/infrastructure/output/midi_file.py`（仅改默认值，第 21-22 行）

#### Phase 2 验收

- [ ] P2-E1 所有输出 MIDI 的路径（`session`、`composite`、`midi_file`、自动附加）轨道名均为 `Melody/Accompaniment`

---

### Phase 3：测试

**目标**：防止回归，确保功能可持续。

#### Task 3.1：工厂测试

- [ ] **3.1.1** 测试 `output_type=console` + session_manager 存在 → 返回 `CompositeOutputSink`（包含原 sink + MIDI sink）
- [ ] **3.1.2** 测试 `output_type=audio` + session_manager 存在 → 同上
- [ ] **3.1.3** 测试 `output_type=json_log` → 返回 `JsonLoggerOutputSink`（不附加 MIDI）
- [ ] **3.1.4** 测试 `output_type=session` → 返回 `SessionLoggerOutputSink`（已有 MIDI，不重复附加）
- [ ] **3.1.5** 测试 `output_type=console` + session_manager 为 None → 返回原 sink（无 MIDI，因为没有输出目录）

涉及文件：
- `tests/unit/application/factories/test_output_factory.py`

#### Task 3.2：BPM 透传测试

- [ ] **3.2.1** 测试 `SessionLoggerOutputSink` 使用传入的 bpm（非默认 120）
- [ ] **3.2.2** 测试自动附加的 `MidiFileOutputSink` 使用 `TempoConfig` 的 bpm

#### Task 3.3：集成测试

- [ ] **3.3.1** 短 `max_ticks` 运行后，session 目录包含 `combined.mid`
- [ ] **3.3.2** `combined.mid` 包含两轨且都有事件
- [ ] **3.3.3** 轨道名为 `Melody` 和 `Accompaniment`

涉及文件：
- `tests/integration/test_simulator_midi_output.py`

#### Phase 3 验收

- [ ] P3-E1 所有新增测试通过
- [ ] P3-E2 原有测试无回归

---

### Phase 4：文档

**目标**：用户知道 MIDI 产物的存在和位置。

- [ ] **4.1** 更新 `docs/user-guide/running-realtime.md`：说明默认 MIDI 输出行为和输出位置
- [ ] **4.2** 更新 `docs/reference/cli-reference.md`：在 output-type 表格中标注哪些 type 会自动产出 MIDI
- [ ] **4.3** 更新 CLI `--help` 中 `--output-type` 的 help 文案，提示默认 MIDI 行为

涉及文件：
- `docs/user-guide/running-realtime.md`
- `docs/reference/cli-reference.md`
- `src/streammuse/presentation/cli/config_parser.py`（help 文案）

#### Phase 4 验收

- [ ] P4-E1 文档描述与实际行为一致

---

## 4. 验证命令（实施后）

### 4.1 Console + 自动 MIDI

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --model-name lekai \
  --output-type console \
  --max-ticks 256
```

期望：
1. 终端有事件输出。
2. `logs/2026-04-09/session_143052/combined.mid` 存在且包含两条轨道（路径含日期层级，session 名简化为纯时间）。

### 4.2 json_log（不附加 MIDI）

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --model-name lekai \
  --output-type json_log \
  --max-ticks 256
```

期望：
1. session 目录有 JSON 文件。
2. **没有** `combined.mid`。

### 4.3 Session（已有 MIDI，BPM 正确）

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --model-name lekai \
  --output-type session \
  --max-ticks 256
```

期望：
1. `combined.mid` 存在。
2. MIDI 文件的 tempo header 为 110 BPM（与输入文件一致，不是 120）。

## 5. 修改文件总览

| 文件 | Phase | 修改内容 |
|------|-------|---------|
| `domain/logging/session_manager.py` | 1.0 | 修改 `create_session_directory()` 添加日期层级 |
| `infrastructure/output/session_logger.py` | 1 | 新增 bpm/ticks_per_beat 参数，替换硬编码 |
| `application/factories/output_factory.py` | 1 | 自动附加 MIDI sink 逻辑 + 透传 tempo |
| `presentation/cli/cli.py` | 1 | 扩展 session_manager 创建条件 |
| `infrastructure/output/midi_file.py` | 2 | 默认轨道名改为 Melody/Accompaniment |
| `tests/unit/.../test_output_factory.py` | 3 | 新增/扩展 |
| `tests/integration/test_simulator_midi_output.py` | 3 | 新增 |
| `docs/user-guide/running-realtime.md` | 4 | 更新 |
| `docs/reference/cli-reference.md` | 4 | 更新 |
| `presentation/cli/config_parser.py` | 4 | help 文案 |

## 6. 风险与回滚

风险：
1. 扩展 session_manager 创建条件后，原本不产生日志目录的 output_type（console/audio）现在会创建 `logs/<date>/session_xxx/` 目录。这是**可接受的副作用**（目录里只有 `combined.mid` 和 `session_config.json`），但用户可能对多出的目录感到意外。
2. 全局改轨道名会影响所有 MIDI 输出场景的轨道名。
3. **日期目录变更的兼容性**：旧 session 恢复时可能期望 `logs/session_xxx/` 路径。需要确保 resume 逻辑能正确找到旧路径，或者明确说明这是破坏性变更（旧 session 不再 resume，只能导出备份）。

回滚策略：
1. 工厂的自动附加逻辑集中在一个 if 分支，删除即可恢复。
2. 轨道名改的是默认值，恢复默认值即可。

## 7. Definition of Done

全部满足才算完成：
1. `console`/`audio`/`websocket` output_type 运行后，自动产出 `combined.mid`。
2. `json_log` 行为不变（无 MIDI）。
3. `session`/`composite` 行为不变，但 BPM 现在正确透传。
4. MIDI 轨道名为 `Melody/Accompaniment`。
5. 测试通过，文档同步。
