# Detailed Inference Logging Plan

## 1. 背景与目标

当前 `inferences.json` 记录的是**摘要信息**（如 `melody_notes_count`、`accompaniment_notes_count`），不包含完整推理请求/响应内容。

本计划目标：引入可控的“详细日志模式（full）”，在需要排查时记录完整 `request/response`，默认仍保持轻量摘要模式，避免性能与磁盘压力。

---

## 2. 需求定义

### 2.1 功能需求

- 新增日志粒度开关（至少两档）：
  - `summary`（默认，兼容当前行为）
  - `full`（记录完整 request/response）
- 在 `full` 模式下，`inferences.json` 中每条推理记录需包含：
  - 完整请求关键信息：
    - `generation_start_tick`
    - `generation_length_frames`
    - `prompt_length_ticks`（若有）
    - `generation_interval_ticks`
    - `model_name`
    - `inference_mode`
    - `melody_notes`（完整事件列表）
  - 完整响应关键信息：
    - `accompaniment`（完整事件列表）
    - `timings`（完整时延字段）
- `summary` 模式下保持当前输出结构（只记录计数与核心指标），保证向后兼容。

### 2.2 非功能需求

- 默认行为不变（老命令不需要修改）。
- 不改变现有输出类型语义（`console/json_log/session/composite`）。
- 保持 Clean Architecture 边界，不让 Application 层依赖 Infrastructure 序列化工具。
- 对大日志风险提供治理策略（见第 7 节）。

---

## 3. 设计方案（高层）

### 3.1 配置入口

在 CLI 增加参数：

- `--inference-log-detail`：`summary | full`，默认 `summary`

配置流：

`CLI args -> ApplicationConfig(OutputConfig) -> OutputSinkFactory -> JsonLoggerOutputSink/SessionLoggerOutputSink`

### 3.2 日志写入职责

- `RealTimeMusicService._inference_worker` 负责收集“本次推理上下文”：
  - 请求输入（melody snapshot + generation args）
  - 响应输出（acc events + timing_info）
- Sink 只负责“按 detail 级别落盘”，不再推断业务字段。

### 3.3 schema 策略

同一个 `inferences.json`，通过 `request_data/response_data` 内容差异表示模式：

- `summary`：保持当前字段集合
- `full`：扩展为完整事件数组 + timings

不新增第二个文件（如 `inferences_full.json`），以减少维护复杂度；通过 `request_data.log_detail` 标识当前模式。

---

## 4. 逐文件实施计划

## Phase 1 — 配置与参数打通

### 4.1 `src/streammuse/application/config/models.py`

- 在 `OutputConfig` 新增字段：
  - `inference_log_detail: Literal["summary", "full"] = "summary"`

### 4.2 `src/streammuse/presentation/cli/config_parser.py`

- 新增 CLI 参数：
  - `--inference-log-detail`（choices: `summary`, `full`）
- 在 `args_to_config()` 中将参数映射到 `OutputConfig.inference_log_detail`
- 更新 `--help` 文案：说明 `full` 会显著增加日志体积

### 4.3 `src/streammuse/application/factories/output_factory.py`

- 创建 `JsonLoggerOutputSink` / `SessionLoggerOutputSink` 时传递 `inference_log_detail`
- 保证未传时默认 `summary`

---

## Phase 2 — Sink 能力扩展

### 4.4 `src/streammuse/infrastructure/output/json_logger.py`

- 构造函数新增：
  - `inference_log_detail: str = "summary"`
- `log_inference()` 增强：
  - 接受并记录完整字典（已支持）
  - 追加统一元字段：
    - `request_data.log_detail`
  - 当 `summary` 模式时，只保留摘要字段（当前逻辑）
  - 当 `full` 模式时，保留完整 `request/response`（包括事件列表）

### 4.5 `src/streammuse/infrastructure/output/session_logger.py`

- 构造函数新增同名参数并透传给 `JsonLoggerOutputSink`
- 保持 `SessionLoggerOutputSink.log_inference()` API 不变

### 4.6 `src/streammuse/infrastructure/output/composite.py`

- 无结构改动，保持 fan-out；仅确认 `log_inference` 透传链不丢字段

---

## Phase 3 — Application 层构造完整 payload

### 4.7 `src/streammuse/application/services/real_time_music_service.py`

在 `_inference_worker` 中重构 `request_data/response_data` 构造逻辑：

- 统一先构造“完整数据对象”：
  - request_full:
    - `timestamp`
    - `generation_start_tick`
    - `generation_length_frames`
    - `melody_notes_count`
    - `melody_notes`（完整事件列表）
  - response_full:
    - `timestamp`
    - `accompaniment_notes_count`
    - `accompaniment`（完整事件列表）
    - `timings`（从 `timing_info` 映射）
- 再根据 sink detail 级别选择：
  - `summary`：降采样为当前字段
  - `full`：写入完整对象

> 实现方式建议：
> - 新增私有 helper：
>   - `_event_to_log_dict(ev: MusicalEvent) -> dict`
>   - `_timing_to_log_dict(t: TimingInfo) -> dict`
> - 不依赖 infrastructure 层 `event_to_dict`，保持架构边界。

### 4.8 级别读取方式

为避免在 Application 层硬耦合具体 sink 类型：

- 在 sink 侧约定可选属性：`inference_log_detail`
- service 侧通过 `getattr(self._output, "inference_log_detail", "summary")` 读取
- 若是 `CompositeOutputSink`，可在其上暴露一个聚合属性（取第一个支持该属性的子 sink）

---

## Phase 4 — 文档与可观测性说明

### 4.9 文档更新

- `docs/reference/cli-reference.md`
  - 增加 `--inference-log-detail`
  - 给出 `summary` 与 `full` 的样例对比
- `docs/architecture/infrastructure/inference/overview.md`（或 logging 相关文档）
  - 增加推理日志粒度说明与文件体积影响提示

---

## 5. 日志字段设计（建议）

`inferences.json` 每条记录结构维持：

```json
{
  "id": "inf_001",
  "timestamp_request": 1773866874.67585,
  "timestamp_response": 1773866874.71901,
  "request_data": { ... },
  "response_data": { ... },
  "latency_ms": 42.47,
  "server_process_ms": 12.34
}
```

`request_data`/`response_data` 在 `full` 下建议最少包含：

- `request_data.log_detail = "full"`
- `request_data.melody_notes`（完整 note_on/off 事件）
- `response_data.accompaniment`（完整 note_on/off 事件）
- `response_data.timings`（6 个 timing 字段）

---

## 6. 测试计划

## 6.1 单元测试

### A. Service 层

新增/扩展测试：

- `summary` 模式：`request_data` 不含完整事件数组
- `full` 模式：`request_data.melody_notes`、`response_data.accompaniment` 存在且非空（有输入时）

### B. JsonLoggerOutputSink

新增测试：

- `log_inference()` 在 `full` 模式下保持完整字典不丢字段
- `save_inferences()` 输出 JSON 可被 `json.load()` 正确解析

### C. Composite/Session 透传

- `composite + session` 场景下，`log_inference` 保留 `full` 内容

## 6.2 集成验证（手工）

命令：

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/manual/nyan_cat.mid \
  --output-type composite \
  --log-dir logs \
  --inference-log-detail full \
  --max-ticks 128
```

验收点：

- `logs/session_*/inferences.json` 出现完整 `melody_notes` 与 `accompaniment`
- `summary` 模式下文件大小显著更小

---

## 7. 风险与治理

### 风险 1：日志体积膨胀

- 表现：长会话下 `inferences.json` 变得很大
- 对策：
  - 默认 `summary`
  - 文档明确 `full` 仅用于排障
  - 可选后续增强：`--inference-log-max-events`（仅在 full 下裁剪每次事件数）

### 风险 2：性能影响

- 表现：JSON 序列化与内存积累导致延迟抖动
- 对策：
  - 保持默认 summary
  - full 模式只在调试时开启
  - 可选后续增强：`inferences.jsonl` 流式写入（避免全量驻留内存）

### 风险 3：敏感信息泄露（未来接入真实用户数据）

- 对策：
  - full 模式仅本地调试使用
  - 后续可加字段白名单/脱敏策略

---

## 8. 交付分批建议

- PR-1：配置参数与 sink 透传（不改业务字段）
- PR-2：service 构造 full payload + summary/full 分支
- PR-3：测试与文档补齐

这样可以确保每一步都可回归、可回滚。

---

## 9. Definition of Done

- 新参数 `--inference-log-detail` 可用，默认 `summary`
- `summary` 行为与现网一致（兼容旧脚本）
- `full` 模式下 `inferences.json` 包含完整 request/response 事件数据
- 单元测试和关键集成验证通过
- CLI 与架构文档同步更新

---

## 10. Detailed To-Do List (Execution Checklist)

> 说明：以下为执行清单，当前仅规划，不代表已实现。

### Phase 0 — Baseline Lock & Scope Confirmation

- [x] D0-1 冻结当前日志行为（确认 `summary` 现状字段集合）
- [x] D0-2 明确 `full` 最小字段集合（request/response/timings）
- [x] D0-3 明确兼容策略：默认 `summary` 且老命令无感知
- [x] D0-4 明确不做项：本轮不引入新日志文件、不改输出类型语义

**Exit criteria**
- [x] D0-E1 形成可执行字段对照表（summary vs full）

### Phase 1 — Config & CLI Plumbing

- [x] D1-1 在 `OutputConfig` 增加 `inference_log_detail` 字段与类型约束
- [x] D1-2 在 `config_parser.py` 添加 `--inference-log-detail` 参数
- [x] D1-3 在 `args_to_config()` 完成参数映射
- [x] D1-4 更新 CLI help 文案（标明 full 模式体积影响）
- [x] D1-5 补充/更新参数解析相关单元测试

**Exit criteria**
- [x] D1-E1 CLI 能正确解析并传递 `summary/full` 到 `ApplicationConfig`

### Phase 2 — Output Factory Wiring

- [x] D2-1 `OutputSinkFactory` 在 `json_log` 分支透传 `inference_log_detail`
- [x] D2-2 `OutputSinkFactory` 在 `session` 分支透传 `inference_log_detail`
- [x] D2-3 `OutputSinkFactory` 在 `composite` + session 分支透传 `inference_log_detail`
- [x] D2-4 保证未设置时回退到默认 `summary`
- [x] D2-5 补充工厂测试覆盖透传路径

**Exit criteria**
- [x] D2-E1 所有含 JSON 日志的输出类型都能收到 detail 配置

### Phase 3 — Sink Detail Mode Support

- [x] D3-1 `JsonLoggerOutputSink.__init__` 增加 `inference_log_detail` 参数
- [x] D3-2 `JsonLoggerOutputSink` 暴露可读取属性 `inference_log_detail`
- [x] D3-3 `log_inference()` 为写入记录追加 `request_data.log_detail`
- [x] D3-4 `summary` 模式下保持当前结构与字段（兼容）
- [x] D3-5 `full` 模式下完整保留 request/response 字典
- [x] D3-6 `SessionLoggerOutputSink` 增加参数并透传到 `JsonLoggerOutputSink`
- [x] D3-7 `CompositeOutputSink` 提供 detail 级别读取能力（聚合子 sink）

**Exit criteria**
- [x] D3-E1 sink 层可稳定区分并落盘 `summary/full` 两种粒度

### Phase 4 — Service Full Payload Construction

- [x] D4-1 在 `RealTimeMusicService` 增加 `_event_to_log_dict()` helper
- [x] D4-2 在 `RealTimeMusicService` 增加 `_timing_to_log_dict()` helper
- [x] D4-3 构建 `request_full`（包含完整 `melody_notes` 与推理参数）
- [x] D4-4 构建 `response_full`（包含完整 `accompaniment` 与 `timings`）
- [x] D4-5 读取 detail 级别并在 `summary/full` 间选择写入内容
- [x] D4-6 为 request/response 显式写入时间戳字段（避免默认回填）

**Exit criteria**
- [x] D4-E1 `full` 模式下每条 inference 都含完整 request/response 内容

### Phase 5 — Schema Validation & Backward Compatibility

- [x] D5-1 验证 `inferences.json` 顶层结构保持不变（id/timestamp/latency 等）
- [x] D5-2 验证 `summary` 文件能被现有分析逻辑消费
- [x] D5-3 验证 `full` 仅扩展字段，不破坏现有字段含义
- [x] D5-4 增加 JSON 序列化安全检查（事件字段可序列化）

**Exit criteria**
- [x] D5-E1 兼容旧脚本且支持新字段扩展

### Phase 6 — Unit Tests

- [x] D6-1 新增/更新 service 测试：`summary` 不包含完整事件数组
- [x] D6-2 新增/更新 service 测试：`full` 包含完整 `melody_notes/accompaniment`
- [x] D6-3 新增 `json_logger` 测试：`log_detail` 标记正确
- [x] D6-4 新增 `json_logger` 测试：`full` 模式不丢字段
- [x] D6-5 新增 `output_factory` 测试：detail 参数透传
- [x] D6-6 新增 `composite/session` 测试：多 sink 场景 detail 一致

**Exit criteria**
- [x] D6-E1 相关单测全部通过且覆盖关键分支

### Phase 7 — Manual Validation

- [x] D7-1 跑 `summary` 命令，验证 `inferences.json` 为摘要模式
- [x] D7-2 跑 `full` 命令，验证 `inferences.json` 包含完整 request/response
- [x] D7-3 对比两种模式文件大小与可读性
- [x] D7-4 检查 `composite` 场景下 session 目录产物完整

**Exit criteria**
- [x] D7-E1 手工验证通过并可稳定复现

### Phase 8 — Documentation Update

- [x] D8-1 更新 `docs/reference/cli-reference.md` 参数说明与示例
- [x] D8-2 更新日志相关架构文档（summary/full 语义）
- [x] D8-3 添加排障说明：何时使用 full、如何回退 summary

**Exit criteria**
- [x] D8-E1 文档可指导新成员独立完成调试与验证

### Phase 9 — Final Readiness & Rollback Notes

- [x] D9-1 汇总变更清单与影响面
- [x] D9-2 给出回滚策略（仅切回 summary/移除参数）
- [x] D9-3 最终回归关键命令（midi_file + composite）

**Exit criteria**
- [x] D9-E1 满足第 9 节 Definition of Done
