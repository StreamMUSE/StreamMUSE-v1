# Clean Test Plan

## 1. 当前测试现状

- 当前可收集测试总数: 124
- 测试文件总数: 26
- 当前基线状态: 全部通过
- 测试结构:
  - integration: 1 个文件
  - unit/application: 3 个文件
  - unit/presentation: 1 个文件
  - unit/domain: 7 个文件
  - unit/infrastructure: 14 个文件

按用例数量看，主要集中在 domain 层，infrastructure 和 application 侧偏轻量。

## 2. 过时测试候选

以下测试不是错误，但价值偏低或与当前架构目标不完全匹配，建议清理。

### 2.1 低价值导入烟雾测试

候选文件:
- tests/unit/infrastructure/test_config_imports.py
- tests/unit/infrastructure/test_tokenization_imports.py

问题:
- 主要断言是 not None 或 hasattr，回归信号弱
- 对业务行为几乎无约束

建议:
- 保留 1 个统一的基础导入 smoke test 即可
- 将其余导入测试替换为行为级测试或删除

### 2.2 低价值协议存在性测试

候选文件:
- tests/unit/infrastructure/input/test_midi_device.py
- tests/unit/infrastructure/input/test_list_input.py
- tests/unit/infrastructure/input/test_keyboard_input.py

问题:
- 多个文件重复测试 has read_events/close
- 断言偏接口存在性，不关注行为差异

建议:
- 合并为 1 个参数化协议一致性测试
- 各输入源文件保留行为测试，不再重复协议测试

### 2.3 集成层与单元层职责混叠

候选文件:
- tests/integration/test_cli_entry_point.py

问题:
- test_service_creation 与 test_service_start_stop 更像单元测试
- 与 tests/unit/application/test_factories_and_service.py 覆盖主题重叠
- 使用 sleep(0.5) 有潜在不稳定性和耗时问题

建议:
- 保留真正 CLI 入口编排测试
- 将 service 层逻辑下沉到 unit
- 去掉定时等待，改为可控同步手段

## 3. 可合并测试清单

### 3.1 Lekai 参数校验测试合并

涉及文件:
- tests/unit/application/test_factories_and_service.py
- tests/unit/infrastructure/inference/test_server_lekai.py

建议:
- 对 interval 和 length 的 multiple-of-4 校验改为参数化用例
- 保留两层测试分工:
  - factory 层: 配置构建与异常信息
  - server 层: HTTP 状态码与响应 detail

### 3.2 CLI config parser 测试合并

涉及文件:
- tests/unit/presentation/test_cli_config_parser.py

问题:
- 每个测试重复构造大段 argparse.Namespace

建议:
- 提取 make_args 基础工厂
- 用参数化覆盖输入模式、输出类型、推理类型
- 减少重复字段，提升可读性

### 3.3 output log detail 测试合并

涉及文件:
- tests/unit/application/test_factories_and_service.py
- tests/unit/infrastructure/output/test_output_sinks.py

建议:
- 构建统一的 log detail 测试矩阵
- 在 output 层专注 sink 行为
- 在 factory 层专注参数透传，不重复检查落盘细节

## 4. 关键测试缺口

当前不是过时问题，而是对近期功能改动的覆盖不足。建议优先补齐。

### 4.1 Lekai 调试修复相关缺口

建议新增测试:
- serialization 层: event_from_dict 对 velocity/channel/program 为 None 的处理
- service 层: 增量发送逻辑，验证 last_sent_index 行为
- lekai backend 层:
  - history extend 不替换
  - generation_length_frames 生效
  - trim_histories 裁剪行为
  - note_off velocity 必为 0

### 4.2 配置链路缺口

建议新增测试:
- checkpoint_path 从 CLI config 到 HTTP payload 的透传
- model_name/inference_mode 在 HTTP payload 的一致性

## 5. 分阶段清理计划

### Phase A: 低风险整理

目标:
- 清理重复和低价值测试，不改变现有行为覆盖

任务:
- 合并协议存在性测试
- 参数化 Lekai multiple-of-4 用例
- 参数化 CLI config parser 用例

验收:
- 测试总数可适度下降
- 可读性提升
- 全量 tests 仍全绿

### Phase B: 覆盖增强

目标:
- 对近期核心改动补上行为回归保护

任务:
- 新增 serialization None 防御测试
- 新增增量发送与历史裁剪测试
- 新增 backend 生成长度与 note_off 语义测试

验收:
- debug 相关行为都有直接测试
- 失败时可定位到具体模块

### Phase C: 集成层减噪

目标:
- 让 integration 测试只做系统级契约验证

任务:
- 重构 tests/integration/test_cli_entry_point.py
- 删除与 unit 重叠的 service 内部行为检查

验收:
- integration 测试更少但更稳定
- 执行时间下降

## 6. 建议的执行顺序

1. 先做 Phase A，快速降低维护成本
2. 再做 Phase B，补齐真正缺口
3. 最后做 Phase C，收敛测试层级边界

## 7. DoD

- 结构上:
  - unit 和 integration 职责清晰
  - 重复测试明显减少
- 质量上:
  - 关键路径都有行为级测试
  - 无仅依赖 hasattr/not None 的冗余核心测试
- 运行上:
  - uv run pytest tests/ 全绿
  - collect-only 结构清晰、命名一致

## 8. Detailed Todo List (Execution Checklist)

> 说明: 以下为执行清单，默认按 Phase A -> B -> C 顺序推进。

### Phase 0 — Baseline 锁定

- [x] CT0-1 记录当前基线统计（112 tests / 22 files）
- [x] CT0-2 保存 `collect-only` 输出作为对比快照
- [x] CT0-3 跑一次 `uv run pytest tests/` 确认基线全绿

**Exit criteria**
- [x] CT0-E1 有可对比的“改动前”测试清单与通过记录

### Phase A — 低风险清理与合并

#### A1. 协议存在性测试合并

- [x] CTA1-1 新建参数化协议一致性测试（覆盖 Keyboard/List/MidiDevice）
- [x] CTA1-2 删除各文件中重复 `has read_events/close` 的断言
- [x] CTA1-3 保留每个输入源的行为测试（不删行为用例）

#### A2. Lekai multiple-of-4 测试参数化

- [x] CTA2-1 在 factory 测试中将 interval/length 校验改为参数化
- [x] CTA2-2 在 server 测试中将 HTTP 422 场景改为参数化
- [x] CTA2-3 保留非-lekai 放行场景（避免误伤）

#### A3. CLI config parser 去重复

- [x] CTA3-1 提取 `make_args()` 基础工厂
- [x] CTA3-2 将输入模式/输出类型/推理类型改为参数化场景
- [x] CTA3-3 删除重复 Namespace 大段构造

#### A4. 输出日志细节测试职责收敛

- [x] CTA4-1 factory 测试仅断言参数透传
- [x] CTA4-2 output sink 测试仅断言落盘与行为
- [x] CTA4-3 去除跨层重复断言

**Exit criteria**
- [x] CTA-E1 用例结构去重完成，且无行为覆盖下降
- [x] CTA-E2 `uv run pytest tests/` 全绿
- [x] CTA-E3 `collect-only` 结构更清晰、重复描述减少

### Phase B — 补关键缺口（优先）

#### B1. Serialization None 防御回归

- [x] CTB1-1 新增 `event_from_dict` 对 `velocity=None` 的测试
- [x] CTB1-2 新增 `channel=None` / `program=None` 的测试
- [x] CTB1-3 覆盖缺省字段与 `None` 共存边界

#### B2. Service 增量发送回归

- [x] CTB2-1 新增 `_last_sent_index` 首轮发送测试
- [x] CTB2-2 新增多轮发送只发送增量测试
- [x] CTB2-3 新增空增量不触发请求测试

#### B3. Lekai backend 行为回归

- [x] CTB3-1 新增 history extend-not-replace 测试
- [x] CTB3-2 新增 generation_length_frames 生效测试
- [x] CTB3-3 新增 `_trim_histories` 裁剪窗口测试
- [x] CTB3-4 新增 note_off velocity=0 语义测试

#### B4. 配置透传链路回归

- [x] CTB4-1 新增 checkpoint_path 从 CLI config 到 HTTP payload 的测试
- [x] CTB4-2 新增 model_name/inference_mode payload 一致性测试

**Exit criteria**
- [x] CTB-E1 debug 相关改动均有直接、可定位的回归测试
- [x] CTB-E2 关键路径失败可在单文件内快速定位

### Phase C — Integration 层减噪

#### C1. 重新界定 integration 边界

- [x] CTC1-1 在 `test_cli_entry_point.py` 仅保留 CLI 编排/流程契约
- [x] CTC1-2 将 service 内部行为断言迁移到 unit 层
- [x] CTC1-3 移除 `sleep(0.5)` 等不稳定等待

#### C2. 稳定性与速度优化

- [x] CTC2-1 使用可控 mock/同步点替代时间等待
- [x] CTC2-2 验证 integration 用例执行时间下降
- [x] CTC2-3 验证 integration 用例波动降低

**Exit criteria**
- [x] CTC-E1 integration 数量更少但信号更强
- [x] CTC-E2 无 unit/integration 职责重叠

### Phase D — Final Validation

- [x] CTD-1 跑 `uv run pytest tests/`
- [x] CTD-2 跑 `uv run pytest tests/ --collect-only -q`
- [x] CTD-3 对比变更前后测试文件数与用例数（112 -> 124，22 -> 26）
- [x] CTD-4 更新本计划中“当前测试现状”统计

**Exit criteria**
- [x] CTD-E1 满足本文件第 7 节 DoD
