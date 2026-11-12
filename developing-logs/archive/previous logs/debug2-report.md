# Debug2 Report

## 1. 执行信息

- 执行日期: 2026-04-02
- 对应计划: informal-docs/debug2-plan.md
- 执行范围: Phase 1 ~ Phase 4
- 执行结果: 已完成

## 2. 结果总览

本次修复完成了 debug2 的核心目标：

1. 修正 Lekai 时长计算逻辑，彻底解耦 `generation_interval_ticks` 与输出长度。
2. 打通 `LEKAI_CHECKPOINT_PATH` 到 server backend 的加载链路。
3. 加固 `beats_to_pianoroll` 的空输入与异常路径处理。
4. 将 shape mismatch 从 warning 提升为 `RuntimeError`（fail-fast）。
5. 同步更新 client/server 校验语义与文档说明。
6. 补齐 tokenizer / inference_adapter / backend 的新增测试。

## 3. 代码改动

### 3.1 核心逻辑修复

- `src/streammuse/infrastructure/inference/lekai_http_backend.py`
  - 新增常量 `TIMESTEPS_PER_BEAT = 4`。
  - `_generate_with_model` 改为按 `generation_length_frames // 4` 计算 beats，不再依赖 interval。
  - 对非 4 倍数长度执行向上取整并打印 warning。
  - `_generate_rule_based` 使用固定 `chord_duration_ticks = 4`，确保输出长度由 `generation_length_frames` 决定。
  - `part1_pianoroll` shape 不匹配时改为 `RuntimeError`。
  - 模型路径默认 BPM 改为环境变量 `LEKAI_DEFAULT_BPM` 可覆盖。

- `src/streammuse/infrastructure/inference/server_lekai.py`
  - `_validate_lekai_constraints` 仅校验 `generation_length_frames % 4 == 0`。
  - 删除 `generation_interval_ticks` 的 4 倍数校验。
  - 模块启动时读取 `LEKAI_CHECKPOINT_PATH` 并传入 `LekaiHttpBackend`。
  - `main()` 输出 checkpoint 与当前模式（real model / rule-based stub）。

- `src/streammuse/application/factories/inference_factory.py`
  - Lekai 分支只校验 `generation_length_frames`，移除 interval 校验。

### 3.2 解码鲁棒性修复

- `src/streammuse/infrastructure/inference/lekai_model/inference_adapter.py`
  - `beats_to_pianoroll` 增加：
    - 空输入 `[]` 返回 `(2, 88, 0)`。
    - 空内部 beat `[]` 安全处理为全零 beat。
    - `bar token [255]`、`empty marker [169]` 的显式处理。
    - 单 beat 解码异常捕获，降级为空 beat，避免整体崩溃。
    - `_adjust_timesteps` 对齐时长。

### 3.3 文档同步

- `docs/reference/cli-reference.md`
  - Lekai 约束更新为仅 `generation-length-frames` 需为 4 的倍数。

- `docs/getting-started/configuration.md`
  - 注意事项改为仅 length 受约束，interval 不受约束。

- `docs/user-guide/running-realtime.md`
  - Lekai 使用说明更新为仅 length 约束。

## 4. 测试补充与更新

### 4.1 新增测试文件

- `tests/unit/infrastructure/inference/lekai_model/test_tokenizer.py`
  - `encode/decode` roundtrip（空输入、有音符、全一、单 patch）。
  - `compress/decompress` roundtrip。

- `tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py`
  - `beats_to_pianoroll` 空列表、空内部列表、bar token、empty marker、真实 beat roundtrip。

### 4.2 更新测试文件

- `tests/unit/infrastructure/inference/test_lekai_http_backend.py`
  - 覆盖 interval=2/4/8 下输出长度恒定由 `generation_length_frames` 决定。
  - 覆盖空 melody 返回空伴奏。

- `tests/unit/infrastructure/inference/test_server_lekai.py`
  - 校验改为 length-only：`interval=3,length=20` 通过，`length=17` 返回 422。

- `tests/unit/application/test_factories_and_service.py`
  - 校验改为 length-only：`interval=3,length=16` 通过，`length=17` 抛错。

- `tests/unit/infrastructure/inference/test_lekai_inference_adapter.py`
  - 增补/对齐 adapter 路径回归测试。

## 5. 验证结果

### 5.1 自动化测试

- `uv run pytest tests/unit/infrastructure/inference/lekai_model/test_tokenizer.py tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py tests/unit/infrastructure/inference/test_lekai_http_backend.py tests/unit/infrastructure/inference/test_server_lekai.py tests/unit/application/test_factories_and_service.py -q`
  - 结果: `30 passed`

- `uv run pytest tests/ -k "stanley" -v`
  - 结果: `3 passed, 137 deselected`

- `uv run pytest tests/ --collect-only`
  - 结果: `140 tests collected`

- `uv run pytest tests/ -q`
  - 结果: `140 passed, 1 warning`

### 5.2 端到端验证（stub 模式）

- 启动 server（无 checkpoint）日志确认:
  - `LEKAI_CHECKPOINT_PATH not set`
  - `Inference mode: rule-based stub`

- HTTP 参数组合验证:
  - `interval=4, length=16` -> `200`
  - `interval=8, length=16` -> `200`
  - `interval=3, length=16` -> `200`
  - `interval=8, length=20` -> `200`
  - `interval=4, length=17` -> `422`

- CLI 参数组合验证（`streammuse-cli` + `input-mode midi_file`）:
  - `interval=2, length=16`：正常启动并运行到 `--max-ticks` 结束
  - `interval=3, length=16`：正常启动并运行到 `--max-ticks` 结束（证明 interval 不受 4 倍数约束）
  - `interval=4, length=17`：启动即抛 `ValueError`（length 必须是 4 的倍数）

- 响应 tick 上界验证:
  - `interval=4, length=16` 最大 tick = `24`（从 start=8 生成 16 ticks）
  - `interval=8, length=16` 最大 tick = `24`（从 start=8 生成 16 ticks）
  - `interval=3, length=16` 最大 tick = `24`（从 start=8 生成 16 ticks）
  - `interval=8, length=20` 最大 tick = `28`（从 start=8 生成 20 ticks）

- 无效 checkpoint 验证:
  - `LEKAI_CHECKPOINT_PATH=/tmp/not_exists_lekai.ckpt` 启动时打印
    - `Checkpoint not found ... using rule-based stub`
    - `Inference mode: rule-based stub`

## 6. 与计划对照

- Phase 1: 已完成（时长逻辑、校验简化、checkpoint 链路、文档同步）。
- Phase 2: 已完成（adapter 鲁棒性 + shape fail-fast）。
- Phase 3: 已完成（新增 tokenizer 与 adapter 测试，backend/校验测试补齐）。
- Phase 4: 已完成（Stanley 回归、Lekai e2e 参数组合验证、debug2 报告）。

## 7. 已知限制

1. 真实模型端到端音频质量未在本次报告中评估（当前主要验证逻辑正确性与链路稳定性）。
2. `pretty_midi` 依赖在测试中仍有 deprecation warning（不影响功能，但建议后续处理依赖版本）。
