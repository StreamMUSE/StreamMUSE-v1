# Test 整理执行 Report（2026-06-12）

对应计划：`developing-logs/plans/2026-06-11-test-cleanup-plan.md`（含执行 Todo List）。

## 结果总览

| 指标 | 整理前 | 整理后 |
|---|---|---|
| 测试文件数 | 35 | 28（-10 删除，+3 新建） |
| 测试数 | 190 passed | 173 passed |
| 运行时间 | ~7s | ~5.5s |

> 注：计划里写的"24 个文件"是当时的统计口径偏小，实际 `git ls-files` 为 35。

每一步之后都跑了 `uv run pytest tests/ -q`，全程绿色；最终 173 passed。删掉的 17 个测试全部是重复覆盖、测死代码、或纯样板（hasattr / import / dataclass 赋值断言），**没有删除任何独立行为覆盖**。

## 决策落地（用户选择方案 ③）

- **删除 `MusicalSequence`**：`src/streammuse/domain/musical/sequence.py` 删除，`domain/musical/__init__.py` 移除 export，`tests/unit/domain/musical/test_sequence.py` 删除（6 个测试）。全仓 grep 确认无残留引用。
- **保留 `domain/events/generic.py`** 及其测试，但按计划删掉了其中两个纯 dataclass 赋值断言测试（`test_text_chunk_payload`、`test_audio_frame_payload`）。

## 删除的文件（10 个）

| 文件 | 原因 |
|---|---|
| `tests/unit/domain/musical/test_sequence.py` | 测死代码（源码一并删除） |
| `tests/integration/test_lekai_runtime_info_integration.py` | `test_server_lekai.py::test_runtime_info_contract` 的严格子集 |
| `tests/unit/infrastructure/test_config_imports.py` | 目录搬迁遗留的 import 烟雾测试 |
| `tests/unit/infrastructure/test_tokenization_imports.py` | 同上 |
| `tests/unit/infrastructure/input/test_input_source_protocol.py` | 仅 hasattr 检查，行为测试已隐含覆盖 |
| `tests/unit/infrastructure/input/test_midi_device.py` | 并入 `test_simple_inputs.py` |
| `tests/unit/infrastructure/input/test_list_input.py` | 并入 `test_simple_inputs.py` |
| `tests/unit/application/test_factories_and_service.py` | 大杂烩，按内容拆到三处（见下） |
| `tests/unit/application/test_real_time_music_service_incremental.py` | 改名/并入 `test_real_time_music_service.py` |
| `tests/unit/application/test_real_time_music_service_logging.py` | 并入 `test_real_time_music_service.py` |

## 新建的文件（3 个 + 3 个辅助）

- **`tests/unit/application/test_real_time_music_service.py`**：原 incremental 文件的 10 个测试（去掉无信息量的 "incremental" 命名）+ 原 factories_and_service 的 service 线程冒烟测试 + 原 logging 文件的 2 个 payload 测试，分节组织。
- **`tests/unit/application/factories/test_inference_factory.py`**：从 factories_and_service 拆出的 4 个 inference factory 测试（lekai 长度校验、非 lekai 跳过校验、checkpoint_path 透传、基础组件冒烟）。
- **`tests/unit/infrastructure/input/test_simple_inputs.py`**：合并 ListInput（2 个）+ MidiDeviceInput（1 个）测试。
- **`tests/unit/application/fakes.py`**：共享的 `NoopInput` / `NoopOutput` / `NoopInference` / `note()`，消除了原本 3 份重复的 service fakes。
- **`tests/unit/application/__init__.py`、`factories/__init__.py`**：补齐包结构使 `from tests.unit.application.fakes import ...` 可用（与 tests 下其他目录的 `__init__.py` 惯例一致）。

## 修改的文件（7 个）

- `tests/unit/application/factories/test_output_factory.py`：移入 `test_output_factory_propagates_inference_log_detail`（参数化 json_log/composite）。
- `tests/unit/domain/events/test_generic_events.py`：删 2 个纯赋值断言测试。
- `tests/unit/infrastructure/inference/test_lekai_inference_adapter.py`：删 `test_tokenizer_roundtrip_preserves_time_length`（shape-only，弱于 lekai_model 的 array-equality 版本）和 `test_beats_to_pianoroll_handles_empty_beats_without_crashing`（移入 lekai_model 文件）。
- `tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py`：并入混合空 beat 场景 `[[255], [250, 171], []]`，删除被其覆盖的 `[[255]]`、`[[]]` 两条细粒度测试（保留 `[[169]]` 和空 list）。
- `tests/unit/domain/musical/test_events.py`：6 个 invalid-field 测试 → 1 个参数化测试（6 cases）。
- `tests/unit/domain/musical/test_notes.py`：4 个 invalid-field 测试 → 1 个参数化测试（4 cases）。
- `tests/unit/domain/timing/test_tempo.py`：3 个 invalid 测试 → 1 个参数化测试（4 cases，原 bpm 测试里 0/-1 两个断言拆成独立 case，**净增 1 个测试**）。
- `src/streammuse/domain/musical/__init__.py`：移除 `MusicalSequence` import/export。

## 与计划的两处偏差

1. **共享 fakes 用了 `fakes.py` 模块而非 `conftest.py`**：service 测试大量以 `class _MetronomeOutput(NoopOutput)` 的方式子类化 fakes，conftest 的 fixture 机制无法干净地导出类供子类化，普通模块 import 更直接。
2. **`integration/test_cli_entry_point.py` 保留本地 mocks**：其 mock 带完整协议签名和 docstring，且测试中用 MagicMock 动态替换方法，与 unit 侧的极简 fakes 形态不同；跨 unit→integration 共享会引入不必要耦合，按计划预留的判断分支选择不动。

## 保持不动（按计划）

- client 侧与 server 侧的 "multiple of 4" 校验测试两份都保留（守护协议两端）。
- `test_output_sinks.py`、`test_lekai_http_backend.py`、两个真 integration 测试、其余 domain/infrastructure 测试。

## 遗留事项（计划附录中的覆盖缺口，本次未处理）

- `ConsoleOutputSink`、`AudioOutputSink` 无测试。
- `stanley_legacy.py`、`generation_logger.py` 无单测。
- `MetricsCalculator` 仅被间接覆盖。
