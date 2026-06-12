# Test 整理计划（2026-06-11）

## 现状

- `tests/` 共 24 个测试文件（不含 `__init__.py`），190 个测试，全部通过，约 7 秒跑完。
- 整体质量不错：绝大多数测试是行为测试、有真实断言。问题主要是**历史遗留的重复、测试死代码、文件组织混乱**三类。

## 一、建议直接删除的测试

### 1. 测试死代码的文件（需要先决策：代码本身留不留）

| 文件 | 测试对象 | 问题 |
|---|---|---|
| `tests/unit/domain/events/test_generic_events.py`（144 行，8 个测试） | `domain/events/generic.py` 的 `Event`/`EventKind`/`TextChunkPayload`/`AudioFramePayload` 及 generic 转换器 | 已 grep 确认：生产代码中**没有任何地方**使用 generic Event 模型（converters.py 里的两个转换函数也只被测试调用） |
| `tests/unit/domain/musical/test_sequence.py`（83 行，6 个测试） | `MusicalSequence` | 同样只在 `__init__.py` 里 export，生产代码无任何调用 |

**决策点**：这两块是"为未来抽象预留"还是真死代码？
- 如果是死代码 → 删代码 + 删测试（推荐，~227 行测试 + 对应源码）。
- 如果要保留代码 → 至少删掉纯构造函数断言类测试（如 `test_text_chunk_payload`、`test_audio_frame_payload`，只是验证 dataclass 赋值）。

### 2. 重复 / 近零价值的测试

| 文件 | 处理 | 理由 |
|---|---|---|
| `tests/integration/test_lekai_runtime_info_integration.py`（16 行） | **删除** | 是 `tests/unit/.../test_server_lekai.py::test_runtime_info_contract` 的严格子集（unit 版检查的 key 更多）。且两者都用同一个 in-process `TestClient`，它根本不算 integration 测试 |
| `tests/unit/infrastructure/test_config_imports.py` | **删除或合并** | 纯 import 烟雾测试，是当年 `schema/` 目录搬迁时写的回归保护，搬迁已完成。其中 `test_model_schema_uses_transformers_roformer` 只断言 `hasattr(vocab_size)` |
| `tests/unit/infrastructure/test_tokenization_imports.py` | **删除或合并** | 同上，`tokenizers/` 搬迁遗留。如果想保留 import 保护，把两个文件合成一个 `test_package_imports.py`（一个参数化测试） |
| `tests/unit/infrastructure/input/test_input_source_protocol.py` | **删除** | 只做 `hasattr(read_events)` / `callable` 检查，每个 adapter 都已有真实行为测试，行为测试本身就会在协议不符时失败 |

### 3. Lekai tokenizer/adapter 三个文件中的重复测试

`test_lekai_inference_adapter.py`（inference/ 下）、`lekai_model/test_inference_adapter.py`、`lekai_model/test_tokenizer.py` 三个文件**各自复制了一份相同的 `_build_tokenizer()` helper**，且互有覆盖重叠：

- `test_lekai_inference_adapter.py::test_tokenizer_roundtrip_preserves_time_length` **删除**：只断言 shape，弱于 `lekai_model/test_tokenizer.py` 里检查 `np.array_equal` 的同场景 roundtrip 测试。
- `test_lekai_inference_adapter.py::test_beats_to_pianoroll_handles_empty_beats_without_crashing` 与 `lekai_model/test_inference_adapter.py` 里 4 个细粒度空 beat 测试（`[]`、`[[]]`、`[[255]]`、`[[169]]`）重叠，**合并**：把混合场景 `[[255], [250, 171], []]` 这一条搬进 `lekai_model/test_inference_adapter.py`，删掉细粒度中被覆盖的（`[[255]]`、`[[]]` 两条可由混合场景代替，保留 `[[169]]` 和空 list 两条）。

## 二、建议合并 / 重组的文件

### 1. 拆掉大杂烩 `tests/unit/application/test_factories_and_service.py`

这个文件混了三类东西，且和已存在的 `factories/test_output_factory.py` 目录结构冲突：

| 现内容 | 去向 |
|---|---|
| `test_output_factory_propagates_inference_log_detail` | 移入 `factories/test_output_factory.py` |
| `test_http_lekai_validates_generation_length_only`、`test_http_non_lekai_skips_multiple_of_4_checks`、`test_http_factory_propagates_checkpoint_path_for_http_client`、`test_factories_create_basic_components` | 新建 `factories/test_inference_factory.py` |
| `test_real_time_music_service_emits_ticks_and_user_events`（唯一真正起线程的 service 冒烟测试） | 移入 `test_real_time_music_service_incremental.py`（并把后者改名，见下） |

完成后删除原文件。

> 注：`test_http_lekai_validates...`（client 侧）与 `test_server_lekai.py::test_lekai_validates...`（server 侧）看似重复，但分别守护协议两端，**有意保留两份**。

### 2. RealTimeMusicService 测试整合

- `test_real_time_music_service_incremental.py`（347 行）和 `test_real_time_music_service_logging.py`（131 行）各自定义了几乎相同的 `_NoopInput` / `_NoopOutput` / `_NoopInference` fakes，`test_factories_and_service.py` 和 `integration/test_cli_entry_point.py` 里还有第三、四份。
  → 在 `tests/unit/application/conftest.py`（或 `tests/helpers.py`）里提取共享 fakes。
- "incremental" 这个名字是当年增量重构的阶段命名，已无信息量 → 改名为 `test_real_time_music_service.py`。logging 那个文件只有 2 个测试，可以直接并进来（用 `# --- logging payload ---` 分节），也可以保留独立文件，倾向并入。

### 3. 输入 adapter 小文件合并

- `test_midi_device.py` 只有 1 个测试（14 行），`test_list_input.py` 只有 2 个（29 行）。
  → 合并成一个 `test_simple_inputs.py`（或把 midi_device 那条并进未来真正的 midi_device 测试文件）。`test_keyboard_input.py` 和 `test_midi_file_input.py` 内容充实，保持独立。

### 4. （可选，低优先级）参数化收紧

- `test_events.py` 的 6 个 invalid-field 测试、`test_notes.py` 的 5 个 invalid-field 测试，可各自收成一个 `@pytest.mark.parametrize` 测试。不减覆盖，纯粹减少样板。
- `test_tempo.py` 的 3 个 invalid 测试同理。

## 三、保持不动的部分

- `test_output_sinks.py`（282 行）：测试多但各管一个行为（metronome 录制、count-in、composite fan-out、log detail 协商），无重复。
- `test_lekai_http_backend.py`（291 行）：覆盖 fallback、裁剪、长度上限等关键路径，质量高。
- `integration/test_simulator_midi_output.py`：仓库里唯一真正端到端（service 线程 → factory sink → 落盘 MIDI 再读回验证）的测试，保留。
- `integration/test_cli_entry_point.py`：虽然 mock 很重（更像 unit），但守护了 injection 顺序（clear → inject → create input）这一关键启动时序，保留；只共享 fakes，不动逻辑。
- web viewer、session manager、prompt repository、runtime_device、serialization、converters、scheduler、tempo、cli_config_parser 等测试均合理。

## 四、执行顺序

1. **先问清决策点**：`domain/events/generic.py` 和 `MusicalSequence` 是否删除（一并删源码）。
2. 删除第一节里的 4 个低价值文件 + lekai 重复测试。
3. 拆 `test_factories_and_service.py`，建 `factories/test_inference_factory.py`。
4. 提取共享 fakes 到 `tests/unit/application/conftest.py`，合并/改名 service 测试文件。
5. 合并输入 adapter 小文件。
6. （可选）参数化收紧 events/notes/tempo 的 invalid 测试。
7. 每步之后 `uv run pytest tests/ -q` 验证；最终确认测试数下降但覆盖场景不变。

## 预期效果

- 文件数：24 → 约 17。
- 测试数：190 → 约 165–175（删掉的全是重复或测死代码的）。
- 消除 4 份重复的 service fakes、3 份重复的 `_build_tokenizer`。
- `tests/unit/application/` 下 factory 测试归位到 `factories/` 子目录，结构与 src 对齐。

## 附：顺手发现的覆盖缺口（本次不处理，仅记录)

- `ConsoleOutputSink`、`AudioOutputSink` 完全没有测试（输出 sink 里只有这两个裸奔）。
- `stanley_legacy.py`（真正的模型 wrapper）、`generation_logger.py` 无单测。
- `MetricsCalculator` 只通过 session logger 间接覆盖，没有直接单测。
