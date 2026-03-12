---
title: LegacyInferenceEngineStanley — ML 引擎层
description: 基于 RoFormerSymbolicTransformer 的 duration-note 推理引擎
---

# LegacyInferenceEngineStanley — ML 引擎层

**源文件**：`src/streammuse/infrastructure/inference/stanley_legacy.py`

直接操作 RoFormer 模型的推理引擎，使用 duration-note 字典和 piano-roll 张量，不依赖 Domain 层。

---

## 初始化

```python
class LegacyInferenceEngineStanley:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        model_size: str,
        generation_length_frames: int,
        max_polyphony: int = 4,
        model_max_seq_len_frames: int = 96,
    ) -> None:
```

初始化时：
1. 验证 `checkpoint_path` 存在，不存在则抛出 `FileNotFoundError`
2. 调用 `RoFormerSymbolicTransformer.load_from_checkpoint()` 加载模型
3. 若 CUDA 可用，将模型移至 `"cuda:0"`，否则在 CPU 上运行
4. 设置 `model.eval()` 模式（关闭 dropout 等训练专用层）
5. 初始化历史记录：`melody_history = []`，`accompaniment_history = []`

### 关键内部状态

| 属性 | 说明 |
|---|---|
| `model` | `RoFormerSymbolicTransformer` 实例 |
| `model_max_seq_len_frames` | context window 帧数（如 96） |
| `prompt_length_ticks` | `model_max_seq_len_frames // 2`（如 48 ticks） |
| `generation_length_frames` | 每次生成帧数（如 20） |
| `max_polyphony` | 最大和弦大小（如 4） |
| `melody_history` | 历史旋律 note dicts |
| `accompaniment_history` | 历史伴奏 note dicts |
| `injection_offset_ticks` | 注入偏移（ticks） |

---

## 数据格式

### Note 字典格式

```python
{"pitch": 60, "tick": 12, "duration": 4, "velocity": 80}
```

`duration` 量化为 `DURATION_TEMPLATES` 中的离散值。

### Piano-roll 张量格式

形状：`(max_tick, max_polyphony, 3)`，各维含义：

| 维度 | 含义 |
|---|---|
| `[tick, slot, 0]` | program（乐器编号，254=空位，255=填充） |
| `[tick, slot, 1]` | pitch（MIDI 音高 0–127） |
| `[tick, slot, 2]` | duration_idx（从 `DURATION_TEMPLATES` 量化的索引） |

EOS_TOKEN 和 PAD_TOKEN 为特殊占位符。

---

## 核心方法

### `generate_accompaniment(melody_notes, generation_start_tick, acc_notes=None, ...) -> tuple`

1. 将 `melody_notes` + 历史数据通过 `_notes_to_rolls()` 转为 piano-roll 张量
2. 将张量输入 `model.generate()`（自回归生成）
3. 通过 `_tensors_to_notes()` 将输出张量转回 note 字典
4. 更新 `melody_history` 和 `accompaniment_history`
5. 返回 `(acc_notes, preprocess_start, inf_start, inf_end, post_start)`（4 个时间戳用于性能分析）

### `_notes_to_rolls(notes, max_tick, max_polyphony, program) -> torch.Tensor`

将 note 字典列表转换为形状 `(max_tick, max_polyphony * 3)` 的张量（reshape 后）。同一 tick 内的多个音符按音高排序，超出 `max_polyphony` 的音符被丢弃。

### `_tensors_to_notes(output_tensors, generation_start_tick) -> list[list[dict]]`

将模型输出张量列表解码为 note 字典列表。跳过 EOS/PAD 标记，将 `pitch_duration` 编码拆分为 pitch（低 7 位）和 duration_idx（高位），通过 `DURATION_TEMPLATES` 还原为实际 duration。

### `clear_history() -> None`

清空 `melody_history` 和 `accompaniment_history`。

### `set_injection_offset(offset_ticks: int) -> None`

设置注入偏移，供 CLI `--injection-file` 功能使用。
